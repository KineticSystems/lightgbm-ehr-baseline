//! Bounded row-group streaming interface for very large rolling-landmark tasks.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use arrow_array::{
    Array, Float32Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::statistics::Statistics;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::Deserialize;

use super::{
    Event, HOUR_US, SegmentKind, column, numeric_statistics, occurrence_statistics,
    segment_bounds, statistic_widths, string_at,
};

const DEFAULT_BATCH_SIZE: usize = 65_536;

pub(crate) type RowConsumer<'a> = &'a mut dyn FnMut(&[u64], &[i64], &[f32]) -> Result<()>;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Representation {
    Summary,
    Temporal,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StreamConfig {
    pub(crate) events_path: PathBuf,
    pub(crate) index_path: PathBuf,
    pub(crate) split: String,
    pub(crate) representation: Representation,
    pub(crate) codes: Vec<String>,
    #[serde(default)]
    pub(crate) occurrence_codes: Vec<String>,
    #[serde(default)]
    pub(crate) ignored_codes: Vec<String>,
    pub(crate) observation_window_hours: f64,
    #[serde(default)]
    pub(crate) segment_kind: SegmentKind,
    /// Cohort-wide split for `VisitHalves`, in hours before prediction time.
    #[serde(default)]
    pub(crate) halves_split_hours: f64,
    /// A profile whose events carry no real elapsed time drops the two
    /// prediction-relative timing statistics and the observation width.
    #[serde(default = "default_true")]
    pub(crate) timed: bool,
    pub(crate) max_component_rows: u64,
    pub(crate) max_chunk_bytes: usize,
    #[serde(default = "default_batch_size")]
    pub(crate) record_batch_rows: usize,
}

fn default_batch_size() -> usize {
    DEFAULT_BATCH_SIZE
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Copy, Debug)]
struct StreamSample {
    sample_id: i64,
    observation_start_us: i64,
    prediction_time_us: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RowGroupInterval {
    index: usize,
    minimum: i64,
    maximum: i64,
    rows: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Component {
    minimum: i64,
    maximum: i64,
    row_groups: Vec<usize>,
    rows: u64,
}

#[derive(Debug)]
struct FeatureLayout {
    numeric_global: Vec<u32>,
    numeric_local: HashMap<u32, usize>,
    occurrence_local: HashMap<u32, usize>,
    occurrence_set: HashSet<u32>,
    ignored_codes: HashSet<String>,
    code_lookup: HashMap<String, u32>,
    aggregate_columns: usize,
    summary_width: usize,
    occurrence_width: usize,
    numeric_summary_columns: usize,
    numeric_temporal_columns: usize,
    feature_count: usize,
    segment_count: usize,
}

type SampleGroups = HashMap<u32, Vec<Event>>;
type ComponentGroups = HashMap<i64, SampleGroups>;

impl FeatureLayout {
    fn new(config: &StreamConfig) -> Result<Self> {
        validate_codes(config)?;
        let occurrence_names: HashSet<&str> =
            config.occurrence_codes.iter().map(String::as_str).collect();
        let numeric_global: Vec<u32> = config
            .codes
            .iter()
            .enumerate()
            .filter(|(_, code)| !occurrence_names.contains(code.as_str()))
            .map(|(index, _)| index as u32)
            .collect();
        let occurrence_global: Vec<u32> = config
            .codes
            .iter()
            .enumerate()
            .filter(|(_, code)| occurrence_names.contains(code.as_str()))
            .map(|(index, _)| index as u32)
            .collect();
        let numeric_local = numeric_global
            .iter()
            .copied()
            .enumerate()
            .map(|(local, global)| (global, local))
            .collect();
        let occurrence_local = occurrence_global
            .iter()
            .copied()
            .enumerate()
            .map(|(local, global)| (global, local))
            .collect();
        let occurrence_set = occurrence_global.iter().copied().collect();
        let ignored_codes = config.ignored_codes.iter().cloned().collect();
        let code_lookup = config
            .codes
            .iter()
            .enumerate()
            .map(|(index, code)| (code.clone(), index as u32))
            .collect();
        let numeric_count = numeric_global.len();
        let occurrence_count = occurrence_global.len();
        let (summary_width, occurrence_width, aggregate_columns) =
            statistic_widths(config.timed);
        let segment_count = config.segment_kind.segment_count();
        let numeric_summary_columns = aggregate_columns
            + checked_mul(numeric_count, summary_width, "summary width")?;
        let numeric_temporal_columns = aggregate_columns
            + checked_mul(
                checked_mul(
                    numeric_count,
                    segment_count,
                    "numeric temporal segment width",
                )?,
                summary_width,
                "numeric temporal statistic width",
            )?;
        let occurrence_summary_columns = checked_mul(
            occurrence_count,
            occurrence_width,
            "occurrence summary width",
        )?;
        let occurrence_temporal_columns = checked_mul(
            checked_mul(
                occurrence_count,
                segment_count,
                "occurrence temporal segment width",
            )?,
            occurrence_width,
            "occurrence temporal statistic width",
        )?;
        let feature_count = match config.representation {
            Representation::Summary => numeric_summary_columns
                .checked_add(occurrence_summary_columns)
                .ok_or_else(|| anyhow!("summary width exceeds addressable memory"))?,
            Representation::Temporal => numeric_temporal_columns
                .checked_add(occurrence_temporal_columns)
                .ok_or_else(|| anyhow!("temporal width exceeds addressable memory"))?,
        };
        if feature_count == 0 {
            bail!("streamed representation has zero features")
        }
        Ok(Self {
            numeric_global,
            numeric_local,
            occurrence_local,
            occurrence_set,
            ignored_codes,
            code_lookup,
            aggregate_columns,
            summary_width,
            occurrence_width,
            numeric_summary_columns,
            numeric_temporal_columns,
            feature_count,
            segment_count,
        })
    }
}

fn checked_mul(left: usize, right: usize, description: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("{description} exceeds addressable memory"))
}

fn validate_codes(config: &StreamConfig) -> Result<()> {
    if config.codes.is_empty() {
        bail!("codes must be nonempty")
    }
    let codes: HashSet<&str> = config.codes.iter().map(String::as_str).collect();
    if codes.len() != config.codes.len() || codes.contains("") {
        bail!("codes must be unique and nonempty")
    }
    let occurrences: HashSet<&str> = config.occurrence_codes.iter().map(String::as_str).collect();
    if occurrences.len() != config.occurrence_codes.len() {
        bail!("occurrence_codes must be unique")
    }
    if !occurrences.is_subset(&codes) {
        bail!("occurrence_codes must be declared codes")
    }
    let ignored: HashSet<&str> = config.ignored_codes.iter().map(String::as_str).collect();
    if ignored.len() != config.ignored_codes.len() || ignored.contains("") {
        bail!("ignored_codes must be unique and nonempty")
    }
    if !ignored.is_disjoint(&codes) {
        bail!("ignored_codes must be disjoint from structured codes")
    }
    let maximum_hours = i64::MAX as f64 / 3_600_000_000.0;
    if !(config.observation_window_hours.is_finite()
        && config.observation_window_hours > 0.0
        && config.observation_window_hours <= maximum_hours)
    {
        bail!("observation_window_hours must be positive, finite, and representable")
    }
    if !config.halves_split_hours.is_finite() || config.halves_split_hours < 0.0 {
        bail!("halves_split_hours must be finite and nonnegative")
    }
    if config.max_component_rows == 0 {
        bail!("max_component_rows must be positive")
    }
    if config.max_chunk_bytes == 0 {
        bail!("max_chunk_bytes must be positive")
    }
    if config.record_batch_rows == 0 {
        bail!("record_batch_rows must be positive")
    }
    Ok(())
}

pub(crate) fn split_at(batch: &RecordBatch, column: usize, row: usize) -> Result<&str> {
    let array = batch.column(column);
    if array.is_null(row) {
        bail!("split cannot be null")
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(values.value(row));
    }
    bail!("split must be an Arrow string or large_string array")
}

fn load_split_index(config: &StreamConfig) -> Result<Vec<StreamSample>> {
    let file = File::open(&config.index_path)
        .with_context(|| format!("cannot open {}", config.index_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let has_split = builder.schema().index_of("split").is_ok();
    let wanted: &[&str] = if has_split {
        &["sample_id", "observation_start", "prediction_time", "split"]
    } else {
        &["sample_id", "observation_start", "prediction_time"]
    };
    let roots: Vec<usize> = wanted
        .iter()
        .map(|name| {
            builder
                .schema()
                .index_of(name)
                .with_context(|| format!("task index missing {name}"))
        })
        .collect::<Result<_>>()?;
    let projection = ProjectionMask::roots(builder.parquet_schema(), roots);
    let reader = builder
        .with_projection(projection)
        .with_batch_size(config.record_batch_rows)
        .build()?;
    let mut samples = Vec::new();
    for batch in reader {
        let batch = batch?;
        let sample_column = column(&batch, "sample_id")?;
        let start_column = column(&batch, "observation_start")?;
        let stop_column = column(&batch, "prediction_time")?;
        let split_column = if has_split { Some(column(&batch, "split")?) } else { None };
        let sample_values = batch
            .column(sample_column)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow!("task index sample_id must be int64"))?;
        let start_values = batch
            .column(start_column)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| anyhow!("task index observation_start must be timestamp[us]"))?;
        let stop_values = batch
            .column(stop_column)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| anyhow!("task index prediction_time must be timestamp[us]"))?;
        for row in 0..batch.num_rows() {
            if let Some(split_column) = split_column
                && split_at(&batch, split_column, row)? != config.split
            {
                continue;
            }
            if sample_values.is_null(row) || start_values.is_null(row) || stop_values.is_null(row) {
                bail!("selected task index identifiers and times cannot be null")
            }
            let start = start_values.value(row);
            let stop = stop_values.value(row);
            let width = stop
                .checked_sub(start)
                .ok_or_else(|| anyhow!("sample observation width overflowed int64"))?;
            let width_hours = width as f64 / 3_600_000_000.0;
            if !(width_hours > 0.0 && width_hours <= config.observation_window_hours) {
                bail!("sample observation window must be positive and no wider than task window")
            }
            samples.push(StreamSample {
                sample_id: sample_values.value(row),
                observation_start_us: start,
                prediction_time_us: stop,
            });
        }
    }
    if samples.is_empty() {
        bail!("selected split has no task index rows")
    }
    samples.sort_unstable_by_key(|sample| sample.sample_id);
    for pair in samples.windows(2) {
        if pair[0].sample_id == pair[1].sample_id {
            bail!("task index sample_id must be unique")
        }
    }
    if samples.len() > u32::MAX as usize {
        bail!("split row count exceeds LightGBM's sampled-row index limit")
    }
    Ok(samples)
}

fn parquet_leaf(metadata: &ParquetMetaData, name: &str) -> Result<usize> {
    metadata
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .position(|column| column.path().string() == name)
        .with_context(|| format!("events missing flat Parquet leaf {name}"))
}

fn exact_int64_interval(statistics: &Statistics, group: usize) -> Result<(i64, i64)> {
    if !statistics.min_is_exact() || !statistics.max_is_exact() {
        bail!("row group {group} sample_id statistics are not exact")
    }
    if statistics.null_count_opt() != Some(0) {
        bail!("row group {group} sample_id statistics have nulls or unknown null count")
    }
    let Statistics::Int64(values) = statistics else {
        bail!("row group {group} sample_id statistics are not int64")
    };
    let minimum = *values
        .min_opt()
        .with_context(|| format!("row group {group} sample_id minimum is missing"))?;
    let maximum = *values
        .max_opt()
        .with_context(|| format!("row group {group} sample_id maximum is missing"))?;
    if minimum > maximum {
        bail!("row group {group} sample_id statistics are reversed")
    }
    Ok((minimum, maximum))
}

fn split_can_match(statistics: &Statistics, split: &str, group: usize) -> Result<bool> {
    if !statistics.min_is_exact() || !statistics.max_is_exact() {
        // Older Parquet writers commonly omit the exactness flags for string
        // statistics even when they store complete values.  Inexact bounds
        // are unsafe for pruning, but they are not a correctness problem: the
        // row reader still checks every event's split value.  Conservatively
        // retain the row group instead of rejecting an otherwise valid task.
        return Ok(true);
    }
    if statistics.null_count_opt() != Some(0) {
        bail!("row group {group} split statistics have nulls or unknown null count")
    }
    let Statistics::ByteArray(values) = statistics else {
        bail!("row group {group} split statistics are not byte array")
    };
    let minimum = values
        .min_opt()
        .with_context(|| format!("row group {group} split minimum is missing"))?
        .data();
    let maximum = values
        .max_opt()
        .with_context(|| format!("row group {group} split maximum is missing"))?
        .data();
    let target = split.as_bytes();
    Ok(minimum <= target && target <= maximum)
}

fn plan_components(
    metadata: &ParquetMetaData,
    split: &str,
) -> Result<(Vec<Component>, usize)> {
    let sample_column = parquet_leaf(metadata, "sample_id")?;
    // Exports written by the shared writer always carry `split`; in-memory
    // indexes (tests, MedsTask.build_representations) may omit it, in which
    // case every row group is a candidate.
    let split_column = parquet_leaf(metadata, "split").ok();
    let mut intervals = Vec::new();
    for (index, group) in metadata.row_groups().iter().enumerate() {
        if let Some(split_column) = split_column {
            let split_statistics = group
                .column(split_column)
                .statistics()
                .with_context(|| format!("row group {index} split statistics are missing"))?;
            if !split_can_match(split_statistics, split, index)? {
                continue;
            }
        }
        let sample_statistics = group
            .column(sample_column)
            .statistics()
            .with_context(|| format!("row group {index} sample_id statistics are missing"))?;
        let (minimum, maximum) = exact_int64_interval(sample_statistics, index)?;
        intervals.push(RowGroupInterval {
            index,
            minimum,
            maximum,
            rows: u64::try_from(group.num_rows()).context("negative Parquet row count")?,
        });
    }
    let candidate_count = intervals.len();
    let components = merge_intervals(intervals)?;
    Ok((components, candidate_count))
}

// Components are true overlap closures of row-group sample intervals and may
// physically exceed `max_component_rows`; an oversized component is executed
// in several passes over disjoint slices of its selected samples, and the
// bound is enforced on the events each pass retains, never on the physical
// rows it scans. An export that interleaves samples across the whole file
// (REMed: 129.5M rows in one component) therefore streams in bounded memory
// instead of being rejected.
fn merge_intervals(mut intervals: Vec<RowGroupInterval>) -> Result<Vec<Component>> {
    intervals.sort_unstable_by_key(|interval| (interval.minimum, interval.maximum, interval.index));
    let mut components: Vec<Component> = Vec::new();
    for interval in intervals {
        if let Some(component) = components.last_mut()
            && interval.minimum <= component.maximum
        {
            component.maximum = component.maximum.max(interval.maximum);
            component.rows = component
                .rows
                .checked_add(interval.rows)
                .ok_or_else(|| anyhow!("component row count overflowed u64"))?;
            component.row_groups.push(interval.index);
            continue;
        }
        components.push(Component {
            minimum: interval.minimum,
            maximum: interval.maximum,
            row_groups: vec![interval.index],
            rows: interval.rows,
        });
    }
    for component in &mut components {
        component.row_groups.sort_unstable();
    }
    Ok(components)
}

struct RowEmitter<'a> {
    consumer: RowConsumer<'a>,
    feature_count: usize,
    chunk_rows: usize,
    positions: Vec<u64>,
    sample_ids: Vec<i64>,
    values: Vec<f32>,
    previous_position: Option<u64>,
    emitted_rows: u64,
}

impl<'a> RowEmitter<'a> {
    fn new(
        consumer: RowConsumer<'a>,
        feature_count: usize,
        max_chunk_bytes: usize,
    ) -> Result<Self> {
        let row_bytes = feature_count
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(16))
            .ok_or_else(|| anyhow!("stream row size exceeds addressable memory"))?;
        let chunk_rows = max_chunk_bytes
            .saturating_sub(8)
            .checked_div(row_bytes)
            .unwrap_or(0)
            .max(1);
        let value_capacity = chunk_rows
            .checked_mul(feature_count)
            .ok_or_else(|| anyhow!("stream chunk size exceeds addressable memory"))?;
        Ok(Self {
            consumer,
            feature_count,
            chunk_rows,
            positions: Vec::with_capacity(chunk_rows),
            sample_ids: Vec::with_capacity(chunk_rows),
            values: Vec::with_capacity(value_capacity),
            previous_position: None,
            emitted_rows: 0,
        })
    }

    fn validate_position(&mut self, position: u64) -> Result<()> {
        if let Some(previous) = self.previous_position
            && previous >= position
        {
            bail!("stream output positions are not strictly increasing")
        }
        self.previous_position = Some(position);
        Ok(())
    }

    fn push_default(
        &mut self,
        position: u64,
        sample: &StreamSample,
        config: &StreamConfig,
        layout: &FeatureLayout,
    ) -> Result<()> {
        self.validate_position(position)?;
        self.positions.push(position);
        self.sample_ids.push(sample.sample_id);
        let start = self.values.len();
        self.values.resize(start + self.feature_count, 0.0);
        fill_default_feature_row(config, layout, sample, &mut self.values[start..])?;
        if self.positions.len() == self.chunk_rows {
            self.flush()?;
        }
        Ok(())
    }

    fn push_parallel(
        &mut self,
        config: &StreamConfig,
        layout: &FeatureLayout,
        pool: &ThreadPool,
        jobs: Vec<(u64, &'_ StreamSample, SampleGroups)>,
    ) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        if !self.positions.is_empty() {
            self.flush()?;
        }
        if jobs.len() > self.chunk_rows {
            bail!("parallel feature jobs exceed the configured chunk bound")
        }
        for (position, sample, _) in &jobs {
            self.validate_position(*position)?;
            self.positions.push(*position);
            self.sample_ids.push(sample.sample_id);
        }
        self.values.resize(jobs.len() * self.feature_count, 0.0);
        pool.install(|| {
            self.values
                .par_chunks_mut(self.feature_count)
                .zip(jobs.into_par_iter())
                .try_for_each(|(target, (_, sample, groups))| {
                    fill_feature_row(config, layout, sample, groups, target)
                })
        })?;
        self.flush()
    }

    fn flush(&mut self) -> Result<()> {
        if !self.positions.is_empty() {
            (self.consumer)(&self.positions, &self.sample_ids, &self.values)?;
            self.emitted_rows += self.positions.len() as u64;
        }
        self.positions.clear();
        self.sample_ids.clear();
        self.values.clear();
        Ok(())
    }
}

fn fill_default_feature_row(
    config: &StreamConfig,
    layout: &FeatureLayout,
    sample: &StreamSample,
    row: &mut [f32],
) -> Result<()> {
    if row.len() != layout.feature_count {
        bail!("dense feature row has the wrong width")
    }
    row.fill(f32::NAN);
    // A sample with no selected event still has an observation width and three
    // zero totals, so the aggregates are defined before any event is seen.
    let aggregates = [0.0, 0.0, 0.0, sample_window_hours(sample)? as f32];
    row[..layout.aggregate_columns]
        .copy_from_slice(&aggregates[..layout.aggregate_columns]);
    // `count` is the only statistic defined for an unobserved numeric code;
    // every occurrence statistic reads 0.0 when the code is unobserved.
    let base = layout.aggregate_columns;
    match config.representation {
        Representation::Summary => {
            for variable in 0..layout.numeric_global.len() {
                row[base + variable * layout.summary_width] = 0.0;
            }
            row[layout.numeric_summary_columns..].fill(0.0);
        }
        Representation::Temporal => {
            for variable in 0..layout.numeric_global.len() {
                for segment in 0..layout.segment_count {
                    row[base
                        + (variable * layout.segment_count + segment) * layout.summary_width] = 0.0;
                }
            }
            row[layout.numeric_temporal_columns..].fill(0.0);
        }
    }
    Ok(())
}

fn sample_window_hours(sample: &StreamSample) -> Result<f64> {
    let width_us = sample
        .prediction_time_us
        .checked_sub(sample.observation_start_us)
        .ok_or_else(|| anyhow!("sample observation width overflowed int64"))?;
    Ok(width_us as f64 / 3_600_000_000.0)
}

fn fill_feature_row(
    config: &StreamConfig,
    layout: &FeatureLayout,
    sample: &StreamSample,
    mut groups: SampleGroups,
    row: &mut [f32],
) -> Result<()> {
    let sample_hours = sample_window_hours(sample)?;
    // The segment boundary is a property of the sample, not of any one code,
    // so it is the latest event time across every code.
    let mut last_event_hours = 0.0_f64;
    let mut instants: HashSet<i64> = HashSet::new();
    let mut event_total = 0usize;
    let mut observed_codes = 0usize;
    for events in groups.values() {
        if events.is_empty() {
            continue;
        }
        observed_codes += 1;
        event_total += events.len();
        for event in events.iter() {
            instants.insert(event.time_us);
            let time = event.time_us as f64 / HOUR_US as f64;
            if time > last_event_hours {
                last_event_hours = time;
            }
        }
    }
    let (segment_count, all_bounds) = segment_bounds(
        config.segment_kind,
        sample_hours,
        last_event_hours,
        config.halves_split_hours,
    );
    let bounds = &all_bounds[..segment_count];
    fill_default_feature_row(config, layout, sample, row)?;
    let totals = [
        event_total as f32,
        observed_codes as f32,
        instants.len() as f32,
    ];
    let counts = layout.aggregate_columns.min(totals.len());
    row[..counts].copy_from_slice(&totals[..counts]);
    for (global_code, events) in &mut groups {
        // Preserve Parquet stream order when both explicit tie-breakers are
        // identical, matching NumPy's stable lexicographic reference path.
        events.sort_by_key(|event| (event.time_us, event.source_row));
        if let Some(&variable) = layout.numeric_local.get(global_code) {
            match config.representation {
                Representation::Summary => {
                    let statistics = numeric_statistics(events, 0.0, sample_hours, sample_hours);
                    let width = layout.summary_width;
                    let base = layout.aggregate_columns + variable * width;
                    row[base..base + width].copy_from_slice(&statistics[..width]);
                }
                Representation::Temporal => {
                    let width = layout.summary_width;
                    for (segment, bound) in bounds.iter().enumerate() {
                        let statistics =
                            numeric_statistics(events, bound.start, bound.stop, sample_hours);
                        let base = layout.aggregate_columns
                            + (variable * segment_count + segment) * width;
                        row[base..base + width].copy_from_slice(&statistics[..width]);
                    }
                }
            }
        } else if let Some(&variable) = layout.occurrence_local.get(global_code) {
            match config.representation {
                Representation::Summary => {
                    let statistics = occurrence_statistics(events, 0.0, sample_hours, sample_hours);
                    let width = layout.occurrence_width;
                    let base = layout.numeric_summary_columns + variable * width;
                    row[base..base + width].copy_from_slice(&statistics[..width]);
                }
                Representation::Temporal => {
                    let width = layout.occurrence_width;
                    for (segment, bound) in bounds.iter().enumerate() {
                        let statistics =
                            occurrence_statistics(events, bound.start, bound.stop, sample_hours);
                        let base = layout.numeric_temporal_columns
                            + (variable * segment_count + segment) * width;
                        row[base..base + width].copy_from_slice(&statistics[..width]);
                    }
                }
            }
        } else {
            bail!("internal code layout disagrees with grouped event")
        }
    }
    if row.iter().any(|value| value.is_infinite()) {
        bail!("generated feature row contains infinity")
    }
    Ok(())
}

fn selected_sample(samples: &[StreamSample], sample_id: i64) -> Result<&StreamSample> {
    samples
        .binary_search_by_key(&sample_id, |sample| sample.sample_id)
        .map(|position| &samples[position])
        .map_err(|_| anyhow!("selected event references unknown sample_id {sample_id}"))
}

fn read_component_events(
    config: &StreamConfig,
    layout: &FeatureLayout,
    samples: &[StreamSample],
    selected_ids: Option<&HashSet<i64>>,
    component: &Component,
) -> Result<(ComponentGroups, u64)> {
    let file = File::open(&config.events_path)
        .with_context(|| format!("cannot open {}", config.events_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let has_split = builder.schema().index_of("split").is_ok();
    let wanted: &[&str] = if has_split {
        &["sample_id", "time", "code", "numeric_value", "source_row", "split"]
    } else {
        &["sample_id", "time", "code", "numeric_value", "source_row"]
    };
    let roots: Vec<usize> = wanted
        .iter()
        .map(|name| {
            builder
                .schema()
                .index_of(name)
                .with_context(|| format!("events missing {name}"))
        })
        .collect::<Result<_>>()?;
    let projection = ProjectionMask::roots(builder.parquet_schema(), roots);
    let reader = builder
        .with_projection(projection)
        .with_row_groups(component.row_groups.clone())
        .with_batch_size(config.record_batch_rows)
        .build()?;
    let mut groups = ComponentGroups::new();
    let mut event_count = 0u64;
    for batch in reader {
        let batch = batch?;
        let sample_column = column(&batch, "sample_id")?;
        let time_column = column(&batch, "time")?;
        let code_column = column(&batch, "code")?;
        let value_column = column(&batch, "numeric_value")?;
        let source_column = column(&batch, "source_row")?;
        let split_column = if has_split { Some(column(&batch, "split")?) } else { None };
        let sample_values = batch
            .column(sample_column)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow!("sample_id must be int64"))?;
        let time_values = batch
            .column(time_column)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| anyhow!("time must be timestamp[us]"))?;
        let numeric_values = batch
            .column(value_column)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow!("numeric_value must be float32"))?;
        let source_values = batch
            .column(source_column)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow!("source_row must be int64"))?;
        for row in 0..batch.num_rows() {
            if let Some(split_column) = split_column
                && split_at(&batch, split_column, row)? != config.split
            {
                continue;
            }
            if sample_values.is_null(row) {
                bail!("sample_id cannot be null")
            }
            let sample_id = sample_values.value(row);
            if sample_id < component.minimum || sample_id > component.maximum {
                bail!("event sample_id lies outside its row-group interval component")
            }
            // An export from the shared writer (it carries `split`) may not
            // reference samples outside its index; an in-memory index used
            // by MedsTask.build_representations may legitimately cover a
            // subset of the file's samples and codes.
            let sample = match selected_sample(samples, sample_id) {
                Ok(sample) => sample,
                Err(_) if !has_split => continue,
                Err(error) => return Err(error),
            };
            if selected_ids.is_some_and(|selected| !selected.contains(&sample_id)) {
                continue;
            }
            let code = string_at(&batch, code_column, row)?;
            let Some(&code_index) = layout.code_lookup.get(code) else {
                if layout.ignored_codes.contains(code) || !has_split {
                    continue;
                }
                bail!("selected split event uses undeclared code {code:?}")
            };
            if time_values.is_null(row) {
                bail!("time cannot be null for selected structured event")
            }
            if source_values.is_null(row) {
                bail!("source_row cannot be null for selected structured event")
            }
            let occurrence = layout.occurrence_set.contains(&code_index);
            let value = if occurrence {
                if !numeric_values.is_null(row) {
                    bail!("occurrence-only structured events must have null numeric_value")
                }
                f32::NAN
            } else {
                if numeric_values.is_null(row) {
                    bail!("numeric structured events must have numeric_value")
                }
                let value = numeric_values.value(row);
                if !value.is_finite() {
                    bail!("numeric structured values must be finite")
                }
                value
            };
            let time_us = time_values
                .value(row)
                .checked_sub(sample.observation_start_us)
                .ok_or_else(|| anyhow!("event time offset overflowed int64"))?;
            let sample_width = sample
                .prediction_time_us
                .checked_sub(sample.observation_start_us)
                .ok_or_else(|| anyhow!("sample observation width overflowed int64"))?;
            if time_us < 0 || time_us > sample_width {
                bail!("selected structured event falls outside its sample window")
            }
            groups
                .entry(sample_id)
                .or_default()
                .entry(code_index)
                .or_default()
                .push(Event {
                    time_us,
                    source_row: source_values.value(row),
                    value,
                });
            event_count = event_count
                .checked_add(1)
                .ok_or_else(|| anyhow!("selected event count overflowed u64"))?;
            if event_count > config.max_component_rows {
                bail!(
                    "component pass retained {} selected event rows, exceeding limit {}",
                    event_count,
                    config.max_component_rows
                )
            }
        }
    }
    Ok((groups, event_count))
}

/// Per-execution audit counters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExecutionStatistics {
    pub(crate) output_rows: u64,
    pub(crate) selected_event_count: u64,
    pub(crate) candidate_row_groups: usize,
    pub(crate) component_count: usize,
    pub(crate) maximum_retained_rows: u64,
}

/// Featurize one split and hand row chunks directly to `consumer`
/// (positions, sample IDs, row-major f32 values). `requested_rows` limits
/// output to those strictly increasing split positions; `None` emits every
/// row in ascending sample-ID order.
/// `threads` sizes the featurization pool (`None` = every core). Row
/// contents never depend on it (see `rayon_num_threads_one_and_four_produce_identical_rows`),
/// so it is deliberately decoupled from the pinned LightGBM trial thread
/// count that model bytes do depend on.
pub(crate) fn execute_rows(
    config: &StreamConfig,
    requested_rows: Option<&[u64]>,
    threads: Option<usize>,
    consumer: RowConsumer<'_>,
) -> Result<ExecutionStatistics> {
    if threads == Some(0) {
        bail!("featurizer_threads must be positive")
    }
    let pool = build_stream_thread_pool(threads)?;
    execute_with_pool(config, requested_rows, consumer, &pool)
}

fn build_stream_thread_pool(rayon_num_threads: Option<usize>) -> Result<ThreadPool> {
    let mut builder = ThreadPoolBuilder::new();
    if let Some(threads) = rayon_num_threads {
        builder = builder.num_threads(threads);
    }
    builder
        .build()
        .context("cannot build stream featurization thread pool")
}

fn execute_with_pool(
    config: &StreamConfig,
    requested_rows: Option<&[u64]>,
    consumer: RowConsumer<'_>,
    pool: &ThreadPool,
) -> Result<ExecutionStatistics> {
    let layout = FeatureLayout::new(config)?;
    let samples = load_split_index(config)?;
    let selected_rows: Vec<u64> = match requested_rows {
        None => (0..samples.len() as u64).collect(),
        Some(rows) => {
            for (index, &row) in rows.iter().enumerate() {
                if row >= samples.len() as u64 {
                    bail!("requested row {row} is outside the split index")
                }
                if index > 0 && rows[index - 1] >= row {
                    bail!("requested rows must be strictly increasing")
                }
            }
            rows.to_vec()
        }
    };
    let selected_ids: Option<HashSet<i64>> = requested_rows.map(|_| {
        selected_rows
            .iter()
            .map(|&position| samples[position as usize].sample_id)
            .collect()
    });
    let file = File::open(&config.events_path)
        .with_context(|| format!("cannot open {}", config.events_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let (components, candidate_groups) = plan_components(builder.metadata(), &config.split)?;
    let mut emitter = RowEmitter::new(consumer, layout.feature_count, config.max_chunk_bytes)?;
    let mut selected_cursor = 0usize;
    let mut event_count = 0u64;
    let mut maximum_retained = 0u64;
    for component in &components {
        while selected_cursor < selected_rows.len() {
            let position = selected_rows[selected_cursor];
            let sample = &samples[position as usize];
            if sample.sample_id >= component.minimum {
                break;
            }
            emitter.push_default(position, sample, config, &layout)?;
            selected_cursor += 1;
        }
        if selected_cursor == selected_rows.len() {
            break;
        }
        let current_sample = samples[selected_rows[selected_cursor] as usize].sample_id;
        if current_sample > component.maximum {
            continue;
        }
        let span_start = selected_cursor;
        let mut span_end = selected_cursor;
        while span_end < selected_rows.len()
            && samples[selected_rows[span_end] as usize].sample_id <= component.maximum
        {
            span_end += 1;
        }
        // A component whose physical rows exceed the bound is executed in
        // several passes over disjoint, consecutive slices of its selected
        // samples: each pass rescans the component's row groups but retains
        // only its own slice's events, so held memory stays bounded while the
        // emitted rows and the total selected event count are identical to a
        // single pass.
        let passes = usize::try_from(component.rows.div_ceil(config.max_component_rows))
            .context("component pass count exceeds addressable memory")?
            .max(1);
        let span = &selected_rows[span_start..span_end];
        let pass_rows = span.len().div_ceil(passes).max(1);
        for pass_positions in span.chunks(pass_rows) {
            let pass_ids: Option<HashSet<i64>> = (passes > 1).then(|| {
                pass_positions
                    .iter()
                    .map(|&position| samples[position as usize].sample_id)
                    .collect()
            });
            // In sampled mode the span already holds only requested rows, so
            // a pass set is the correct intersection in both modes.
            let pass_selected = match (&pass_ids, selected_ids.as_ref()) {
                (Some(ids), _) => Some(ids),
                (None, ids) => ids,
            };
            let (mut groups, pass_events) =
                read_component_events(config, &layout, &samples, pass_selected, component)?;
            event_count = event_count
                .checked_add(pass_events)
                .ok_or_else(|| anyhow!("selected event count overflowed u64"))?;
            maximum_retained = maximum_retained.max(pass_events);
            let mut pass_cursor = 0usize;
            while pass_cursor < pass_positions.len() {
                let mut jobs = Vec::with_capacity(emitter.chunk_rows);
                while pass_cursor < pass_positions.len() && jobs.len() < emitter.chunk_rows {
                    let position = pass_positions[pass_cursor];
                    let sample = &samples[position as usize];
                    let sample_groups = groups.remove(&sample.sample_id).unwrap_or_default();
                    jobs.push((position, sample, sample_groups));
                    pass_cursor += 1;
                }
                if jobs.is_empty() {
                    break;
                }
                emitter.push_parallel(config, &layout, pool, jobs)?;
            }
            if !groups.is_empty() {
                bail!("component retained event groups outside selected output rows")
            }
        }
        selected_cursor = span_end;
    }
    while selected_cursor < selected_rows.len() {
        let position = selected_rows[selected_cursor];
        let sample = &samples[position as usize];
        emitter.push_default(position, sample, config, &layout)?;
        selected_cursor += 1;
    }
    emitter.flush()?;
    let emitted = emitter.emitted_rows;
    drop(emitter);
    if emitted != selected_rows.len() as u64 {
        bail!("stream emitted the wrong number of rows")
    }
    Ok(ExecutionStatistics {
        output_rows: emitted,
        selected_event_count: event_count,
        candidate_row_groups: candidate_groups,
        component_count: components.len(),
        maximum_retained_rows: maximum_retained,
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use super::super::{
        OCCURRENCE_SUMMARY_COUNT, SUMMARY_COUNT, TIMED_AGGREGATE_COUNT,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    use arrow_array::{BooleanArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use parquet::arrow::ArrowWriter;
    use parquet::data_type::ByteArray;
    use parquet::file::statistics::ValueStatistics;

    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "meds-featurizer-stream-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn test_config(root: &Path, representation: Representation) -> StreamConfig {
        StreamConfig {
            events_path: root.join("events.parquet"),
            index_path: root.join("index.parquet"),
            split: "train".into(),
            representation,
            codes: vec!["numeric".into(), "occurrence".into()],
            occurrence_codes: vec!["occurrence".into()],
            ignored_codes: vec![],
            observation_window_hours: 4.0,
            segment_kind: SegmentKind::VisitHalves,
            halves_split_hours: 2.0,
            timed: true,
            max_component_rows: 20,
            max_chunk_bytes: 400,
            record_batch_rows: 2,
        }
    }

    fn index_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("subject_id", DataType::Int64, false),
            Field::new("sample_id", DataType::Int64, false),
            Field::new(
                "observation_start",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "prediction_time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("boolean_value", DataType::Boolean, false),
            Field::new("split", DataType::Utf8, false),
        ]))
    }

    fn write_index(path: &Path) {
        let schema = index_schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![4, 2, 3, 1, 9])),
                Arc::new(Int64Array::from(vec![40, 20, 30, 10, 100])),
                Arc::new(TimestampMicrosecondArray::from(vec![0, 0, 0, 0, 0])),
                Arc::new(TimestampMicrosecondArray::from(vec![
                    14_400_000_000,
                    7_200_000_000,
                    10_800_000_000,
                    14_400_000_000,
                    14_400_000_000,
                ])),
                Arc::new(BooleanArray::from(vec![false, true, false, true, false])),
                Arc::new(StringArray::from(vec![
                    "train", "train", "train", "train", "test",
                ])),
            ],
        )
        .unwrap();
        let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn event_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("subject_id", DataType::Int64, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("code", DataType::Utf8, false),
            Field::new("numeric_value", DataType::Float32, true),
            Field::new("sample_id", DataType::Int64, false),
            Field::new("source_row", DataType::Int64, false),
            Field::new("split", DataType::Utf8, false),
        ]))
    }

    fn event_batch(
        schema: Arc<Schema>,
        sample_ids: Vec<i64>,
        splits: Vec<&str>,
        codes: Vec<&str>,
        values: Vec<Option<f32>>,
        times: Vec<i64>,
        source_rows: Vec<i64>,
    ) -> RecordBatch {
        let count = sample_ids.len();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1; count])),
                Arc::new(TimestampMicrosecondArray::from(times)),
                Arc::new(StringArray::from(codes)),
                Arc::new(Float32Array::from(values)),
                Arc::new(Int64Array::from(sample_ids)),
                Arc::new(Int64Array::from(source_rows)),
                Arc::new(StringArray::from(splits)),
            ],
        )
        .unwrap()
    }

    fn write_events(path: &Path) {
        let schema = event_schema();
        let mut writer =
            ArrowWriter::try_new(File::create(path).unwrap(), schema.clone(), None).unwrap();
        // The identical explicit tie for sample 10 spans nonadjacent physical
        // row groups. The first group bridges samples 10..40, exercising a
        // multi-sample interval component, and the intervening test group must
        // be skipped by split stats.
        writer
            .write(&event_batch(
                schema.clone(),
                vec![10, 40],
                vec!["train", "train"],
                vec!["numeric", "numeric"],
                vec![Some(1.0), Some(4.0)],
                vec![0, 7_200_000_000],
                vec![7, 9],
            ))
            .unwrap();
        writer.flush().unwrap();
        writer
            .write(&event_batch(
                schema.clone(),
                vec![100],
                vec!["test"],
                vec!["numeric"],
                vec![Some(99.0)],
                vec![0],
                vec![0],
            ))
            .unwrap();
        writer.flush().unwrap();
        writer
            .write(&event_batch(
                schema.clone(),
                vec![10],
                vec!["train"],
                vec!["numeric"],
                vec![Some(9.0)],
                vec![0],
                vec![7],
            ))
            .unwrap();
        writer.flush().unwrap();
        writer
            .write(&event_batch(
                schema.clone(),
                vec![20],
                vec!["train"],
                vec!["occurrence"],
                vec![None],
                vec![3_600_000_000],
                vec![8],
            ))
            .unwrap();
        writer.close().unwrap();
    }

    fn fixture(representation: Representation) -> (TestDirectory, StreamConfig) {
        let root = TestDirectory::new();
        write_index(&root.0.join("index.parquet"));
        write_events(&root.0.join("events.parquet"));
        let config = test_config(&root.0, representation);
        (root, config)
    }

    fn collect(
        config: &StreamConfig,
        requested: Option<&[u64]>,
    ) -> (Vec<(u64, i64, Vec<f32>)>, ExecutionStatistics) {
        let mut rows: Vec<(u64, i64, Vec<f32>)> = Vec::new();
        let statistics = execute_rows(config, requested, Some(2), &mut |positions,
                                                                        sample_ids,
                                                                        values| {
            let width = values.len() / positions.len();
            for (index, (&position, &sample_id)) in
                positions.iter().zip(sample_ids).enumerate()
            {
                rows.push((
                    position,
                    sample_id,
                    values[index * width..(index + 1) * width].to_vec(),
                ));
            }
            Ok(())
        })
        .unwrap();
        (rows, statistics)
    }

    fn to_bits(rows: &[(u64, i64, Vec<f32>)]) -> Vec<(u64, i64, Vec<u32>)> {
        rows.iter()
            .map(|(position, sample, values)| {
                (
                    *position,
                    *sample,
                    values.iter().map(|value| value.to_bits()).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn globally_overlapping_intervals_join_nonadjacent_row_groups() {
        let intervals = vec![
            RowGroupInterval {
                index: 0,
                minimum: 10,
                maximum: 20,
                rows: 5,
            },
            RowGroupInterval {
                index: 1,
                minimum: 100,
                maximum: 110,
                rows: 2,
            },
            RowGroupInterval {
                index: 2,
                minimum: 15,
                maximum: 25,
                rows: 3,
            },
        ];
        assert_eq!(
            merge_intervals(intervals).unwrap(),
            vec![
                Component {
                    minimum: 10,
                    maximum: 25,
                    row_groups: vec![0, 2],
                    rows: 8,
                },
                Component {
                    minimum: 100,
                    maximum: 110,
                    row_groups: vec![1],
                    rows: 2,
                },
            ]
        );
    }

    #[test]
    fn inexact_split_statistics_disable_pruning() {
        let exact: Statistics = ValueStatistics::new(
            Some(ByteArray::from("test")),
            Some(ByteArray::from("test")),
            None,
            Some(0),
            false,
        )
        .into();
        assert!(!split_can_match(&exact, "train", 0).unwrap());

        let inexact: Statistics = ValueStatistics::new(
            Some(ByteArray::from("test")),
            Some(ByteArray::from("test")),
            None,
            Some(0),
            false,
        )
        .with_min_is_exact(false)
        .with_max_is_exact(false)
        .into();
        assert!(split_can_match(&inexact, "train", 0).unwrap());
    }

    #[test]
    fn oversized_component_streams_identical_rows_in_bounded_passes() {
        let (_root, config) = fixture(Representation::Summary);
        let (baseline_rows, baseline_stats) = collect(&config, None);

        // The train component holds 4 physical rows, so a bound of 3 forces
        // two passes while each pass's retained events stay within the bound.
        let mut chunked_config = fixture(Representation::Summary).1;
        chunked_config.events_path = config.events_path.clone();
        chunked_config.index_path = config.index_path.clone();
        chunked_config.max_component_rows = 3;
        let (chunked_rows, chunked_stats) = collect(&chunked_config, None);

        // NaN payloads compare by bit pattern, not by float equality.
        assert_eq!(to_bits(&baseline_rows), to_bits(&chunked_rows));
        assert_eq!(baseline_stats.output_rows, chunked_stats.output_rows);
        assert_eq!(
            baseline_stats.selected_event_count,
            chunked_stats.selected_event_count
        );
        assert_eq!(
            baseline_stats.candidate_row_groups,
            chunked_stats.candidate_row_groups
        );
        assert_eq!(baseline_stats.component_count, chunked_stats.component_count);
        assert!(chunked_stats.maximum_retained_rows <= 3);
    }

    #[test]
    fn all_mode_streams_sorted_samples_zero_rows_and_stable_ties() {
        let (_root, config) = fixture(Representation::Summary);
        let (rows, statistics) = collect(&config, None);
        assert_eq!(
            rows.iter()
                .map(|(row, id, _)| (*row, *id))
                .collect::<Vec<_>>(),
            vec![(0, 10), (1, 20), (2, 30), (3, 40)]
        );
        // event count, distinct code count, distinct instant count, window
        // hours; then count, mean, std, min, max, median, latest_observed,
        // delta, first_recency, recency; then the occurrence statistics.
        let numeric = TIMED_AGGREGATE_COUNT;
        let occurrence = numeric + SUMMARY_COUNT;
        assert_eq!(rows[0].2[..numeric], [2.0, 1.0, 1.0, 4.0]);
        assert_eq!(rows[0].2[numeric], 2.0);
        assert_eq!(rows[0].2[numeric + 3], 1.0);
        assert_eq!(rows[0].2[numeric + 4], 9.0);
        assert_eq!(rows[0].2[numeric + 5], 5.0);
        assert_eq!(rows[0].2[numeric + 6], 9.0);
        assert_eq!(rows[0].2[numeric + 7], 8.0);
        assert_eq!(rows[1].2[occurrence], 1.0);
        assert_eq!(rows[2].2[..numeric], [0.0, 0.0, 0.0, 3.0]);
        assert_eq!(rows[2].2[numeric], 0.0);
        assert!(rows[2].2[numeric + 2].is_nan());
        assert_eq!(rows[2].2[occurrence..], [0.0; OCCURRENCE_SUMMARY_COUNT]);
        assert_eq!(statistics.output_rows, 4);
        assert_eq!(statistics.selected_event_count, 4);
        assert_eq!(statistics.candidate_row_groups, 3);
        assert_eq!(statistics.component_count, 1);
    }

    #[test]
    fn temporal_segments_preserve_tie_order_and_split() {
        let (_root, config) = fixture(Representation::Temporal);
        let layout = FeatureLayout::new(&config).unwrap();
        let (rows, _statistics) = collect(&config, None);
        assert_eq!(rows[0].2.len(), layout.feature_count);
        // Sample 10 has a fully identical explicit tie, so stream order decides
        // which value is latest; both of its events sit at hour 0, which the
        // cohort-wide two-hour split places in `earlier` and never in `recent`.
        let latest = layout.aggregate_columns + 6;
        assert_eq!(rows[0].2[latest], 9.0);
        assert_eq!(rows[0].2[SUMMARY_COUNT + latest], 9.0);
        assert_eq!(rows[0].2[layout.aggregate_columns + 2 * SUMMARY_COUNT], 0.0);
        assert!(rows[0].2[2 * SUMMARY_COUNT + latest].is_nan());
    }

    #[test]
    fn rayon_num_threads_one_and_four_produce_identical_rows() {
        let (_root, config) = fixture(Representation::Summary);
        let one_thread = build_stream_thread_pool(Some(1)).unwrap();
        let four_threads = build_stream_thread_pool(Some(4)).unwrap();
        let collect_with = |pool: &ThreadPool| {
            let mut rows: Vec<(u64, i64, Vec<f32>)> = Vec::new();
            execute_with_pool(&config, None, &mut |positions, ids, values| {
                let width = values.len() / positions.len();
                for (index, (&position, &sample)) in
                    positions.iter().zip(ids).enumerate()
                {
                    rows.push((
                        position,
                        sample,
                        values[index * width..(index + 1) * width].to_vec(),
                    ));
                }
                Ok(())
            }, pool)
            .unwrap();
            rows
        };
        let one = collect_with(&one_thread);
        let four = collect_with(&four_threads);
        assert_eq!(to_bits(&one), to_bits(&four));
    }

    #[test]
    fn requested_rows_emit_only_those_rows_including_zero_event() {
        let (_root, config) = fixture(Representation::Summary);
        let (rows, statistics) = collect(&config, Some(&[0, 2]));
        assert_eq!(
            rows.iter()
                .map(|(row, id, _)| (*row, *id))
                .collect::<Vec<_>>(),
            vec![(0, 10), (2, 30)]
        );
        assert_eq!(rows[0].2[TIMED_AGGREGATE_COUNT + 6], 9.0);
        assert_eq!(rows[1].2[TIMED_AGGREGATE_COUNT], 0.0);
        assert_eq!(statistics.output_rows, 2);
        assert_eq!(statistics.selected_event_count, 2);
    }

    #[test]
    fn requested_rows_must_be_increasing_and_in_range() {
        let (_root, config) = fixture(Representation::Summary);
        let mut sink = |_p: &[u64], _i: &[i64], _v: &[f32]| Ok(());
        assert!(
            execute_rows(&config, Some(&[2, 1]), None, &mut sink)
                .unwrap_err()
                .to_string()
                .contains("strictly increasing")
        );
        assert!(
            execute_rows(&config, Some(&[0, 99]), None, &mut sink)
                .unwrap_err()
                .to_string()
                .contains("outside the split index")
        );
    }

}
