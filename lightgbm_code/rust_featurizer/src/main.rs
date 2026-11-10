use std::path::Path;

use anyhow::{Context, Result, bail};
use arrow_array::{Array, LargeStringArray, RecordBatch, StringArray};
use serde::Deserialize;

mod lightgbm_ffi;
mod sha256;
mod stream;
mod train;

#[cfg(not(target_endian = "little"))]
compile_error!("meds-featurizer's raw-array interchange requires a little-endian target");

const SUMMARY_COUNT: usize = 10;
const OCCURRENCE_SUMMARY_COUNT: usize = 3;
const HOUR_US: i64 = 3_600_000_000;

/// How the segmented representation derives its three per-sample boundaries.
///
/// `VisitHalves` splits at one cohort-wide duration before prediction time, so
/// "recent" denotes the same elapsed span for every sample. `VisitBuckets`
/// splits at the sample's own last event time, which for a task carrying one
/// timestamp per visit is exactly the current visit.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
#[allow(clippy::enum_variant_names)]
enum SegmentKind {
    #[default]
    VisitHalves,
    VisitBuckets,
}

impl SegmentKind {
    fn segment_count(self) -> usize {
        SEGMENT_COUNT
    }
}

/// Every segmented representation carries exactly three segments.
const SEGMENT_COUNT: usize = 3;
const MAX_SEGMENTS: usize = SEGMENT_COUNT;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Segment {
    start: f64,
    stop: f64,
}
/// Per-sample totals prepended to every representation: event count, distinct
/// observed code count, distinct event-instant count, and — for a timed
/// profile only — the elapsed observation width in hours.
const AGGREGATE_COUNT: usize = 3;
const TIMED_AGGREGATE_COUNT: usize = AGGREGATE_COUNT + 1;

fn statistic_widths(timed: bool) -> (usize, usize, usize) {
    if timed {
        (SUMMARY_COUNT, OCCURRENCE_SUMMARY_COUNT, TIMED_AGGREGATE_COUNT)
    } else {
        (SUMMARY_COUNT - 2, OCCURRENCE_SUMMARY_COUNT - 2, AGGREGATE_COUNT)
    }
}

#[derive(Clone, Copy)]
struct Event {
    time_us: i64,
    source_row: i64,
    value: f32,
}

fn segment_bounds(
    kind: SegmentKind,
    sample_hours: f64,
    last_event_hours: f64,
    split_hours: f64,
) -> (usize, [Segment; MAX_SEGMENTS]) {
    let all = |start: f64, stop: f64| Segment { start, stop };
    match kind {
        SegmentKind::VisitHalves => {
            let cut = (sample_hours - split_hours).max(0.0);
            (3, [all(0.0, sample_hours), all(0.0, cut), all(cut, sample_hours)])
        }
        SegmentKind::VisitBuckets => (
            3,
            [
                all(0.0, sample_hours),
                all(0.0, last_event_hours),
                all(last_event_hours, sample_hours),
            ],
        ),
    }
}

fn in_window(time: f64, start: f64, stop: f64, hours: f64) -> bool {
    time >= start
        && if stop == hours {
            time <= stop
        } else {
            time < stop
        }
}


fn numeric_statistics(
    events: &[Event],
    window_start: f64,
    window_stop: f64,
    hours: f64,
) -> [f32; SUMMARY_COUNT] {
    let mut selected = Vec::with_capacity(events.len());
    for event in events {
        let time = event.time_us as f64 / 3_600_000_000.0;
        if in_window(time, window_start, window_stop, hours) {
            selected.push((time, event.value as f64));
        }
    }
    if selected.is_empty() {
        let mut result = [f32::NAN; SUMMARY_COUNT];
        result[0] = 0.0;
        return result;
    }
    let count = selected.len() as f64;
    let mut sum = 0.0;
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut values = Vec::with_capacity(selected.len());
    for &(_time, value) in &selected {
        sum += value;
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        values.push(value);
    }
    let mean = sum / count;
    let variance = selected
        .iter()
        .map(|(_, value)| (value - mean) * (value - mean))
        .sum::<f64>()
        / count;
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    let median = if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    };
    let first = selected[0];
    let latest = selected[selected.len() - 1];
    [
        count as f32,
        mean as f32,
        variance.sqrt() as f32,
        minimum as f32,
        maximum as f32,
        median as f32,
        latest.1 as f32,
        (latest.1 - first.1) as f32,
        // Prediction-relative timing only; see reference_features.SUMMARY_NAMES.
        (hours - first.0) as f32,
        (hours - latest.0) as f32,
    ]
}


fn occurrence_statistics(
    events: &[Event],
    window_start: f64,
    window_stop: f64,
    hours: f64,
) -> [f32; OCCURRENCE_SUMMARY_COUNT] {
    let selected: Vec<f64> = events
        .iter()
        .map(|event| event.time_us as f64 / 3_600_000_000.0)
        .filter(|&time| in_window(time, window_start, window_stop, hours))
        .collect();
    if selected.is_empty() {
        return [0.0; OCCURRENCE_SUMMARY_COUNT];
    }
    let first = selected[0];
    let latest = selected[selected.len() - 1];
    [
        selected.len() as f32,
        (hours - first) as f32,
        (hours - latest) as f32,
    ]
}

fn string_at(batch: &RecordBatch, column: usize, row: usize) -> Result<&str> {
    let array = batch.column(column);
    if array.is_null(row) {
        bail!("code cannot be null")
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(values.value(row));
    }
    bail!("code must be an Arrow string or large_string array")
}

fn column(batch: &RecordBatch, name: &str) -> Result<usize> {
    batch
        .schema()
        .index_of(name)
        .with_context(|| format!("projected event batch is missing {name}"))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    const USAGE: &str =
        "usage: meds-featurizer (--train-config | --predict-config | --dump-config) CONFIG.json";
    if args.len() != 3 {
        bail!(USAGE)
    }
    match args[1].as_str() {
        "--train-config" => train::run_train(Path::new(&args[2])),
        "--predict-config" => train::run_predict(Path::new(&args[2])),
        "--dump-config" => train::run_dump(Path::new(&args[2])),
        _ => bail!(USAGE),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_bounds_are_three_time_windows() {
        let (count, bounds) = segment_bounds(SegmentKind::VisitBuckets, 10.0, 8.0, 0.0);
        assert_eq!(count, 3);
        assert_eq!(bounds[0], Segment { start: 0.0, stop: 10.0 });
        assert_eq!(bounds[1], Segment { start: 0.0, stop: 8.0 });
        assert_eq!(bounds[2], Segment { start: 8.0, stop: 10.0 });

        // VisitHalves cuts one cohort-wide span before prediction time, and
        // clamps to zero when the sample is narrower than that span.
        let (count, bounds) = segment_bounds(SegmentKind::VisitHalves, 10.0, 8.0, 4.0);
        assert_eq!(count, 3);
        assert_eq!(bounds[1], Segment { start: 0.0, stop: 6.0 });
        assert_eq!(bounds[2], Segment { start: 6.0, stop: 10.0 });
        let (_, bounds) = segment_bounds(SegmentKind::VisitHalves, 3.0, 2.0, 8.0);
        assert_eq!(bounds[1], Segment { start: 0.0, stop: 0.0 });

        assert_eq!(SegmentKind::VisitHalves.segment_count(), 3);
        assert_eq!(SegmentKind::VisitBuckets.segment_count(), 3);
    }

    #[test]
    fn numeric_statistics_match_frozen_definition() {
        let events = [
            Event {
                time_us: 0,
                source_row: 0,
                value: 1.0,
            },
            Event {
                time_us: 3_600_000_000,
                source_row: 1,
                value: 3.0,
            },
        ];
        // Two events, at hours 0 and 1, valued 1.0 and 3.0, in a 48-hour window.
        // Order: count, mean, std, min, max, median, latest_observed, delta,
        // first_recency, recency.
        let statistics = numeric_statistics(&events, 0.0, 48.0, 48.0);
        assert_eq!(statistics[0], 2.0);
        assert_eq!(statistics[1], 2.0);
        assert_eq!(statistics[2], 1.0);
        assert_eq!(statistics[3], 1.0);
        assert_eq!(statistics[4], 3.0);
        assert_eq!(statistics[5], 2.0);
        assert_eq!(statistics[6], 3.0);
        assert_eq!(statistics[7], 2.0);
        assert_eq!(statistics[8], 48.0);
        assert_eq!(statistics[9], 47.0);
    }

    #[test]
    fn empty_numeric_statistics_keep_only_zero_count() {
        let statistics = numeric_statistics(&[], 0.0, 48.0, 48.0);
        assert_eq!(statistics[0], 0.0);
        assert!(statistics[1..].iter().all(|value| value.is_nan()));
    }

    #[test]
    fn stable_tie_order_preserves_input_order() {
        let mut events = [
            Event {
                time_us: 100,
                source_row: 7,
                value: 2.0,
            },
            Event {
                time_us: 100,
                source_row: 7,
                value: 9.0,
            },
        ];
        events.sort_by_key(|event| (event.time_us, event.source_row));
        assert_eq!(events[0].value, 2.0);
        assert_eq!(events[1].value, 9.0);
    }
}
