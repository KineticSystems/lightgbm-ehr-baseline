//! In-process LightGBM trial executor.
//!
//! One invocation handles one representation: it streams the train and
//! validation splits from the featurizer in-process, builds the native
//! Datasets through LightGBM's public C API with exactly the call sequence
//! the audited Python executor used, runs the pre-planned trials with
//! replicated early-stopping semantics, and spills every trial's validation
//! prediction and model string to disk for Python-side selection. Parameter
//! strings arrive pre-serialized by Python so both executors feed LightGBM
//! byte-identical configuration.

use std::collections::HashSet;
use std::ffi::CString;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use arrow_array::{Array, BooleanArray, Int64Array};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;
use serde_json::json;

use super::SegmentKind;
use super::lightgbm_ffi::{
    BoosterHandle, C_API_DTYPE_FLOAT32, C_API_PREDICT_NORMAL, DatasetHandle,
    LightGbm,
};
use super::sha256::Sha256;
use super::stream::{self, Representation, StreamConfig};

const ZERO_THRESHOLD: f64 = 1e-35;
const ZERO_VACATING_OFFSET: f32 = 0.1;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TrainConfig {
    lightgbm_lib: PathBuf,
    /// Every representation the study searches. All are constructed before any
    /// trial runs, so workers can cross a representation boundary instead of
    /// idling through the ragged tail of the one they are on.
    representations: Vec<RepresentationConfig>,
    trials: Vec<TrialConfig>,
    num_boost_round: usize,
    early_stopping_rounds: usize,
    parallel_trials: usize,
    spill_dir: PathBuf,
    /// Featurization + Dataset-construction thread count (None = all cores).
    #[serde(default)]
    featurizer_threads: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RepresentationConfig {
    name: String,
    train_stream: StreamConfig,
    validation_stream: StreamConfig,
    dataset_parameters_str: String,
    spill_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrialConfig {
    number: u64,
    parameters_str: String,
    /// Index into TrainConfig::representations.
    representation: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PredictConfig {
    lightgbm_lib: PathBuf,
    stream: StreamConfig,
    model_paths: Vec<PathBuf>,
    spill_dir: PathBuf,
    #[serde(default)]
    featurizer_threads: Option<usize>,
}

struct SharedDataset(DatasetHandle);
unsafe impl Send for SharedDataset {}
unsafe impl Sync for SharedDataset {}

fn load_split_labels(index_path: &Path, split: &str) -> Result<Vec<f32>> {
    let file = File::open(index_path)
        .with_context(|| format!("cannot open {}", index_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let roots: Vec<usize> = ["sample_id", "boolean_value", "split"]
        .iter()
        .map(|name| {
            schema
                .index_of(name)
                .with_context(|| format!("task index missing {name}"))
        })
        .collect::<Result<_>>()?;
    let projection = ProjectionMask::roots(builder.parquet_schema(), roots);
    let reader = builder.with_projection(projection).build()?;
    let mut entries: Vec<(i64, f32)> = Vec::new();
    for batch in reader {
        let batch = batch?;
        let ids = batch
            .column(batch.schema().index_of("sample_id")?)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow!("sample_id must be int64"))?
            .clone();
        let labels = batch
            .column(batch.schema().index_of("boolean_value")?)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| anyhow!("boolean_value must be bool"))?
            .clone();
        let split_column = batch.schema().index_of("split")?;
        for row in 0..batch.num_rows() {
            if stream::split_at(&batch, split_column, row)? != split {
                continue;
            }
            if ids.is_null(row) || labels.is_null(row) {
                bail!("selected task index IDs and labels cannot be null")
            }
            entries.push((ids.value(row), if labels.value(row) { 1.0 } else { 0.0 }));
        }
    }
    if entries.is_empty() {
        bail!("task index split {split:?} has no rows")
    }
    entries.sort_by_key(|(sample_id, _)| *sample_id);
    for pair in entries.windows(2) {
        if pair[0].0 == pair[1].0 {
            bail!("task index sample_id must be unique within the split")
        }
    }
    Ok(entries.into_iter().map(|(_, label)| label).collect())
}

fn segment_names(config: &StreamConfig) -> &'static [&'static str] {
    match (config.representation, config.segment_kind) {
        (Representation::Summary, _) => &["full"],
        (Representation::Temporal, SegmentKind::VisitHalves) => {
            &["full", "earlier", "recent"]
        }
        (Representation::Temporal, SegmentKind::VisitBuckets) => {
            &["all", "prior", "current"]
        }
    }
}

/// Feature-name derivation mirroring `reference_features` exactly; the
/// byte-identity parity tests hold this to the Python implementation.
fn feature_names(config: &StreamConfig) -> Vec<String> {
    let timed = config.timed;
    let aggregates: &[&str] = if timed {
        &[
            "__sample__event_count",
            "__sample__code_count",
            "__sample__distinct_time_count",
            "__sample__window_hours",
        ]
    } else {
        &[
            "__sample__event_count",
            "__sample__code_count",
            "__sample__distinct_time_count",
        ]
    };
    let numeric_stats: &[&str] = if timed {
        &[
            "count",
            "mean",
            "std",
            "min",
            "max",
            "median",
            "latest_observed",
            "delta",
            "first_recency",
            "recency",
        ]
    } else {
        &[
            "count", "mean", "std", "min", "max", "median", "latest_observed",
            "delta",
        ]
    };
    let occurrence_stats: &[&str] = if timed {
        &["count", "first_recency", "recency"]
    } else {
        &["count"]
    };
    let segments = segment_names(config);
    let occurrence: HashSet<&str> =
        config.occurrence_codes.iter().map(String::as_str).collect();
    let mut names: Vec<String> =
        aggregates.iter().map(|name| name.to_string()).collect();
    for code in config
        .codes
        .iter()
        .filter(|code| !occurrence.contains(code.as_str()))
    {
        for segment in segments {
            for stat in numeric_stats {
                names.push(format!("{code}__{segment}__{stat}"));
            }
        }
    }
    for code in config
        .codes
        .iter()
        .filter(|code| occurrence.contains(code.as_str()))
    {
        for segment in segments {
            for stat in occurrence_stats {
                names.push(format!("{code}__{segment}__{stat}"));
            }
        }
    }
    names
}

const NAME_FORBIDDEN: &[char] = &['[', ']', '{', '}', '"', ':', ',', '\\'];

fn percent_encode(name: &str) -> String {
    let mut encoded = String::with_capacity(name.len() * 3);
    for byte in name.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-' | b'~') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn lightgbm_feature_names(names: &[String]) -> Vec<String> {
    let clean = names.iter().all(|name| {
        !name.is_empty()
            && name.is_ascii()
            && !name.chars().any(|c| c.is_whitespace())
            && !name.chars().any(|c| NAME_FORBIDDEN.contains(&c))
    });
    if clean {
        names.to_vec()
    } else {
        names
            .iter()
            .map(|name| format!("mxgb_{}", percent_encode(name)))
            .collect()
    }
}

const OFFSET_EXEMPT_STATS: &[&str] = &[
    "count",
    "event_count",
    "code_count",
    "distinct_time_count",
    "window_hours",
];

fn offset_mask(names: &[String], occurrence_codes: &[String]) -> Result<Vec<bool>> {
    let occurrence: HashSet<&str> =
        occurrence_codes.iter().map(String::as_str).collect();
    names
        .iter()
        .map(|name| {
            let mut parts = name.rsplitn(3, "__");
            let stat = parts.next().unwrap_or("");
            let segment = parts.next();
            let code = parts.next();
            let (Some(_), Some(code)) = (segment, code) else {
                bail!("feature name is not code__segment__statistic: {name:?}")
            };
            Ok(!(occurrence.contains(code) || OFFSET_EXEMPT_STATS.contains(&stat)))
        })
        .collect()
}

fn apply_offset(values: &mut [f32], mask: &[bool], feature_count: usize) {
    for row in values.chunks_mut(feature_count) {
        for (value, &marked) in row.iter_mut().zip(mask) {
            if !marked {
                continue;
            }
            if *value >= 0.0 {
                *value += ZERO_VACATING_OFFSET;
            } else if value.is_nan() {
                *value = 0.0;
            }
        }
    }
}

/// Bridge the featurizer's chunk callback to owned value buffers.
fn run_stream(
    config: &StreamConfig,
    requested_rows: Option<&[u64]>,
    threads: Option<usize>,
    mut on_rows: impl FnMut(&[u64], Vec<f32>) -> Result<()>,
) -> Result<stream::ExecutionStatistics> {
    stream::execute_rows(config, requested_rows, threads, &mut |positions, _sample_ids, values| {
        on_rows(positions, values.to_vec())
    })
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Flat column-sorted bin sample, mirroring the Python wide-COO layout.
struct FlatSample {
    values: Vec<f64>,
    rows: Vec<i32>,
    bounds: Vec<usize>,
    sampled_rows: usize,
}

fn collect_bin_sample(
    train_stream: &StreamConfig,
    feature_count: usize,
    mask: &[bool],
    indices: &[i32],
    threads: Option<usize>,
) -> Result<FlatSample> {
    let requested: Vec<u64> = indices.iter().map(|&row| row as u64).collect();
    let mut coo_columns: Vec<i64> = Vec::new();
    let mut coo_rows: Vec<i32> = Vec::new();
    let mut coo_values: Vec<f64> = Vec::new();
    let mut sampled_rows = 0usize;
    run_stream(train_stream, Some(&requested), threads, |positions, mut values| {
        apply_offset(&mut values, mask, feature_count);
        for (local, _position) in positions.iter().enumerate() {
            let row = &values[local * feature_count..(local + 1) * feature_count];
            for (column, &value) in row.iter().enumerate() {
                let wide = value as f64;
                if wide.abs() > ZERO_THRESHOLD || wide.is_nan() {
                    coo_columns.push(column as i64);
                    coo_rows.push((sampled_rows + local) as i32);
                    coo_values.push(wide);
                }
            }
        }
        sampled_rows += positions.len();
        Ok(())
    })?;
    if sampled_rows != indices.len() {
        bail!("bin sample collected an unexpected row count")
    }

    // Counting sort by column: O(nnz + ncol) and stable, so each column's
    // rows stay ascending exactly as the streamed row order delivered them.
    let mut counts = vec![0usize; feature_count + 1];
    for &column in &coo_columns {
        counts[column as usize + 1] += 1;
    }
    for column in 0..feature_count {
        counts[column + 1] += counts[column];
    }
    let bounds = counts.clone();
    let mut values = vec![0f64; coo_values.len()];
    let mut rows = vec![0i32; coo_rows.len()];
    let mut cursors = counts;
    for position in 0..coo_columns.len() {
        let column = coo_columns[position] as usize;
        let target = cursors[column];
        cursors[column] += 1;
        values[target] = coo_values[position];
        rows[target] = coo_rows[position];
    }
    Ok(FlatSample {
        values,
        rows,
        bounds,
        sampled_rows,
    })
}

#[allow(clippy::too_many_arguments)]
fn push_split(
    lgbm: &LightGbm,
    handle: DatasetHandle,
    config: &StreamConfig,
    feature_count: usize,
    labels_sorted: &[f32],
    mask: &[bool],
    expected_rows: usize,
    threads: Option<usize>,
) -> Result<()> {
    lgbm.check(
        unsafe { (lgbm.dataset_init_streaming)(handle, 0, 0, 0, 1, 1, -1) },
        "LGBM_DatasetInitStreaming",
    )?;
    let mut cursor = 0usize;
    run_stream(config, None, threads, |positions, mut values| {
        apply_offset(&mut values, mask, feature_count);
        let labels: Vec<f32> = positions
            .iter()
            .map(|&position| labels_sorted[position as usize])
            .collect();
        lgbm.check(
            unsafe {
                (lgbm.dataset_push_rows_with_metadata)(
                    handle,
                    values.as_ptr() as *const c_void,
                    C_API_DTYPE_FLOAT32,
                    positions.len() as c_int,
                    feature_count as c_int,
                    cursor as c_int,
                    labels.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                )
            },
            "LGBM_DatasetPushRowsWithMetadata",
        )?;
        cursor += positions.len();
        Ok(())
    })?;
    if cursor != expected_rows {
        bail!("stream pushed {cursor} rows, expected {expected_rows}")
    }
    lgbm.check(
        unsafe { (lgbm.dataset_mark_finished)(handle) },
        "LGBM_DatasetMarkFinished",
    )
}

fn set_feature_names(
    lgbm: &LightGbm,
    handle: DatasetHandle,
    encoded_names: &[String],
) -> Result<()> {
    let storage: Vec<CString> = encoded_names
        .iter()
        .map(|name| CString::new(name.as_str()).context("feature name has NUL"))
        .collect::<Result<_>>()?;
    let pointers: Vec<*const c_char> =
        storage.iter().map(|name| name.as_ptr()).collect();
    lgbm.check(
        unsafe {
            (lgbm.dataset_set_feature_names)(
                handle,
                pointers.as_ptr(),
                pointers.len() as c_int,
            )
        },
        "LGBM_DatasetSetFeatureNames",
    )
}

struct TrialOutcome {
    number: u64,
    representation: usize,
    best_iteration: usize,
    prediction_path: PathBuf,
    prediction_sha256: String,
    model_path: PathBuf,
    model_sha256: String,
}

/// Mirror of the Python executor's narrowly recoverable LightGBM pathology:
/// sampled splits can produce an empty child, which fails only that trial.
fn recoverable_trial_failure(error: &anyhow::Error) -> Option<String> {
    let message = format!("{error}");
    let recoverable = ["left", "right"].iter().any(|side| {
        message.contains(&format!("Check failed: (best_split_info.{side}_count) > (0)"))
    });
    recoverable.then_some(message)
}

#[allow(clippy::too_many_arguments)]
fn run_trial(
    lgbm: &LightGbm,
    train: DatasetHandle,
    validation: DatasetHandle,
    trial: &TrialConfig,
    num_boost_round: usize,
    early_stopping_rounds: usize,
    validation_rows: usize,
    spill_dir: &Path,
) -> Result<TrialOutcome> {
    let parameters = CString::new(trial.parameters_str.as_str())
        .context("trial parameters contain NUL")?;
    let mut booster: BoosterHandle = std::ptr::null_mut();
    lgbm.check(
        unsafe { (lgbm.booster_create)(train, parameters.as_ptr(), &mut booster) },
        "LGBM_BoosterCreate",
    )?;
    let result = run_trial_with_booster(
        lgbm,
        booster,
        validation,
        trial,
        num_boost_round,
        early_stopping_rounds,
        validation_rows,
        spill_dir,
    );
    let free_code = unsafe { (lgbm.booster_free)(booster) };
    lgbm.check(free_code, "LGBM_BoosterFree")?;
    result
}

#[allow(clippy::too_many_arguments)]
fn run_trial_with_booster(
    lgbm: &LightGbm,
    booster: BoosterHandle,
    validation: DatasetHandle,
    trial: &TrialConfig,
    num_boost_round: usize,
    early_stopping_rounds: usize,
    validation_rows: usize,
    spill_dir: &Path,
) -> Result<TrialOutcome> {
    lgbm.check(
        unsafe { (lgbm.booster_add_valid_data)(booster, validation) },
        "LGBM_BoosterAddValidData",
    )?;
    let mut eval_counts: c_int = 0;
    lgbm.check(
        unsafe { (lgbm.booster_get_eval_counts)(booster, &mut eval_counts) },
        "LGBM_BoosterGetEvalCounts",
    )?;
    if eval_counts != 1 {
        bail!("trial expects exactly one evaluation metric, found {eval_counts}")
    }

    // Exact replica of lightgbm.callback._EarlyStoppingCallback semantics:
    // iteration 0 seeds the best score, later iterations must be strictly
    // greater (ties keep the earliest best), stopping triggers when
    // `iteration - best_iteration >= stopping_rounds`, and the final planned
    // iteration also resolves best_iteration; the engine then stores
    // `best_iter + 1`.
    let mut best_score = f64::NEG_INFINITY;
    let mut best_iter = 0usize;
    let mut best_iteration = 0usize;
    for iteration in 0..num_boost_round {
        let mut finished: c_int = 0;
        lgbm.check(
            unsafe { (lgbm.booster_update_one_iter)(booster, &mut finished) },
            "LGBM_BoosterUpdateOneIter",
        )?;
        let mut out_len: c_int = 0;
        let mut score: c_double = 0.0;
        lgbm.check(
            unsafe { (lgbm.booster_get_eval)(booster, 1, &mut out_len, &mut score) },
            "LGBM_BoosterGetEval",
        )?;
        if out_len != 1 {
            bail!("validation evaluation returned {out_len} values")
        }
        if iteration == 0 || score > best_score {
            best_score = score;
            best_iter = iteration;
        }
        if iteration - best_iter >= early_stopping_rounds {
            best_iteration = best_iter + 1;
            break;
        }
        if iteration + 1 == num_boost_round {
            best_iteration = best_iter + 1;
        }
        if finished != 0 {
            // LightGBM stopped growing (no further splits); mirror the
            // Python path, where training also ends and the callback's
            // final-iteration check resolves the best iteration.
            best_iteration = best_iter + 1;
            break;
        }
    }

    let mut current: c_int = 0;
    lgbm.check(
        unsafe { (lgbm.booster_get_current_iteration)(booster, &mut current) },
        "LGBM_BoosterGetCurrentIteration",
    )?;
    while (current as usize) > best_iteration {
        lgbm.check(
            unsafe { (lgbm.booster_rollback_one_iter)(booster) },
            "LGBM_BoosterRollbackOneIter",
        )?;
        current -= 1;
    }

    let mut predict_len: i64 = 0;
    lgbm.check(
        unsafe { (lgbm.booster_get_num_predict)(booster, 1, &mut predict_len) },
        "LGBM_BoosterGetNumPredict",
    )?;
    if predict_len as usize != validation_rows {
        bail!(
            "validation prediction length {predict_len} differs from rows {validation_rows}"
        )
    }
    let mut predictions = vec![0f64; predict_len as usize];
    let mut written: i64 = 0;
    lgbm.check(
        unsafe {
            (lgbm.booster_get_predict)(
                booster,
                1,
                &mut written,
                predictions.as_mut_ptr(),
            )
        },
        "LGBM_BoosterGetPredict",
    )?;
    if written != predict_len {
        bail!("LightGBM wrote {written} predictions into a {predict_len}-value buffer")
    }

    let mut buffer_len: i64 = 1 << 20;
    let model = loop {
        let mut buffer = vec![0u8; buffer_len as usize];
        let mut out_len: i64 = 0;
        lgbm.check(
            unsafe {
                (lgbm.booster_save_model_to_string)(
                    booster,
                    0,
                    -1,
                    0,
                    buffer_len,
                    &mut out_len,
                    buffer.as_mut_ptr() as *mut c_char,
                )
            },
            "LGBM_BoosterSaveModelToString",
        )?;
        if out_len <= buffer_len {
            buffer.truncate((out_len - 1).max(0) as usize);
            break buffer;
        }
        buffer_len = out_len + 1;
    };

    let prediction_bytes = bytemuck::cast_slice::<f64, u8>(&predictions);
    let prediction_path = spill_dir.join(format!("trial_{}.pred.f64", trial.number));
    std::fs::write(&prediction_path, prediction_bytes)?;
    let model_path = spill_dir.join(format!("trial_{}.model.txt", trial.number));
    std::fs::write(&model_path, &model)?;
    Ok(TrialOutcome {
        number: trial.number,
        representation: trial.representation,
        best_iteration,
        prediction_sha256: hex(&Sha256::digest(prediction_bytes)),
        prediction_path,
        model_sha256: hex(&Sha256::digest(&model)),
        model_path,
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DumpConfig {
    stream: StreamConfig,
    #[serde(default)]
    requested_rows: Option<Vec<u64>>,
    output_path: PathBuf,
    #[serde(default)]
    featurizer_threads: Option<usize>,
}

/// Test-support mode: featurize one split and write the row-major f32 matrix
/// to a flat file, so the Python parity suite compares matrices against the
/// oracle without any wire-protocol reader.
pub(crate) fn run_dump(config_path: &Path) -> Result<()> {
    let raw = std::fs::read(config_path)
        .with_context(|| format!("cannot read {}", config_path.display()))?;
    let config: DumpConfig = serde_json::from_slice(&raw)?;
    let names = feature_names(&config.stream);
    let feature_count = names.len();
    let mut writer = BufWriter::new(File::create(&config.output_path)?);
    let mut digest = Sha256::new();
    let mut total_rows = 0usize;
    run_stream(
        &config.stream,
        config.requested_rows.as_deref(),
        config.featurizer_threads,
        |positions, values| {
            let bytes = bytemuck::cast_slice::<f32, u8>(&values);
            digest.update(bytes);
            writer.write_all(bytes)?;
            total_rows += positions.len();
            Ok(())
        },
    )?;
    writer.flush()?;
    let report = json!({
        "rows": total_rows,
        "feature_count": feature_count,
        "values_sha256": hex(&digest.finalize()),
        "feature_names": names,
    });
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

/// One representation's constructed Datasets, held for the whole study so that
/// trials from different representations can run at the same time.
struct BuiltRepresentation {
    name: String,
    train: SharedDataset,
    validation: SharedDataset,
    train_rows: usize,
    validation_rows: usize,
    feature_count: usize,
    sampled_rows: usize,
    sample_staging_bytes: usize,
    construction_seconds: f64,
    spill_dir: PathBuf,
}

fn build_representation(
    lgbm: &LightGbm,
    rep: &RepresentationConfig,
    featurizer_threads: Option<usize>,
) -> Result<BuiltRepresentation> {
    std::fs::create_dir_all(&rep.spill_dir)?;
    let names = feature_names(&rep.train_stream);
    let encoded_names = lightgbm_feature_names(&names);
    let mask = offset_mask(&names, &rep.train_stream.occurrence_codes)?;
    let feature_count = names.len();

    let train_labels =
        load_split_labels(&rep.train_stream.index_path, &rep.train_stream.split)?;
    let validation_labels = load_split_labels(
        &rep.validation_stream.index_path,
        &rep.validation_stream.split,
    )?;
    let train_rows = train_labels.len();
    let validation_rows = validation_labels.len();

    let dataset_parameters = CString::new(rep.dataset_parameters_str.as_str())
        .context("dataset parameters contain NUL")?;
    let construction_started = std::time::Instant::now();
    let mut sample_count: c_int = 0;
    lgbm.check(
        unsafe {
            (lgbm.get_sample_count)(
                train_rows as c_int,
                dataset_parameters.as_ptr(),
                &mut sample_count,
            )
        },
        "LGBM_GetSampleCount",
    )?;
    let mut indices = vec![0i32; sample_count as usize];
    let mut actual: c_int = 0;
    lgbm.check(
        unsafe {
            (lgbm.sample_indices)(
                train_rows as c_int,
                dataset_parameters.as_ptr(),
                indices.as_mut_ptr() as *mut c_void,
                &mut actual,
            )
        },
        "LGBM_SampleIndices",
    )?;
    indices.truncate(actual as usize);
    if indices.windows(2).any(|pair| pair[0] > pair[1]) {
        bail!("LightGBM returned unsorted sample indices")
    }

    let sample = collect_bin_sample(
        &rep.train_stream,
        feature_count,
        &mask,
        &indices, featurizer_threads,
    )?;
    let value_pointers: Vec<*mut c_double> = sample
        .bounds
        .iter()
        .take(feature_count)
        .map(|&offset| unsafe { sample.values.as_ptr().add(offset) as *mut c_double })
        .collect();
    let index_pointers: Vec<*mut c_int> = sample
        .bounds
        .iter()
        .take(feature_count)
        .map(|&offset| unsafe { sample.rows.as_ptr().add(offset) as *mut c_int })
        .collect();
    let num_per_col: Vec<c_int> = sample
        .bounds
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as c_int)
        .collect();
    let mut train_handle: DatasetHandle = std::ptr::null_mut();
    lgbm.check(
        unsafe {
            (lgbm.dataset_create_from_sampled_column)(
                value_pointers.as_ptr() as *mut *mut c_double,
                index_pointers.as_ptr() as *mut *mut c_int,
                feature_count as c_int,
                num_per_col.as_ptr(),
                sample.sampled_rows as c_int,
                train_rows as c_int,
                train_rows as i64,
                dataset_parameters.as_ptr(),
                &mut train_handle,
            )
        },
        "LGBM_DatasetCreateFromSampledColumn",
    )?;
    drop((value_pointers, index_pointers, num_per_col));
    let sample_staging_bytes =
        sample.values.len() * 8 + sample.rows.len() * 4 + sample.bounds.len() * 8;
    let sampled_rows = sample.sampled_rows;
    drop(sample);

    push_split(
        &lgbm,
        train_handle,
        &rep.train_stream,
        feature_count,
        &train_labels,
        &mask,
        train_rows,
        featurizer_threads,
    )?;
    set_feature_names(&lgbm, train_handle, &encoded_names)?;

    let mut validation_handle: DatasetHandle = std::ptr::null_mut();
    lgbm.check(
        unsafe {
            (lgbm.dataset_create_by_reference)(
                train_handle,
                validation_rows as i64,
                &mut validation_handle,
            )
        },
        "LGBM_DatasetCreateByReference",
    )?;
    push_split(
        &lgbm,
        validation_handle,
        &rep.validation_stream,
        feature_count,
        &validation_labels,
        &mask,
        validation_rows,
        featurizer_threads,
    )?;
    set_feature_names(&lgbm, validation_handle, &encoded_names)?;

    let construction_seconds = construction_started.elapsed().as_secs_f64();
    Ok(BuiltRepresentation {
        name: rep.name.clone(),
        train: SharedDataset(train_handle),
        validation: SharedDataset(validation_handle),
        train_rows,
        validation_rows,
        feature_count,
        sampled_rows,
        sample_staging_bytes,
        construction_seconds,
        spill_dir: rep.spill_dir.clone(),
    })
}

pub(crate) fn run_train(config_path: &Path) -> Result<()> {
    let raw = std::fs::read(config_path)
        .with_context(|| format!("cannot read {}", config_path.display()))?;
    let config: TrainConfig = serde_json::from_slice(&raw)?;
    if config.parallel_trials == 0 || config.num_boost_round == 0 {
        bail!("parallel_trials and num_boost_round must be positive")
    }
    std::fs::create_dir_all(&config.spill_dir)?;
    let lgbm = LightGbm::load(&config.lightgbm_lib)?;

    if config.representations.is_empty() {
        bail!("at least one representation is required")
    }
    for (index, trial) in config.trials.iter().enumerate() {
        if trial.representation >= config.representations.len() {
            bail!(
                "trial {} at index {index} names representation {}, but only {} were given",
                trial.number,
                trial.representation,
                config.representations.len()
            )
        }
    }

    // Construct every representation before any trial starts. Trials are handed
    // out in the order given, which the caller sorts by representation, so a
    // worker stays on one Dataset until that representation is exhausted and
    // only then crosses into the next.
    let mut built = Vec::with_capacity(config.representations.len());
    for rep in &config.representations {
        built.push(build_representation(&lgbm, rep, config.featurizer_threads)?);
    }

    let next_trial = AtomicUsize::new(0);
    let outcomes: Mutex<Vec<Result<TrialOutcome, (u64, usize, String)>>> =
        Mutex::new(Vec::new());
    let failure: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let built_ref = &built;
    std::thread::scope(|scope| {
        for _ in 0..config.parallel_trials.min(config.trials.len().max(1)) {
            scope.spawn(|| {
                loop {
                    let index = next_trial.fetch_add(1, Ordering::SeqCst);
                    let Some(trial) = config.trials.get(index) else {
                        return;
                    };
                    let rep = &built_ref[trial.representation];
                    match run_trial(
                        &lgbm,
                        rep.train.0,
                        rep.validation.0,
                        trial,
                        config.num_boost_round,
                        config.early_stopping_rounds,
                        rep.validation_rows,
                        &rep.spill_dir,
                    ) {
                        Ok(outcome) => outcomes.lock().unwrap().push(Ok(outcome)),
                        Err(error) => {
                            if let Some(message) = recoverable_trial_failure(&error) {
                                outcomes
                                    .lock()
                                    .unwrap()
                                    .push(Err((trial.number, trial.representation, message)));
                                continue;
                            }
                            failure.lock().unwrap().get_or_insert(error);
                            return;
                        }
                    }
                }
            });
        }
    });
    for rep in &built {
        let free_valid = unsafe { (lgbm.dataset_free)(rep.validation.0) };
        let free_train = unsafe { (lgbm.dataset_free)(rep.train.0) };
        lgbm.check(free_valid, "LGBM_DatasetFree(validation)")?;
        lgbm.check(free_train, "LGBM_DatasetFree(train)")?;
    }
    if let Some(error) = failure.into_inner().unwrap() {
        return Err(error);
    }

    let mut results = outcomes.into_inner().unwrap();
    results.sort_by_key(|outcome| match outcome {
        Ok(success) => success.number,
        Err((number, _, _)) => *number,
    });
    let report = json!({
        "executor": "rust-lightgbm-v1",
        "featurizer_threads": config.featurizer_threads,
        "representations": built
            .iter()
            .map(|rep| json!({
                "name": rep.name,
                "feature_count": rep.feature_count,
                "train_rows": rep.train_rows,
                "validation_rows": rep.validation_rows,
                "sampled_rows": rep.sampled_rows,
                "bin_sample_staging_bytes": rep.sample_staging_bytes,
                "construction_seconds": rep.construction_seconds,
            }))
            .collect::<Vec<_>>(),
        "trials": results
            .iter()
            .map(|outcome| match outcome {
                Ok(success) => json!({
                    "number": success.number,
                    "representation": success.representation,
                    "best_iteration": success.best_iteration,
                    "prediction_path": success.prediction_path,
                    "prediction_sha256": success.prediction_sha256,
                    "model_path": success.model_path,
                    "model_sha256": success.model_sha256,
                }),
                Err((number, representation, message)) => json!({
                    "number": number,
                    "representation": representation,
                    "failure": message,
                }),
            })
            .collect::<Vec<_>>(),
    });
    let results_path = config.spill_dir.join("results.json");
    std::fs::write(&results_path, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", results_path.display());
    Ok(())
}

pub(crate) fn run_predict(config_path: &Path) -> Result<()> {
    let raw = std::fs::read(config_path)
        .with_context(|| format!("cannot read {}", config_path.display()))?;
    let config: PredictConfig = serde_json::from_slice(&raw)?;
    std::fs::create_dir_all(&config.spill_dir)?;
    let lgbm = LightGbm::load(&config.lightgbm_lib)?;

    let names = feature_names(&config.stream);
    let mask = offset_mask(&names, &config.stream.occurrence_codes)?;
    let feature_count = names.len();

    let mut boosters: Vec<BoosterHandle> = Vec::new();
    for path in &config.model_paths {
        let model = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        let model_c = CString::new(model).context("model string contains NUL")?;
        let mut iterations: c_int = 0;
        let mut handle: BoosterHandle = std::ptr::null_mut();
        lgbm.check(
            unsafe {
                (lgbm.booster_load_model_from_string)(
                    model_c.as_ptr(),
                    &mut iterations,
                    &mut handle,
                )
            },
            "LGBM_BoosterLoadModelFromString",
        )?;
        boosters.push(handle);
    }

    let empty_parameters = CString::new("").unwrap();
    let mut writers: Vec<BufWriter<File>> = config
        .model_paths
        .iter()
        .enumerate()
        .map(|(rank, _)| {
            let path = config.spill_dir.join(format!("member_{rank}.pred.f64"));
            Ok(BufWriter::new(File::create(path)?))
        })
        .collect::<Result<_>>()?;
    let mut digests: Vec<Sha256> = boosters.iter().map(|_| Sha256::new()).collect();
    let mut total_rows = 0usize;
    run_stream(&config.stream, None, config.featurizer_threads, |positions, mut values| {
        apply_offset(&mut values, mask.as_slice(), feature_count);
        let nrow = positions.len();
        for (rank, &booster) in boosters.iter().enumerate() {
            let mut out = vec![0f64; nrow];
            let mut out_len: i64 = 0;
            lgbm.check(
                unsafe {
                    (lgbm.booster_predict_for_mat)(
                        booster,
                        values.as_ptr() as *const c_void,
                        C_API_DTYPE_FLOAT32,
                        nrow as c_int,
                        feature_count as c_int,
                        1,
                        C_API_PREDICT_NORMAL,
                        0,
                        -1,
                        empty_parameters.as_ptr(),
                        &mut out_len,
                        out.as_mut_ptr(),
                    )
                },
                "LGBM_BoosterPredictForMat",
            )?;
            if out_len as usize != nrow {
                bail!("prediction returned {out_len} values for {nrow} rows")
            }
            let bytes = bytemuck::cast_slice::<f64, u8>(&out);
            digests[rank].update(bytes);
            writers[rank].write_all(bytes)?;
        }
        total_rows += nrow;
        Ok(())
    })?;
    for mut writer in writers {
        writer.flush()?;
    }
    for &booster in &boosters {
        let code = unsafe { (lgbm.booster_free)(booster) };
        lgbm.check(code, "LGBM_BoosterFree")?;
    }
    let report = json!({
        "executor": "rust-lightgbm-v1",
        "rows": total_rows,
        "feature_count": feature_count,
        "members": digests
            .into_iter()
            .enumerate()
            .map(|(rank, digest)| {
                json!({
                    "rank": rank,
                    "prediction_path":
                        config.spill_dir.join(format!("member_{rank}.pred.f64")),
                    "prediction_sha256": hex(&digest.finalize()),
                })
            })
            .collect::<Vec<_>>(),
    });
    let results_path = config.spill_dir.join("results.json");
    std::fs::write(&results_path, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", results_path.display());
    Ok(())
}
