# Common task format

Each paper's exporter writes one MEDS 0.4 event dataset per prespecified
binary AUROC task.
It preserves the earliest deterministic event stream in the authors' released
cohort path that can still be proved equivalent to their model input.
Paper-specific binning, filling, normalization, and tensor padding are verified
views, not the persistent cross-paper representation.

The paper exporter builds explicitly typed Polars frames and writes this
directory directly. It does not emit NPZ, a dense feature bundle, or any other
intermediate format, and it does not import the `meds` package. Polars may
materialize UTF-8 columns as Arrow `large_string`; shared validation treats
that as equivalent to Arrow `string`.

```text
paper_tasks/<paper>/<task>/
├── manifest.json
├── data/events.parquet
├── metadata/dataset.json
├── metadata/codes.parquet
├── metadata/subject_splits.parquet
├── tasks/<task>/labels.parquet
├── audit/task_index.parquet
├── audit/cohort_reconstruction.json
└── audit/full_batch_recreation.json
```

## Modalities

The persistent common task supports exactly two modality declarations:

- `["structured"]`
- `["structured", "notes"]`

`task.note_codes` is the complete exception list. Every MEDS code not in
`note_codes` is a structured code and is available to shared featurization.
There is no structured-code allowlist, generic auxiliary modality, configurable
input specification, or row-level feature filter. For a structured-only task,
`note_codes` is empty. For a note-using task, every note event has raw UTF-8 in
`text_value`; shared text modeling orders slots by `source_row` and joins them
with two newlines. A sample with no note rows becomes an empty document; note
availability is not an additional cohort filter unless the selected source
experiment explicitly made it one.

Token IDs, attention masks, frozen text embeddings, and paper-specific note
masks are not common-format data. When needed for source parity,
`check_batch.sh` regenerates them with the released loader/tokenizer or compares
them transiently. Imaging pixels, embeddings, times, masks, and modality flags
are never retained.

A minimal task declaration is:

```json
{
  "id": "mortality_48h",
  "name": "In-hospital mortality from first 48 ICU hours",
  "primary_metric": "AUROC",
  "published_auroc": 0.82,
  "observation_window_hours": 48.0,
  "modalities": ["structured", "notes"],
  "note_codes": ["NOTE"]
}
```

The writer derives `modalities` from `note_codes`; paper exporters supply the
task metadata, full MEDS code vocabulary, and note-code exception list.
Tasks may additionally set `temporal_profile` to declare how their events
carry time. The default is `single_visit_timed`: one stay, real elapsed time
between events. `multi_visit_untimed` is for a task whose released model is
shown visit *order* only, with one timestamp per visit and no inter-visit
duration. `multi_visit_timed` has both multiple visits and real time inside
each one. The profile decides which representations the shared runner searches
and which statistics it is allowed to compute; see below.

## MEDS events and labels

`events.parquet` uses the MEDS fields `subject_id int64`, `time timestamp[us]`,
`code string`, and `numeric_value float32`. Required extensions are `sample_id`,
the prediction unit, and `source_row`, deterministic order for tied times.
Exporters may retain `sequence_row` and an exact floating coordinate in
`source_time` when these are needed to prove source-loader parity. Structured
numeric measurements have finite values. A non-note row with null numeric and
null text is an occurrence-only categorical event. Encode a categorical value
in the code itself, such as `blood_type//A`. A structured code cannot mix
numeric and occurrence-only rows. Note events have null numeric values and
non-null `text_value`.

Each observed structured measurement is one row. Missing measurements are
absent events, never zero-filled fake observations. The complete released code
vocabulary lives in `metadata/codes.parquet`, including codes not observed in a
particular split. That table always contains `code` and may contain a MEDS-style
UTF-8 `description` when a compact event identifier needs a lossless,
human-readable or source-reconstructing payload. Descriptions are provenance,
not additional LightGBM inputs; shared featurization consumes only `code`.

The binary label file contains `subject_id`, `sample_id`, `prediction_time`,
and `boolean_value`. The audit index adds `observation_start` and the exact
released `train`, `validation`, or `test` assignment. The shared reader asserts
that labels and subject/split memberships exactly match this index. It may
retain source identifiers for verification, but these are never model inputs.

The released split is a fidelity target. If a source assigns stays rather than
patients and one subject crosses partitions, the task preserves those exact
sample assignments, records every `(subject_id, split)` membership, and reports
the overlap in its manifest.

For rolling prediction, each landmark is an independent sample with a unique
`sample_id` and a duplicated copy of exactly the eligible event prefix. For a
paper evaluated over multiple folds or source split seeds, each fold/seed is a
fully duplicated task replica of the same clinical endpoint. Each replica is
trained independently; it is never collapsed into one artificial split.

## Shared views and verification

Shared featurization always constructs a `summary` view from every non-note
event code, and — where the task's temporal profile allows one — a second,
segmented view:

- `summary`: ten statistics per numeric code and three per occurrence-only
  code, over the full observation window;
- `visit_halves` (`single_visit_timed`): the same statistics over three
  segments — full, earlier, and recent — split at one cohort-wide duration
  before prediction time;
- `visit_buckets` (`multi_visit_untimed`): the same statistics over all,
  prior, and current, split at the sample's own last event time, which for a
  task carrying one timestamp per visit is exactly the current visit;
- `visit_scope` and `visit_scope_halved` (`multi_visit_timed`, when the
  export carries visit identity — `visit_id` on events and
  `current_visit_id` on the task index): segments by encounter membership
  rather than time. `visit_scope` is all / prior / current, where an event is
  current when its `visit_id` equals the sample's `current_visit_id` and
  prior otherwise (events tied to no encounter count as prior).
  `visit_scope_halved` additionally splits the current visit at the midpoint
  between its first event and prediction time (all / prior / current_early /
  current_late). Both are searched for a `multi_visit_timed` task; a task of
  that profile without visit identity gets `summary` only, because distinct
  event time is not the visit key there and a last-event boundary would split
  measurement instants rather than visits.

Every timing statistic is measured backwards from `prediction_time`, never
forwards from `observation_start`. Only a prediction-relative anchor is
comparable across samples whose windows differ in length, so a feature counted
from the start of the window is not part of the contract.

The `visit_halves` boundary is one absolute duration, fitted once on the
training split as half the median training window width and then applied
unchanged to every split. A per-sample fraction would make "recent" mean 12
hours for a 24-hour stay and 100 for a 200-hour one, which is exactly the
window-length dependence prediction-relative anchoring exists to remove.

Every view is prefixed by the same per-sample aggregate block: total event
count, distinct observed code count, and distinct event-instant count. A timed
profile adds the elapsed observation width in hours. That width is the history
available at prediction time, never the eventual length of stay, so it carries
no future information.

A `multi_visit_untimed` task drops both prediction-relative timing statistics
and the observation width, because its released model provably cannot see
elapsed time and those features would be an unfair advantage rather than a
fair baseline. Every remaining statistic summarizes values, not instants. For
a task whose vocabulary is entirely occurrence-only, that leaves exactly
`count` per code and segment.

The observation window runs from each sample's `observation_start` through its
`prediction_time`. This matters for rolling landmarks and variable-length
stays. The task-level `observation_window_hours` is only the verified maximum
and determines a common matrix width; it is never used as a replacement
denominator for a shorter sample's recency.

The complete released vocabulary remains in `metadata/codes.parquet`, but a
declared code with no event in any split does not materialize a constant model
column. Codes observed only outside training are retained and typed from those
events, so their missingness in training is preserved without guessing whether
an entirely unobserved code was numeric or occurrence-only.

The shared multithreaded Rust featurizer streams Parquet record batches and
maps event codes to integer vocabulary positions before grouping. Event-level
strings are never materialized as Python objects. Every statistic, including
the median, is exact; observed `(sample, code)` series are sorted by event time
and `source_row`, processed in parallel across samples, and released before
LightGBM fitting begins.

A separate slow Python reference implements the same contract with direct
nested loops and is never used for model fitting. Tests compare its complete
matrices with Rust on deterministic randomized mixed streams at
`rtol=atol=1e-6`, including one- and four-thread executions.

Numeric statistics for an unobserved code remain `NaN` apart from a zero
`count`, allowing LightGBM to use its native missing-value branches.
High-cardinality occurrence codes use sparse count/timing features; a zero
count denotes absence.

`export.sh` constructs the task. `check_batch.sh` independently reopens it,
recreates the authors' complete retained structured/raw-note projection, and
asserts equality. It must also assert that the event code vocabulary is exact,
that the `note_codes` exception list is exact, and that no imaging or other
undeclared modality entered the task. A passing test records and hashes
`audit/full_batch_recreation.json`.

Every paper additionally provides `check_cohort.sh`. That paper-specific
command compares MEDS identifiers, labels, counts, prevalence, splits, and all
other available paper/artifact or attributable construction-path facts and
records `audit/cohort_reconstruction.json`. Both the cohort and batch checks
are mandatory and must report exact status before shared training.

The export manifest is immutable. Checkers write these reports without
rewriting it, and shared training records both hashes in its result manifest.

The shared runner always fits on `train`, uses `validation` for joint
representation/hyperparameter selection and early stopping, and evaluates the
selected model once on `test`.
