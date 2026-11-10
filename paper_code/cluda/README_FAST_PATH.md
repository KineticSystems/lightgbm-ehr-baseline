# CLUDA exact LOS optimization and split-state reconstruction

The released `load_ICU_datasets/process_miiv.R` is unchanged. Its LOS loop
compares every stay against the complete, repeatedly shrinking time-series
table. `cluda_los_filter_fast.R` provides a one-pass equivalent only
when runtime assertions prove that every released comparison is scalar and
total. It refuses duplicate or missing stay mappings, duplicate admission/LOS
mappings, nonfinite identifiers, nonfinite chart times, and nonfinite selected
LOS values. Consequently it never emulates R's position-dependent recycling
behavior.

`prepare_miiv_cohort_fast.R` integrity-checks both the complete released script
and the exact skipped loop, creates a temporary derived script, and intercepts
only the canonical wrapper's single `sys.source(process_miiv.R)` call. The
released script remains unchanged. The canonical `export.sh` invokes this
wrapper after verifying the exact isolated R runtime, all six imported
MIMIC-IV v1.0 tables, the complete released-script hash, and the skipped-block
hash.

Run the bounded fixtures with:

```bash
paper_code/cluda/.export_assets/r-env/bin/Rscript --vanilla \
  paper_code/cluda/test_los_filter_fast.R
```

The production path is:

```bash
paper_code/cluda/export.sh
paper_code/cluda/check_cohort.sh
paper_code/cluda/check_batch.sh
```

The source manifest records `los_fast_path.json`, its digest, the released
script/block hashes, the derived-script digest, and the real-data scalar/total
invariants. Both checkers revalidate that seal through `require_source()`.

The released script also consumes random numbers while RICU generates
temporary column names.  On the pinned runtime, its single static and
time-series loads reach a 37-name-draw RNG state immediately before
`createDataPartition()`.  That state does not reproduce the jointly released
downstream signatures: loader split counts, rolling-window counts, and the
printed training-age mean and standard deviation.  Those signatures instead
identify one state exactly: 72 name draws, equivalent to one additional
35-draw invocation of the released 41-channel time-series loader.  This is a
paper-artifact-triangulated state, not a claim that the standalone released
script records the missing invocation.

`cluda_split_rng_compat.R` therefore implements a bounded, fail-closed
transition at the split boundary.  It first requires the exact serialized
37-draw RNG hash, performs exactly 35 calls to RICU's temporary-name generator,
and then requires the exact serialized 72-draw hash before allowing the
released `createDataPartition()` call.  The temporary derived script and both
RNG hashes are recorded in `los_fast_path.json` and revalidated through the
source manifest.  `test_split_rng_compat.R` proves the transition and its
wrong-state rejection behavior.

The common task retains two distinct time coordinates.  Tensor slicing,
padding, `stay_hour`, and `sequence_row` use the row positions consumed by the
released `ICUDataset`.  MEDS event/landmark timestamps and `source_time` use
the absolute ICU-relative `charttime` retained by RICU.  This matters for
stays whose first retained grid row is after hour zero; shifting both events
and landmarks to their true source clock leaves prediction-relative
LightGBM statistics unchanged while preserving the source timestamp.  The
synthetic integration test includes a grid beginning at hour 3 and requires
both coordinate systems to reconstruct exactly.
