# What we changed in the authors' code

`td_icu_mortality` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff td_icu_mortality`.

```
paper: td_icu_mortality
upstream: https://github.com/tdgfrost/td-icu-mortality.git
commit: 235ca921bb200dffe6c854ef16949a94435758f4
path: .
tree: 1e7265dc5ce8d8b45656dd954d01707a70014254
modified_upstream_files: 1
deleted_upstream_files: 0
additive_local_files: 11

MODIFIED UPSTREAM FILES

--- utils/preprocessing/tools.py (mode 100644 -> 100644)
--- upstream/utils/preprocessing/tools.py
+++ local/utils/preprocessing/tools.py
@@ -516,7 +516,8 @@
 
 
 def create_final_dataframe(data_option: str, encoded_input_data, labels, encodings, train_patient_ids, val_patient_ids,
-                           test_patient_ids, sorting_columns, grouping_columns, context_length):
+                           test_patient_ids, sorting_columns, grouping_columns, context_length,
+                           export_chunk_callback=None):
     """
     This is the final pipeline for taking the encoded data and labels, and creating dedicated input data
     for the model.
@@ -746,7 +747,7 @@
 
             final_target_data = pl.concat([has_next_state, no_next_state])
 
-            (
+            final_target_data = (
                 final_target_data
                 # Again, sort out the "nulls" for each in way that is consistent with the dtypes
                 .with_columns([
@@ -761,10 +762,19 @@
                     .then(pl.lit([np.nan for _ in range(context_length)]).alias('next_' + col))
                     .otherwise('next_' + col) for col in input_feature_columns if 'value' in col
                 ])
-                # Collect and save the data
+                # Collect the data
                 .collect()
-                .write_parquet(os.path.join(dataframe_path, f'dataframe{idx:03}.parquet'))
             )
+
+            if export_chunk_callback is None:
+                final_target_data.write_parquet(
+                    os.path.join(dataframe_path, f'dataframe{idx:03}.parquet')
+                )
+            else:
+                # The exporter captures the exact author-produced row before the
+                # identifier-dropping HDF5 conversion.  The default released
+                # path above remains unchanged.
+                export_chunk_callback(group, idx, final_target_data)
 
 
 def create_hdf5_array(target_path, batch_dir, features, label_features, context_window):

ADDITIVE LOCAL FILES
.python-version
check_batch.py
check_batch.sh
check_cohort.py
check_cohort.sh
cohort_source.py
derive_outcomes.py
export.sh
export_source.py
requirements-export.txt
test_export.py

status: UPSTREAM_FILES_CHANGED
```
