# What we changed in the authors' code

`medpatch` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff medpatch`.

```
paper: medpatch
upstream: https://github.com/nyuad-cai/MedPatch.git
commit: f33c5ed61c0ea00c28051bb2b7b50ed50485b90b
path: .
tree: 34f1cc3f6a317f63fa9dcce01a33c338eade7182
modified_upstream_files: 1
deleted_upstream_files: 64
additive_local_files: 11

MODIFIED UPSTREAM FILES

--- mimic4extract/mimic3benchmark/mimic3csv.py (mode 100644 -> 100644)
--- upstream/mimic4extract/mimic3benchmark/mimic3csv.py
+++ local/mimic4extract/mimic3benchmark/mimic3csv.py
@@ -179,6 +179,7 @@
             self.curr_obs = []
 
     data_stats = DataStats()
+    retained_rows = 0
 
     def write_current_observations():
         dn = os.path.join(output_path, str(data_stats.curr_subject_id))
@@ -222,6 +223,8 @@
             write_current_observations()
         data_stats.curr_obs.append(row_out)
         data_stats.curr_subject_id = row['subject_id']
+        retained_rows += 1
 
     if data_stats.curr_subject_id != '':
         write_current_observations()
+    print('retained_event_rows {}: {}'.format(table.lower(), retained_rows))

DELETED UPSTREAM FILES
datasets/__pycache__/DataFusion.cpython-36.pyc
datasets/__pycache__/Quatrimodal_Fusion.cpython-36.pyc
datasets/__pycache__/__init__.cpython-36.pyc
datasets/__pycache__/__init__.cpython-39.pyc
datasets/__pycache__/cxr_dataset.cpython-36.pyc
datasets/__pycache__/cxr_dataset.cpython-39.pyc
datasets/__pycache__/ehr_dataset.cpython-36.pyc
datasets/__pycache__/ehr_dataset.cpython-39.pyc
datasets/__pycache__/fusion.cpython-36.pyc
datasets/__pycache__/fusion.cpython-39.pyc
ehr_utils/__pycache__/__init__.cpython-36.pyc
ehr_utils/__pycache__/__init__.cpython-39.pyc
ehr_utils/__pycache__/preprocessing.cpython-36.pyc
ehr_utils/__pycache__/preprocessing.cpython-39.pyc
models/__pycache__/__init__.cpython-36.pyc
models/__pycache__/__init__.cpython-39.pyc
models/__pycache__/classifier.cpython-36.pyc
models/__pycache__/customtransformer.cpython-36.pyc
models/__pycache__/cxr_encoder.cpython-36.pyc
models/__pycache__/cxr_models.cpython-36.pyc
models/__pycache__/cxr_models.cpython-39.pyc
models/__pycache__/dn_encoder.cpython-36.pyc
models/__pycache__/ehr_encoder.cpython-36.pyc
models/__pycache__/ehr_models.cpython-36.pyc
models/__pycache__/ehr_models.cpython-39.pyc
models/__pycache__/fusion.cpython-36.pyc
models/__pycache__/fusion.cpython-39.pyc
models/__pycache__/fusion_daft.cpython-36.pyc
models/__pycache__/fusion_daft.cpython-39.pyc
models/__pycache__/fusion_mmtm.cpython-36.pyc
models/__pycache__/fusion_mmtm.cpython-39.pyc
models/__pycache__/healnet.cpython-36.pyc
models/__pycache__/loss.cpython-36.pyc
models/__pycache__/loss.cpython-39.pyc
models/__pycache__/loss_set.cpython-36.pyc
models/__pycache__/rr_encoder.cpython-36.pyc
models/__pycache__/text_models.cpython-36.pyc
trainers/__pycache__/Calibration.cpython-36.pyc
trainers/__pycache__/DHF_trainer.cpython-36.pyc
trainers/__pycache__/MSMA_trainer.cpython-36.pyc
trainers/__pycache__/__init__.cpython-36.pyc
trainers/__pycache__/__init__.cpython-39.pyc
trainers/__pycache__/daft_trainer.cpython-36.pyc
trainers/__pycache__/daft_trainer.cpython-39.pyc
trainers/__pycache__/ensemble_average.cpython-36.pyc
trainers/__pycache__/ensemble_late.cpython-36.pyc
trainers/__pycache__/ensemble_trainer.cpython-36.pyc
trainers/__pycache__/frozen_trainer.cpython-36.pyc
trainers/__pycache__/frozen_trainer.cpython-39.pyc
trainers/__pycache__/fusion_trainer.cpython-36.pyc
trainers/__pycache__/fusion_trainer.cpython-39.pyc
trainers/__pycache__/hierarchical_ensemble_trainer.cpython-36.pyc
trainers/__pycache__/lstm_trainer.cpython-36.pyc
trainers/__pycache__/mmtm_trainer.cpython-36.pyc
trainers/__pycache__/mmtm_trainer.cpython-39.pyc
trainers/__pycache__/staged_Trainer.cpython-36.pyc
trainers/__pycache__/trainer.cpython-36.pyc
trainers/__pycache__/trainer.cpython-39.pyc
trainers/__pycache__/triple_ensemble.cpython-36.pyc
trainers/__pycache__/unimodal_trainer.cpython-36.pyc
trainers/__pycache__/utils.cpython-36.pyc
trainers/__pycache__/utils.cpython-39.pyc
trainers/__pycache__/weighted_ensemble.cpython-36.pyc
trainers/__pycache__/weighted_trainer.cpython-36.pyc

ADDITIVE LOCAL FILES
.python-version
check_batch.py
check_batch.sh
check_cohort.py
check_cohort.sh
cohort_source.py
export.sh
export_source.py
prepare_cohort_source.py
radiology_notes.py
requirements-export.txt

status: UPSTREAM_FILES_CHANGED
```
