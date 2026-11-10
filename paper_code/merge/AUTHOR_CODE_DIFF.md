# What we changed in the authors' code

`merge` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff merge`.

```
paper: merge
upstream: https://github.com/aaronhan223/merge.git
commit: 75de1f059030e23a9629ed9fdd0d31b56e29ffe4
path: .
tree: 9295ad02c787edf065bcf96e3f15d661edc77f8d
modified_upstream_files: 3
deleted_upstream_files: 1
additive_local_files: 32

MODIFIED UPSTREAM FILES

--- mimiciv/data_preprocess/create_ihm_task.py (mode 100644 -> 100644)
--- upstream/mimiciv/data_preprocess/create_ihm_task.py
+++ local/mimiciv/data_preprocess/create_ihm_task.py
@@ -80,7 +80,7 @@
                     curr_stay_dict['text_data'] = curr_stay_notes['text'].tolist()
                     curr_stay_dict['text_time'] = curr_stay_notes['icu_time_delta'].values
                     if not use_raw_data:
-                        curr_stay_dict['text_embeddings'] = [np.mean(chunk_embs, axis=0) for chunk_embs in curr_stay_notes['biobert_embeddings']] # average over chunk embeddings
+                        curr_stay_dict['text_embeddings'] = [np.mean(np.stack(chunk_embs), axis=0) for chunk_embs in curr_stay_notes['biobert_embeddings']] # average over chunk embeddings
                     else:
                         curr_stay_dict['text_embeddings'] = []
                     curr_stay_dict['text_missing'] = 0

--- mimiciv/train_mimiciv_multimodal.py (mode 100644 -> 100644)
--- upstream/mimiciv/train_mimiciv_multimodal.py
+++ local/mimiciv/train_mimiciv_multimodal.py
@@ -167,7 +167,7 @@
             converted_sample = {}
             for modality in self.modality_names:
                 if modality in sample:
-                    features, mask = sample[modality]
+                    features, mask = sample.pop(modality)
                     # Convert numpy arrays to torch tensors
                     features_tensor = torch.from_numpy(features).float()
                     mask_tensor = torch.from_numpy(mask).bool()
@@ -543,9 +543,11 @@
     print(f"Loading train data from {args.train_data_path}...")
     train_stays = pickle.load(open(args.train_data_path, 'rb'))
     train_multimodal_reg_ts, train_labels = preprocess_mimiciv_data(train_stays, modality_dim_dict)
+    del train_stays
     print(f"Loading val data from {args.val_data_path}...")
     val_stays = pickle.load(open(args.val_data_path, 'rb'))
     val_multimodal_reg_ts, val_labels = preprocess_mimiciv_data(val_stays, modality_dim_dict)
+    del val_stays
 
     # Load RUS data
     modality_names = sorted(list(modality_dim_dict.keys()))

--- utils/evaluation_utils.py (mode 100644 -> 100644)
--- upstream/utils/evaluation_utils.py
+++ local/utils/evaluation_utils.py
@@ -46,7 +46,7 @@
     # Confusion matrix
     print(f"\nConfusion Matrix:")
     cm = results['confusion_matrix']
-    print(f"{'Actual \\ Predicted':<20}", end="")
+    print("{:<20}".format("Actual \\ Predicted"), end="")
     for i, class_name in enumerate(class_names):
         if i < cm.shape[1]:
             print(f"{class_name:<10}", end="")

DELETED UPSTREAM FILES
MERGE.pdf

ADDITIVE LOCAL FILES
.python-version
acquire_encoders.py
acquire_encoders.sh
build_ihm_inputs.sh
check_batch.sh
check_cohort.py
check_cohort.sh
check_encoders.py
check_encoders.sh
cohort_source.py
export.sh
export_source.py
materialize_embeddings.py
prepare_embedding_inputs.py
requirements-export.txt
requirements-gpu.txt
run_embedding_phase.sh
run_reproduction_all.sh
run_reproduction_seed.sh
run_rus_reproduction.sh
run_training_smoke.sh
setup_gpu_env.sh
summarize_reproduction.py
verify_full_batch.py
verify_ihm_inputs.py
verify_ihm_inputs.sh
verify_reproduction_seed.py
verify_rus.py
verify_rus.sh
verify_training_dtype_boundary.py
verify_training_dtype_boundary.sh
verify_training_smoke.py

status: UPSTREAM_FILES_CHANGED
```
