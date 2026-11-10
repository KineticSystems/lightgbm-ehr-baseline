# What we changed in the authors' code

`remed` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff remed`.

```
paper: remed
upstream: https://github.com/starmpcc/REMed.git
commit: eb444fad98b842be3c4d10bb690fd2f85f6ad90e
path: .
tree: d27e7fec817497bf8601d304c95102cbcd66120d
upstream: https://github.com/Jwoo5/integrated-ehr-pipeline.git
commit: 3b7f68663d6c4c95b3a79e0649ad4593a63cb5cb
path: dependencies/integrated-ehr-pipeline
tree: 78f8a53dac0fc7232bd54cceb73ac51b0f89ec83
modified_upstream_files: 1
deleted_upstream_files: 0
additive_local_files: 28

MODIFIED UPSTREAM FILES

--- src/trainer/base.py (mode 100644 -> 100644)
--- upstream/src/trainer/base.py
+++ local/src/trainer/base.py
@@ -295,7 +295,7 @@
         return metric_dict
 
     def dataloader_set(self, split):
-        if split is None:
+        if self.args.src_data == "meds" and split is None:
             return None
         dataset = self.dataset(self.args, split, self.data, self.df)
         return DataLoader(

ADDITIVE LOCAL FILES
check_batch.py
check_batch.sh
check_cohort.py
check_cohort.sh
cohort_source.py
derive_outcomes.py
evaluate_reproduction_seed.py
export.sh
export_source.py
plain_events.py
prepare_cohort_source.py
prepare_reproduction_input.py
record_gpu_environment.py
record_reproduction_stage.py
requirements-export.txt
requirements-gpu.txt
run_encoded_cache_verification.sh
run_exact_maximum_batch.sh
run_reproduction_evaluation.sh
run_reproduction_smoke.sh
run_reproduction_stage.sh
setup_gpu_env.sh
smoke_reproduction.py
test_export_adapter.py
verify_encode_events_loader_repair.py
verify_encoded_cache.py
verify_exact_maximum_batch.py
verify_reproduction_config.py

status: UPSTREAM_FILES_CHANGED
```
