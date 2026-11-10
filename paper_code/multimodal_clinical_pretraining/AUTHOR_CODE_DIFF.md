# What we changed in the authors' code

`multimodal_clinical_pretraining` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff multimodal_clinical_pretraining`.

```
paper: multimodal_clinical_pretraining
upstream: https://github.com/kingrc15/multimodal-clinical-pretraining.git
commit: 655c26a23880950cc270df5681b981e6869e26df
path: .
tree: c62b2c8801e935be10a70e0d99ffe2c43aa1546a
modified_upstream_files: 0
deleted_upstream_files: 0
additive_local_files: 13

ADDITIVE LOCAL FILES
check_batch.py
check_batch.sh
check_cohort.py
check_cohort.sh
cohort_source.py
export.sh
export_source.py
prepare_cohort_source.py
reproduce_ihm.py
requirements-export.txt
requirements-gpu.txt
run_ihm_reproduction.sh
setup_gpu_env.sh

status: UPSTREAM_FILES_UNCHANGED
```
