# What we changed in the authors' code

`garlic` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff garlic`.

```
paper: garlic
upstream: https://github.com/SCAI-Lab/GARLIC.git
commit: aa2cd2c2d434fc80b6705eec66d54060327995de
path: .
tree: 21d96b9e25cc0e5bd49398d6f51194403ef59336
modified_upstream_files: 0
deleted_upstream_files: 2
additive_local_files: 22

DELETED UPSTREAM FILES
.DS_Store
data/.DS_Store

ADDITIVE LOCAL FILES
.python-version
benchmark_epoch.py
check_batch.sh
check_cohort.py
check_cohort.sh
cohort_source.py
export.sh
export_source.py
prepare_reproduction_inputs.py
prepare_reproduction_inputs.sh
reproduction_data.py
requirements-export.txt
run_epoch_benchmark.sh
run_gpu_smoke.sh
run_reproduction.py
run_reproduction.sh
smoke_reproduction.py
test_export_resume.py
verify_full_batch.py
verify_reproduction.py
verify_reproduction_inputs.py
verify_reproduction_inputs.sh

status: UPSTREAM_FILES_CHANGED
```
