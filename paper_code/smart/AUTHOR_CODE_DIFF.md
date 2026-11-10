# What we changed in the authors' code

`smart` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff smart`.

```
paper: smart
upstream: https://github.com/yzhHoward/SMART.git
commit: f0c89f9757f098feec8eb2aaeb0a8f27c37cc80b
path: .
tree: 27250fd978be29cef1c141a932a787a6d3187159
modified_upstream_files: 0
deleted_upstream_files: 0
additive_local_files: 21

ADDITIVE LOCAL FILES
.python-version
check_batch.py
check_batch.sh
check_batch_decompensation.py
check_cohort.py
check_cohort.sh
check_cohort_decompensation.py
cohort_decompensation.py
cohort_source.py
evaluate_reproduction.py
export.sh
export_decompensation.py
export_source.py
materialize_reproduction.py
prepare_cohort_source.py
prepare_decompensation_source.py
requirements-export.txt
requirements-gpu.txt
run_reproduction.sh
setup_gpu_env.sh
test_cohort_source.py

status: UPSTREAM_FILES_UNCHANGED
```
