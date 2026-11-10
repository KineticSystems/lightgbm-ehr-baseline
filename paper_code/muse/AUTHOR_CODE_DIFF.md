# What we changed in the authors' code

`muse` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff muse`.

```
paper: muse
upstream: https://github.com/zzachw/MUSE.git
commit: cdec2aceff40bad0679df2dc2716cdc13e8ae315
path: .
tree: 6e04933323fc65ec28cbf2219efe8839245e9a8e
modified_upstream_files: 0
deleted_upstream_files: 0
additive_local_files: 24

ADDITIVE LOCAL FILES
.python-version
_readmission_limits.sh
check_batch.py
check_batch.sh
check_batch_readmission.py
check_cohort.py
check_cohort.sh
check_cohort_readmission.py
check_muse_plus_inputs.py
check_muse_plus_inputs.sh
cohort_readmission.py
cohort_source.py
evaluate_temporal_cheat.py
export.sh
export_readmission.py
export_source.py
measure_split_leakage.py
reproduce_muse.py
requirements-export.txt
requirements-gpu.txt
run_temporal_cheat.sh
setup_gpu_env.sh
smoke_muse_plus.py
smoke_muse_plus.sh

status: UPSTREAM_FILES_UNCHANGED
```
