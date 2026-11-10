# What we changed in the authors' code

`semi_supervised_meta_learning` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff semi_supervised_meta_learning`.

```
paper: semi_supervised_meta_learning
upstream: https://github.com/lidazhang/ssml-time-series-heterogeneity.git
commit: 426c1766e287dec5782855a430ec14c5cacc3a94
path: .
tree: a59b865e55e2d9d7916212d796c27f371cab9db9
modified_upstream_files: 0
deleted_upstream_files: 0
additive_local_files: 18

ADDITIVE LOCAL FILES
.python-version
check_batch.py
check_batch.sh
check_batch_ihm.py
check_batchgen.py
check_cohort.py
check_cohort.sh
check_cohort_ihm.py
cohort_source.py
cohort_source_ihm.py
derive_sampled.py
export.sh
export_source.py
export_source_ihm.py
measure_precision.py
measure_precision_ihm.py
prepare_cohort_source.py
requirements-export.txt

status: UPSTREAM_FILES_UNCHANGED
```
