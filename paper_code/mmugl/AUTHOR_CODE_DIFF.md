# What we changed in the authors' code

`mmugl` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff mmugl`.

```
paper: mmugl
upstream: https://github.com/ratschlab/mmugl.git
commit: 73556144b14958d41b1fc1d7541c61b51fedf8b3
path: .
tree: 29a519d8a674de16f22a57f85743bf015fe287cf
modified_upstream_files: 0
deleted_upstream_files: 0
additive_local_files: 11

ADDITIVE LOCAL FILES
.python-version
check_batch.py
check_batch.sh
check_cohort.py
check_cohort.sh
cohort_source.py
export.sh
export_source.py
requirements-export.txt
requirements-gpu.txt
setup_gpu_env.sh

status: UPSTREAM_FILES_UNCHANGED
```
