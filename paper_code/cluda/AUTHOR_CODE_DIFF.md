# What we changed in the authors' code

`cluda` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff cluda`.

```
paper: cluda
upstream: https://github.com/oezyurty/CLUDA
commit: 60e0b10f0e967fbada08ee6c164fc7c97932c57b
path: .
tree: 6cd8bac5e20ba35f6e18a322c885221cb30b12fe
modified_upstream_files: 0
deleted_upstream_files: 0
additive_local_files: 25

ADDITIVE LOCAL FILES
.python-version
README_FAST_PATH.md
analyze_split_probe.R
check_batch.py
check_batch.sh
check_cohort.py
check_cohort.sh
cluda_los_filter_fast.R
cluda_split_rng_compat.R
cohort_source.py
diagnose_split_protocol.R
diagnose_split_protocol.py
environment-r.yml
export.sh
export_source.py
import_miiv_table.R
prepare_cohort_source.py
prepare_miiv_cohort.R
prepare_miiv_cohort_fast.R
probe_split_state.R
requirements-export.txt
split_rng_search_audit.json
test_export.py
test_los_filter_fast.R
test_split_rng_compat.R

status: UPSTREAM_FILES_UNCHANGED
```
