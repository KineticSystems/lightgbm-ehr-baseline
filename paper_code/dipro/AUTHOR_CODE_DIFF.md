# What we changed in the authors' code

`dipro` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff dipro`.

```
paper: dipro
upstream: https://github.com/Chenliu-svg/DiPro.git
commit: 64b12ec6fc9455ba873af0597804500f0d25afc2
path: .
tree: 6e7361f7c7e8c5502b9a534e75f3b741a6af1b3e
modified_upstream_files: 0
deleted_upstream_files: 2
additive_local_files: 17

DELETED UPSTREAM FILES
figures/DiPro_poster.pdf
figures/DiPro_slides.pdf

ADDITIVE LOCAL FILES
.python-version
check_batch.sh
check_cohort.py
check_cohort.sh
check_real_multimodal_loader.py
check_real_multimodal_loader.sh
cohort_source.py
export.sh
export_source.py
prepare_multimodal_reproduction.py
reproduce_mortality_checkpoints.py
requirements-export.txt
requirements-gpu.txt
run_mortality_checkpoints.sh
setup_gpu_env.sh
verify_full_batch.py
verify_mortality_reproduction.py

status: UPSTREAM_FILES_CHANGED
```
