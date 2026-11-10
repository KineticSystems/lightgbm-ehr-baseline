# What we changed in the authors' code

`oc_distill` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff oc_distill`.

```
paper: oc_distill
upstream: https://github.com/the-chen-lab/OC-Distill.git
commit: 9599f6ec7dcf2d74cc1b7ac049b59053ece337b9
path: .
tree: eb9eb4bae50e56cfb2f2b6b1c86b5c5e378cc272
modified_upstream_files: 2
deleted_upstream_files: 0
additive_local_files: 9

MODIFIED UPSTREAM FILES

--- environment.yml (mode 100644 -> 100644)
--- upstream/environment.yml
+++ local/environment.yml
@@ -16,3 +16,5 @@
   - pip:
     - transformers>=4.20.0
     - rdflib>=6.0.0
+    - pyarrow==17.0.0
+    - polars==0.20.31

--- requirements.txt (mode 100644 -> 100644)
--- upstream/requirements.txt
+++ local/requirements.txt
@@ -5,3 +5,5 @@
 scikit-learn>=1.0.0
 rdflib>=6.0.0
 tqdm>=4.60.0
+polars==0.20.31
+pyarrow==17.0.0

ADDITIVE LOCAL FILES
.python-version
check_batch.sh
check_cohort.py
check_cohort.sh
export.sh
export_source.py
prepare_student_auxiliary.py
requirements-export.txt
verify_full_batch.py

status: UPSTREAM_FILES_CHANGED
```
