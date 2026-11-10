# What we changed in the authors' code

`fusemoe` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff fusemoe`.

```
paper: fusemoe
upstream: https://github.com/aaronhan223/FuseMoE.git
commit: 52ea655e2103b4daf1033572ca354cabc496bd87
path: .
tree: 532edf2cc628c38cfe1f27763536cc5511332f4e
modified_upstream_files: 2
deleted_upstream_files: 3
additive_local_files: 22

MODIFIED UPSTREAM FILES

--- requirements.txt (mode 100644 -> 100644)
--- upstream/requirements.txt
+++ local/requirements.txt
@@ -15,4 +15,6 @@
 torchvision==0.8.2
 tqdm==4.62.3
 transformers==4.16.2
-nltk==3.8.1+nltk==3.8.1
+polars==0.20.31
+pyarrow==17.0.0

--- src/core/module.py (mode 100644 -> 100644)
--- upstream/src/core/module.py
+++ local/src/core/module.py
@@ -589,7 +589,6 @@
             num_modalities=args.num_modalities,
             gating=args.gating_function[0])
             self.moe = MoE(moe_config)
-            self.moe = self.moe.to('cuda:0')
         elif args.cross_method == 'hme':
             moe_config = MoEConfig(
             num_experts=args.num_of_experts,
@@ -601,7 +600,6 @@
             num_modalities=args.num_modalities,
             gating=args.gating_function)
             self.moe = HierarchicalMoE(moe_config)
-            self.moe = self.moe.to('cuda:0')
         
     def forward(self, x_list, modality):
         """

DELETED UPSTREAM FILES
src/utils/ihm.pdf
src/utils/los.pdf
src/utils/pheno.pdf

ADDITIVE LOCAL FILES
check_batch.sh
check_cohort.py
check_cohort.sh
check_gpu_env.py
check_gpu_env.sh
export.sh
export_source.py
prepare_reproduction_inputs.py
prepare_reproduction_inputs.sh
reproduce_table4.py
requirements-export.txt
requirements-gpu.txt
run_table4_all.sh
run_table4_seed.sh
setup_gpu_env.sh
smoke_table4.py
smoke_table4.sh
verify_full_batch.py
verify_reproduction_inputs.py
verify_reproduction_inputs.sh
verify_table4_runs.py
verify_table4_runs.sh

status: UPSTREAM_FILES_CHANGED
```
