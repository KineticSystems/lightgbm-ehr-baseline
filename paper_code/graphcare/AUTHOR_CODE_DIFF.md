# What we changed in the authors' code

`graphcare` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff graphcare`.

```
paper: graphcare
upstream: https://github.com/pat-jj/GraphCare.git
commit: fe92aa67add80b62d4b5108d570b6483ee69a5d4
path: .
tree: c60e6aec8a3742d36015ff01e8d13f5ca2f2dcfa
modified_upstream_files: 2
deleted_upstream_files: 0
additive_local_files: 32

MODIFIED UPSTREAM FILES

--- graphcare_/model.py (mode 100644 -> 100644)
--- upstream/graphcare_/model.py
+++ local/graphcare_/model.py
@@ -2,7 +2,13 @@
 import torch.nn as nn
 import torch.nn.functional as F
 from torch_geometric.nn import GINEConv, GATConv, GINConv
-from pyhealth.models import RETAINLayer
+try:
+    # RETAINLayer is not used anywhere in this module.  Keep the released
+    # import when the paper environment is active, but allow the model itself
+    # to run in a modern CUDA environment on GPUs unsupported by Torch 1.12.
+    from pyhealth.models import RETAINLayer
+except ImportError:  # pragma: no cover - compatibility path for GPU replay
+    RETAINLayer = None
 from torch_geometric.nn.inits import reset
 
 from typing import Callable, Optional, Union
@@ -98,6 +104,7 @@
             out_channels, layers=3, dropout=0.5, decay_rate=0.03, node_emb=None, rel_emb=None,
             freeze=False, patient_mode="joint", use_alpha=True, use_beta=True, use_edge_attn=True, 
             self_attn=0., gnn="BAT", attn_init=None, drop_rate=0.,
+            mask_padded_visits=False,
         ):
         super(GraphCare, self).__init__()
 
@@ -112,6 +119,7 @@
         self.num_nodes = num_nodes
         self.num_rels = num_rels
         self.max_visit = max_visit
+        self.mask_padded_visits = mask_padded_visits
 
         j = torch.arange(max_visit).float()
         self.lambda_j = torch.exp(self.decay_rate * (max_visit - j)).unsqueeze(0).reshape(1, max_visit, 1).float()
@@ -207,11 +215,18 @@
         for layer in range(1, self.layers+1):
             if self.use_alpha:
                 # alpha = masked_softmax((self.leakyrelu(self.alpha_attn[str(layer)](visit_node.float()))), mask=visit_node>1, dim=1)
-                alpha = torch.softmax((self.alpha_attn[str(layer)](visit_node.float())), dim=1)  # (batch, max_visit, num_nodes)
+                alpha_logits = self.alpha_attn[str(layer)](visit_node.float())
+                if self.mask_padded_visits:
+                    visit_mask = torch.any(visit_node != 0, dim=2, keepdim=True)
+                    alpha_logits = alpha_logits.masked_fill(~visit_mask, float('-inf'))
+                alpha = torch.softmax(alpha_logits, dim=1)  # (batch, max_visit, num_nodes)
 
             if self.use_beta:
                 # beta = masked_softmax((self.leakyrelu(self.beta_attn[str(layer)](visit_node.float()))), mask=visit_node>1, dim=0) * self.lambda_j
                 beta = torch.tanh((self.beta_attn[str(layer)](visit_node.float()))) * self.lambda_j
+                if self.mask_padded_visits:
+                    visit_mask = torch.any(visit_node != 0, dim=2, keepdim=True)
+                    beta = beta * visit_mask
 
             if self.use_alpha and self.use_beta:
                 attn = alpha * beta
@@ -275,5 +290,3 @@
             return logits, self.alpha_weights, self.beta_weights,  self.attention_weights, self.edge_weights
         else:
             return logits
-
-

--- run.sh (mode 100644 -> 100755)

ADDITIVE LOCAL FILES
.python-version
GPU_REPRODUCTION.md
GPU_REPRODUCTION_RESULTS.md
_export_phase2.sh
_phase2_common.sh
acquire_mimic4_resources.py
build_parse_cache.py
build_parse_cache.sh
check_batch.py
check_batch.sh
check_batch_phase2.py
check_batch_phase2.sh
check_cohort.py
check_cohort.sh
check_cohort_phase2.py
check_cohort_phase2.sh
check_raw_roundtrip.py
check_raw_roundtrip.sh
cohort_phase2.py
cohort_source.py
export.sh
export_phase2.py
export_source.py
investigate_crosswalks.py
reproduce_gpu.py
requirements-export.txt
requirements-gpu.txt
run_crosswalk_audit.sh
run_gpu_reproduction.sh
setup_gpu_env.sh
summarize_reproduction.py
summarize_reproduction.sh

status: UPSTREAM_FILES_CHANGED
```
