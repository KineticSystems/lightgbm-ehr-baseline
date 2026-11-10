# What we changed in the authors' code

`dpmm` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff dpmm`.

```
paper: dpmm
upstream: https://github.com/HKU-MedAI/DPMM.git
commit: 17d0314ff0b21dde31f3f932e02d4683826029aa
path: .
tree: 2e72f0ef970822a0b3171a2a4831d53038a01cb5
modified_upstream_files: 2
deleted_upstream_files: 11
additive_local_files: 23

MODIFIED UPSTREAM FILES

--- mimic3extract/mimic3benchmark/preprocessing.py (mode 100644 -> 100644)
--- upstream/mimic3extract/mimic3benchmark/preprocessing.py
+++ local/mimic3extract/mimic3benchmark/preprocessing.py
@@ -201,8 +201,11 @@
     ''' The two following lines implement the code that was used to create the benchmark dataset that the paper used.
     This works with both python 2 and python 3.
     '''
-    is_str = np.array(map(lambda x: type(x) == str,
-                      list(df.VALUE)), dtype=bool)
+    # Python 3's map is an iterator.  Materialize it so this released
+    # Python-2-compatible cleaning rule remains executable under the paper's
+    # declared Python 3.11 environment.
+    is_str = np.array(list(map(lambda x: type(x) == str,
+                      list(df.VALUE))), dtype=bool)
     idx = df.VALUEUOM.fillna('').apply(
         lambda s: 'torr' not in s.lower()) & (is_str | (~is_str & (v > 1.0)))
 

--- models/loss.py (mode 100644 -> 100644)
--- upstream/models/loss.py
+++ local/models/loss.py
@@ -341,7 +341,7 @@
 
         # Using reparameterization tricks
 
-        x_samples = [MultivariateNormal(self.mu_x[k], scale_tril=torch.diag(cov[k])).rsample(sample_shape=n_samples).T * pi[:, k] for k in range(self.K)]
+        x_samples = [MultivariateNormal(self.mu[k], scale_tril=torch.diag(cov[k])).rsample(sample_shape=n_samples).T * pi[:, k] for k in range(self.K)]
         x_samples = torch.stack(x_samples, dim=0).sum(dim=0).T.float()
 
         return x_samples

DELETED UPSTREAM FILES
mimic4extract/mimic3benchmark/resources/2018 Code Descriptions/icd10OrderFiles.pdf
mimic4extract/mimic3benchmark/resources/2018 Code Descriptions/icd10cmCodesFile.pdf
mimic4extract/mimic3benchmark/resources/diagnosis_gems_2018/Dxgem_guide_2018.pdf
mimic4extract/mimic3benchmark/resources/diagnosis_gems_2018/GemsTechDoc_2018.pdf
mimic4extract/mimic3benchmark/tests/resources/decomp.pkl
mimic4extract/mimic3benchmark/tests/resources/ihm.pkl
mimic4extract/mimic3benchmark/tests/resources/los.pkl
mimic4extract/mimic3benchmark/tests/resources/multitasking.pkl
mimic4extract/mimic3benchmark/tests/resources/pheno.pkl
mimic4extract/mimic3benchmark/tests/resources/root-after-validating.pkl
mimic4extract/mimic3benchmark/tests/resources/root-final.pkl

ADDITIVE LOCAL FILES
check_batch.py
check_batch.sh
check_batch_mimic4.py
check_batch_readmission.py
check_cohort.py
check_cohort.sh
check_cohort_mimic4.py
check_cohort_readmission.py
check_reproduction_data.py
cohort_readmission.py
cohort_source.py
cohort_source_mimic4.py
export.sh
export_mimic4.py
export_readmission.py
export_source.py
prepare_mimic4_source.py
reproduce_dpmm.py
reproduction_data.py
requirements-export.txt
requirements-gpu.txt
run_reproduction.sh
setup_gpu_env.sh

status: UPSTREAM_FILES_CHANGED
```
