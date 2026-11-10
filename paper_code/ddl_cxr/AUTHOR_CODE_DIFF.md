# What we changed in the authors' code

`ddl_cxr` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff ddl_cxr`.

```
paper: ddl_cxr
upstream: https://github.com/Chenliu-svg/DDL-CXR.git
commit: 0ddd5485d3f4baf0334e206618f0a3d9cd0e04a5
path: .
tree: c7c303ba3c387e0b6a3d70dcf362fde2d43280a5
modified_upstream_files: 4
deleted_upstream_files: 9
additive_local_files: 35

MODIFIED UPSTREAM FILES

--- ldm/models/ts_transformer.py (mode 100755 -> 100755)
--- upstream/ldm/models/ts_transformer.py
+++ local/ldm/models/ts_transformer.py
@@ -132,7 +132,8 @@
         super(TransformerBatchNormEncoderLayer, self).__setstate__(state)
 
     def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
-                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
+                src_key_padding_mask: Optional[Tensor] = None,
+                is_causal: bool = False) -> Tensor:
         r"""Pass the input through the encoder layer.
 
         Args:

--- mimic4extract/mimic3benchmark/mimic3csv.py (mode 100755 -> 100755)
--- upstream/mimic4extract/mimic3benchmark/mimic3csv.py
+++ local/mimic4extract/mimic3benchmark/mimic3csv.py
@@ -2,6 +2,7 @@
 from __future__ import print_function
 
 import csv
+import gzip
 import numpy as np
 import os
 import pandas as pd
@@ -10,8 +11,18 @@
 from mimic3benchmark.util import dataframe_from_csv
 
 
+def resolve_csv(path):
+    """Accept the compressed files distributed by PhysioNet without copying them."""
+    if os.path.isfile(path):
+        return path
+    compressed = path + '.gz'
+    if os.path.isfile(compressed):
+        return compressed
+    raise FileNotFoundError(path)
+
+
 def read_patients_table(path):
-    pats = pd.read_csv(path)#dataframe_from_csv(path)  
+    pats = pd.read_csv(resolve_csv(path))#dataframe_from_csv(path)  
     columns = ['subject_id', 'gender', 'anchor_age', 'dod']  
     pats = pats[columns]
 
@@ -23,7 +34,7 @@
 def read_admissions_table(path):
 
     # admits = dataframe_from_csv(path)
-    admits = pd.read_csv(path) #header=header, index_col=index_col
+    admits = pd.read_csv(resolve_csv(path)) #header=header, index_col=index_col
     # admits = admits[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'ethnicity']] # missing DIAGNOSIS
     admits = admits[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime']]
     admits.admittime = pd.to_datetime(admits.admittime)
@@ -33,7 +44,7 @@
 
 
 def read_icustays_table(path):
-    stays = pd.read_csv(path)#dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'))
+    stays = pd.read_csv(resolve_csv(path))#dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'))
     # subject_id	hadm_id	stay_id	first_careunit	last_careunit	intime	outtime	los
     stays.intime = pd.to_datetime(stays.intime)
     stays.outtime = pd.to_datetime(stays.outtime)
@@ -42,10 +53,10 @@
 
 def read_icd_diagnoses_table(path):
 
-    codes = pd.read_csv(f'{path}/d_icd_diagnoses.csv')
+    codes = pd.read_csv(resolve_csv(f'{path}/d_icd_diagnoses.csv'))
     # dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_DIAGNOSES.csv'))
     codes = codes[['icd_code', 'long_title']]
-    diagnoses = pd.read_csv(f'{path}/diagnoses_icd.csv')
+    diagnoses = pd.read_csv(resolve_csv(f'{path}/diagnoses_icd.csv'))
     diagnoses = diagnoses.merge(codes, how='inner', left_on='icd_code', right_on='icd_code')
     diagnoses[['subject_id', 'hadm_id', 'seq_num']] = diagnoses[['subject_id', 'hadm_id', 'seq_num']].astype(int)
     return diagnoses
@@ -58,11 +69,14 @@
     # nb_rows = {'chartevents': 329499788, 'labevents': 122103667, 'outputevents': 4457381}
     csv_files = {'chartevents': 'icu/chartevents.csv', 'labevents': 'hosp/labevents.csv', 'outputevents': 'icu/outputevents.csv','procedureevents': 'icu/procedureevents.csv'}
     # nb_rows = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219}
-    reader = csv.DictReader(open(os.path.join(mimic3_path, csv_files[table.lower()]), 'r'))
-    for i, row in enumerate(reader):
-        if 'stay_id' not in row:
-            row['stay_id'] = ''
-        yield row, i, nb_rows[table.lower()]
+    path = resolve_csv(os.path.join(mimic3_path, csv_files[table.lower()]))
+    opener = gzip.open if path.endswith('.gz') else open
+    with opener(path, 'rt', newline='') as handle:
+        reader = csv.DictReader(handle)
+        for i, row in enumerate(reader):
+            if 'stay_id' not in row:
+                row['stay_id'] = ''
+            yield row, i, nb_rows[table.lower()]
 
 
 def count_icd_codes(diagnoses, output_path=None):
@@ -218,13 +232,17 @@
         
         # import pdb; pdb.set_trace()
         # value = row['valueuom'] if table=='OUTPUTEVENTS' else row['valuenum']
+        charttime = row.get('charttime') or row.get('starttime')
+        if not charttime:
+            raise KeyError('event row has neither charttime nor starttime')
         row_out = {'subject_id': row['subject_id'],
                    'hadm_id': row['hadm_id'],
                    'stay_id': '' if 'stay_id' not in row else row['stay_id'],
-                   'charttime': row['charttime'],
+                   'charttime': charttime,
                    'itemid': row['itemid'],
                    'value': row['value'],
-                   'valuenum': row['valueuom'] if table=='OUTPUTEVENTS' else row['valuenum']}
+                   'valuenum': (row['valueuom'] if table=='OUTPUTEVENTS'
+                                else row.get('valuenum', row['value']))}
 
 
         # 将一个病人的数据收集好了后，一次性写入

--- mimic4extract/mimic3benchmark/scripts/extract_subjects_iv.py (mode 100755 -> 100755)
--- upstream/mimic4extract/mimic3benchmark/scripts/extract_subjects_iv.py
+++ local/mimic4extract/mimic3benchmark/scripts/extract_subjects_iv.py
@@ -18,6 +18,8 @@
                     help='YAML file with phenotype definitions.')
 parser.add_argument('--itemids_file', '-i', type=str,default=os.path.join(os.path.dirname(__file__),'../resources/itemid_to_variable_map.csv'),
                     help='CSV containing list of ITEMIDs to keep.')
+parser.add_argument('--subjects_file', type=str,
+                    help='Optional newline-delimited subject IDs to retain while reading events.')
 
 parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Verbosity in output')
 parser.add_argument('--quiet', '-q', dest='verbose', action='store_false', help='Suspend printing of details')
@@ -95,6 +97,17 @@
     print('Using only', stays.shape[0], 'stays and only', args.event_tables[0], 'table')
 
 subjects = stays.subject_id.unique()
+if args.subjects_file:
+    with open(args.subjects_file) as subject_handle:
+        requested_subjects = {int(line) for line in subject_handle if line.strip()}
+    available_subjects = set(subjects)
+    missing_subjects = requested_subjects - available_subjects
+    if missing_subjects:
+        raise ValueError('Requested subjects absent from the released stay cohort: {}'.format(
+            sorted(missing_subjects)[:10]))
+    subjects = np.asarray([subject for subject in subjects if subject in requested_subjects])
+    if args.verbose:
+        print('STAGE 3 COHORT FILTER:\n\tsubject_ids: {}'.format(len(subjects)))
 break_up_stays_by_subject(stays, args.output_path, subjects=subjects)
 break_up_diagnoses_by_subject(phenotypes, args.output_path, subjects=subjects)
 items_to_keep = set(

--- pre_processing.py (mode 100644 -> 100644)
--- upstream/pre_processing.py
+++ local/pre_processing.py
@@ -29,6 +29,15 @@
 zero_timedelta = timedelta(0)
 one_day=timedelta(days=1)
 
+
+def resolve_csv(path):
+    if os.path.isfile(path):
+        return path
+    compressed = path + '.gz'
+    if os.path.isfile(compressed):
+        return compressed
+    raise FileNotFoundError(path)
+
 disease_cols=['Atelectasis', 'Cardiomegaly',
        'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
        'Lung Lesion', 'Lung Opacity',  'Pleural Effusion',
@@ -90,6 +99,8 @@
     parser.add_argument('--mimic_iv_csv_dir', required=True, help='The path to the origin mimic-iv CSVs ')
     parser.add_argument('--output_csv_dir', required=True, help='The path for the output CSV from the pre-processing ')
     parser.add_argument('--random_seed', default=1, help='random seed ')
+    parser.add_argument('--prediction_only', action='store_true',
+                        help='Stop after constructing the prediction cohort.')
     
     
     args = parser.parse_args()
@@ -116,9 +127,9 @@
    
     
     ## handling cxr: get the latest AP  for each study case
-    cxr=pd.read_csv(os.path.join(args.mimic_cxr_jpg_dir,'mimic-cxr-2.0.0-metadata.csv'))
-    
-    chexpert=pd.read_csv(os.path.join(args.mimic_cxr_jpg_dir,'mimic-cxr-2.0.0-chexpert.csv'))[cols]
+    cxr=pd.read_csv(resolve_csv(os.path.join(args.mimic_cxr_jpg_dir,'mimic-cxr-2.0.0-metadata.csv')))
+    
+    chexpert=pd.read_csv(resolve_csv(os.path.join(args.mimic_cxr_jpg_dir,'mimic-cxr-2.0.0-chexpert.csv')))[cols]
           
     # convert time to pd.datetime
     cxr['StudyTime'] = cxr['StudyTime'].apply(lambda x: f'{int(float(x)):06}' )
@@ -192,7 +203,7 @@
     splitted_df={'train':train_df,'validate':val_df,'test':test_df}
     
     # add static info from patients.csv
-    patients=pd.read_csv(os.path.join(args.mimic_iv_csv_dir,'hosp/patients.csv'))
+    patients=pd.read_csv(resolve_csv(os.path.join(args.mimic_iv_csv_dir,'hosp/patients.csv')))
     # add phenotype labels 
     pheno_df=pd.read_csv(os.path.join(args.mimic_iv_subjects_dir,'phenotype_labels.csv'))
     pheno_cols=pheno_df.columns.tolist()
@@ -208,6 +219,9 @@
         pred_df_add_satic_add_pheno=pred_df_add_satic_add_pheno[cols_tosave].reset_index()
         
         pred_df_add_satic_add_pheno.to_csv(os.path.join(args.output_csv_dir,f'{s}_pred.csv'))
+
+    if args.prediction_only:
+        raise SystemExit(0)
 
     
     ### Build the dataset for DM
@@ -357,4 +371,3 @@
     
     
    
-

DELETED UPSTREAM FILES
data_pre/test_autoencoder_augmented_label.csv
data_pre/test_dm_labels.csv
data_pre/test_pred.csv
data_pre/train_autoencoder_augmented_label.csv
data_pre/train_dm_labels.csv
data_pre/train_pred.csv
data_pre/validate_autoencoder_augmented_label.csv
data_pre/validate_dm_labels.csv
data_pre/validate_pred.csv

ADDITIVE LOCAL FILES
.python-version
assemble_verify_predictor_inputs.py
assemble_verify_predictor_inputs.sh
benchmark_ldm_batch48.sh
benchmark_ldm_batch48_200steps.sh
check_batch.py
check_batch.sh
check_cohort.py
check_cohort.sh
check_gpu_env.py
check_gpu_env.sh
cohort_pickle_prediction.py
cohort_source.py
export.sh
export_source.py
fetch_resnet34_weight.sh
generate_released_latents.py
generate_released_latents.sh
inspect_checkpoints.py
inspect_checkpoints.sh
prepare_cohort_source.py
prepare_released_generation_ehr.py
prepare_released_generation_ehr.sh
requirements-export.txt
requirements-gpu.txt
setup_gpu_env.sh
smoke_ldm.py
smoke_ldm.sh
smoke_predictor.py
smoke_predictor.sh
smoke_source_trainer.sh
train_predictor.py
train_predictor.sh
verify_predictor_runs.py
verify_predictor_runs.sh

status: UPSTREAM_FILES_CHANGED
```
