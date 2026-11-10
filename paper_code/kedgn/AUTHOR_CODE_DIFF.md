# What we changed in the authors' code

`kedgn` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff kedgn`.

```
paper: kedgn
upstream: https://github.com/qianlima-lab/KEDGN.git
commit: c82bc004cd1d1305fc58ca7150369d78682fccad
path: .
tree: 2e0dd2ad02c5705d5acd86b0298449510b0cd4cf
modified_upstream_files: 0
deleted_upstream_files: 46
additive_local_files: 10

DELETED UPSTREAM FILES
data/P12/P12_bart_var_rep_gpt_source.pt
data/P12/P12_bert_var_rep_gpt_source.pt
data/P12/P12_bert_var_rep_name_source.pt
data/P12/P12_bert_var_rep_wiki_source.pt
data/P12/P12_gpt2_var_rep_gpt_source.pt
data/P12/P12_led_var_rep_gpt_source.pt
data/P12/P12_pegasus_var_rep_gpt_source.pt
data/P12/P12_t5_var_rep_gpt_source.pt
data/P12/processed_data/arr_outcomes.npy
data/P12/processed_data/extended_static_params.npy
data/P12/processed_data/static_params.npy
data/P12/processed_data/ts_params.npy
data/P12/splits/phy12_split1.npy
data/P19/P19_bart_var_rep_gpt_source.pt
data/P19/P19_bert_var_rep_gpt_source.pt
data/P19/P19_bert_var_rep_name_source.pt
data/P19/P19_bert_var_rep_wiki_source.pt
data/P19/P19_gpt2_var_rep_gpt_source.pt
data/P19/P19_led_var_rep_gpt_source.pt
data/P19/P19_pegasus_var_rep_gpt_source.pt
data/P19/P19_t5_var_rep_gpt_source.pt
data/P19/processed_data/arr_outcomes_6.npy
data/P19/processed_data/labels_demogr.npy
data/P19/processed_data/labels_ts.npy
data/P19/splits/phy19_split1_new.npy
data/mimic3/mimic3_bart_var_rep_gpt_source.pt
data/mimic3/mimic3_bert_var_rep_gpt_source.pt
data/mimic3/mimic3_bert_var_rep_name_source.pt
data/mimic3/mimic3_bert_var_rep_wiki_source.pt
data/mimic3/mimic3_gpt2_var_rep_gpt_source.pt
data/mimic3/mimic3_led_var_rep_gpt_source.pt
data/mimic3/mimic3_pegasus_var_rep_gpt_source.pt
data/mimic3/mimic3_t5_var_rep_gpt_source.pt
data/physionet/physionet_bart_var_rep_gpt_source.pt
data/physionet/physionet_bert_var_rep_gpt_source.pt
data/physionet/physionet_bert_var_rep_name_source.pt
data/physionet/physionet_bert_var_rep_wiki_source.pt
data/physionet/physionet_gpt2_var_rep_gpt_source.pt
data/physionet/physionet_led_var_rep_gpt_source.pt
data/physionet/physionet_pegasus_var_rep_gpt_source.pt
data/physionet/physionet_t5_var_rep_gpt_source.pt
data/physionet/processed_data/arr_outcomes.npy
data/physionet/processed_data/extended_static_params.npy
data/physionet/processed_data/static_params.npy
data/physionet/processed_data/ts_params.npy
data/physionet/splits/phy12_split5.npy

ADDITIVE LOCAL FILES
.python-version
check_batch.py
check_batch.sh
check_cohort.py
check_cohort.sh
cohort_source.py
export.sh
export_source.py
prepare_cohort_source.py
requirements-export.txt

status: UPSTREAM_FILES_CHANGED
```
