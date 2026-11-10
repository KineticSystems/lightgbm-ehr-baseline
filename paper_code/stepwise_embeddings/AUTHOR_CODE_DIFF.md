# What we changed in the authors' code

`stepwise_embeddings` vendors the authors' released repository at the exact commit named
below. This file is the complete difference between that commit and the copy in
this directory, so every deviation from the published code is visible without
cloning anything.

It has three parts. **Modified upstream files** are the authors' files we had to
change, shown as unified diffs. **Deleted upstream files** are files we removed.
**Additive local files** are the files we wrote: the exporter, its checks, and
their support code. Nothing else in this directory differs from the authors'
release.

Regenerate with `scripts/check-paper-code-diff stepwise_embeddings`.

```
paper: stepwise_embeddings
upstream: https://github.com/ratschlab/clinical-embeddings.git
commit: e66cbdde0aff7dde7197d4520ec9df29b3edd58a
path: .
tree: c65aad1f1a6cb7376e7963654424215b8a642fce
modified_upstream_files: 0
deleted_upstream_files: 0
additive_local_files: 142

ADDITIVE LOCAL FILES
.python-version
check_batch.py
check_batch.sh
check_batch_ihm.py
check_cohort.py
check_cohort.sh
check_cohort_ihm.py
cohort_ihm.py
cohort_source.py
configs/reproduce/mimic/embeddings/ftt-no-group/one_epoch_no_test.gin
configs/reproduce/mimic/embeddings/ftt-no-group/transformer_ftt_decomp.gin
evaluate_reproduction_seed.py
export.sh
export_ihm.py
export_source.py
m3b_gzip.patch
preflight_export_size.py
prepare_cohort_source.py
record_gpu_environment.py
record_gpu_environment.sh
requirements-export.txt
requirements-gpu.txt
run_reproduction_all.sh
run_reproduction_epoch_benchmark.sh
run_reproduction_seed.sh
run_reproduction_smoke.sh
setup_gpu_env.sh
smoke_reproduction.py
summarize_reproduction.py
test_export.py
tls/.gitignore
tls/LICENSE
tls/Makefile
tls/README.md
tls/configs/hirid/GRU.gin
tls/configs/hirid/surv_GRU.gin
tls/configs/hirid/surv_transformer.gin
tls/configs/hirid/transformer.gin
tls/configs/mimic3-benchmark/surv_transformer.gin
tls/configs/mimic3-benchmark/transformer.gin
tls/environment.yml
tls/preprocessing/resources/channel_to_id_m3.pkl
tls/preprocessing/resources/matching_dict_m3.pkl
tls/preprocessing/resources/mimic3_resources/channel_to_id_m3.pkl
tls/preprocessing/resources/mimic3_resources/matching_dict_m3.pkl
tls/preprocessing/resources/mimic3_resources/var_range_m3.pkl
tls/preprocessing/resources/split.tsv
tls/preprocessing/resources/var_range_m3.pkl
tls/preprocessing/resources/varref.tsv
tls/run_scripts/baseline/Circ/GRU.sh
tls/run_scripts/baseline/Circ/GRU_LS.sh
tls/run_scripts/baseline/Circ/GRU_TLS_exp_param.sh
tls/run_scripts/baseline/Circ/GRU_TLS_step_ablation.sh
tls/run_scripts/baseline/Circ/GRU_ddrsa.sh
tls/run_scripts/baseline/Circ/GRU_landmarking.sh
tls/run_scripts/baseline/Circ/GRU_tcsr.sh
tls/run_scripts/baseline/Circ/multihorizon_GRU.sh
tls/run_scripts/baseline/Decomp/multihorizon_transformer.sh
tls/run_scripts/baseline/Decomp/transformer.sh
tls/run_scripts/baseline/Decomp/transformer_LS.sh
tls/run_scripts/baseline/Decomp/transformer_TLS_exp_param.sh
tls/run_scripts/baseline/Decomp/transformer_TLS_step_ablation.sh
tls/run_scripts/baseline/Decomp/transformer_ddrsa.sh
tls/run_scripts/baseline/Decomp/transformer_landmarking.sh
tls/run_scripts/baseline/Decomp/transformer_tcsr.sh
tls/run_scripts/baseline/Resp/multihorizon_transformer.sh
tls/run_scripts/baseline/Resp/transformer.sh
tls/run_scripts/baseline/Resp/transformer_LS.sh
tls/run_scripts/baseline/Resp/transformer_TLS_exp_param.sh
tls/run_scripts/baseline/Resp/transformer_TLS_step_ablation.sh
tls/run_scripts/baseline/Vent/GRU.sh
tls/run_scripts/baseline/Vent/GRU_TLS_exp.sh
tls/run_scripts/baseline/Vent/GRU_TLS_step_ablation.sh
tls/run_scripts/baseline/Vent/multihorizon_GRU.sh
tls/run_scripts/hp-search/architecture/Circ/GRU.sh
tls/run_scripts/hp-search/architecture/Decomp/transformer.sh
tls/run_scripts/hp-search/architecture/Resp/transformer.sh
tls/run_scripts/hp-search/architecture/Vent/GRU.sh
tls/run_scripts/hp-search/focal_strength/Circ/GRU.sh
tls/run_scripts/hp-search/focal_strength/Decomp/transformer.sh
tls/run_scripts/hp-search/focal_strength/Resp/transformer.sh
tls/run_scripts/hp-search/ls_strength/Circ/GRU_LS.sh
tls/run_scripts/hp-search/ls_strength/Decomp/transformer_LS.sh
tls/run_scripts/hp-search/smoothing_function/Circ/GRU_TLS_exp.sh
tls/run_scripts/hp-search/smoothing_function/Circ/GRU_TLS_sigmoid.sh
tls/run_scripts/hp-search/smoothing_function/Decomp/transformer_TLS_exp.sh
tls/run_scripts/hp-search/smoothing_function/Decomp/transformer_TLS_sigmoid.sh
tls/run_scripts/hp-search/smoothing_function/Resp/transformer_TLS_exp.sh
tls/run_scripts/hp-search/smoothing_function/Resp/transformer_TLS_sigmoid.sh
tls/run_scripts/hp-search/smoothing_function/Vent/GRU_TLS_exp.sh
tls/run_scripts/hp-search/smoothing_function/Vent/GRU_TLS_sigmoid.sh
tls/run_scripts/preprocess/hirid.sh
tls/run_scripts/preprocess/mimic3.sh
tls/setup.cfg
tls/setup.py
tls/tests/__init__.py
tls/tests/data/.gitkeep
tls/tests/data/test_features_extraction.py
tls/tests/endpoints/test_endpoints.py
tls/tests/labels/test_labels.py
tls/tests/preprocessing/test_preprocess.py
tls/tests/test_tls.py
tls/tls/__init__.py
tls/tls/common/__init__.py
tls/tls/common/constants.py
tls/tls/common/datasets.py
tls/tls/common/lookups.py
tls/tls/common/processing.py
tls/tls/common/reference_data.py
tls/tls/common/resampling.py
tls/tls/common/utils.py
tls/tls/data/__init__.py
tls/tls/data/endpoint_generation.py
tls/tls/data/extended_general_table_generation.py
tls/tls/data/feature_extraction.py
tls/tls/data/imputation_for_endpoints.py
tls/tls/data/labels.py
tls/tls/data/loader.py
tls/tls/data/preprocess.py
tls/tls/data/schemata.py
tls/tls/data/utils.py
tls/tls/endpoints/__init__.py
tls/tls/endpoints/endpoint_benchmark.py
tls/tls/imputation/forward_filling.py
tls/tls/imputation/impute_one_batch.py
tls/tls/labels/label_benchmark.py
tls/tls/labels/utils.py
tls/tls/models/__init__.py
tls/tls/models/encoders.py
tls/tls/models/layers.py
tls/tls/models/train.py
tls/tls/models/utils.py
tls/tls/models/wrappers.py
tls/tls/preprocessing/__init__.py
tls/tls/preprocessing/merge.py
tls/tls/preprocessing/preprocess_pharma.py
tls/tls/run.py
tls/tls/synthetic_data/__init__.py
tls/tls/synthetic_data/collect_stats.py
tls/tls/synthetic_data/generate_simple_fake_data.py
verify_epoch_benchmark.py
verify_reproduction_config.py

status: UPSTREAM_FILES_UNCHANGED
```
