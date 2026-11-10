# GraphCare GPU reproduction

This runner trains the released GraphCare BAT implementation from randomly
initialized weights on the exact exported MIMIC-III subsequent-visit mortality
cohort.  It is designed to answer two separate questions without conflating
them:

1. Can the released BAT model be trained and independently tested on the
   authors' reconstructed cohort and split?
2. How close can the public release get to Table 2 when the paper's private KG
   artifacts are unavailable?

The answer to the first question is directly testable.  The second remains a
qualified approximation because the concept-specific KG files, global graph,
GPT entity/relation embeddings, cluster centroids, and task attention vector
used for the paper were not released.

## Environment

The frozen paper environment remains Python 3.8, Torch 1.12, and CUDA 10.2.
It cannot execute kernels on the workstation's RTX 5090 (`sm_120`).  The setup
script therefore creates a second ignored environment under
`.export_assets/gpu_venv` with pinned Torch 2.8.0/CUDA 12.8 and PyG 2.6.1:

```bash
paper_code/graphcare/setup_gpu_env.sh
```

The default path changes no BAT equation.  The model source has an optional
guard around its unused `RETAINLayer` import, plus an opt-in padding-mask
ablation described below.

## Two released attention paths

The public repository does not provide a single unambiguous temporal model.
`run.sh` invokes `graphcare.py`, whose model constructor sets both `use_alpha`
and `use_beta` to false for either CLI value.  That primary released path is
order-invariant: prior visits only enlarge the union of patient nodes.

`graphcare_analysis.py` correctly passes the CLI attention flags and matches
the paper's advertised node/visit attention.  However, `model.py` applies its
softmax and tanh operations over the whole 28-row tensor without masking the
zero-padded visits.  The default runner exercises this paper-intended path;
passing `--disable-alpha --disable-beta` reproduces the primary driver's flag
bug.  Outputs must keep those configurations separate.

Passing `--mask-padded-visits` runs an explicit repaired-attention ablation.
It excludes zero rows from alpha's visit-wise softmax and zeros beta on those
rows; it is not represented as a reproduction of the released equations.

## Fidelity profiles

`direct_compact` is the primary public-artifact profile.  It maps every exact
CCS-CM, CCS-PROC, and ATC-3 occurrence through the released code-to-cluster
files.  The 785 source codes collapse to the 489 released clusters containing
direct EHR concepts.  Each direct node has a self edge, so BAT's node, visit,
and relation-attention path executes, but no invented knowledge nodes or
external edges are introduced.

`direct_released_space` keeps the paper's 4,599-node tensor width.  The 4,110
non-direct nodes stay inactive because their graph membership, topology, and
embeddings are unreleased.  This is useful as a dimensional-fidelity
sensitivity, but is substantially slower and does not restore the missing KG.

Both profiles use random unit-norm node/relation embeddings and train them
from scratch.  This replaces the unavailable GPT initialization and is
recorded in every result manifest.

## Commands

First verify the already-exported cohort and author batch boundary:

```bash
paper_code/graphcare/check_cohort.sh
paper_code/graphcare/check_batch.sh
```

Inspect the reconstructed neural-model boundary without training:

```bash
paper_code/graphcare/run_gpu_reproduction.sh --dry-run
```

Run the deliberately simple raw-data round trip. This reads the exact
MIMIC-III 1.4 diagnosis, procedure, and prescription tables, applies the pinned
GraphCare crosswalks, and compares every reconstructed sample/visit/namespace
cell with the export:

```bash
paper_code/graphcare/check_raw_roundtrip.sh
```

Run a short CUDA smoke test:

```bash
paper_code/graphcare/run_gpu_reproduction.sh \
  --profile direct_compact --seeds 0 --epochs 1 --max-train-batches 4 \
  --output-dir paper_results/graphcare/gpu_reproduction/smoke
```

Run the paper Table 10 mortality protocol (batch 4, hidden 128, learning rate
and weight decay 1e-5, decay 0.01, one BAT layer, 50 epochs, patience 10):

```bash
paper_code/graphcare/run_gpu_reproduction.sh \
  --profile direct_compact --seeds 0,1,2,3,4,5,6,7,8,9 \
  --output-dir paper_results/graphcare/gpu_reproduction/direct_compact
```

Run the same approximation with the primary released driver's disabled
alpha/beta attention behavior:

```bash
paper_code/graphcare/run_gpu_reproduction.sh \
  --profile direct_compact --disable-alpha --disable-beta \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --output-dir paper_results/graphcare/gpu_reproduction/released_driver
```

`--patient-mode node` supplies an explicitly cross-sectional control that
uses the released direct-node pooling head.  It is useful for testing whether
the graph/visit path adds anything beyond an unordered code set.

Every seed reloads the validation-selected checkpoint and evaluates the full
independent test split.  It also records the released-code validation metric
with `drop_last=True`.  The 6,186 length-one rows are cumulative prefix
examples: one observed admission predicts mortality on the next admission.
They are valid supervised examples, but contain no longitudinal transition.
Only 1,836 of the 6,186 subjects ever contribute a prefix of length two, and
only 752 contribute one of length three.  Evaluation is therefore stratified
into one-, exactly-two-, two-or-more-, three-or-more-, and five-or-more-visit
groups.  Results, checkpoints, and restricted predictions are written beneath
the chosen ignored output directory.

Run the equal-budget simple temporal LightGBM sensitivity with:

```bash
lightgbm_code/run_graphcare_sensitivity.sh \
  --trials 50 --tuning-seed 0 --overwrite
```

Each representation receives 50 paired hyperparameter draws. Repeat with
different tuning seeds and distinct output directories to assess validation
selection stability.

Run the training-only raw ICD sensitivity with its default five tuning seeds
and 50 exactly paired draws per representation:

```bash
lightgbm_code/run_graphcare_icd_sensitivity.sh --overwrite
```

It compares mapped CCS-CM/CCS-PROC/ATC3, raw diagnosis and procedure category
prefixes with ATC3, mapped codes augmented with those prefixes, and normalized
full raw diagnosis/procedure codes with ATC3. Every representation uses only a
current-visit bag and an any-prior-visit bag. Validation/test-only vocabulary
is ignored rather than learned transductively.

Run the plainer raw-code medication factorial with:

```bash
lightgbm_code/run_graphcare_plain_lightgbm.sh --overwrite
```

This is a two-by-three comparison: full raw ICD versus full raw ICD plus
category-prefix expansion, crossed with no medication features, GraphCare's
ATC3 medication mapping, or MIMIC's native Generic Sequence Number (GSN). GSN
is used as the compact RxNorm-like alternative; raw NDC product/package codes
are intentionally not used. The same 50 parameter draws are enforced across
all six representations for each of five tuning seeds.

After the ten-seed intended and released-driver runs and five LightGBM tuning
seeds are present, consolidate metrics, patient-clustered bootstrap intervals,
history strata, crosswalk retention, and cluster collapse with:

```bash
paper_code/graphcare/summarize_reproduction.sh
```

The findings and exact output locations are reported in
`GPU_REPRODUCTION_RESULTS.md`.
