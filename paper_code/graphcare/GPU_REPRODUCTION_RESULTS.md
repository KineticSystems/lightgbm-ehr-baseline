# GraphCare mortality reproduction results

## Bottom line

The exact Table 2 GraphCare result cannot be regenerated from the released
repository because the complete concept-specific knowledge graphs, global
graph, entity/relation dictionaries, GPT embeddings, attention initialization,
processed samples, checkpoints, and original split/result artifacts are absent.
The repository explicitly says that the embeddings are not included, and it
contains only clustering maps and notebooks where the required graph assets
should have been.

The strongest attributable public-artifact approximation uses the exact Stage
3 cohort and split, the released direct code-to-cluster maps, the released BAT
model equations, the paper's Table 10 mortality hyperparameters, self edges in
place of unobservable KG edges, and randomly initialized trainable embeddings.
Across ten GPU training seeds it obtains test AUROC `0.6548 +/- 0.0018` and
AUPRC `0.1281 +/- 0.0025`, versus the paper's `0.703` and `0.167`.

This is not labeled an exact GraphCare reproduction. It is a bounded
public-artifact reproduction that makes every missing input explicit.

The new stabilized comparison does **not** support a claim that this neural
approximation beats LightGBM in AUROC. The ten-model neural probability
ensemble obtains `0.6551/0.1264`; a five-model, validation-selected LightGBM
ensemble using training-only CCS/ATC vocabulary and only current/any-prior
features obtains `0.6541/0.1099`. The paired patient bootstrap for neural minus
LightGBM is `+0.0010` AUROC (95% interval `[-0.0635, 0.0646]`) and `+0.0165`
AUPRC (`[-0.0416, 0.0755]`). With only 54 test deaths, neither difference is
distinguishable from zero.

## Main comparison

| Model or representation | Validation AUROC | Test AUROC | Test AUPRC | Repetitions |
|---|---:|---:|---:|---:|
| Paper GraphCare/BAT, Table 2 | not reported | 0.703 | 0.167 | paper says 100 runs |
| Public approximation, intended alpha/beta attention | 0.6654 +/- 0.0020 | 0.6548 +/- 0.0018 | 0.1281 +/- 0.0025 | 10 training seeds |
| Public approximation, mean-probability ensemble | -- | 0.6551 | 0.1264 | 10-model ensemble |
| Same approximation, released 4,599-node tensor width | 0.6642 | 0.6544 | 0.1278 | 1 training seed |
| Public approximation, primary released driver flags | 0.6535 +/- 0.0023 | 0.6444 +/- 0.0052 | 0.1209 +/- 0.0089 | 10 training seeds |
| Existing shared LightGBM, 5,113 features | 0.6865 | 0.6368 | 0.1047 | one validation-selected search |
| LightGBM mapped CCS/ATC, current + any prior | 0.6776 +/- 0.0029 | 0.6331 +/- 0.0158 | 0.1027 +/- 0.0082 | 5 validation-selected models |
| Same mapped LightGBM, mean-probability ensemble | -- | 0.6541 | 0.1099 | 5-model ensemble |
| LightGBM full raw ICD + ATC3 ensemble | -- | 0.6509 | 0.1136 | 5-model ensemble |
| LightGBM raw ICD prefixes + ATC3 ensemble | -- | 0.6286 | 0.1135 | 5-model ensemble |
| LightGBM mapped codes + raw ICD prefixes ensemble | -- | 0.6296 | 0.1230 | 5-model ensemble |

The repetitions have different meanings: the GraphCare rows vary model
initialization and minibatch order, while the simple LightGBM rows vary the
random hyperparameter search. The fixed patient split and its test set do not
change. Each LightGBM representation receives the same 50 parameter draws per
tuning seed, and no test label is used for model or representation selection.
The ensembles are a useful stability comparison, not a preregistered estimate
from the original paper.

The ten-seed intended-attention mean-probability ensemble has test AUROC
`0.6551`, with a patient-clustered 95% bootstrap interval of
`[0.5842, 0.7278]`. The paper value is inside that wide sampling interval, but
the ten independently fitted point estimates are all near `0.655`; this is not
evidence that the public approximation reproduced `0.703`.

Comparing only the mean of individually selected LightGBM models (`0.6331`)
does reproduce neural-greater-than-LightGBM to a degree. That contrast is not
stable to averaging those five independently validation-selected predictions:
the LightGBM AUROC becomes `0.6541`. The defensible conclusion is therefore
that an individual LightGBM search is unstable on this small-event test set,
not that GraphCare has established an architectural advantage.

## Recommended LightGBM time representation

Use two binary code bags:

1. code present in the current observed visit;
2. code present in any earlier observed visit.

This is the smallest temporal representation that is stable in this cohort.
In the new inductive comparison it has 1,286 features: 643 codes seen in
training patients, each represented once for the current visit and once for
any prior visit. Its individual test AUROC ranges from `0.6162` to `0.6491`,
while its probability ensemble reaches `0.6541`.

The data boundary has ordinal visit position but no clinical time gaps. More
elaborate recency bins therefore create sparsity without recovering elapsed
time. The two-bin representation asks the only robust temporal question the
released data can answer: is the code current, or merely historical?

This representation does not apply GraphCare's code-to-node cluster collapse.
It retains the 643 training-observed CCS-CM, CCS-PROC, and ATC3 codes and drops
only held-out-only tokens. That makes it a clean structured-EHR baseline, not
a reconstruction of GraphCare's external-KG input.

## Raw ICD feature sensitivity

The raw-code experiment uses the same exact cohort, patient split, current vs
any-prior time bins, five tuning seeds, and 50 paired hyperparameter draws per
representation. Vocabulary is fit on training patients only. The diagnosis
prefix is the three-character ICD-9-CM category (four for E codes), procedure
prefix is the two-digit ICD-9 procedure category, and drugs remain ATC3 so the
test isolates the suspected ICD/CCS conversion problem.

Raw prefixes do not improve the mapped baseline. Relative to the mapped
CCS/ATC ensemble, raw-prefix + ATC3 changes AUROC by `-0.0255` (paired 95%
interval `[-0.0715, 0.0216]`); mapped + prefixes changes it by `-0.0245`
(`[-0.0702, 0.0190]`). Full raw ICD + ATC3 is nearly identical at `-0.0032`
(`[-0.0470, 0.0393]`). The enriched representations often win on validation
(mapped never wins the across-representation choice in the five searches) but
do not carry that advantage to test, a direct sign of selection noise rather
than a useful missing feature family.

So the CCS conversion is lossy, but it does not explain the low LightGBM
AUROC. Keeping full raw diagnosis and procedure codes recovers no measurable
test advantage; coarse prefixing is worse in AUROC here.

The primary ten-seed GPU estimate uses the compact 489-observed-cluster alpha
layer. Because the released model defines alpha as a dense node-by-node layer,
a full 4,599-node run was also trained to convergence. Its test result,
`0.6544/0.1278`, is effectively the same as the compact estimate. Thus the
compact parameterization is not responsible for the performance gap in this
sensitivity. The full-width run still cannot activate or reconstruct the 4,110
KG-only nodes that are absent from the release.

## Why the short histories matter

There are 9,717 prefix samples from 6,186 patients. History length is extremely
short:

- 6,186 samples (63.7%) have one observed visit;
- 8,022 (82.6%) have at most two;
- 8,774 (90.3%) have at most three;
- only 595 (6.1%) have at least five.

A one-visit row is still a valid supervised example: one observed admission is
used to predict mortality on the immediately subsequent admission. It is not a
patient with no target visit. Every patient contributes one such prefix, while
only 1,836 patients ever contribute a prefix with two observed visits and only
752 ever contribute one with three.

The independent test set contains 996 prefixes and only 54 deaths. Of those,
619/38 are one-visit rows, 377/16 have at least two visits, 200/9 have at least
three, and 96/2 have at least five. For the intended-attention ensemble:

| Test stratum | Rows / deaths | AUROC | AUPRC |
|---|---:|---:|---:|
| One observed visit | 619 / 38 | 0.6427 | 0.1449 |
| At least two | 377 / 16 | 0.6543 | 0.1078 |
| Exactly two | 177 / 7 | 0.7992 | 0.1700 |
| At least three | 200 / 9 | 0.5858 | 0.0607 |
| At least five | 96 / 2 | 0.4521 | 0.0277 |

The exactly-two result is based on seven deaths and the five-plus result on two;
neither is a reliable model comparison. There is no empirical support here for
a claim that long-sequence modeling drives the aggregate result.

## Conversion audit

The simplest raw-MIMIC check independently reads the three MIMIC-III 1.4 gzip
tables, applies the pinned GraphCare crosswalk CSVs, forms per-admission sets,
and compares them with every right-aligned input cell in the exported task. It
checks 9,717 selected admissions, 18,249 sample/visit cells, and 54,747
sample/visit/namespace cells. The result is exact: zero mismatched cells, zero
missing mapped events, zero unexpected mapped events, and zero orphan export
cells. This rules out a raw-row-to-export implementation discrepancy relative
to the released crosswalks; it does not assert that those crosswalks are
clinically optimal.

The ICD/CCS and procedure conversion is coarse but not broadly broken on the
selected visits:

- diagnoses: 137,748 raw occurrences, zero unmapped, 266 CCS-CM outputs;
- procedures: 46,036 raw occurrences, zero unmapped, 195 CCS-PROC outputs;
- drugs: 941,995 raw prescription rows, 166,138 unmapped (17.64%), 192 ATC3
  outputs.

Of the unmapped drug rows, 130,448 are the invalid placeholder NDC `0`.
Excluding that placeholder, the remaining unmapped drug rows are 35,690, or
3.79% of all raw prescription rows. Large within-visit duplicate counts are an
expected consequence of converting prescription records to a set of codes per
visit, not evidence of random strings entering the model.

The conversion is aggressively many-to-one. Of the mapped targets observed in
the selected visits, 249/266 CCS-CM targets, 138/195 CCS procedure targets, and
172/192 ATC3 targets have multiple raw sources. The median mapped target
collects 10 diagnosis codes, 3 procedure codes, or 12 NDC codes; the respective
maxima are 113, 57, and 144. The exact round trip therefore says the pipeline
implemented its maps consistently, not that no information was discarded.

GraphCare's released cluster mapping is a more substantial information
bottleneck for the direct-node branch. The 785 declared paper codes, of which
653 occur in this cohort, map to 489 direct clusters. Forty-six clusters are
shared across code namespaces and five are shared across all three. Within a
namespace, 36.0% of condition events, 49.9% of procedure events, and 48.7% of
drug events map to a cluster that represents multiple released codes. The
unreleased KG topology could still distinguish codes through different
neighbors; the public direct-node approximation cannot.

## Released implementation defects

1. `run.sh` invokes `graphcare.py`, which hard-codes both `use_alpha` and
   `use_beta` to false when constructing the model. Command-line attention
   flags are ignored. Restoring the intended flags improves the ten-seed test
   mean by about 0.0104 AUROC and reduces seed variance.
2. The alternate analysis driver passes the flags, but the released model
   applies attention over all 28 visit rows without masking right-side zero
   padding. A one-visit prefix therefore has one real row and 27 zero rows in
   the visit softmax. An opt-in padding-mask ablation was added without changing
   the default equations; its single-seed result did not improve this public
   approximation.
3. The released binary driver creates a test loader but never evaluates it. It
   reports validation and does not reload a best checkpoint for an independent
   test evaluation. The reproduction runner retains released drop-last behavior
   for training and validation selection, then reloads the best checkpoint and
   evaluates full validation and test sets.
4. The paper's Table 10 mortality configuration specifies hidden size 128, 50
   epochs, and patience 10, while the released shell command uses hidden size
   512 and 100 epochs. The primary reproduction follows the paper configuration.

## Reproduction artifacts

- `reproduce_gpu.py`: exact-cohort adapter, GPU training, checkpoint selection,
  prediction export, and per-history evaluation.
- `graphcare_/model.py`: released model plus an opt-in padded-visit mask; the
  default path is unchanged.
- `graphcare_sensitivity.py`: equal-budget current/prior LightGBM sensitivity.
- `graphcare_icd_sensitivity.py`: inductive mapped, raw-prefix, augmented, and
  full-raw ICD LightGBM comparison with paired search draws.
- `graphcare_plain_lightgbm.py`: raw-feature LightGBM comparison.

  These three lived in `lightgbm_code` and were removed once this reproduction
  left the manuscript's retraining table; nothing in the paper reads their
  output. Their results below stand as recorded, and the code is in Git history.
  They are not needed to reproduce anything the paper reports: GraphCare's
  campaign numbers come from `missing-xgboost-run-lightgbm`, like every other
  paper's.
- `check_raw_roundtrip.py`: direct raw MIMIC-III-to-export cell comparison.
- `investigate_crosswalks.py`: raw MIMIC-to-export crosswalk accounting.
- `summarize_reproduction.py`: multi-seed ensemble, clustered bootstrap,
  LightGBM aggregation, and cluster-collapse accounting.
- `paper_results/graphcare/reproduction_summary.json`: generated consolidated
  evidence.

The modern environment is intentionally separate from the frozen paper
environment because Torch 1.12/CUDA 10.2 cannot execute kernels for the RTX
5090 (`sm_120`). The modern runner uses Torch 2.8.0 with CUDA 12.8 and PyG
2.6.1; CUDA forward and backward execution were verified before the full runs.
