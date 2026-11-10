# A Simple LightGBM Baseline Beats Most Deep Learning Methods for Structured EHR Prediction

Over the past several years, many deep learning methods have been developed to improve risk prediction using structured electronic health record (EHR) data. However, papers proposing such methods typically only evaluate them against other deep learning methods, overlooking simple yet powerful techniques such as gradient-boosted trees (e.g., XGBoost and LightGBM).

Our ML4H 2026 paper "A Simple LightGBM Baseline Beats Most Deep Learning Methods for Structured EHR Prediction" investigates this. We do a literature review and find that only 20.9% of deep learning papers include a gradient-boosted tree baseline. We determine the extent of the problem by adding LightGBM baselines to published papers, and find that our LightGBM baselines achieve higher AUROC scores than 81% of the proposed deep learning methods.

This repository contains our code for running the LightGBM experiments in our paper, including our LightGBM baseline as well as our code for extracting cohorts from published work.

## Setup

1. Obtain credentialed access to the MIMIC datasets from PhysioNet. Each paper
   requires a specific release; the exporter for that paper checks the release
   you point it at and refuses to run against a substitute.

2. Copy `workflows/data_paths.example.sh` to `workflows/data_paths.local.sh`
   and set each variable to your local copy of the corresponding dataset. Only
   the datasets used by the papers you intend to export need to be set. Every
   exporter reads that file, so nothing else needs configuring.

3. Install the baseline. It needs a Rust toolchain and `uv`:

       cd lightgbm_code
       uv sync --frozen
       cd rust_featurizer && cargo build --release

   `uv` reads `.python-version` and fetches that interpreter. The pin is not
   incidental: the lock resolves a different NumPy and SciPy on Python 3.11
   than on 3.12, which moves reported AUROCs in the third decimal place.

   Install it before exporting anything: each exporter validates the cohort it
   writes by calling the baseline.

## Convert a paper's cohort to MEDS

Each directory under `paper_code/` is one paper: the authors' released code,
plus the exporter we added. Start from `EXPORT_GUIDE.md` inside that directory.
It names the MIMIC release that paper needs, the tasks it exports, and the
commands, in order:

       cd paper_code/<paper>
       uv venv --seed --python "$(cat .python-version)" .venv
       .venv/bin/pip install torch==<pinned>+cpu \
           --index-url https://download.pytorch.org/whl/cpu
       uv pip install --python .venv/bin/python -r requirements-export.txt
       ./export.sh
       ./check_cohort.sh
       ./check_batch.sh

Take the exact commands from the guide; the shape above is only the outline.

Every paper builds its own virtualenv at `.venv` in its own directory. Their
interpreters span Python 3.8 to 3.11 and their pinned dependencies conflict, so
the environments are never shared, and none of them is the one the baseline runs
in. `requirements-export.txt` is the frozen environment that paper's export was
produced in.

Nothing here needs a GPU. Eighteen of the twenty papers pin torch, and their
guides install the CPU build first, as above: it is several times smaller than
the default build and the requirements file accepts it. The two that pin no
torch omit that step and need no `--seed`.

Three papers need something outside Python, which their guides explain: CLUDA
builds its cohort with the authors' R pipeline and ships an `environment-r.yml`
for it, REMed's source tables come out of a PySpark job that needs a JDK, and
kedgn carries a caveat about loading torch on hardened kernels.

The export writes a MEDS cohort containing exactly the patients, features,
labels and splits the paper's own model was evaluated on. Verify it before you
use it: `check_cohort.sh` reconstructs the cohort independently and compares it
against the paper's reported statistics, and `check_batch.sh` replays the
paper's own dataloader and compares every exported row against what it produces.
Both exit non-zero on any mismatch and leave an audit record beside the cohort.
The baseline reads those records, notes in `result.json` what it found, and
warns if you fit a cohort that was never verified.

Expect this to take a while. One of the smaller papers, td_icu_mortality, takes
about 25 minutes to export, writes 2 GB, and takes another 20 to check.

`AUTHOR_CODE_DIFF.md` in each paper directory is the complete difference between
the authors' released code at its pinned commit and the copy here, so every
change we made is visible without cloning anything.

## Run the baseline

The baseline reads any MEDS cohort and needs to know nothing about the paper it
came from:

       lightgbm_code/.venv/bin/missing-xgboost-run-lightgbm \
           --task-dir <exported cohort> \
           --output-dir <where to write results>

It featurizes the events, runs a 50-trial random search over LightGBM
hyperparameters tuned on the paper's own validation split, and reports the
logit-averaged ensemble of the five best configurations on the test split.
`result.json` holds the AUROC and its confidence interval.

Two environment variables matter. `MXGB_THREADS` defaults to 2 and the fitted
model depends on it, so leave it alone to reproduce our numbers.
`MXGB_PARALLEL_TRIALS` sets how many trials run at once and is free to raise on
a larger machine.

Run it once per exported cohort. For the papers whose method consumes clinical
text, `missing-xgboost-run-stacking` fits the note-stacking variant the same
way.
