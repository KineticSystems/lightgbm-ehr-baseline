# Exporting td_icu_mortality to MEDS

Paper: Robust Real-Time Mortality Prediction in the Intensive Care Unit using Temporal Difference Learning

## Tasks this directory exports

- `mortality_28d` -- 28-day mortality at rolling ICU measurement landmarks

## 1. Build this paper's environment

This paper's code runs on Python 3.11.15. Each paper pins its own
interpreter and dependency set, several of which conflict with each
other, so every paper gets its own virtualenv and none of them are
shared. The scripts here require exactly one thing: that
`.venv/bin/python` exists in this directory.

```
uv venv --python 3.11.15 .venv
uv pip install --python .venv/bin/python -r requirements-export.txt
```

`requirements-export.txt` is the frozen environment these exports were
produced in, pinned to exact versions. Any tool that can create a
virtualenv and install those pins will do; `uv` is only a convenience.

## 2. Point it at the data

Set the following in `workflows/data_paths.local.sh` at the release
root (copy it from `data_paths.example.sh` beside it):

- `MIMIC_IV_3_0_ROOT` -- MIMIC-IV v3.0

The exporter verifies the release you point it at and refuses to run
against a substitute, because the paper's cohort is defined against
that exact version. Every script below sources this file itself.

## 3. Export the cohort

```
./export.sh
```

This writes a MEDS cohort under `paper_tasks/td_icu_mortality/` containing
exactly the patients, features, labels and splits the paper's own model
was evaluated on. It refuses to overwrite an existing export, so remove
the task directory first if you are re-running it.

## 4. Verify it

```
./check_cohort.sh
./check_batch.sh
```

`check_cohort.sh` reconstructs the cohort independently and compares
it against the statistics the paper reports.
`check_batch.sh` replays the paper's own dataloader and compares every
exported row against what it produces.

Both exit non-zero on any mismatch. They are the evidence that the
exported cohort is the paper's cohort and not an approximation of it.

## Then run the baseline

The exported cohort is the input to the LightGBM baseline; see the
top-level `README.md`.

## The other files here

`README.md` is the original authors' own and describes their pipeline,
not this export. `AUTHOR_CODE_DIFF.md` is the complete difference between
their released code and the copy in this directory.
