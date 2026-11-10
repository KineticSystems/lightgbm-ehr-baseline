# Exporting multimodal_clinical_pretraining to MEDS

Paper: Multimodal Pretraining of Medical Time Series and Notes

## Tasks this directory exports

- `mortality_48h` -- Full-label in-hospital mortality from first 48 ICU hours

## 1. Build this paper's environment

This paper's code runs on Python 3.9.16. Each paper pins its own
interpreter and dependency set, several of which conflict with each
other, so every paper gets its own virtualenv and none of them are
shared. The scripts here require exactly one thing: that
`.venv/bin/python` exists in this directory.

```
uv venv --seed --python 3.9.16 .venv
.venv/bin/pip install torch==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements-export.txt
```

Install torch's CPU build first. Nothing in this directory needs a
GPU, and the default build from PyPI carries the CUDA runtime, which
is several times larger. The requirements file pins torch without a
build tag, so it accepts the CPU build and leaves it alone.

That one step uses `pip` rather than `uv` because PyTorch publishes
its CPU wheels on its own index, which a resolver cannot merge with
PyPI for the same package; `--seed` is what puts `pip` in the
virtualenv.

`requirements-export.txt` is the frozen environment these exports were
produced in, pinned to exact versions. Any tool that can create a
virtualenv and install those pins will do; `uv` is only a convenience.

Beyond Python, this paper needs:

The authors' listfiles, which define the cohort and are not derivable
from MIMIC. Their `README.md` in this directory links the Google Drive
folder they published them in; the exporter expects the
`in-hospital-mortality` directory from it. Point `MCP_LIST_ROOT` at
that directory, or place it at the default path the scripts name.

## 2. Point it at the data

Set the following in `workflows/data_paths.local.sh` at the release
root (copy it from `data_paths.example.sh` beside it):

- `MIMIC_III_1_4_ROOT` -- MIMIC-III v1.4

The exporter verifies the release you point it at and refuses to run
against a substitute, because the paper's cohort is defined against
that exact version. Every script below sources this file itself.

## 3. Build this paper's source artifacts

The exporter reads intermediate artifacts under `.export_assets/`,
here the MIMIC-III benchmark build. That directory is not part of the release: it runs
to tens of gigabytes per paper and is derived from credentialed
MIMIC, so it has to be rebuilt locally.

```
./.venv/bin/python reproduce_ihm.py
```

It is resumable and skips work already done, so re-running after an
interruption is safe. Everything below assumes it has finished.

## 4. Export the cohort

```
./export.sh
```

This writes a MEDS cohort under `paper_tasks/multimodal_clinical_pretraining/` containing
exactly the patients, features, labels and splits the paper's own model
was evaluated on. It refuses to overwrite an existing export, so remove
the task directory first if you are re-running it.

## 5. Verify it

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
