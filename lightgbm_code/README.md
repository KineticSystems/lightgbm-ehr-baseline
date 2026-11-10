# Shared binary LightGBM runner

This package is the only model-training implementation used by the binary
reproductions. Paper repositories do not contain LightGBM training code. Each
paper contributes one thin command that exports its released event stream,
labels, identifiers, and train/validation/test partitions into the
[MEDS common task format](common_format/README.md). Exporters write those
Parquet files directly; there is no legacy dense or NPZ staging format.

Install, build, and test the shared package once:

```bash
uv sync --extra test
cargo build --release -j 2 \
  --locked --manifest-path rust_featurizer/Cargo.toml
uv run pytest
```

The optimized executable is checked against the deliberately slow, literal
Python oracle in `src/missing_xgboost_lightgbm/reference_features.py`. The
parity suite covers numeric and occurrence-only codes, absent variables,
segment boundaries, multiplicity, stable exact ties, the untimed projection,
and identical output with one versus four Rayon threads. The Rust crate also has direct unit tests;
`cargo clippy --all-targets -- -D warnings` must pass.

Then run any paper export with the same training command:

```bash
.venv/bin/missing-xgboost-run-lightgbm \
  --task-dir ../paper_tasks/<paper>/<task> \
  --output-dir /path/to/fresh/output
```

The runner consumes exactly one task folder per invocation. From its MEDS
events it builds every representation the task's temporal profile allows —
always `summary`, plus `visit_halves` or `visit_buckets` where the profile
admits one — and searches the representation jointly with LightGBM
hyperparameters using an Optuna `RandomSampler`. Selection uses validation
AUROC only; test labels are not loaded by the tuning objective. See
[the common task format](common_format/README.md) for what each profile
allows and, for `multi_visit_untimed`, which statistics it withholds so that
LightGBM is never shown elapsed time the published method cannot see.

Every trial is fit on the exported training partition with the exported
validation partition used for early stopping. All trials are planned before
any training, then executed by `MXGB_PARALLEL_TRIALS` worker threads
(default 4) over one shared native Dataset per representation, with
`MXGB_THREADS` LightGBM threads per trial (default 2); planning-before-
execution keeps every fitted model independent of worker count and
completion order. The search space floors (`num_leaves >= 8`,
`min_child_samples >= 10`, `max_depth <= 12`) exist because validation-argmax
selection harvests fragile extremes: without them the argmax will take a
degenerate configuration -- boosted stumps, or leaves holding a single sample
-- that scores highest on validation and worst on test.

The representation is selected by the single best validation-AUROC trial,
but the shipped model is the uniform logit-averaged ensemble of the top-5
validation trials within that representation (Caruana et al., ICML 2004,
sorted initialization; auto-sklearn, NeurIPS 2015). The ensemble's members
are written as `model_member_<rank>.txt` with rank 0 the selected trial. The
test partition is opened only after model and representation selection is
complete.

High-dimensional sparse tasks remain SciPy CSR through LightGBM's native
sparse input path. LightGBM's disposable histogram cache is capped at 1 GiB;
cache eviction only triggers deterministic histogram recomputation and does
not alter the feature contract or parameter search.

## Structured-plus-notes stacking for note-using papers

When—and only when—the proposed paper method itself consumes clinical notes,
an audited MEDS task declares raw text codes and runs the shared stacked model:

```bash
.venv/bin/missing-xgboost-run-stacking \
  --task-dir ../paper_tasks/<paper>/<task> \
  --structured-result-dir /path/to/that-task/structured-result \
  --output-dir /path/to/fresh/output
```

The fixed note learner is word unigram/bigram TF-IDF (`min_df=5`,
`max_df=0.98`, at most 50,000 terms, sublinear TF, L2 normalization) followed
by balanced L2 logistic regression. The vocabulary is fit on training only.
Each prespecified `C` in `10^-4, 10^-3, ..., 10^4` is fit on training, and the
single candidate with maximum validation AUROC is retained; there is no
cross-validation. Exact ties choose the smallest `C`. The only combined model
is an ordinary unregularized logistic regression on structured and text
log-odds. That three-parameter model is fit once on every validation row, with
no folds or hyperparameter tuning, before test is loaded.

The note vocabulary and classifier are fit on training only. Appending the
note prediction to LightGBM and fixed-weight averaging are not part of the
protocol. Stacking is forbidden for a structured-only source method: cohort
filtering by note availability does not make note content a fair input.
