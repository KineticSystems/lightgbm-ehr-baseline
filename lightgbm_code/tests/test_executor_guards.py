"""Executor guardrails.

The Rust executor drives LightGBM through its public C API and shares const
Dataset handles across trial threads. That sharing is safe only because the
C++ side takes const pointers and keeps no mutable members, which is a
property of the pinned version, so the pin below demands a re-audit of those
call paths before upgrading.
The determinism test proves the selected results independent of the Rust
executor's worker count.
"""

import lightgbm
import numpy as np

from missing_xgboost_lightgbm.experiment import run

from test_experiment import _write_task


AUDITED_LIGHTGBM_VERSION_PREFIX = "4.7."


def test_lightgbm_version_matches_thread_safety_audit():
    assert lightgbm.__version__.startswith(AUDITED_LIGHTGBM_VERSION_PREFIX), (
        f"LightGBM {lightgbm.__version__} is outside the audited series; "
        "re-audit the Rust executor's shared-Dataset threading before "
        "upgrading"
    )


def test_rust_executor_worker_count_does_not_change_results(tmp_path, monkeypatch):
    task_dir = tmp_path / "mortality"
    _write_task(task_dir)
    monkeypatch.setenv("MXGB_EXECUTOR", "rust")
    for name, workers in (("serial", "1"), ("parallel", "4")):
        monkeypatch.setenv("MXGB_PARALLEL_TRIALS", workers)
        run(
            task_dir,
            tmp_path / name,
            trials=6,
            threads=1,
            bootstrap_replicates=10,
        )
    with np.load(tmp_path / "serial" / "predictions.npz") as left, np.load(
        tmp_path / "parallel" / "predictions.npz"
    ) as right:
        for key in left.files:
            assert np.array_equal(left[key], right[key], equal_nan=True), key
    # The histogram pool is a fixed per-trial budget rather than a total
    # divided across workers, so the fitted trees are worker-invariant too and
    # the saved models must match byte for byte.
    left_models = sorted((tmp_path / "serial").glob("model_member_*.txt"))
    right_models = sorted((tmp_path / "parallel").glob("model_member_*.txt"))
    assert left_models and len(left_models) == len(right_models)
    for one, other in zip(left_models, right_models):
        assert one.read_text() == other.read_text(), one.name
