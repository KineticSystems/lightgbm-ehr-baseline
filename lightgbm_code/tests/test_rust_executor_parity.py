"""Cross-executor checks.

The Rust executor's bit-level anchor is byte-identity against sealed
artifacts (verified on FuseMoE) and its own parallel/sequential determinism.
The fully-Python reference executor computes features through the literal
oracle, whose arithmetic is tolerance-equal (not bit-pinned) to Rust's, so
the cross-path expectation here is structural agreement: same selection,
same shapes, and closely matching quality — not byte equality.
"""

import numpy as np

from missing_xgboost_lightgbm.experiment import run

from test_experiment import _write_task


def test_python_reference_and_rust_executors_agree(tmp_path, monkeypatch):
    task_dir = tmp_path / "mortality"
    _write_task(task_dir)

    results = {}
    for executor in ("python", "rust"):
        monkeypatch.setenv("MXGB_EXECUTOR", executor)
        results[executor] = run(
            task_dir,
            tmp_path / executor,
            trials=6,
            threads=1,
            bootstrap_replicates=10,
        )
    monkeypatch.delenv("MXGB_EXECUTOR", raising=False)

    python_result, rust_result = results["python"], results["rust"]
    assert python_result["execution"]["executor"] == "python-reference-v1"
    assert rust_result["execution"]["executor"] == "rust-lightgbm-v1"
    assert python_result["selection"]["selected_representation"] == (
        rust_result["selection"]["selected_representation"]
    )
    assert python_result["selection"]["selected_trial"] == (
        rust_result["selection"]["selected_trial"]
    )
    python_members = [
        member["trial"]
        for member in python_result["selection"]["ensemble"]["members"]
    ]
    rust_members = [
        member["trial"] for member in rust_result["selection"]["ensemble"]["members"]
    ]
    assert python_members == rust_members
    assert abs(python_result["test"]["auroc"] - rust_result["test"]["auroc"]) < 0.02
    assert (
        abs(
            python_result["selection"]["ensemble"]["validation_auroc"]
            - rust_result["selection"]["ensemble"]["validation_auroc"]
        )
        < 0.02
    )
    with np.load(tmp_path / "python" / "predictions.npz") as left, np.load(
        tmp_path / "rust" / "predictions.npz"
    ) as right:
        assert set(left.files) == set(right.files)
        for key in left.files:
            assert left[key].shape == right[key].shape, key
        assert np.allclose(
            left["test_probability"], right["test_probability"], atol=0.05
        )


def test_python_reference_executor_is_deterministic(tmp_path, monkeypatch):
    task_dir = tmp_path / "mortality"
    _write_task(task_dir)
    monkeypatch.setenv("MXGB_EXECUTOR", "python")
    for name in ("first", "second"):
        run(
            task_dir,
            tmp_path / name,
            trials=4,
            threads=1,
            bootstrap_replicates=10,
        )
    monkeypatch.delenv("MXGB_EXECUTOR", raising=False)
    with np.load(tmp_path / "first" / "predictions.npz") as left, np.load(
        tmp_path / "second" / "predictions.npz"
    ) as right:
        for key in left.files:
            assert np.array_equal(left[key], right[key], equal_nan=True), key
    assert (tmp_path / "first" / "model_member_0.txt").read_bytes() == (
        tmp_path / "second" / "model_member_0.txt"
    ).read_bytes()
