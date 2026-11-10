import json

from lightgbm.basic import LightGBMError
import numpy as np
import polars as pl
import pytest

from missing_xgboost_lightgbm.common_format import open_task
from missing_xgboost_lightgbm.experiment import (
    _audit_inventory,
    _cross_split_group_overlap,
    _recoverable_lightgbm_trial_failure,
    _validate_partitions,
    run,
)


@pytest.mark.parametrize("side", ["left", "right"])
def test_only_lightgbm_empty_child_trial_failure_is_recoverable(side):
    assert _recoverable_lightgbm_trial_failure(
        LightGBMError(
            f"Check failed: (best_split_info.{side}_count) > (0) at source.cpp"
        )
    )
    assert not _recoverable_lightgbm_trial_failure(
        LightGBMError("Dataset was constructed with parameters that disagree")
    )
    assert not _recoverable_lightgbm_trial_failure(
        RuntimeError(
            f"Check failed: (best_split_info.{side}_count) > (0) at source.cpp"
        )
    )
from missing_xgboost_lightgbm.multimodal_experiment import run as run_stacking
from missing_xgboost_lightgbm.polars_meds import write_meds_task


def _write_task(task_dir, *, with_notes=False):
    count = 160
    codes = ("a", "b", "c", "d")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(count, len(codes))).astype(np.float32)
    y = (x[:, 0] > 0).astype(bool)
    y[[0, 1, 80, 81, 120, 121]] = [False, True] * 3
    starts = np.asarray(["2100-01-01T00:00:00"] * count, dtype="datetime64[us]")
    split = np.asarray(["train"] * 80 + ["validation"] * 40 + ["test"] * 40)
    samples = pl.DataFrame(
        {
            "subject_id": pl.Series(np.arange(count), dtype=pl.Int64),
            "sample_id": pl.Series(np.arange(count), dtype=pl.Int64),
            "observation_start": pl.Series(starts, dtype=pl.Datetime("us")),
            "prediction_time": pl.Series(
                starts + np.timedelta64(48, "h"), dtype=pl.Datetime("us")
            ),
            "boolean_value": pl.Series(y, dtype=pl.Boolean),
            "split": pl.Series(split, dtype=pl.String),
        }
    )
    events = pl.DataFrame(
        {
            "subject_id": pl.Series(np.repeat(np.arange(count), 4), dtype=pl.Int64),
            "time": pl.Series(
                np.repeat(starts, 4) + np.tile(np.arange(1, 5), count).astype("timedelta64[h]"),
                dtype=pl.Datetime("us"),
            ),
            "code": pl.Series(np.tile(codes, count), dtype=pl.String),
            "numeric_value": pl.Series(x.reshape(-1), dtype=pl.Float32),
            "sample_id": pl.Series(np.repeat(np.arange(count), 4), dtype=pl.Int64),
            "source_row": pl.Series(np.arange(count * 4), dtype=pl.Int64),
        }
    )
    codes_for_export = codes
    note_codes = ()
    if with_notes:
        events = events.with_columns(
            pl.lit(None, dtype=pl.String).alias("text_value")
        )
        notes = pl.DataFrame(
            {
                "subject_id": pl.Series(np.arange(count), dtype=pl.Int64),
                "time": pl.Series(
                    starts + np.timedelta64(5, "h"), dtype=pl.Datetime("us")
                ),
                "code": pl.Series(["NOTE"] * count, dtype=pl.String),
                "numeric_value": pl.Series([None] * count, dtype=pl.Float32),
                "sample_id": pl.Series(np.arange(count), dtype=pl.Int64),
                "source_row": pl.Series(
                    np.arange(count * 4, count * 5), dtype=pl.Int64
                ),
                "text_value": pl.Series(
                    ["critical high risk" if label else "stable normal" for label in y],
                    dtype=pl.String,
                ),
            }
        )
        events = pl.concat([events, notes])
        codes_for_export = (*codes, "NOTE")
        note_codes = ("NOTE",)
    write_meds_task(
        task_dir,
        paper={"key": "example", "title": "Example"},
        task={
            "id": "mortality",
            "name": "Mortality",
            "primary_metric": "AUROC",
            "published_auroc": None,
            "observation_window_hours": 48.0,
        },
        codes=codes_for_export,
        note_codes=note_codes,
        events=events,
        samples=samples,
        source={"loader_entrypoint": "test fixture"},
        dataset_name="test",
        dataset_version="1",
    )
    (task_dir / "audit" / "full_batch_recreation.json").write_text(
        json.dumps({"status": "exact"}) + "\n"
    )
    (task_dir / "audit" / "cohort_reconstruction.json").write_text(
        json.dumps({"status": "exact"}) + "\n"
    )


def test_audit_inventory_records_exact_cohort_and_batch_reports(tmp_path):
    task_dir = tmp_path / "mortality"
    _write_task(task_dir)
    source = open_task(task_dir)
    inventory = _audit_inventory(source)
    assert set(inventory) == {"batch", "cohort"}
    assert all(entry["exact_source"] == "true" for entry in inventory.values())


def test_audit_inventory_reports_a_missing_check_without_stopping(tmp_path, capsys):
    task_dir = tmp_path / "mortality"
    _write_task(task_dir)
    source = open_task(task_dir)
    (task_dir / "audit" / "cohort_reconstruction.json").unlink()

    inventory = _audit_inventory(source)

    assert inventory["cohort"] == {"status": "missing", "command": "check_cohort.sh"}
    assert inventory["batch"]["exact_source"] == "true"
    warning = capsys.readouterr().err
    assert "has not been verified" in warning
    assert "check_cohort.sh" in warning


def test_audit_inventory_reports_an_inexact_check_without_stopping(tmp_path, capsys):
    task_dir = tmp_path / "mortality"
    _write_task(task_dir)
    (task_dir / "audit" / "cohort_reconstruction.json").write_text(
        json.dumps({"status": "approximate"}) + "\n"
    )

    inventory = _audit_inventory(open_task(task_dir))

    assert inventory["cohort"]["exact_source"] == "false"
    assert inventory["cohort"]["status"] == "approximate"
    assert "not exact-source" in capsys.readouterr().err


def test_audit_inventory_can_be_required(tmp_path, monkeypatch):
    task_dir = tmp_path / "mortality"
    _write_task(task_dir)
    source = open_task(task_dir)
    (task_dir / "audit" / "cohort_reconstruction.json").unlink()
    monkeypatch.setenv("MXGB_REQUIRE_AUDITS", "1")

    with pytest.raises(ValueError, match="check_cohort.sh"):
        _audit_inventory(source)


def test_audit_inventory_records_explicit_exact_source_qualification(tmp_path):
    task_dir = tmp_path / "mortality"
    _write_task(task_dir)
    cohort_path = task_dir / "audit" / "cohort_reconstruction.json"
    cohort_path.write_text(
        json.dumps({"status": "exact_source_with_published_count_mismatch"}) + "\n"
    )

    inventory = _audit_inventory(open_task(task_dir))

    assert inventory["cohort"]["status"] == (
        "exact_source_with_published_count_mismatch"
    )


def test_audit_inventory_accepts_exact_released_loader_boundary(tmp_path):
    task_dir = tmp_path / "mortality"
    _write_task(task_dir)
    batch_path = task_dir / "audit" / "full_batch_recreation.json"
    batch_path.write_text(
        json.dumps({"status": "exact_at_released_ehrdataset_input"}) + "\n"
    )

    inventory = _audit_inventory(open_task(task_dir))

    assert inventory["batch"]["status"] == "exact_at_released_ehrdataset_input"


def test_audit_inventory_accepts_exact_structured_source_boundary(tmp_path):
    task_dir = tmp_path / "mortality"
    _write_task(task_dir)
    batch_path = task_dir / "audit" / "full_batch_recreation.json"
    batch_path.write_text(
        json.dumps({"status": "exact_structured_source_boundary"}) + "\n"
    )

    inventory = _audit_inventory(open_task(task_dir))

    assert inventory["batch"]["status"] == "exact_structured_source_boundary"

    # The structured-source boundary is a batch-only allowance; for the cohort
    # it is not exact source, which the inventory records rather than raising.
    cohort_path = task_dir / "audit" / "cohort_reconstruction.json"
    cohort_path.write_text(
        json.dumps({"status": "exact_structured_source_boundary"}) + "\n"
    )
    assert _audit_inventory(open_task(task_dir))["cohort"]["exact_source"] == "false"


def test_audit_inventory_rejects_generic_qualified_status(tmp_path):
    task_dir = tmp_path / "mortality"
    _write_task(task_dir)
    cohort_path = task_dir / "audit" / "cohort_reconstruction.json"
    cohort_path.write_text(json.dumps({"status": "qualified"}) + "\n")

    inventory = _audit_inventory(open_task(task_dir))

    assert inventory["cohort"]["exact_source"] == "false"


def test_exact_author_partitions_may_overlap_groups():
    partitions = {
        "train": {"y": np.array([0, 1]), "group_id": np.array([1, 2])},
        "validation": {"y": np.array([0, 1]), "group_id": np.array([2, 3])},
        "test": {"y": np.array([0, 1]), "group_id": np.array([1, 4])},
    }
    _validate_partitions("source_exact", partitions)
    assert _cross_split_group_overlap(partitions) == {
        "train_validation": 1,
        "train_test": 1,
        "validation_test": 0,
    }


def test_single_task_end_to_end_smoke(tmp_path):
    task_dir = tmp_path / "mortality"
    _write_task(task_dir)
    output_dir = tmp_path / "output"

    result = run(
        task_dir,
        output_dir,
        trials=2,
        tuning_seed=0,
        threads=1,
        bootstrap_replicates=10,
    )

    assert result["task_id"] == "mortality"
    assert result["training"]["fit_split"] == "train"
    assert result["training"]["early_stopping_split"] == "validation"
    assert result["training"]["training_rows"] == 80
    assert result["training"]["model_seed"] == 0
    assert result["selection"]["trials"] == 2
    assert result["selection"]["selected_representation"] == "summary"
    assert result["selection"]["joint_search"] is False
    assert set(result["selection"]["by_representation"]) == {
        "summary",
        "visit_halves",
    }
    ensemble = result["selection"]["ensemble"]
    assert ensemble["weighting"] == "uniform_logit_average"
    assert 1 <= ensemble["size"] <= 5
    assert ensemble["members"][0]["trial"] == result["selection"]["selected_trial"]
    assert len(result["training"]["member_iterations"]) == ensemble["size"]
    assert (output_dir / "result.json").is_file()
    for rank in range(ensemble["size"]):
        assert (output_dir / f"model_member_{rank}.txt").is_file()
    assert set(result["restricted_local_files"]) == {
        "predictions.npz",
        *(f"model_member_{rank}.txt" for rank in range(ensemble["size"])),
    }
    with np.load(output_dir / "predictions.npz") as archive:
        assert archive["validation_probability"].shape == (40,)
        assert archive["validation_sample_id"].shape == (40,)
        assert archive["test_probability"].shape == (40,)


def test_note_stacker_reuses_saved_structured_predictions(tmp_path):
    task_dir = tmp_path / "mortality"
    _write_task(task_dir, with_notes=True)
    structured_dir = tmp_path / "structured"
    run(
        task_dir,
        structured_dir,
        trials=1,
        threads=1,
        bootstrap_replicates=10,
    )
    stacked_dir = tmp_path / "stacked"
    result = run_stacking(
        task_dir,
        structured_dir,
        stacked_dir,
        bootstrap_replicates=10,
    )

    assert result["stacker"]["test"]["auroc"] > 0.9
    assert result["structured_model"]["validation_prediction_source"] == "predictions.npz"
    assert json.loads((stacked_dir / "result.json").read_text()).keys() == {
        "paper",
        "stacked_auroc",
        "task_id",
    }
    with np.load(stacked_dir / "predictions.npz") as archive:
        assert "validation_stacked_probability" not in archive
