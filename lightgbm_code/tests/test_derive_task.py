import json

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from missing_xgboost_lightgbm.common_format import open_task
from missing_xgboost_lightgbm.derive_task import derive_task

from test_meds_format import _samples, _structured_events, _write


def test_derived_sibling_task_reuses_events_and_passes_the_audit_gate(tmp_path):
    base = tmp_path / "mortality_48h"
    starts, samples = _samples()
    _write(base, _structured_events(starts), samples)
    # Stand-in audits as a real export would carry them.
    (base / "audit" / "full_batch_recreation.json").write_text(
        json.dumps({"status": "exact", "samples": 6}) + "\n"
    )

    # New outcome on four of the six samples, flipped labels, same anchors.
    labels = pl.DataFrame(
        {
            "sample_id": [11, 12, 13, 15],
            "boolean_value": [True, False, True, False],
            "split": ["train", "train", "validation", "test"],
        }
    )
    # Every split needs both classes for the writer; add the missing ones.
    labels = pl.concat(
        [labels, pl.DataFrame({"sample_id": [14, 16], "boolean_value": [False, True],
                               "split": ["validation", "test"]})]
    )
    derived_dir = tmp_path / "readmission_30d"
    manifest = derive_task(
        base,
        derived_dir,
        task={
            "id": "readmission_30d",
            "name": "Readmission within 30 days",
            "published_auroc": 0.7,
            "positive_class": "readmitted",
        },
        labels=labels,
        derivation={"label_source": "test fixture column"},
    )
    assert manifest["task"]["id"] == "readmission_30d"
    assert manifest["task"]["observation_window_hours"] == 48.0
    assert manifest["source"]["derived_from"]["task_id"] == "mortality_48h"

    task = open_task(derived_dir)
    assert task.load("train")["sample_id"].tolist() == [11, 12]
    assert task.load("train")["y"].tolist() == [1, 0]
    events = pq.read_table(derived_dir / "data" / "events.parquet")
    assert sorted(set(events["sample_id"].to_pylist())) == [11, 12, 13, 14, 15, 16]
    base_events = pq.read_table(base / "data" / "events.parquet")
    assert events.num_rows == base_events.num_rows
    cohort = json.loads((derived_dir / "audit" / "cohort_reconstruction.json").read_text())
    assert cohort["status"].startswith("exact_source_with_")
    batch = json.loads((derived_dir / "audit" / "full_batch_recreation.json").read_text())
    assert batch["status"] == "exact" and "derived_from" in batch


def test_derivation_rejects_unknown_samples(tmp_path):
    base = tmp_path / "mortality_48h"
    starts, samples = _samples()
    _write(base, _structured_events(starts), samples)
    (base / "audit" / "full_batch_recreation.json").write_text('{"status": "exact"}\n')
    labels = pl.DataFrame(
        {"sample_id": [11, 12, 13, 14, 15, 999], "boolean_value": [True, False] * 3,
         "split": ["train", "train", "validation", "validation", "test", "test"]}
    )
    try:
        derive_task(base, tmp_path / "other", task={"id": "other", "name": "x",
                    "published_auroc": None, "positive_class": "x"},
                    labels=labels, derivation={})
    except ValueError as error:
        assert "not in the base task index" in str(error)
    else:
        raise AssertionError("unknown sample accepted")
