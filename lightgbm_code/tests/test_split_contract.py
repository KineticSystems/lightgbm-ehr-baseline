"""Featurizer parity and metadata-contract tests.

The Rust binary owns feature construction end to end; nothing row-shaped
crosses the process boundary in production. These tests therefore compare
whole matrices written by the binary's `--dump-config` test-support mode
against the literal Python oracle, and pin the metadata contract
(`SplitContract`) both executors share.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from missing_xgboost_lightgbm.reference_features import (
    build_reference_event_representations,
    cohort_halves_split_hours,
)
import missing_xgboost_lightgbm.split_contract as split_contract
from missing_xgboost_lightgbm.split_contract import (
    SplitContract,
    restore_original_order,
    to_sorted_sample_order,
)


_HOUR_US = 3_600_000_000
_CODES = ("numeric", "occurrence")
_OCCURRENCE_CODES = ("occurrence",)
_TASK_HOURS = 4.0
_TRAIN_WINDOW_HOURS = (4.0, 2.0, 3.0, 4.0)
_HALVES_SPLIT_HOURS = cohort_halves_split_hours(_TRAIN_WINDOW_HOURS)


def _timestamps(values: np.ndarray) -> pa.Array:
    return pa.array(values, type=pa.timestamp("us"))


@pytest.fixture
def synthetic_task(tmp_path: Path) -> dict[str, object]:
    starts = {
        10: 1_000_000_000_000,
        20: 2_000_000_000_000,
        30: 3_000_000_000_000,
        40: 4_000_000_000_000,
        100: 5_000_000_000_000,
    }
    widths = {10: 4.0, 20: 2.0, 30: 3.0, 40: 4.0, 100: 4.0}

    # Deliberately retain a non-sample-ID physical order. The Rust featurizer
    # must sort the selected split, while the bridge remembers how to undo it.
    index_ids = np.asarray([40, 20, 100, 10, 30], dtype=np.int64)
    index_path = tmp_path / "task_index.parquet"
    pq.write_table(
        pa.table(
            {
                "subject_id": pa.array(index_ids, type=pa.int64()),
                "sample_id": pa.array(index_ids, type=pa.int64()),
                "observation_start": _timestamps(
                    np.asarray([starts[int(value)] for value in index_ids])
                ),
                "prediction_time": _timestamps(
                    np.asarray(
                        [
                            starts[int(value)]
                            + int(widths[int(value)] * _HOUR_US)
                            for value in index_ids
                        ]
                    )
                ),
                "boolean_value": pa.array(
                    [False, True, True, True, False], type=pa.bool_()
                ),
                "split": pa.array(
                    ["train", "train", "test", "train", "train"],
                    type=pa.string(),
                ),
            }
        ),
        index_path,
    )

    # Three physical row groups make split pruning observable. The middle
    # group is test-only; sample 30 intentionally has no event rows.
    event_ids = np.asarray([10, 10, 10, 20, 20, 100, 40, 40, 40], dtype=np.int64)
    relative_hours = np.asarray(
        [0.0, 1.0, 4.0, 0.5, 1.75, 1.0, 2.0, 3.0, 3.5], dtype=np.float64
    )
    event_codes = np.asarray(
        [
            "numeric",
            "occurrence",
            "numeric",
            "numeric",
            "occurrence",
            "numeric",
            "numeric",
            "occurrence",
            "NOTE",
        ]
    )
    event_values = np.asarray(
        [1.0, np.nan, 5.0, 2.0, np.nan, 99.0, 4.0, np.nan, np.nan],
        dtype=np.float32,
    )
    event_splits = np.asarray(
        [
            "train",
            "train",
            "train",
            "train",
            "train",
            "test",
            "train",
            "train",
            "train",
        ]
    )
    event_times = np.asarray(
        [
            starts[int(sample)] + int(hour * _HOUR_US)
            for sample, hour in zip(event_ids, relative_hours, strict=True)
        ],
        dtype=np.int64,
    )
    event_table = pa.table(
        {
            "subject_id": pa.array(event_ids, type=pa.int64()),
            "time": _timestamps(event_times),
            "code": pa.array(event_codes, type=pa.string()),
            "numeric_value": pa.array(
                [None if np.isnan(value) else value for value in event_values],
                type=pa.float32(),
            ),
            "sample_id": pa.array(event_ids, type=pa.int64()),
            "source_row": pa.array(np.arange(len(event_ids)), type=pa.int64()),
            "split": pa.array(event_splits, type=pa.string()),
        }
    )
    events_path = tmp_path / "events.parquet"
    with pq.ParquetWriter(events_path, event_table.schema) as writer:
        writer.write_table(event_table.slice(0, 5), row_group_size=5)
        writer.write_table(event_table.slice(5, 1), row_group_size=1)
        writer.write_table(event_table.slice(6, 3), row_group_size=3)

    selected = (event_splits == "train") & np.isin(event_codes, _CODES)
    sorted_ids = np.asarray([10, 20, 30, 40], dtype=np.int64)
    reference = build_reference_event_representations(
        {
            "sample_id": event_ids[selected],
            "time_hours": relative_hours[selected],
            "code": event_codes[selected],
            "value": event_values[selected],
            "source_row": np.arange(len(event_ids), dtype=np.int64)[selected],
        },
        sorted_ids,
        _CODES,
        _TASK_HOURS,
        occurrence_codes=_OCCURRENCE_CODES,
        sample_window_hours=np.asarray(_TRAIN_WINDOW_HOURS),
    )
    return {
        "events_path": events_path,
        "index_path": index_path,
        "reference": reference,
    }


def _contract(
    synthetic_task: dict[str, object],
    representation: str = "summary",
    timed: bool = True,
) -> SplitContract:
    return SplitContract(
        events_path=synthetic_task["events_path"],
        index_path=synthetic_task["index_path"],
        split="train",
        representation=representation,
        codes=_CODES,
        occurrence_codes=_OCCURRENCE_CODES,
        ignored_codes=("NOTE",),
        observation_window_hours=_TASK_HOURS,
        halves_split_hours=_HALVES_SPLIT_HOURS,
        timed=timed,
        max_component_rows=16,
        max_chunk_bytes=1_024,
        record_batch_rows=2,
    )


def _run_dump(
    contract: SplitContract,
    tmp_path: Path,
    tag: str,
    requested_rows: list[int] | None = None,
    mutate_stream_config=None,
) -> subprocess.CompletedProcess[bytes]:
    """Invoke the binary's --dump-config mode on the contract's exact config."""
    work_dir = tmp_path / f"dump_{tag}"
    work_dir.mkdir()
    stream_config = contract._config()
    if mutate_stream_config is not None:
        mutate_stream_config(stream_config, work_dir)
    config: dict[str, object] = {
        "stream": stream_config,
        "output_path": str(work_dir / "matrix.f32"),
    }
    if requested_rows is not None:
        config["requested_rows"] = requested_rows
    config_path = work_dir / "dump_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return subprocess.run(
        [str(split_contract._binary()), "--dump-config", str(config_path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )


def _dump_matrix(
    contract: SplitContract,
    tmp_path: Path,
    tag: str,
    requested_rows: list[int] | None = None,
    mutate_stream_config=None,
) -> tuple[np.ndarray, dict[str, object]]:
    completed = _run_dump(
        contract, tmp_path, tag, requested_rows, mutate_stream_config
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    report = json.loads(completed.stdout)
    matrix = np.fromfile(
        tmp_path / f"dump_{tag}" / "matrix.f32", dtype="<f4"
    ).reshape(report["rows"], report["feature_count"])
    return matrix, report


@pytest.mark.parametrize("representation", ["summary", "visit_halves"])
def test_full_and_sampled_dumps_match_reference(
    synthetic_task: dict[str, object], representation: str, tmp_path: Path
) -> None:
    contract = _contract(synthetic_task, representation)
    expected = synthetic_task["reference"][representation]
    assert contract.feature_names == expected.names
    assert contract.feature_count == len(expected.names)
    assert contract.sample_ids.tolist() == [10, 20, 30, 40]
    np.testing.assert_array_equal(
        contract.labels, np.asarray([1, 1, 0, 0], np.float32)
    )

    actual, report = _dump_matrix(contract, tmp_path, "full")
    assert report["feature_names"] == list(contract.feature_names)
    assert (report["rows"], report["feature_count"]) == (4, contract.feature_count)
    np.testing.assert_allclose(
        actual, expected.values, rtol=1e-6, atol=1e-6, equal_nan=True
    )
    assert not np.any(np.isinf(actual))

    sampled, sampled_report = _dump_matrix(
        contract, tmp_path, "sampled", requested_rows=[0, 2]
    )
    assert sampled_report["rows"] == 2
    np.testing.assert_allclose(
        sampled, expected.values[[0, 2]], rtol=1e-6, atol=1e-6, equal_nan=True
    )

    # A replay must reproduce the matrix byte-identically.
    _, replay_report = _dump_matrix(contract, tmp_path, "replay")
    assert replay_report["values_sha256"] == report["values_sha256"]


def test_requested_rows_are_strict(
    synthetic_task: dict[str, object], tmp_path: Path
) -> None:
    contract = _contract(synthetic_task)
    decreasing = _run_dump(contract, tmp_path, "decreasing", requested_rows=[2, 1])
    assert decreasing.returncode != 0
    assert b"strictly increasing" in decreasing.stderr
    outside = _run_dump(contract, tmp_path, "outside", requested_rows=[4])
    assert outside.returncode != 0
    assert b"outside the split index" in outside.stderr


def test_rust_rejects_unknown_configuration_field(
    synthetic_task: dict[str, object], tmp_path: Path
) -> None:
    contract = _contract(synthetic_task)

    def add_unknown_field(config: dict[str, object], work_dir: Path) -> None:
        config["paper_specific_override"] = True

    completed = _run_dump(
        contract, tmp_path, "unknown", mutate_stream_config=add_unknown_field
    )
    assert completed.returncode != 0
    assert b"unknown field" in completed.stderr


def test_invalid_events_fail_with_nonzero_exit(
    synthetic_task: dict[str, object], tmp_path: Path
) -> None:
    # Treating the occurrence code as numeric is structurally valid JSON, but
    # makes its native null value invalid while Rust reads selected events.
    contract = SplitContract(
        events_path=synthetic_task["events_path"],
        index_path=synthetic_task["index_path"],
        split="train",
        representation="summary",
        codes=_CODES,
        occurrence_codes=(),
        observation_window_hours=_TASK_HOURS,
        max_component_rows=16,
        max_chunk_bytes=1_024,
        record_batch_rows=2,
    )
    completed = _run_dump(contract, tmp_path, "invalid")
    assert completed.returncode != 0
    assert b"numeric structured events must have numeric_value" in completed.stderr


def test_prediction_order_round_trip(synthetic_task: dict[str, object]) -> None:
    contract = _contract(synthetic_task)
    np.testing.assert_array_equal(contract.sorted_to_original, [2, 1, 3, 0])
    sorted_predictions = np.asarray(
        [[10.0, 0.1], [20.0, 0.2], [30.0, 0.3], [40.0, 0.4]]
    )
    expected_original = np.asarray(
        [[40.0, 0.4], [20.0, 0.2], [10.0, 0.1], [30.0, 0.3]]
    )
    restored = restore_original_order(
        sorted_predictions, contract.sorted_to_original
    )
    np.testing.assert_array_equal(restored, expected_original)
    np.testing.assert_array_equal(
        to_sorted_sample_order(restored, contract.sorted_to_original),
        sorted_predictions,
    )
    np.testing.assert_array_equal(
        contract.predictions_to_sorted_order(
            contract.restore_prediction_order(sorted_predictions)
        ),
        sorted_predictions,
    )


def test_untimed_features_withhold_elapsed_time(
    synthetic_task: dict[str, object], tmp_path: Path
) -> None:
    """The untimed layout must drop exactly what an untimed task may not see."""
    timed = _contract(synthetic_task, "visit_buckets")
    untimed = _contract(synthetic_task, "visit_buckets", timed=False)
    assert "__sample__window_hours" in timed.feature_names
    assert "__sample__window_hours" not in untimed.feature_names
    assert not any(
        name.endswith(("__recency", "__first_recency"))
        for name in untimed.feature_names
    )
    assert untimed.feature_count == len(untimed.feature_names)
    kept = [
        position
        for position, name in enumerate(timed.feature_names)
        if name in set(untimed.feature_names)
    ]
    assert [timed.feature_names[position] for position in kept] == list(
        untimed.feature_names
    )
    full, full_report = _dump_matrix(timed, tmp_path, "timed")
    projected, projected_report = _dump_matrix(untimed, tmp_path, "untimed")
    # The binary must agree with the contract about each layout's columns.
    assert full_report["feature_names"] == list(timed.feature_names)
    assert projected_report["feature_names"] == list(untimed.feature_names)
    np.testing.assert_array_equal(
        np.isnan(full[:, kept]), np.isnan(projected)
    )
    np.testing.assert_allclose(
        full[:, kept], projected, rtol=1e-6, atol=1e-6, equal_nan=True
    )
