import json
import hashlib
import subprocess

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from missing_xgboost_lightgbm.reference_features import (
    OCCURRENCE_SUMMARY_NAMES,
    SUMMARY_NAMES,
)
from missing_xgboost_lightgbm.common_format import open_task
from missing_xgboost_lightgbm.polars_meds import write_meds_task
from missing_xgboost_lightgbm.reference_features import (
    build_reference_event_representations,
)
from missing_xgboost_lightgbm.reference_features import (
    TIMED_AGGREGATE_NAMES,
    cohort_halves_split_hours,
)
from missing_xgboost_lightgbm.rust_features import build_rust_event_representations


def _task(hours=48.0):
    return {
        "id": "mortality_48h",
        "name": "Mortality 48 hours",
        "primary_metric": "AUROC",
        "published_auroc": None,
        "observation_window_hours": hours,
    }


def _samples(subjects=(1, 2, 3, 4, 5, 6)):
    starts = np.asarray(["2100-01-01T00:00:00"] * 6, dtype="datetime64[us]")
    return starts, pl.from_arrow(pa.table(
        {
            "subject_id": pa.array(subjects, type=pa.int64()),
            "sample_id": pa.array([11, 12, 13, 14, 15, 16], type=pa.int64()),
            "observation_start": pa.array(starts, type=pa.timestamp("us")),
            "prediction_time": pa.array(
                starts + np.timedelta64(48, "h"), type=pa.timestamp("us")
            ),
            "boolean_value": pa.array([False, True] * 3, type=pa.bool_()),
            "split": pa.array(
                ["train", "train", "validation", "validation", "test", "test"]
            ),
        }
    ))


def _structured_events(starts, subjects=(1, 2, 3, 4, 5, 6)):
    return pl.from_arrow(pa.table(
        {
            "subject_id": pa.array(subjects, type=pa.int64()),
            "time": pa.array(
                starts + np.timedelta64(1, "h"), type=pa.timestamp("us")
            ),
            "code": pa.array(["heart_rate"] * 6),
            "numeric_value": pa.array(np.arange(6), type=pa.float32()),
            "sample_id": pa.array([11, 12, 13, 14, 15, 16], type=pa.int64()),
            "source_row": pa.array(np.arange(6), type=pa.int64()),
        }
    ))


def _write(root, events, samples, *, codes=("heart_rate",), note_codes=()):
    return write_meds_task(
        root,
        paper={"key": "example", "title": "Example"},
        task=_task(),
        codes=codes,
        note_codes=note_codes,
        events=events,
        samples=samples,
        source={"loader_entrypoint": "test"},
        dataset_name="test",
        dataset_version="1",
    )


def test_event_summary_uses_observed_stream_and_native_missingness():
    representations = build_reference_event_representations(
        {
            "sample_id": np.array([10, 10, 10]),
            "time_hours": np.array([0.5, 2.5, 47.0]),
            "code": np.array(["a", "a", "b"]),
            "value": np.array([1.0, 3.0, 7.0]),
            "source_row": np.arange(3),
        },
        np.array([10]),
        ("a", "b", "never_observed"),
        48.0,
    )

    names = representations["summary"].names
    row = representations["summary"].values[0]
    assert row[names.index("a__full__count")] == 2
    assert row[names.index("a__full__mean")] == 2
    assert row[names.index("a__full__delta")] == 2
    assert row[names.index("b__full__recency")] == 1
    assert row[names.index("never_observed__full__count")] == 0
    assert np.isnan(row[names.index("never_observed__full__mean")])
    # One cohort sample of width 48 h puts the visit_halves cut at 24 h, so
    # `a` (0.5 h and 2.5 h) is entirely earlier and `b` (47 h) entirely recent.
    halves = representations["visit_halves"]
    assert halves.values.shape == (
        1,
        len(TIMED_AGGREGATE_NAMES) + 3 * len(SUMMARY_NAMES) * 3,
    )
    row = halves.values[0]
    assert row[halves.names.index("a__earlier__count")] == 2
    assert row[halves.names.index("a__recent__count")] == 0
    assert row[halves.names.index("b__earlier__count")] == 0
    assert row[halves.names.index("b__recent__count")] == 1
    assert row[halves.names.index("b__recent__recency")] == 1


def test_source_row_preserves_multiplicity_and_resolves_time_ties():
    representations = build_reference_event_representations(
        {
            "sample_id": np.array([10, 10]),
            "time_hours": np.array([1.0, 1.0]),
            "code": np.array(["a", "a"]),
            "value": np.array([9.0, 1.0]),
            "source_row": np.array([2, 1]),
        },
        np.array([10]),
        ("a",),
        48.0,
    )
    summary = representations["summary"]
    halves = representations["visit_halves"]
    assert summary.values[0, summary.names.index("a__full__count")] == 2
    assert summary.values[0, summary.names.index("a__full__latest_observed")] == 9
    # Both events sit at hour 1 of a 48 h window, so the 24 h cut puts them in
    # `earlier`, where the identical explicit tie resolves to stream order.
    assert halves.values[0, halves.names.index("a__earlier__count")] == 2
    assert halves.values[0, halves.names.index("a__earlier__latest_observed")] == 9
    assert halves.values[0, halves.names.index("a__recent__count")] == 0


def test_rust_timeline_matches_reference_event_featurization(tmp_path):
    root = tmp_path / "mortality_48h"
    starts, samples = _samples()
    event_sample = np.array([11, 11, 11, 11, 11, 12, 12, 13, 14, 15, 16])
    event_subject = np.array([1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6])
    event_hours = np.array([0.5, 1.0, 1.0, 2.5, 47.0, 3.0, 40.0, 1, 1, 1, 1])
    event_codes = np.array(["a", "a", "a", "a", "b", "a", "b", "a", "a", "a", "a"])
    event_values = np.array([2, 9, 1, 3, 7, 4, 8, 5, 6, 10, 11], dtype=np.float32)
    source_rows = np.array([0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10])
    events = pl.DataFrame(
        {
            "subject_id": pl.Series(event_subject, dtype=pl.Int64),
            "time": pl.Series(
                starts[event_sample - 11]
                + (event_hours * 3_600_000_000).astype("timedelta64[us]"),
                dtype=pl.Datetime("us"),
            ),
            "code": pl.Series(event_codes, dtype=pl.String),
            "numeric_value": pl.Series(event_values, dtype=pl.Float32),
            "sample_id": pl.Series(event_sample, dtype=pl.Int64),
            "source_row": pl.Series(source_rows, dtype=pl.Int64),
        }
    )
    _write(root, events, samples, codes=("a", "b"))

    actual = open_task(root).build_representations("train")
    selected = event_sample <= 12
    reference = build_reference_event_representations(
        {
            "sample_id": event_sample[selected],
            "time_hours": event_hours[selected],
            "code": event_codes[selected],
            "value": event_values[selected],
            "source_row": source_rows[selected],
        },
        np.array([11, 12]),
        ("a", "b"),
        48.0,
    )
    for representation in ("summary", "visit_halves"):
        assert actual[representation].names == reference[representation].names
        np.testing.assert_allclose(
            actual[representation].values,
            reference[representation].values,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )


def test_rust_randomized_mixed_stream_matches_reference_and_threads(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(20260829)
    hours = 13.5
    all_samples = np.arange(101, 107, dtype=np.int64)
    selected_samples = all_samples[:4]
    starts_us = (
        np.datetime64("2100-01-01T00:00:00", "us").astype(np.int64)
        + np.arange(len(all_samples), dtype=np.int64) * 86_400_000_000
    )
    event_count = 600
    event_samples = rng.choice(all_samples, size=event_count)
    event_codes = rng.choice(
        np.asarray(["numeric_a", "numeric_b", "occurrence_x", "NOTE"]),
        size=event_count,
        p=[0.35, 0.30, 0.25, 0.10],
    )
    random_us = rng.integers(
        0, int(hours * 3_600_000_000) + 1, size=event_count, dtype=np.int64
    )
    boundary_us = np.asarray(
        [
            0,
            int(0.10 * hours * 3_600_000_000),
            int(0.25 * hours * 3_600_000_000),
            int(0.50 * hours * 3_600_000_000),
            int(0.75 * hours * 3_600_000_000),
            int(0.90 * hours * 3_600_000_000),
            int(hours * 3_600_000_000),
        ],
        dtype=np.int64,
    )
    random_us[: len(boundary_us)] = boundary_us
    event_times_us = starts_us[event_samples - all_samples[0]] + random_us
    source_rows = rng.integers(0, 80, size=event_count, dtype=np.int64)
    numeric_values = rng.normal(size=event_count).astype(np.float32)
    numeric_values[np.isin(event_codes, ["occurrence_x", "NOTE"])] = np.nan

    # Exercise exact duplicate tie-breakers. Both the NumPy reference and Rust
    # path must preserve their Parquet stream order for the last-value grid.
    event_samples[:2] = selected_samples[0]
    event_codes[:2] = "numeric_a"
    event_times_us[:2] = starts_us[0] + 3_600_000_000
    source_rows[:2] = 7
    numeric_values[:2] = [2.0, 9.0]

    events_path = tmp_path / "events.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": pa.array(event_samples, type=pa.int64()),
                "time": pa.array(event_times_us, type=pa.timestamp("us")),
                "code": pa.array(event_codes, type=pa.string()),
                "numeric_value": pa.array(
                    [None if np.isnan(value) else value for value in numeric_values],
                    type=pa.float32(),
                ),
                "source_row": pa.array(source_rows, type=pa.int64()),
            }
        ),
        events_path,
    )
    index = pa.table(
        {
            "sample_id": pa.array(selected_samples, type=pa.int64()),
            "observation_start": pa.array(
                starts_us[: len(selected_samples)], type=pa.timestamp("us")
            ),
            "prediction_time": pa.array(
                starts_us[: len(selected_samples)]
                + int(hours * 3_600_000_000),
                type=pa.timestamp("us"),
            ),
        }
    )
    codes = ("numeric_a", "numeric_b", "occurrence_x", "never_observed")
    occurrence_codes = ("occurrence_x",)

    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    one_thread = build_rust_event_representations(
        events_path,
        index,
        codes,
        hours,
        occurrence_codes=occurrence_codes,
        halves_split_hours=0.5 * hours,
    )
    monkeypatch.setenv("RAYON_NUM_THREADS", "4")
    four_threads = build_rust_event_representations(
        events_path,
        index,
        codes,
        hours,
        occurrence_codes=occurrence_codes,
        halves_split_hours=0.5 * hours,
    )

    selected = np.isin(event_samples, selected_samples) & np.isin(
        event_codes, codes
    )
    reference = build_reference_event_representations(
        {
            "sample_id": event_samples[selected],
            "time_hours": (
                event_times_us[selected]
                - starts_us[event_samples[selected] - all_samples[0]]
            )
            / 3_600_000_000.0,
            "code": event_codes[selected],
            "value": numeric_values[selected],
            "source_row": source_rows[selected],
        },
        selected_samples,
        codes,
        hours,
        occurrence_codes=occurrence_codes,
    )
    for representation in ("summary", "visit_halves"):
        assert one_thread[representation].names == reference[representation].names
        assert four_threads[representation].names == reference[representation].names
        expected = reference[representation].values
        if hasattr(expected, "toarray"):
            expected = np.asarray(expected)
        for actual in (one_thread[representation].values, four_threads[representation].values):
            if hasattr(actual, "toarray"):
                actual = np.asarray(actual)
            np.testing.assert_allclose(
                actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True
            )


def test_variable_sample_windows_match_reference(tmp_path):
    task_hours = 240.0
    sample_ids = np.asarray([101, 102], dtype=np.int64)
    starts_us = np.asarray([0, 1_000_000_000_000], dtype=np.int64)
    sample_hours = np.asarray([12.0, 240.0])
    event_samples = np.repeat(sample_ids, 4)
    event_hours = np.tile(np.asarray([1.0, 6.0, 12.0, 6.0]), 2)
    event_codes = np.tile(
        np.asarray(["numeric", "numeric", "numeric", "occurrence"]), 2
    )
    event_values = np.tile(np.asarray([1.0, 2.0, 3.0, np.nan], np.float32), 2)
    source_rows = np.arange(len(event_samples), dtype=np.int64)
    event_starts = starts_us[event_samples - sample_ids[0]]
    events_path = tmp_path / "variable_windows.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": pa.array(event_samples, type=pa.int64()),
                "time": pa.array(
                    event_starts
                    + (event_hours * 3_600_000_000).astype(np.int64),
                    type=pa.timestamp("us"),
                ),
                "code": pa.array(event_codes, type=pa.string()),
                "numeric_value": pa.array(
                    [None if np.isnan(value) else value for value in event_values],
                    type=pa.float32(),
                ),
                "source_row": pa.array(source_rows, type=pa.int64()),
            }
        ),
        events_path,
    )
    index = pa.table(
        {
            "sample_id": pa.array(sample_ids, type=pa.int64()),
            "observation_start": pa.array(starts_us, type=pa.timestamp("us")),
            "prediction_time": pa.array(
                starts_us
                + (sample_hours * 3_600_000_000).astype(np.int64),
                type=pa.timestamp("us"),
            ),
        }
    )
    codes = ("numeric", "occurrence")
    actual = build_rust_event_representations(
        events_path,
        index,
        codes,
        task_hours,
        occurrence_codes=("occurrence",),
        halves_split_hours=cohort_halves_split_hours(sample_hours),
    )
    reference = build_reference_event_representations(
        {
            "sample_id": event_samples,
            "time_hours": event_hours,
            "code": event_codes,
            "value": event_values,
            "source_row": source_rows,
        },
        sample_ids,
        codes,
        task_hours,
        occurrence_codes=("occurrence",),
        sample_window_hours=sample_hours,
    )
    for representation in ("summary", "visit_halves"):
        assert actual[representation].names == reference[representation].names
        observed = np.asarray(actual[representation].values)
        expected = reference[representation].values
        np.testing.assert_array_equal(np.isnan(observed), np.isnan(expected))
        finite = np.isfinite(expected)
        np.testing.assert_array_max_ulp(
            observed[finite], expected[finite], maxulp=4
        )

    summary = actual["summary"]
    summary_values = np.asarray(summary.values)
    assert summary_values[0, summary.names.index("numeric__full__recency")] == 0
    assert summary_values[1, summary.names.index("numeric__full__recency")] == 228
    # Widths are 12 h and 240 h, so the cohort median is 126 h and the shared
    # cut lands 63 h before prediction time: the 12 h sample is entirely recent
    # and the 240 h sample, whose events all sit before hour 12, entirely
    # earlier. That is the whole point of a dataset-wide split.
    halves = actual["visit_halves"]
    halves_values = np.asarray(halves.values)
    assert halves_values[0, halves.names.index("numeric__recent__count")] == 3
    assert halves_values[0, halves.names.index("numeric__earlier__count")] == 0
    assert halves_values[1, halves.names.index("numeric__recent__count")] == 0
    assert halves_values[1, halves.names.index("numeric__earlier__count")] == 3


def test_short_and_tiny_sample_windows_match_reference(tmp_path):
    task_hours = 48.0
    sample_ids = np.asarray([201, 202], dtype=np.int64)
    starts_us = np.asarray([0, 1_000_000_000_000], dtype=np.int64)
    widths_us = np.asarray([12 * 3_600_000_000, 1], dtype=np.int64)
    event_offsets_us = np.asarray([6 * 3_600_000_000, 1], dtype=np.int64)
    event_hours = event_offsets_us / 3_600_000_000.0
    events_path = tmp_path / "short_windows.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": pa.array(sample_ids, type=pa.int64()),
                "time": pa.array(
                    starts_us + event_offsets_us, type=pa.timestamp("us")
                ),
                "code": pa.array(["numeric", "numeric"], type=pa.string()),
                "numeric_value": pa.array([1.0, 2.0], type=pa.float32()),
                "source_row": pa.array([0, 1], type=pa.int64()),
            }
        ),
        events_path,
    )
    index = pa.table(
        {
            "sample_id": pa.array(sample_ids, type=pa.int64()),
            "observation_start": pa.array(starts_us, type=pa.timestamp("us")),
            "prediction_time": pa.array(
                starts_us + widths_us, type=pa.timestamp("us")
            ),
        }
    )
    actual = build_rust_event_representations(
        events_path,
        index,
        ("numeric",),
        task_hours,
        halves_split_hours=cohort_halves_split_hours(widths_us / 3_600_000_000.0),
    )
    reference = build_reference_event_representations(
        {
            "sample_id": sample_ids,
            "time_hours": event_hours,
            "code": np.asarray(["numeric", "numeric"]),
            "value": np.asarray([1.0, 2.0], dtype=np.float32),
            "source_row": np.asarray([0, 1], dtype=np.int64),
        },
        sample_ids,
        ("numeric",),
        task_hours,
        sample_window_hours=widths_us / 3_600_000_000.0,
    )
    for representation in ("summary", "visit_halves"):
        np.testing.assert_allclose(
            actual[representation].values,
            reference[representation].values,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )

    summary = actual["summary"]
    assert summary.values[0, summary.names.index("numeric__full__recency")] == 6
    assert summary.values[1, summary.names.index("numeric__full__recency")] == 0
    # Widths are 12 h and one microsecond, so the cohort median is ~6 h and the
    # cut is ~3 h back. The 12 h sample's only event, at hour 6, is earlier;
    # the microsecond sample has no earlier span at all.
    halves = actual["visit_halves"]
    assert halves.values[0, halves.names.index("numeric__earlier__mean")] == 1
    assert halves.values[0, halves.names.index("numeric__recent__count")] == 0
    assert halves.values[1, halves.names.index("numeric__recent__mean")] == 2
    assert halves.values[1, halves.names.index("numeric__earlier__count")] == 0


@pytest.mark.parametrize(
    ("time_hours", "value"),
    [(49.0, 1.0), (1.0, np.inf), (1.0, None)],
)
def test_rust_binary_rejects_invalid_selected_numeric_events(
    tmp_path, time_hours, value
):
    start = np.datetime64("2100-01-01T00:00:00", "us")
    events_path = tmp_path / "invalid.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": pa.array([11], type=pa.int64()),
                "time": pa.array(
                    [
                        start
                        + np.timedelta64(int(time_hours * 3_600_000_000), "us")
                    ],
                    type=pa.timestamp("us"),
                ),
                "code": pa.array(["heart_rate"], type=pa.string()),
                "numeric_value": pa.array([value], type=pa.float32()),
                "source_row": pa.array([0], type=pa.int64()),
            }
        ),
        events_path,
    )
    index = pa.table(
        {
            "sample_id": pa.array([11], type=pa.int64()),
            "observation_start": pa.array([start], type=pa.timestamp("us")),
            "prediction_time": pa.array(
                [start + np.timedelta64(48, "h")], type=pa.timestamp("us")
            ),
        }
    )
    with pytest.raises(subprocess.CalledProcessError):
        build_rust_event_representations(
            events_path,
            index,
            ("heart_rate",),
            48.0,
            halves_split_hours=24.0,
        )


def test_write_and_open_structured_only_meds_task(tmp_path):
    root = tmp_path / "mortality_48h"
    starts, samples = _samples()
    _write(root, _structured_events(starts).reverse(), samples)

    task = open_task(root)
    assert task.modalities == ("structured",)
    assert task.note_codes == ()
    assert task.load("validation")["sample_id"].tolist() == [13, 14]
    assert task.build_representations("train")["summary"].values.shape == (
        2,
        len(TIMED_AGGREGATE_NAMES) + len(SUMMARY_NAMES),
    )
    assert pq.read_table(root / "data" / "events.parquet")[
        "sample_id"
    ].to_pylist() == [11, 12, 13, 14, 15, 16]
    manifest = json.loads((root / "manifest.json").read_text())
    assert "input" not in manifest["task"]
    assert json.loads((root / "metadata" / "dataset.json").read_text())[
        "meds_version"
    ] == "0.4"


def test_exact_sample_split_preserves_cross_split_subject_overlap(tmp_path):
    root = tmp_path / "mortality_48h"
    subjects = (1, 2, 3, 4, 5, 1)
    starts, samples = _samples(subjects)
    manifest = _write(root, _structured_events(starts, subjects), samples)

    assert manifest["split_assignment_unit"] == "sample_id"
    assert manifest["cross_split_subject_overlap"] == {
        "train_validation": 0,
        "train_test": 1,
        "validation_test": 0,
        "subjects_in_multiple_splits": 1,
    }
    assert open_task(root).load("test")["group_id"].tolist() == [1, 5]
    subject_splits = pq.read_table(root / "metadata" / "subject_splits.parquet")
    subject_one = subject_splits.filter(pc.equal(subject_splits["subject_id"], 1))
    assert sorted(subject_one["split"].to_pylist()) == ["test", "train"]


def test_note_codes_are_the_only_nonstructured_codes(tmp_path):
    root = tmp_path / "mortality_48h"
    starts, samples = _samples()
    subject = [1, 2, 3, 4, 5, 6] * 2
    sample = [11, 12, 13, 14, 15, 16] * 2
    events = pl.from_arrow(pa.table(
        {
            "subject_id": pa.array(subject, type=pa.int64()),
            "time": pa.array(
                np.concatenate(
                    [
                        starts + np.timedelta64(1, "h"),
                        starts + np.timedelta64(3, "h"),
                    ]
                ),
                type=pa.timestamp("us"),
            ),
            "code": pa.array(["heart_rate"] * 6 + ["NOTE"] * 6),
            "numeric_value": pa.array(
                list(range(6)) + [None] * 6, type=pa.float32()
            ),
            "sample_id": pa.array(sample, type=pa.int64()),
            "source_row": pa.array(np.arange(12), type=pa.int64()),
            "sequence_row": pa.array([0] * 12, type=pa.int32()),
            "text_value": pa.array(
                [None] * 6 + [f"note {index}" for index in range(1, 7)],
                type=pa.string(),
            ),
        }
    ))
    _write(
        root,
        events,
        samples,
        codes=("heart_rate", "NOTE"),
        note_codes=("NOTE",),
    )

    task = open_task(root)
    assert task.modalities == ("structured", "notes")
    representation = task.build_representations("train")["summary"]
    assert representation.values.shape == (
        2,
        len(TIMED_AGGREGATE_NAMES) + len(SUMMARY_NAMES),
    )
    assert representation.values[
        0, representation.names.index("heart_rate__full__mean")
    ] == 0
    assert task.load_documents("train") == ["note 1", "note 2"]
    dataset_metadata = json.loads((root / "metadata" / "dataset.json").read_text())
    assert set(dataset_metadata) == {"dataset_name", "dataset_version", "meds_version"}


def test_note_event_cannot_carry_a_numeric_value(tmp_path):
    root = tmp_path / "mortality_48h"
    starts, samples = _samples()
    structured = _structured_events(starts)
    note = pl.from_arrow(pa.table(
        {
            "subject_id": pa.array([1], type=pa.int64()),
            "time": pa.array(starts[:1] + np.timedelta64(2, "h"), type=pa.timestamp("us")),
            "code": pa.array(["NOTE"]),
            "numeric_value": pa.array([999.0], type=pa.float32()),
            "sample_id": pa.array([11], type=pa.int64()),
            "source_row": pa.array([6], type=pa.int64()),
            "text_value": pa.array(["note"], type=pa.string()),
        }
    ))
    structured = structured.with_columns(pl.lit(None, dtype=pl.String).alias("text_value"))
    with pytest.raises(ValueError, match="note events require text and no numeric"):
        _write(
            root,
            pl.concat([structured, note], how="diagonal"),
            samples,
            codes=("heart_rate", "NOTE"),
            note_codes=("NOTE",),
        )


def test_occurrence_only_structured_codes_use_sparse_features(tmp_path):
    root = tmp_path / "mortality_48h"
    starts, samples = _samples()
    events = _structured_events(starts).with_columns(
        pl.lit(None, dtype=pl.Float32).alias("numeric_value"),
        pl.lit("diagnosis//I10").alias("code"),
    )
    _write(root, events, samples, codes=("diagnosis//I10",))

    task = open_task(root)
    representations = task.build_representations("train")
    assert task.occurrence_codes == ("diagnosis//I10",)
    assert representations["summary"].values.shape == (
        2,
        len(TIMED_AGGREGATE_NAMES) + len(OCCURRENCE_SUMMARY_NAMES),
    )
    summary = representations["summary"]
    assert (
        np.asarray(summary.values)[
            0, summary.names.index("diagnosis//I10__full__count")
        ]
        == 1
    )
    assert (
        "diagnosis//I10__earlier__count" in representations["visit_halves"].names
    )


def test_unobserved_vocabulary_codes_do_not_create_dense_features(tmp_path):
    root = tmp_path / "mortality_48h"
    starts, samples = _samples()
    events = _structured_events(starts).with_columns(
        pl.lit(None, dtype=pl.Float32).alias("numeric_value"),
        pl.lit("diagnosis//I10").alias("code"),
    )
    _write(
        root,
        events,
        samples,
        codes=("diagnosis//I10", "diagnosis//NEVER_OBSERVED"),
    )

    task = open_task(root)
    representations = task.build_representations("train")

    assert task._codes == ("diagnosis//I10", "diagnosis//NEVER_OBSERVED")
    assert task.occurrence_codes == ("diagnosis//I10",)
    assert task._structured_codes() == ("diagnosis//I10",)
    # The featurizer returns a dense float32 matrix; occurrence codes still
    # contribute only their three occurrence statistics.
    assert isinstance(representations["summary"].values, np.ndarray)
    assert representations["summary"].values.dtype == np.float32
    assert representations["summary"].values.shape == (
        2,
        len(TIMED_AGGREGATE_NAMES) + len(OCCURRENCE_SUMMARY_NAMES),
    )


def test_structured_code_cannot_mix_numeric_and_occurrence_rows(tmp_path):
    root = tmp_path / "mortality_48h"
    starts, samples = _samples()
    events = _structured_events(starts).with_columns(
        pl.when(pl.col("sample_id") == 11)
        .then(None)
        .otherwise(pl.col("numeric_value"))
        .alias("numeric_value")
    )
    _write(root, events, samples)
    with pytest.raises(ValueError, match="cannot mix numeric and occurrence"):
        open_task(root)


def test_note_using_task_allows_samples_without_note_rows(tmp_path):
    root = tmp_path / "mortality_48h"
    starts, samples = _samples()
    structured = _structured_events(starts).with_columns(
        pl.lit(None, dtype=pl.String).alias("text_value")
    )
    note = pl.DataFrame(
        {
            "subject_id": pl.Series([1, 1, 1], dtype=pl.Int64),
            "time": pl.Series(
                np.repeat(starts[:1] + np.timedelta64(2, "h"), 3),
                dtype=pl.Datetime("us"),
            ),
            "code": pl.Series(["NOTE"] * 3, dtype=pl.String),
            "numeric_value": pl.Series([None] * 3, dtype=pl.Float32),
            "sample_id": pl.Series([11] * 3, dtype=pl.Int64),
            # Physical row order is deliberately different from exporter order.
            "source_row": pl.Series([8, 6, 7], dtype=pl.Int64),
            "text_value": pl.Series(["third", "first", "second"], dtype=pl.String),
        }
    )
    _write(
        root,
        pl.concat([structured, note]),
        samples,
        codes=("heart_rate", "NOTE"),
        note_codes=("NOTE",),
    )
    assert open_task(root).load_documents("train") == [
        "first\n\nsecond\n\nthird",
        "",
    ]


def _refresh_manifest_hash(root, record):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = root / manifest["meds"][record]["file"]
    manifest["meds"][record]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_open_rejects_label_file_that_disagrees_with_task_index(tmp_path):
    root = tmp_path / "mortality_48h"
    starts, samples = _samples()
    _write(root, _structured_events(starts), samples)
    path = root / "tasks" / "mortality_48h" / "labels.parquet"
    labels = pl.read_parquet(path).with_columns(
        pl.when(pl.col("sample_id") == 11)
        .then(~pl.col("boolean_value"))
        .otherwise(pl.col("boolean_value"))
        .alias("boolean_value")
    )
    labels.write_parquet(path)
    _refresh_manifest_hash(root, "labels")
    with pytest.raises(ValueError, match="labels.parquet disagrees"):
        open_task(root)


def test_open_rejects_subject_split_file_that_disagrees_with_index(tmp_path):
    root = tmp_path / "mortality_48h"
    starts, samples = _samples()
    _write(root, _structured_events(starts), samples)
    path = root / "metadata" / "subject_splits.parquet"
    memberships = pl.read_parquet(path).filter(pl.col("subject_id") != 1)
    memberships.write_parquet(path)
    _refresh_manifest_hash(root, "subject_splits")
    with pytest.raises(ValueError, match="subject_splits.parquet disagrees"):
        open_task(root)


def test_untimed_profile_drops_timing_statistics_and_window(tmp_path):
    """An untimed task must not receive elapsed-time features it cannot have.

    A multi_visit_untimed source model is shown visit order only, so every
    prediction-relative statistic and the observation width are withheld. For
    an occurrence-only vocabulary that leaves exactly one statistic per code.
    """
    task_hours = 240.0
    sample_ids = np.asarray([301, 302], dtype=np.int64)
    starts_us = np.asarray([0, 500_000_000_000], dtype=np.int64)
    widths_us = np.asarray([120 * 3_600_000_000, 240 * 3_600_000_000], dtype=np.int64)
    event_samples = np.asarray([301, 301, 302], dtype=np.int64)
    event_hours = np.asarray([10.0, 100.0, 5.0])
    events_path = tmp_path / "untimed.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": pa.array(event_samples, type=pa.int64()),
                "time": pa.array(
                    starts_us[event_samples - sample_ids[0]]
                    + (event_hours * 3_600_000_000).astype(np.int64),
                    type=pa.timestamp("us"),
                ),
                "code": pa.array(["visit//A"] * 3, type=pa.string()),
                "numeric_value": pa.array([None] * 3, type=pa.float32()),
                "source_row": pa.array(np.arange(3), type=pa.int64()),
            }
        ),
        events_path,
    )
    index = pa.table(
        {
            "sample_id": pa.array(sample_ids, type=pa.int64()),
            "observation_start": pa.array(starts_us, type=pa.timestamp("us")),
            "prediction_time": pa.array(
                starts_us + widths_us, type=pa.timestamp("us")
            ),
        }
    )
    actual = build_rust_event_representations(
        events_path,
        index,
        ("visit//A",),
        task_hours,
        occurrence_codes=("visit//A",),
        segment_kind="visit_buckets",
        timed=False,
    )
    reference = build_reference_event_representations(
        {
            "sample_id": event_samples,
            "time_hours": event_hours,
            "code": np.asarray(["visit//A"] * 3),
            "value": np.full(3, np.nan, dtype=np.float32),
            "source_row": np.arange(3, dtype=np.int64),
        },
        sample_ids,
        ("visit//A",),
        task_hours,
        occurrence_codes=("visit//A",),
        sample_window_hours=widths_us / 3_600_000_000.0,
        temporal_profile="multi_visit_untimed",
    )
    for representation in ("summary", "visit_buckets"):
        assert actual[representation].names == reference[representation].names
        np.testing.assert_allclose(
            np.asarray(actual[representation].values),
            reference[representation].values,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )

    names = actual["summary"].names
    assert not any(name.endswith(("__recency", "__first_recency")) for name in names)
    assert "__sample__window_hours" not in names
    assert names == (
        "__sample__event_count",
        "__sample__code_count",
        "__sample__distinct_time_count",
        "visit//A__full__count",
    )
    buckets = actual["visit_buckets"]
    values = np.asarray(buckets.values)
    # Sample 301's last event is at hour 100, so one of its two events falls in
    # the current visit and the other in the prior span.
    assert values[0, buckets.names.index("visit//A__all__count")] == 2
    assert values[0, buckets.names.index("visit//A__prior__count")] == 1
    assert values[0, buckets.names.index("visit//A__current__count")] == 1
    assert values[1, buckets.names.index("visit//A__current__count")] == 1


def test_events_may_carry_a_paper_verified_column(tmp_path):
    """A column a paper's own checks read, which no representation does.

    GraphCare's batch check asserts every event still names its encounter, so
    its export carries visit_id even though no representation reads it. The
    contract permits exactly that column; a genuinely unknown one still fails.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    task_dir = tmp_path / "mortality_48h"
    starts, samples = _samples()
    _write(task_dir, _structured_events(starts), samples)
    events_path = task_dir / "data" / "events.parquet"
    def rewrite(events):
        pq.write_table(events, events_path)
        manifest_path = task_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        digest = hashlib.sha256(events_path.read_bytes()).hexdigest()
        manifest["meds"]["events"]["sha256"] = digest
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    table = pq.read_table(events_path)
    with_legacy = table.append_column(
        "visit_id", pa.array([1] * table.num_rows, type=pa.int64())
    )
    rewrite(with_legacy)

    source = open_task(task_dir)
    assert "visit_id" not in source.representations
    assert source.load("train")["sample_id"].size > 0

    with_unknown = with_legacy.append_column(
        "not_a_contract_column", pa.array([0] * table.num_rows, type=pa.int64())
    )
    rewrite(with_unknown)
    with pytest.raises(ValueError, match="not_a_contract_column"):
        open_task(task_dir)
