from __future__ import annotations

import numpy as np
import polars as pl

from check_batch import reconstruct_prefix
from check_cohort import m3b_base_decomp_counts
from preflight_export_size import cutoff_chunks, length_band_indexes
from cohort_source import (
    ANCHOR_US,
    HOUR_US,
    _event_frame,
    _prefix_block_coordinates,
    paper_encoded_stay,
    parse_filename,
)


def test_full_episode_parser_and_released_surrogate_bug() -> None:
    assert parse_filename("123_episode12_timeseries.csv") == (123, 12)
    assert paper_encoded_stay(123, 12) == 20_000_123


def test_m3b_base_mask_counts_are_separate_from_tls_horizon(tmp_path) -> None:
    listfile = tmp_path / "listfile.csv"
    listfile.write_text(
        "decompensation task (masks;labels)\n"
        "1;0;1;0;0;1\n"
        "0;1;0;1\n"
    )
    assert m3b_base_decomp_counts(listfile) == {"positive": 2, "negative": 1}


def test_preflight_strata_are_disjoint_and_deterministic() -> None:
    chunks = cutoff_chunks(np.arange(10, dtype=np.int64))
    np.testing.assert_array_equal(np.concatenate(chunks), np.arange(10))
    bands = length_band_indexes(np.asarray([9, 1, 8, 2, 7, 3, 6, 4, 5]))
    assert set(np.concatenate(list(bands.values())).tolist()) == set(range(9))
    assert [len(bands[name]) for name in ("short", "median", "long")] == [3, 3, 3]


def test_event_frame_preserves_occurrence_null() -> None:
    buffer = {
        "subject_id": [np.asarray([7, 7], dtype=np.int64)],
        "time": [np.asarray([ANCHOR_US, ANCHOR_US], dtype=np.int64)],
        "code": [np.asarray(["numeric", "category"], dtype=object)],
        "numeric_value": [np.asarray([1.25, 0.0], dtype=np.float32)],
        "numeric_valid": [np.asarray([True, False])],
        "sample_id": [np.asarray([1, 1], dtype=np.int64)],
        "source_row": [np.asarray([0, 1], dtype=np.int64)],
        "sequence_row": [np.asarray([0, 0], dtype=np.int32)],
        "source_time": [np.asarray([0.0, 0.0], dtype=np.float64)],
    }
    frame = _event_frame(buffer)
    assert frame.schema["numeric_value"] == pl.Float32
    assert frame["numeric_value"].to_list() == [1.25, None]


def test_vectorized_prefix_coordinates_match_simple_reference() -> None:
    target_rows = np.asarray([1, 3], dtype=np.int64)
    cat_rows = np.asarray([0, 2, 3], dtype=np.int32)
    cat_features = np.asarray([2, 3, 2], dtype=np.int32)
    rows, features, samples, numeric = _prefix_block_coordinates(
        target_rows, cat_rows, cat_features, numeric_count=2
    )
    actual = sorted(zip(samples.tolist(), rows.tolist(), features.tolist(), numeric.tolist()))
    expected = []
    for sample, target in enumerate(target_rows):
        for row in range(int(target) + 1):
            expected.extend((sample, row, feature, True) for feature in range(2))
        expected.extend(
            (sample, int(row), int(feature), False)
            for row, feature in zip(cat_rows, cat_features)
            if row <= target
        )
    assert actual == sorted(expected)


def test_meds_occurrences_reconstruct_exact_dense_prefix() -> None:
    events = pl.DataFrame(
        {
            "source_row": pl.Series([0, 2, 3], dtype=pl.Int64),
            "sequence_row": pl.Series([0, 1, 1], dtype=pl.Int32),
            "source_time": pl.Series([0.0, 1.0, 1.0], dtype=pl.Float64),
            "time": pl.Series(
                np.asarray(
                    [ANCHOR_US, ANCHOR_US + HOUR_US, ANCHOR_US + HOUR_US],
                    dtype=np.int64,
                ),
                dtype=pl.Datetime("us"),
            ),
            "code": pl.Series(["numeric", "numeric", "category"], dtype=pl.String),
            "numeric_value": pl.Series([2.0, -1.0, None], dtype=pl.Float32),
        }
    )
    actual, summary = reconstruct_prefix(
        events,
        columns=("numeric", "category"),
        numeric_count=1,
        author_row=1,
    )
    np.testing.assert_array_equal(
        actual, np.asarray([[2.0, 0.0], [-1.0, 1.0]], dtype=np.float32)
    )
    assert summary["numeric_events"] == 2
    assert summary["occurrence_events"] == 1
