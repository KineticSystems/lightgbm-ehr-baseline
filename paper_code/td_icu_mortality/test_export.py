"""Small deterministic tests for the TD-ICU the adapter."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

import export_source
from check_batch import (
    assert_event_identity,
    author_row_positions,
    author_event_count,
    reconstruct_delta_arrays,
    reconstruct_primary_arrays,
)
from cohort_source import CONTEXT_LENGTH


class ExportAdapterTest(unittest.TestCase):
    def test_sparse_subsample_source_rows_map_to_dense_batch_positions(self) -> None:
        current = pl.DataFrame(
            {
                "source_sample_row": pl.Series([5, 9], dtype=pl.Int64),
            }
        )
        tokens = np.asarray(
            [5 * CONTEXT_LENGTH, 9 * CONTEXT_LENGTH + 17], dtype=np.int64
        )
        np.testing.assert_array_equal(author_row_positions(current, tokens), [0, 1])

    def test_event_batches_remove_only_padding(self) -> None:
        values = [60.0, 1.25] + [float("nan")] * (CONTEXT_LENGTH - 2)
        features = [0, 1] + [-1] * (CONTEXT_LENGTH - 2)
        times = [0, 30] + [-1] * (CONTEXT_LENGTH - 2)
        delta_time = [-1, 20] + [-1] * (CONTEXT_LENGTH - 2)
        delta_value = [float("nan"), 0.5] + [float("nan")] * (CONTEXT_LENGTH - 2)
        frame = pl.DataFrame(
            {
                "subject_id": [99],
                "sample_id": [7],
                "source_sample_row": [5],
                "split": ["test"],
                "labeltime": [datetime(2020, 1, 8, 12)],
                "encoded_feature": [features],
                "featuretime": [times],
                "value": [values],
                "delta_time": [delta_time],
                "delta_value": [delta_value],
                "1-day-died": [0],
                "3-day-died": [0],
                "7-day-died": [0],
                "14-day-died": [0],
                "28-day-died": [1],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current.parquet"
            frame.write_parquet(path)
            with patch.object(export_source, "current_files", return_value=[path]):
                events = list(
                    export_source.event_batches(
                        ("age", "Lactate"), np.asarray([7], dtype=np.int64)
                    )
                )[0]
        self.assertEqual(events.height, 4)
        self.assertEqual(
            events["code"].to_list(),
            ["age", "Lactate", "DELTA_TIME//Lactate", "DELTA_VALUE//Lactate"],
        )
        self.assertEqual(events["sequence_row"].to_list(), [0, 1, 1, 1])
        self.assertEqual(events["source_row"].to_list(), [6000, 6003, 6004, 6005])
        self.assertEqual(events["numeric_value"].dtype, pl.Float32)
        self.assertEqual(
            events["time"].to_numpy()[1],
            events["time"].to_numpy()[0] - np.timedelta64(30, "m"),
        )

    def test_meds_events_recreate_author_padding_and_deltas(self) -> None:
        feature = [0, -1, -1, 1, 1, 2, 2] + [-1] * (CONTEXT_LENGTH - 7)
        timepoint = [0, -1, -1, 0, 5, 10, 20] + [-1] * (CONTEXT_LENGTH - 7)
        value = [60.0, np.nan, np.nan, 1.25, 1.25, 5.0, 4.0] + [
            np.nan
        ] * (CONTEXT_LENGTH - 7)
        current = pl.DataFrame(
            {
                "subject_id": [99],
                "sample_id": [7],
                "source_sample_row": [5],
                "split": ["test"],
                "labeltime": [datetime(2020, 1, 8, 12)],
                "encoded_feature": [feature],
                "featuretime": [timepoint],
                "value": [value],
                "delta_time": [[-1, -1, -1, 0, -1, 10, -1] + [-1] * (CONTEXT_LENGTH - 7)],
                "delta_value": [[np.nan, np.nan, np.nan, 0.0, np.nan, 1.0, np.nan] + [np.nan] * (CONTEXT_LENGTH - 7)],
                "1-day-died": [0],
                "3-day-died": [0],
                "7-day-died": [0],
                "14-day-died": [0],
                "28-day-died": [1],
            }
        )
        events = export_source.events_from_current(
            current, ("age", "Noradrenaline rate", "Lactate")
        )
        self.assertEqual(author_event_count(current), 9)
        assert_event_identity(events.reverse(), current)
        recreated = reconstruct_primary_arrays(
            events, current, {"age": 0, "Noradrenaline rate": 1, "Lactate": 2}
        )
        np.testing.assert_array_equal(recreated[0][0], np.asarray(feature, np.int16))
        np.testing.assert_array_equal(recreated[1][0], np.asarray(timepoint, np.int16))
        np.testing.assert_array_equal(
            recreated[2][0], np.asarray(value, np.float32)
        )
        delta_time, delta_value = reconstruct_delta_arrays(
            events,
            current,
            {"age": 0, "Noradrenaline rate": 1, "Lactate": 2},
        )
        self.assertEqual(delta_time[0, 3], 0)
        self.assertEqual(delta_time[0, 5], 10)
        self.assertEqual(delta_value[0, 3], 0.0)
        self.assertEqual(delta_value[0, 5], 1.0)
        self.assertTrue(np.isnan(delta_value[0, 4]))


if __name__ == "__main__":
    unittest.main()
