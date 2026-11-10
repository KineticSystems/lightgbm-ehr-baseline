"""Bounded restart/cache tests; these never access MIMIC."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import export_source
import cohort_source


class ResumeCacheTest(unittest.TestCase):
    @staticmethod
    def _cohort() -> list[dict]:
        return [
            {
                "source_position": 0,
                "hadm_id": 101,
                "subject_id": 11,
                "label": 1,
                "split": "train",
                "observation_start_us": 1_000_000,
            },
            {
                "source_position": 1,
                "hadm_id": 102,
                "subject_id": 12,
                "label": 0,
                "split": "test",
                "observation_start_us": 2_000_000,
            },
        ]

    @staticmethod
    def _features() -> tuple[str, ...]:
        removed = set(cohort_source.REMOVE_INDICES)
        return tuple(
            feature
            for index, feature in enumerate(cohort_source.DECLARED_FEATURES)
            if index not in removed
        )

    def _pass1_cache(self, cache_path: Path) -> None:
        connection = cohort_source._initialize_database(cache_path)
        try:
            cohort_source._store_pass1_checkpoint(
                connection,
                self._cohort(),
                full_feature_order=cohort_source.DECLARED_FEATURES,
                features=self._features(),
            )
        finally:
            connection.close()

    def test_resume_after_pass1_does_not_rerun_first_pass(self) -> None:
        class ReachedSecondPass(Exception):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "cache.sqlite"
            self._pass1_cache(cache_path)
            (root / "required.csv.gz").touch()

            with (
                mock.patch.object(cohort_source, "RAW_FILES", ("required.csv.gz",)),
                mock.patch.object(cohort_source, "EXPECTED_COHORT_ROWS", 2),
                mock.patch.object(cohort_source, "EXPECTED_POSITIVE", 1),
                mock.patch.object(
                    cohort_source,
                    "author_adult_icu",
                    return_value=pd.DataFrame(),
                ),
                mock.patch.object(cohort_source, "_first_pass") as first_pass,
                mock.patch.object(
                    cohort_source,
                    "_second_pass",
                    side_effect=ReachedSecondPass,
                ) as second_pass,
            ):
                with self.assertRaises(ReachedSecondPass):
                    cohort_source.build_cache(root, cache_path)

            first_pass.assert_not_called()
            second_pass.assert_called_once()
            connection = cohort_source._initialize_database(cache_path)
            try:
                self.assertEqual(
                    cohort_source.checkpoint(connection, "pass1_complete")[
                        "cohort_rows"
                    ],
                    2,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM cohort").fetchone()[0],
                    2,
                )
            finally:
                connection.close()

    def test_incomplete_cache_requires_explicit_resume_and_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "cache.sqlite"
            self._pass1_cache(cache_path)

            with self.assertRaisesRegex(ValueError, "--resume"):
                export_source.load_cache(
                    root,
                    cache_path,
                    resume=False,
                    restart_stage=None,
                )

            connection = cohort_source._initialize_database(cache_path)
            try:
                self.assertIsNotNone(
                    cohort_source.checkpoint(connection, "pass1_complete")
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM cohort").fetchone()[0],
                    2,
                )
            finally:
                connection.close()

    def test_complete_cache_writes_meds_without_source_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "cache.sqlite"
            output_path = root / "task"
            self._pass1_cache(cache_path)
            summary = {
                "status": "complete",
                "cohort_rows": 2,
                "positive": 1,
                "features": list(self._features()),
                "candidate_events": 0,
                "retained_events": 0,
                "retained_length": {
                    "minimum": 0,
                    "maximum": 0,
                    "mean": 0.0,
                    "samples_without_observed_cells": 2,
                },
                "splits": {
                    "train": {"count": 1, "positive": 1, "subjects": 1},
                    "validation": {"count": 0, "positive": 0, "subjects": 0},
                    "test": {"count": 1, "positive": 0, "subjects": 1},
                },
                "cross_split_subject_overlap": {
                    "train_validation": 0,
                    "train_test": 0,
                    "validation_test": 0,
                },
                "normalization": [],
                "raw_artifacts": {},
                "released_source": {},
            }
            connection = cohort_source._initialize_database(cache_path)
            try:
                cohort_source._put_metadata(connection, "cache_summary", summary)
                cohort_source._checkpoint(
                    connection,
                    "cache_complete",
                    {
                        "cohort_rows": 2,
                        "candidate_events": 0,
                        "retained_events": 0,
                    },
                )
                connection.commit()
            finally:
                connection.close()

            with mock.patch.object(
                export_source,
                "build_cache",
                side_effect=AssertionError("source build must not run"),
            ) as source_build:
                connection, loaded = export_source.load_cache(
                    root / "deliberately-missing-source",
                    cache_path,
                    resume=True,
                    restart_stage=None,
                )
            source_build.assert_not_called()
            try:
                with mock.patch.object(export_source, "write_meds_task") as writer:
                    export_source.write_task_from_cache(
                        connection,
                        loaded,
                        output_path,
                    )
                writer.assert_called_once()
                call = writer.call_args
                self.assertEqual(call.args[0], output_path)
                self.assertEqual(call.kwargs["samples"].height, 2)
                self.assertEqual(len(call.kwargs["codes"]), 103)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
