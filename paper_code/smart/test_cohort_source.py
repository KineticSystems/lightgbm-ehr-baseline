from __future__ import annotations

import unittest
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cohort_source import (
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TOTAL,
    SEEDS,
    code_vocabulary,
    replica_task_id,
    split_assignment,
)


class SmartExportSourceTest(unittest.TestCase):
    def test_released_vocabulary_is_complete_and_unique(self):
        vocabulary = code_vocabulary()
        self.assertEqual(len(vocabulary), 59)
        self.assertEqual(len(set(vocabulary)), 59)
        self.assertIn("Heart Rate", vocabulary)
        self.assertIn("Glascow coma scale eye opening//4 Spontaneously", vocabulary)

    def test_all_three_released_seed_redraws_are_deterministic(self):
        cohort = pl.DataFrame(
            {
                "source_position": range(EXPECTED_TOTAL),
                "sample_id": range(1, EXPECTED_TOTAL + 1),
            },
            schema={"source_position": pl.Int64, "sample_id": pl.Int64},
        )
        assignments = []
        for seed in SEEDS:
            first = split_assignment(cohort, seed)
            second = split_assignment(cohort, seed)
            self.assertTrue(first.equals(second))
            self.assertEqual(
                dict(first.group_by("split").len().iter_rows()),
                EXPECTED_SPLIT_COUNTS,
            )
            assignments.append(first["split"].to_list())
        self.assertNotEqual(assignments[0], assignments[1])
        self.assertNotEqual(assignments[1], assignments[2])

    def test_replica_ids_are_stable(self):
        self.assertEqual(
            [replica_task_id(value) for value in (1, 2, 3)],
            [
                "mortality_48h_replica_01",
                "mortality_48h_replica_02",
                "mortality_48h_replica_03",
            ],
        )


if __name__ == "__main__":
    unittest.main()
