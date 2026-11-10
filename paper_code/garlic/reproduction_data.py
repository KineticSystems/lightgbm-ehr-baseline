"""Bounded exact-data adapter for released GARLIC training and evaluation."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from cohort_source import (  # noqa: E402
    EXPECTED_COHORT_ROWS,
    EXPECTED_FEATURES,
    MAX_MODEL_TIMESTAMPS,
    author_adult_icu,
    cohort_rows,
    open_complete_cache,
)
from utils import get_data_split  # noqa: E402


SPLITS = ("train", "validation", "test")


def reconstruct_ordered_splits(
    cache_path: Path, mimic_root: Path
) -> tuple[dict[str, list[int]], dict[int, dict[str, Any]], dict[str, Any]]:
    """Replay only the released small-table stratified split in exact order."""
    connection, summary = open_complete_cache(cache_path)
    try:
        rows = cohort_rows(connection)
    finally:
        connection.close()
    if len(rows) != EXPECTED_COHORT_ROWS:
        raise AssertionError("GARLIC cohort size changed")

    adult_icu = author_adult_icu(mimic_root)
    static_by_subject: dict[int, tuple[float, float]] = {}
    for row in adult_icu.itertuples(index=False):
        subject = int(row.SUBJECT_ID)
        if subject in static_by_subject:
            continue
        gender = 0.0 if row.GENDER == "F" else 1.0 if row.GENDER == "M" else np.nan
        age = float(row.AGE) if not np.isnan(row.AGE) else np.nan
        static_by_subject[subject] = (gender, age)

    candidates = []
    by_hadm = {}
    for row in rows:
        subject = int(row["subject_id"])
        gender, age = static_by_subject.get(subject, (np.nan, np.nan))
        candidate = {
            "id": int(row["hadm_id"]),
            "hadm_id": int(row["hadm_id"]),
            "subject_id": subject,
            "label": int(row["label"]),
            "static": [gender, age],
            "split": str(row["split"]),
            "source_position": int(row["source_position"]),
        }
        candidates.append(candidate)
        by_hadm[int(row["hadm_id"])] = candidate

    train, validation, test = get_data_split(candidates)
    values = {"train": train, "validation": validation, "test": test}
    orders = {
        split: [int(row["hadm_id"]) for row in values[split]] for split in SPLITS
    }
    all_ids = [hadm for split in SPLITS for hadm in orders[split]]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(by_hadm):
        raise AssertionError("released split replay lost or duplicated admissions")
    for split in SPLITS:
        expected = {
            int(row["hadm_id"]) for row in rows if str(row["split"]) == split
        }
        if set(orders[split]) != expected:
            raise AssertionError(f"released {split} membership changed")
    return orders, by_hadm, summary


class GarlicSplitSequence(Sequence[dict[str, Any]]):
    """List-like split whose slices materialize only the requested batch."""

    def __init__(self, store: "GarlicDataStore", split: str, ids: list[int]):
        self.store = store
        self.split = split
        self.ids = tuple(int(value) for value in ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            selected = self.ids[index]
            return self.store.load_ids(selected)
        position = int(index)
        if position < 0:
            position += len(self.ids)
        if position < 0 or position >= len(self.ids):
            raise IndexError(index)
        return self.store.load_ids((self.ids[position],))[0]


class GarlicDataStore:
    """Read-only SQLite-backed source of exact pre-`prepare_batch` samples."""

    def __init__(self, cache_path: Path, order_path: Path):
        payload = json.loads(order_path.read_text())
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported GARLIC ordered-split schema")
        orders = payload.get("ordered_hadm_ids", {})
        if set(orders) != set(SPLITS):
            raise ValueError("GARLIC ordered-split artifact is incomplete")

        cache_uri = f"file:{cache_path.resolve()}?mode=ro"
        self.connection = sqlite3.connect(cache_uri, uri=True)
        complete = self.connection.execute(
            "SELECT payload FROM checkpoints WHERE name='cache_complete'"
        ).fetchone()
        if complete is None:
            self.connection.close()
            raise ValueError("GARLIC source cache is incomplete")
        self.rows = {
            int(row[0]): {
                "hadm_id": int(row[0]),
                "subject_id": int(row[1]),
                "label": int(row[2]),
                "split": str(row[3]),
                "source_position": int(row[4]),
            }
            for row in self.connection.execute(
                "SELECT hadm_id, subject_id, label, split, source_position "
                "FROM cohort ORDER BY source_position"
            )
        }
        self.normalization = {
            int(row[0]): (float(row[1]), float(row[2]))
            for row in self.connection.execute(
                "SELECT feature_idx, mean_value, std_value FROM normalization "
                "ORDER BY feature_idx"
            )
        }
        if len(self.rows) != EXPECTED_COHORT_ROWS:
            raise AssertionError("GARLIC cache cohort changed")
        if len(self.normalization) != EXPECTED_FEATURES:
            raise AssertionError("GARLIC normalization vocabulary changed")

        self.splits = {
            split: GarlicSplitSequence(self, split, list(orders[split]))
            for split in SPLITS
        }
        for split, sequence in self.splits.items():
            if any(self.rows[hadm]["split"] != split for hadm in sequence.ids):
                raise AssertionError(f"GARLIC {split} order crosses split membership")

    def close(self) -> None:
        self.connection.close()

    def source_order(self) -> GarlicSplitSequence:
        ids = [
            hadm
            for hadm, _ in sorted(
                self.rows.items(), key=lambda item: item[1]["source_position"]
            )
        ]
        return GarlicSplitSequence(self, "source", ids)

    def load_ids(self, ids: Sequence[int]) -> list[dict[str, Any]]:
        ids = tuple(int(value) for value in ids)
        if not ids:
            return []
        if len(ids) > 900:
            raise ValueError("GARLIC adapter batch exceeds safe SQLite variable count")
        if len(set(ids)) != len(ids):
            raise AssertionError("GARLIC adapter batch contains duplicate admissions")
        for hadm in ids:
            if hadm not in self.rows:
                raise KeyError(hadm)

        placeholders = ",".join("?" for _ in ids)
        cursor = self.connection.execute(
            "SELECT hadm_id, sequence_row, feature_idx, raw_value, source_time "
            f"FROM raw_cells WHERE hadm_id IN ({placeholders}) "
            "ORDER BY hadm_id, sequence_row, feature_idx",
            ids,
        )
        cells_by_hadm: dict[int, list[tuple[int, int, float, float]]] = {
            hadm: [] for hadm in ids
        }
        for hadm, sequence, feature, raw, source_time in cursor:
            cells_by_hadm[int(hadm)].append(
                (int(sequence), int(feature), float(raw), float(source_time))
            )

        result = []
        for hadm in ids:
            arr = np.zeros(
                (MAX_MODEL_TIMESTAMPS, EXPECTED_FEATURES), dtype=np.float64
            )
            mask = np.zeros_like(arr)
            times = np.zeros((MAX_MODEL_TIMESTAMPS, 1), dtype=np.float64)
            for sequence, feature, raw, source_time in cells_by_hadm[hadm]:
                if not (0 <= sequence < MAX_MODEL_TIMESTAMPS):
                    raise AssertionError("GARLIC sequence row is out of range")
                mean_value, std_value = self.normalization[feature]
                normalized = raw - mean_value
                if std_value != 0:
                    normalized /= std_value
                arr[sequence, feature] = normalized
                mask[sequence, feature] = 1.0
                candidate = source_time * 60.0
                if times[sequence, 0] not in (0.0, candidate):
                    raise AssertionError("GARLIC features disagree on timestamp")
                times[sequence, 0] = candidate
            row = self.rows[hadm]
            result.append(
                {
                    "id": hadm,
                    "arr": arr,
                    "mask": mask,
                    "time": times,
                    "label": row["label"],
                }
            )
        return result


def load_reproduction_splits(
    cache_path: Path, order_path: Path
) -> tuple[GarlicDataStore, GarlicSplitSequence, GarlicSplitSequence, GarlicSplitSequence]:
    store = GarlicDataStore(cache_path, order_path)
    return (
        store,
        store.splits["train"],
        store.splits["validation"],
        store.splits["test"],
    )

