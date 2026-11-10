"""Deterministic resource-limited sampling of rolling prediction landmarks."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl


LANDMARK_SAMPLE_SEED = 20_260_829
LANDMARK_SAMPLE_CAPS = {
    "train": 250_000,
    "validation": 100_000,
    "test": 250_000,
}

_SPLIT_SALTS = {
    "train": np.uint64(0x243F6A8885A308D3),
    "validation": np.uint64(0x13198A2E03707344),
    "test": np.uint64(0xA4093822299F31D0),
}


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Return a stable bijective 64-bit mixing of unsigned integers."""
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        return values ^ (values >> np.uint64(31))


def _ordered_id_sha256(values: np.ndarray) -> str:
    ids = np.asarray(values, dtype="<u8")
    digest = hashlib.sha256()
    digest.update(np.asarray(ids.shape, dtype="<i8").tobytes())
    digest.update(ids.tobytes())
    return digest.hexdigest()


def membership_mask(candidate_ids: np.ndarray, selected_ids: np.ndarray) -> np.ndarray:
    """Mark candidate IDs present in an ascending unique selected-ID array."""
    candidates = np.asarray(candidate_ids, dtype=np.int64)
    selected = np.asarray(selected_ids, dtype=np.int64)
    if selected.ndim != 1 or (len(selected) > 1 and np.any(selected[1:] <= selected[:-1])):
        raise ValueError("selected_ids must be a strictly increasing one-dimensional array")
    positions = np.searchsorted(selected, candidates)
    valid = positions < len(selected)
    matched = np.zeros(len(candidates), dtype=bool)
    matched[valid] = selected[positions[valid]] == candidates[valid]
    return matched


def select_uniform_landmarks(
    samples: pl.DataFrame,
    *,
    caps: Mapping[str, int] = LANDMARK_SAMPLE_CAPS,
    seed: int = LANDMARK_SAMPLE_SEED,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Uniformly select at most the configured number of landmarks per split.

    Selection is without replacement and is independent of labels, subjects,
    stay length, and row order. Original sample IDs and source ordering are
    retained. The SplitMix64 permutation makes membership reproducible across
    Python, NumPy, and Polars versions.
    """
    required = {"sample_id", "subject_id", "boolean_value", "split"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"samples lack required sampling columns: {sorted(missing)}")
    if samples.is_empty():
        raise ValueError("cannot sample an empty landmark table")
    if samples["sample_id"].null_count() or samples["sample_id"].n_unique() != samples.height:
        raise ValueError("sample_id must be non-null and unique before sampling")
    if samples["sample_id"].min() < 0:
        raise ValueError("sample_id must be non-negative")
    if set(caps) != set(_SPLIT_SALTS):
        raise ValueError(f"caps must define exactly {sorted(_SPLIT_SALTS)}")
    if any(not isinstance(value, int) or value <= 0 for value in caps.values()):
        raise ValueError("every landmark cap must be a positive integer")
    observed_splits = set(samples["split"].unique().to_list())
    if observed_splits != set(caps):
        raise ValueError(
            f"landmark splits differ from sampling contract: {sorted(observed_splits)}"
        )

    indexed = samples.with_row_index("__sampling_source_row")
    selected_parts: list[pl.DataFrame] = []
    split_audit: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        source = indexed.filter(pl.col("split") == split)
        ids = source["sample_id"].to_numpy().astype(np.int64, copy=False)
        cap = int(caps[split])
        if source.height > cap:
            unsigned = ids.astype(np.uint64, copy=False)
            scores = _splitmix64(
                unsigned ^ np.uint64(seed) ^ _SPLIT_SALTS[split]
            )
            positions = np.argpartition(scores, cap - 1)[:cap]
            keep = np.zeros(source.height, dtype=bool)
            keep[positions] = True
            selected = source.filter(pl.Series("__keep", keep))
        else:
            selected = source
        selected_parts.append(selected)
        selected_ids = selected["sample_id"].to_numpy().astype(np.int64, copy=False)
        split_audit[split] = {
            "source_landmarks": source.height,
            "source_subjects": source["subject_id"].n_unique(),
            "source_positive": int(source["boolean_value"].sum()),
            "source_prevalence": float(source["boolean_value"].mean()),
            "cap": cap,
            "selected_landmarks": selected.height,
            "selected_subjects": selected["subject_id"].n_unique(),
            "selected_positive": int(selected["boolean_value"].sum()),
            "selected_prevalence": float(selected["boolean_value"].mean()),
            "ordered_selected_sample_id_sha256": _ordered_id_sha256(selected_ids),
        }

    selected = (
        pl.concat(selected_parts, how="vertical")
        .sort("__sampling_source_row")
        .drop("__sampling_source_row")
    )
    audit = {
        "status": "deterministic_resource_limited_landmark_subsample",
        "sampling_unit": "released prediction landmark",
        "method": "lowest SplitMix64 scores within each released split",
        "without_replacement": True,
        "label_stratified": False,
        "subject_or_stay_balancing": False,
        "sample_weights": None,
        "seed": int(seed),
        "caps": {name: int(caps[name]) for name in ("train", "validation", "test")},
        "source_landmarks": samples.height,
        "selected_landmarks": selected.height,
        "splits": split_audit,
        "ordered_selected_sample_id_sha256": _ordered_id_sha256(
            selected["sample_id"].to_numpy().astype(np.int64, copy=False)
        ),
    }
    return selected, audit
