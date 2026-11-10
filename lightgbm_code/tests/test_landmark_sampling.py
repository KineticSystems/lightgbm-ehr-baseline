from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from missing_xgboost_lightgbm.landmark_sampling import (
    membership_mask,
    select_uniform_landmarks,
)


def samples(rows_per_split: int = 20) -> pl.DataFrame:
    splits = [name for name in ("train", "validation", "test") for _ in range(rows_per_split)]
    ids = np.arange(len(splits), dtype=np.int64)
    return pl.DataFrame(
        {
            "sample_id": ids,
            "subject_id": ids // 3,
            "boolean_value": ids % 5 == 0,
            "split": splits,
            "source_order": ids,
        }
    )


def test_selection_is_exact_deterministic_and_retains_source_order() -> None:
    source = samples()
    caps = {"train": 7, "validation": 8, "test": 9}
    first, audit = select_uniform_landmarks(source, caps=caps, seed=17)
    second, repeated = select_uniform_landmarks(source, caps=caps, seed=17)
    assert first.equals(second)
    assert audit == repeated
    assert first["source_order"].to_list() == sorted(first["source_order"].to_list())
    assert first.group_by("split").len().sort("split")["len"].to_list() == [9, 7, 8]
    assert audit["selected_landmarks"] == 24
    assert audit["label_stratified"] is False
    assert audit["sample_weights"] is None


def test_selection_ignores_labels_and_input_row_order() -> None:
    source = samples()
    caps = {"train": 7, "validation": 8, "test": 9}
    selected, _ = select_uniform_landmarks(source, caps=caps, seed=19)
    changed = source.with_columns((~pl.col("boolean_value")).alias("boolean_value"))
    changed = changed.reverse()
    selected_changed, _ = select_uniform_landmarks(changed, caps=caps, seed=19)
    assert sorted(selected["sample_id"].to_list()) == sorted(
        selected_changed["sample_id"].to_list()
    )


def test_membership_mask_requires_sorted_unique_selection() -> None:
    actual = membership_mask(
        np.asarray([9, 2, 5, 1]), np.asarray([1, 5, 9], dtype=np.int64)
    )
    np.testing.assert_array_equal(actual, [True, False, True, True])
    with pytest.raises(ValueError, match="strictly increasing"):
        membership_mask(np.asarray([1]), np.asarray([2, 2]))
