"""The per-split contract both executors must agree on.

A :class:`SplitContract` pins everything about one task split that the Rust
and Python executors have to share exactly: the feature-name contract (and
therefore column order), the zero-vacating offset mask, labels and sample
order from the task index, and the featurizer configuration handed to the
Rust binary. Feature values themselves never pass through this module — the
Rust binary computes and consumes them in-process, and the parity suite
compares its `--dump-config` matrices against the Python oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .reference_features import (
    SEGMENTS_BY_REPRESENTATION,
    aggregate_names,
    statistic_names,
)
__all__ = [
    "ContractError",
    "SplitContract",
    "restore_original_order",
    "to_sorted_sample_order",
]


_SEGMENTED_REPRESENTATIONS = tuple(SEGMENTS_BY_REPRESENTATION)
# The featurizer knows one temporal kind; `segment_kind` selects boundaries.
_FEATURIZER_REPRESENTATION_NAME = {
    "summary": "summary",
    "visit_halves": "temporal",
    "visit_buckets": "temporal",
}
_MAX_RECORD_BATCH_ROWS = 1_000_000
_MAX_CHUNK_BYTES = 512 << 20


class ContractError(RuntimeError):
    """A task split violates the shared executor contract."""


@dataclass(frozen=True)
class _IndexOrder:
    sample_ids: np.ndarray
    labels: np.ndarray
    sorted_to_original: np.ndarray


def _require_positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _binary() -> Path:
    """Return the current release binary, refusing absent or stale builds."""
    crate = Path(__file__).resolve().parents[2] / "rust_featurizer"
    binary = crate / "target" / "release" / "meds-featurizer"
    if not binary.is_file():
        raise RuntimeError(
            "Rust MEDS featurizer is not built; run cargo "
            "build --release --manifest-path "
            "lightgbm_code/rust_featurizer/Cargo.toml"
        )
    if not os.access(binary, os.X_OK):
        raise RuntimeError(f"Rust MEDS featurizer is not executable: {binary}")
    source_files = [
        crate / "Cargo.toml",
        crate / "Cargo.lock",
        *(crate / "src").rglob("*.rs"),
    ]
    newest_source = max(path.stat().st_mtime_ns for path in source_files)
    if binary.stat().st_mtime_ns < newest_source:
        raise RuntimeError(
            "Rust MEDS featurizer is stale; rebuild it with the documented "
            "cargo build --release command"
        )
    return binary


# LightGBM's sparse initialization cannot distinguish "absent" from "zero"
# unless `zero_as_missing` is enabled, which reinterprets every *stored* zero as
# missing. Columns whose absent value is exactly 0.0 are unaffected by that
# reinterpretation: for them "missing" and "zero" denote the same state, so the
# tree simply learns one default direction. Columns that are NaN when the code
# is absent but can legitimately be 0.0 when it is present would otherwise
# collide, so their nonnegative values are shifted by a constant that vacates
# 0.0. The shift is strictly monotone within a column, so every split threshold
# maps across unchanged and the fitted tree structure is invariant.
# `count` is the only numeric statistic that fills 0.0 rather than NaN for an
# unobserved code. Occurrence codes are exempt wholesale: all three of their
# statistics fill 0.0. `first_recency` and `recency` appear in both statistic
# lists, so exemption is decided by the owning code, never by the name alone.
# The per-sample aggregates are counts on the same footing, and the observation
# width is never 0.0, so `zero_as_missing` cannot misread any of them.
_OFFSET_EXEMPT_STATISTICS = frozenset({"count"}) | frozenset(
    name.rsplit("__", 1)[1] for name in aggregate_names(True)
)


def _offset_columns(
    feature_names: Sequence[str],
    occurrence_codes: tuple[str, ...],
) -> np.ndarray:
    """Mark columns whose 0.0 must be vacated before `zero_as_missing`."""
    occurrence = frozenset(occurrence_codes)
    mask = np.zeros(len(feature_names), dtype=bool)
    for position, name in enumerate(feature_names):
        parts = name.rsplit("__", 2)
        if len(parts) != 3:
            raise ContractError(f"feature name is not code__segment__statistic: {name!r}")
        code, _, statistic = parts
        if code in occurrence or statistic in _OFFSET_EXEMPT_STATISTICS:
            continue
        mask[position] = True
    return mask


def _feature_names(
    codes: tuple[str, ...],
    occurrence_codes: tuple[str, ...],
    representation: str,
    timed: bool,
) -> tuple[str, ...]:
    occurrence = set(occurrence_codes)
    numeric = tuple(code for code in codes if code not in occurrence)
    categorical = tuple(code for code in codes if code in occurrence)
    segments = SEGMENTS_BY_REPRESENTATION.get(representation, ("full",))
    summary_statistics, occurrence_statistics = statistic_names(timed)
    return tuple(
        list(aggregate_names(timed))
        + [
            f"{code}__{segment}__{statistic}"
            for code in numeric
            for segment in segments
            for statistic in summary_statistics
        ]
        + [
            f"{code}__{segment}__{statistic}"
            for code in categorical
            for segment in segments
            for statistic in occurrence_statistics
        ]
    )


_LIGHTGBM_FORBIDDEN_NAME_CHARACTERS = frozenset('[]{}":,\\')


def lightgbm_feature_names(names) -> list[str]:
    """Collision-free names accepted by LightGBM's JSON model writer."""
    from urllib.parse import quote

    if all(
        name
        and name.isascii()
        and not any(character.isspace() for character in name)
        and not (_LIGHTGBM_FORBIDDEN_NAME_CHARACTERS & set(name))
        for name in names
    ):
        return list(names)
    return [f"mxgb_{quote(name, safe='')}" for name in names]


def _validate_permutation(permutation: np.ndarray, row_count: int) -> np.ndarray:
    permutation = np.asarray(permutation)
    if permutation.ndim != 1 or len(permutation) != row_count:
        raise ValueError("sorted_to_original must contain one position per row")
    if not np.issubdtype(permutation.dtype, np.integer):
        raise TypeError("sorted_to_original must be an integer array")
    permutation = np.ascontiguousarray(permutation, dtype=np.int64)
    if not np.array_equal(np.sort(permutation), np.arange(row_count)):
        raise ValueError("sorted_to_original is not a permutation")
    return permutation


def restore_original_order(
    sorted_values: np.ndarray, sorted_to_original: np.ndarray
) -> np.ndarray:
    """Restore values aligned to ascending sample ID to physical index order."""
    values = np.asarray(sorted_values)
    if values.ndim == 0:
        raise ValueError("sorted_values must have a row dimension")
    permutation = _validate_permutation(sorted_to_original, len(values))
    restored = np.empty_like(values)
    restored[permutation] = values
    return restored


def to_sorted_sample_order(
    original_values: np.ndarray, sorted_to_original: np.ndarray
) -> np.ndarray:
    """Convert values in physical index order to ascending-sample-ID order."""
    values = np.asarray(original_values)
    if values.ndim == 0:
        raise ValueError("original_values must have a row dimension")
    permutation = _validate_permutation(sorted_to_original, len(values))
    return values[permutation]


def _load_index_order(index_path: Path, split: str) -> _IndexOrder:
    parquet = pq.ParquetFile(index_path)
    schema = parquet.schema_arrow
    for name in ("sample_id", "boolean_value", "split"):
        if name not in schema.names:
            raise ValueError(f"task index is missing {name}")
    if schema.field("sample_id").type != pa.int64():
        raise TypeError("task index sample_id must be int64")
    if schema.field("boolean_value").type != pa.bool_():
        raise TypeError("task index boolean_value must be bool")
    split_type = schema.field("split").type
    if not (pa.types.is_string(split_type) or pa.types.is_large_string(split_type)):
        raise TypeError("task index split must be string or large_string")

    sample_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for batch in parquet.iter_batches(
        batch_size=250_000,
        columns=["sample_id", "boolean_value", "split"],
    ):
        table = pa.Table.from_batches([batch])
        if table["split"].null_count:
            raise ValueError("task index split cannot be null")
        selected = pc.equal(table["split"], split)
        if pc.any(selected).as_py():
            samples = pc.filter(table["sample_id"], selected)
            labels = pc.filter(table["boolean_value"], selected)
            if samples.null_count or labels.null_count:
                raise ValueError("selected task index IDs and labels cannot be null")
            sample_parts.append(samples.to_numpy(zero_copy_only=False))
            label_parts.append(labels.to_numpy(zero_copy_only=False))
    if not sample_parts:
        raise ValueError(f"task index split {split!r} has no rows")
    original_ids = np.concatenate(sample_parts).astype(np.int64, copy=False)
    original_labels = np.concatenate(label_parts).astype(np.float32, copy=False)
    order = np.argsort(original_ids, kind="stable")
    sorted_ids = np.ascontiguousarray(original_ids[order], dtype=np.int64)
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError("task index sample_id must be unique within the split")
    labels = np.ascontiguousarray(original_labels[order], dtype=np.float32)
    order = np.ascontiguousarray(order, dtype=np.int64)
    for array in (sorted_ids, labels, order):
        array.setflags(write=False)
    return _IndexOrder(sorted_ids, labels, order)


class SplitContract:
    """Validated metadata and executor config for one task split.

    Feature construction itself happens inside the Rust binary (or, on the
    reference path, in the Python oracle); this class owns everything both
    executors must agree on up front: the feature-name contract, the
    zero-vacating offset mask, labels and sample order from the task index,
    and the stream configuration handed to the binary.
    """

    def __init__(
        self,
        *,
        events_path: Path,
        index_path: Path,
        split: str,
        representation: str,
        codes: Sequence[str],
        observation_window_hours: float,
        occurrence_codes: Sequence[str] = (),
        ignored_codes: Sequence[str] = (),
        halves_split_hours: float = 0.0,
        timed: bool = True,
        max_component_rows: int = 5_000_000,
        max_chunk_bytes: int = 64 << 20,
        record_batch_rows: int = 65_536,
    ) -> None:
        self.events_path = Path(events_path).resolve()
        self.index_path = Path(index_path).resolve()
        if not self.events_path.is_file():
            raise ValueError(f"events Parquet is missing: {self.events_path}")
        if not self.index_path.is_file():
            raise ValueError(f"task index Parquet is missing: {self.index_path}")
        if not isinstance(split, str) or not split:
            raise ValueError("split must be a nonempty string")
        if representation not in _FEATURIZER_REPRESENTATION_NAME:
            raise ValueError(
                f"representation must be one of "
                f"{sorted(_FEATURIZER_REPRESENTATION_NAME)}"
            )
        codes = tuple(codes)
        occurrence_codes = tuple(occurrence_codes)
        ignored_codes = tuple(ignored_codes)
        if (
            not codes
            or any(not isinstance(code, str) or not code for code in codes)
            or len(set(codes)) != len(codes)
        ):
            raise ValueError("codes must be unique nonempty strings")
        if (
            any(not isinstance(code, str) or not code for code in occurrence_codes)
            or len(set(occurrence_codes)) != len(occurrence_codes)
            or not set(occurrence_codes).issubset(codes)
        ):
            raise ValueError("occurrence_codes must be unique declared codes")
        if (
            any(not isinstance(code, str) or not code for code in ignored_codes)
            or len(set(ignored_codes)) != len(ignored_codes)
            or not set(ignored_codes).isdisjoint(codes)
        ):
            raise ValueError("ignored_codes must be unique nonempty codes disjoint from codes")
        hours = float(observation_window_hours)
        maximum_hours = np.iinfo(np.int64).max / 3_600_000_000.0
        if not (math.isfinite(hours) and 0.0 < hours <= maximum_hours):
            raise ValueError(
                "observation_window_hours must be positive, finite, and representable"
            )
        halves_split_hours = float(halves_split_hours)
        if not math.isfinite(halves_split_hours) or halves_split_hours < 0.0:
            raise ValueError("halves_split_hours must be finite and nonnegative")
        if representation == "visit_halves" and halves_split_hours <= 0.0:
            raise ValueError("visit_halves requires a positive cohort-wide split")
        max_component_rows = _require_positive_integer(
            "max_component_rows", max_component_rows
        )
        max_chunk_bytes = _require_positive_integer(
            "max_chunk_bytes", max_chunk_bytes
        )
        record_batch_rows = _require_positive_integer(
            "record_batch_rows", record_batch_rows
        )
        if record_batch_rows > _MAX_RECORD_BATCH_ROWS:
            raise ValueError(
                f"record_batch_rows exceeds {_MAX_RECORD_BATCH_ROWS:,}"
            )
        if max_chunk_bytes > _MAX_CHUNK_BYTES:
            raise ValueError(
                f"max_chunk_bytes exceeds {_MAX_CHUNK_BYTES:,}"
            )
        if max_component_rows > np.iinfo(np.uint64).max:
            raise ValueError("max_component_rows exceeds the featurizer's u64 range")

        self.split = split
        self.representation = representation
        self.codes = codes
        self.occurrence_codes = occurrence_codes
        self.ignored_codes = ignored_codes
        self.observation_window_hours = hours
        self.halves_split_hours = halves_split_hours
        self.timed = bool(timed)
        self.max_component_rows = max_component_rows
        self.max_chunk_bytes = max_chunk_bytes
        self.record_batch_rows = record_batch_rows
        self.feature_names = _feature_names(
            codes, occurrence_codes, representation, self.timed
        )
        self.feature_count = len(self.feature_names)
        if self.feature_count == 0 or self.feature_count > np.iinfo(np.uint32).max:
            raise ValueError("feature width is outside the featurizer's u32 range")
        self.offset_columns = _offset_columns(self.feature_names, occurrence_codes)
        index = _load_index_order(self.index_path, split)
        self.sample_ids = index.sample_ids
        self.labels = index.labels
        self.sorted_to_original = index.sorted_to_original
        self.row_count = len(self.sample_ids)
        if self.row_count > np.iinfo(np.int32).max:
            raise ValueError("split row count exceeds LightGBM's int32 range")

    def restore_prediction_order(self, predictions: np.ndarray) -> np.ndarray:
        return restore_original_order(predictions, self.sorted_to_original)

    def predictions_to_sorted_order(self, predictions: np.ndarray) -> np.ndarray:
        return to_sorted_sample_order(predictions, self.sorted_to_original)

    def _config(self) -> dict[str, object]:
        config: dict[str, object] = {
            "events_path": str(self.events_path),
            "index_path": str(self.index_path),
            "split": self.split,
            "representation": _FEATURIZER_REPRESENTATION_NAME[self.representation],
            "codes": list(self.codes),
            "occurrence_codes": list(self.occurrence_codes),
            "ignored_codes": list(self.ignored_codes),
            "observation_window_hours": self.observation_window_hours,
            "timed": self.timed,
            "max_component_rows": self.max_component_rows,
            "max_chunk_bytes": self.max_chunk_bytes,
            "record_batch_rows": self.record_batch_rows,
        }
        if self.representation in _SEGMENTED_REPRESENTATIONS:
            config["segment_kind"] = self.representation
            config["halves_split_hours"] = self.halves_split_hours
        return config


def featurizer_threads() -> int:
    """Thread count for featurization and LightGBM Dataset construction.

    Decoupled from ``MXGB_THREADS`` (the per-trial LightGBM thread count that
    model bytes depend on): featurized rows and bin mappers are identical for
    any thread count, so construction uses every core unless
    ``MXGB_FEATURIZER_THREADS`` says otherwise.
    """
    raw = os.environ.get("MXGB_FEATURIZER_THREADS", "").strip()
    if raw:
        value = int(raw)
        if value < 1:
            raise ValueError("MXGB_FEATURIZER_THREADS must be positive")
        return value
    return os.cpu_count() or 1
