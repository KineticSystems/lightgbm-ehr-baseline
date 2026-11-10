"""Exhaustively recreate the TD-ICU current-state author tensors from MEDS."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from export_source import (  # noqa: E402
    DELTA_TIME_PREFIX,
    DELTA_VALUE_PREFIX,
    TASK_ID,
    meds_feature_names,
)
from missing_xgboost_lightgbm.common_format import write_json  # noqa: E402
from missing_xgboost_lightgbm.landmark_sampling import (  # noqa: E402
    membership_mask,
    select_uniform_landmarks,
)
from cohort_source import (  # noqa: E402
    CONTEXT_LENGTH,
    CURRENT_BUILD_PATH,
    DATA_ROOT,
    LABEL_COLUMNS,
    SAMPLES_PATH,
    SOURCE_BUILD_PATH,
    UPSTREAM_COMMIT,
    current_files,
    read_json,
    sha256_file,
    source_feature_names,
)


class ArrowRowStream:
    def __init__(self, path: Path):
        self._iterator: Iterator[pa.RecordBatch] = iter(
            pq.ParquetFile(path).iter_batches(batch_size=250_000)
        )
        self._buffer: pa.Table | None = None

    def take(self, rows: int) -> pa.Table:
        pieces: list[pa.Table] = []
        remaining = rows
        while remaining:
            if self._buffer is None or len(self._buffer) == 0:
                try:
                    self._buffer = pa.Table.from_batches([next(self._iterator)])
                except StopIteration as error:
                    raise AssertionError("MEDS events ended before author traversal") from error
            take = min(remaining, len(self._buffer))
            pieces.append(self._buffer.slice(0, take))
            self._buffer = self._buffer.slice(take)
            remaining -= take
        return pa.concat_tables(pieces) if len(pieces) > 1 else pieces[0]

    def assert_exhausted(self) -> None:
        if self._buffer is not None and len(self._buffer):
            raise AssertionError("MEDS has events after exhaustive author traversal")
        try:
            next(self._iterator)
        except StopIteration:
            return
        raise AssertionError("MEDS has events after exhaustive author traversal")


def update_digest(digest: hashlib._Hash, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array)
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def list_matrix(frame: pl.DataFrame, name: str, dtype: np.dtype) -> np.ndarray:
    values = frame.select(name).explode(name).to_numpy().reshape(
        frame.height, CONTEXT_LENGTH
    )
    return values.astype(dtype, copy=False)


def author_row_positions(
    current: pl.DataFrame, token_source_row: np.ndarray
) -> np.ndarray:
    """Map persistent source-row coordinates onto a possibly sparse shard."""
    author_rows = current["source_sample_row"].to_numpy().astype(np.int64)
    if len(author_rows) > 1 and np.any(author_rows[1:] <= author_rows[:-1]):
        raise AssertionError("author shard source rows are not strictly increasing")
    source_samples = token_source_row // CONTEXT_LENGTH
    positions = np.searchsorted(author_rows, source_samples)
    valid = positions < len(author_rows)
    if not valid.all() or not np.array_equal(
        author_rows[positions], source_samples
    ):
        raise AssertionError("event source_row points outside its author shard")
    return positions


def reconstruct_primary_arrays(
    events: pl.DataFrame,
    current: pl.DataFrame,
    code_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = current.height
    feature = np.full((rows, CONTEXT_LENGTH), -1, dtype=np.int16)
    timepoint = np.full((rows, CONTEXT_LENGTH), -1, dtype=np.int16)
    value = np.full((rows, CONTEXT_LENGTH), np.nan, dtype=np.float32)
    ordered = (
        events.filter(pl.col("source_row") % 3 == 0).sort("source_row")
        .with_columns(
            pl.col("code")
            .replace_strict(code_to_id, return_dtype=pl.Int16)
            .alias("feature_id")
        )
    )
    token_source_row = ordered["source_row"].to_numpy().astype(np.int64) // 3
    row_index = author_row_positions(current, token_source_row)
    sequence = ordered["sequence_row"].to_numpy().astype(np.int64)
    if np.any(row_index < 0) or np.any(row_index >= rows):
        raise AssertionError("event source_row points outside its author shard")
    if len(np.unique(row_index * CONTEXT_LENGTH + sequence)) != len(ordered):
        raise AssertionError("MEDS duplicates an author token position")
    expected_token_source_row = (
        current["source_sample_row"].to_numpy().astype(np.int64)[row_index]
        * CONTEXT_LENGTH
        + sequence
    )
    if not np.array_equal(expected_token_source_row, token_source_row):
        raise AssertionError("MEDS source_row no longer encodes author row/token order")
    feature[row_index, sequence] = ordered["feature_id"].to_numpy().astype(np.int16)
    timepoint[row_index, sequence] = ordered["source_time"].to_numpy().astype(
        np.int16
    )
    value[row_index, sequence] = ordered["numeric_value"].to_numpy().astype(
        np.float32
    )
    return feature, timepoint, value


def reconstruct_delta_arrays(
    events: pl.DataFrame,
    current: pl.DataFrame,
    code_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    rows = current.height
    delta_time = np.full((rows, CONTEXT_LENGTH), -1, dtype=np.int16)
    delta_value = np.full((rows, CONTEXT_LENGTH), np.nan, dtype=np.float32)
    for channel, prefix, destination, dtype in (
        (1, DELTA_TIME_PREFIX, delta_time, np.int16),
        (2, DELTA_VALUE_PREFIX, delta_value, np.float32),
    ):
        selected = events.filter(pl.col("source_row") % 3 == channel).sort(
            "source_row"
        )
        if selected.is_empty():
            continue
        source_row = selected["source_row"].to_numpy().astype(np.int64)
        token_source_row = source_row // 3
        row_index = author_row_positions(current, token_source_row)
        sequence = selected["sequence_row"].to_numpy().astype(np.int64)
        if np.any(row_index < 0) or np.any(row_index >= rows):
            raise AssertionError("delta event points outside its author shard")
        if len(np.unique(row_index * CONTEXT_LENGTH + sequence)) != len(selected):
            raise AssertionError("MEDS duplicates an author delta token position")
        if selected.filter(~pl.col("code").str.starts_with(prefix)).height:
            raise AssertionError("MEDS delta code has the wrong channel prefix")
        source_ids = (
            selected
            .select(
                pl.col("code")
                .str.strip_prefix(prefix)
                .replace_strict(code_to_id, return_dtype=pl.Int16)
            )
            .to_numpy()
            .reshape(-1)
        )
        author_features = list_matrix(current, "encoded_feature", np.int16)
        if not np.array_equal(author_features[row_index, sequence], source_ids):
            raise AssertionError("MEDS delta code disagrees with author feature ID")
        values = selected["numeric_value"].to_numpy().astype(np.float32)
        if channel == 1:
            if not np.array_equal(values, values.astype(np.int16).astype(np.float32)):
                raise AssertionError("MEDS delta_time is not an exact int16 minute value")
        destination[row_index, sequence] = values.astype(dtype)
    return delta_time, delta_value


def author_event_count(current: pl.DataFrame) -> int:
    return int(current.select(
        (
            pl.col("encoded_feature")
            .list.eval((pl.element() >= 0).cast(pl.Int64))
            .list.sum()
            + pl.col("delta_time")
            .list.eval((pl.element() >= 0).cast(pl.Int64))
            .list.sum()
            + pl.col("delta_value")
            .list.eval(pl.element().is_finite().cast(pl.Int64))
            .list.sum()
        ).sum()
    ).item())


def assert_event_identity(actual: pl.DataFrame, current: pl.DataFrame) -> None:
    required = {
        "subject_id",
        "time",
        "code",
        "numeric_value",
        "sample_id",
        "source_row",
        "sequence_row",
        "source_time",
        "split",
    }
    if set(actual.columns) != required:
        raise AssertionError("MEDS event columns differ from selected source boundary")
    ordered = actual.sort("source_row")
    source_row = ordered["source_row"].to_numpy().astype(np.int64)
    token_source_row = source_row // 3
    row_index = author_row_positions(current, token_source_row)
    sequence = ordered["sequence_row"].to_numpy().astype(np.int64)
    if np.any(row_index < 0) or np.any(row_index >= current.height):
        raise AssertionError("event source_row points outside its exact author shard")
    if not np.array_equal(token_source_row % CONTEXT_LENGTH, sequence):
        raise AssertionError("event sequence_row differs from exact author token order")
    if not np.array_equal(
        ordered["subject_id"].to_numpy(),
        current["subject_id"].to_numpy()[row_index],
    ):
        raise AssertionError("MEDS changed author subject identity")
    if not np.array_equal(
        ordered["sample_id"].to_numpy(),
        current["sample_id"].to_numpy()[row_index],
    ):
        raise AssertionError("MEDS changed author final-row identity")
    if (
        ordered["split"].n_unique() != 1
        or current["split"].n_unique() != 1
        or ordered["split"].item(0) != current["split"].item(0)
    ):
        raise AssertionError("MEDS changed author source split")
    source_minutes = ordered["source_time"].to_numpy()
    if not np.array_equal(source_minutes, source_minutes.astype(np.int16)):
        raise AssertionError("MEDS changed exact integer relative-minute coordinate")
    expected_time = (
        current["labeltime"].to_numpy()[row_index].astype("datetime64[us]")
        - source_minutes.astype("timedelta64[m]")
    )
    if not np.array_equal(
        ordered["time"].to_numpy().astype("datetime64[us]"), expected_time
    ):
        raise AssertionError("MEDS wall-clock time does not recreate author relative time")


def check(task_dir: Path) -> dict:
    manifest_path = task_dir / "manifest.json"
    manifest_hash_before = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    task = manifest["task"]
    if task_dir.name != TASK_ID or task["id"] != TASK_ID:
        raise AssertionError("selected task identity changed")
    if task["modalities"] != ["structured"] or task["note_codes"] != []:
        raise AssertionError("TD-ICU current-state projection must be structured-only")
    if manifest["source"]["commit"] != UPSTREAM_COMMIT:
        raise AssertionError("upstream commit changed")
    if not manifest["source"].get("all_non_note_codes_are_intended_structured_inputs"):
        raise AssertionError("manifest no longer declares the full structured boundary")

    features = source_feature_names()
    codes = tuple(
        pl.read_parquet(task_dir / "metadata" / "codes.parquet")["code"].to_list()
    )
    expected_codes = meds_feature_names(features)
    if codes != expected_codes:
        raise AssertionError("MEDS code vocabulary/order differs from exact source channels")
    code_to_id = {name: index for index, name in enumerate(features)}
    source_index = pl.read_parquet(SAMPLES_PATH).sort("sample_id")
    expected_index, sampling_audit = select_uniform_landmarks(source_index)
    expected_index = expected_index.sort("sample_id")
    actual_index = pl.read_parquet(task_dir / "audit" / "task_index.parquet").sort(
        "sample_id"
    )
    if actual_index.schema != expected_index.schema or not actual_index.equals(
        expected_index
    ):
        raise AssertionError("task index differs from deterministic landmark subsample")
    if manifest["source"].get("sampling") != sampling_audit:
        raise AssertionError("manifest sampling record differs from deterministic contract")
    index_sample_ids = actual_index["sample_id"].to_numpy().astype(np.int64)
    if len(index_sample_ids) > 1 and np.any(index_sample_ids[1:] <= index_sample_ids[:-1]):
        raise AssertionError("sampled source IDs are not strictly increasing")
    index_labels = actual_index["boolean_value"].to_numpy().astype(np.uint8)

    current_record = read_json(CURRENT_BUILD_PATH)
    source_record = read_json(SOURCE_BUILD_PATH)
    if sha256_file(SAMPLES_PATH) != current_record["samples_sha256"]:
        raise AssertionError("exact author final-row sample index changed")
    source_hashes: dict[str, str] = {}
    event_stream = ArrowRowStream(task_dir / "data" / "events.parquet")
    digests = {
        field: hashlib.sha256()
        for field in (
            "features",
            "timepoints",
            "values",
            "delta_time",
            "delta_value",
            "labels_28_day",
            "sample_id",
        )
    }
    reconstructed_digests = {
        field: hashlib.sha256()
        for field in ("features", "timepoints", "values", "delta_time", "delta_value")
    }
    shapes: dict[str, set[tuple[tuple[int, ...], str]]] = defaultdict(set)
    observed_codes: set[str] = set()
    examples = 0
    events = 0
    shards = 0
    max_delta_value_abs = 0.0
    selected_ids = np.sort(index_sample_ids)

    for path in current_files():
        relative = str(path.relative_to(CURRENT_BUILD_PATH.parent))
        source_hashes[relative] = sha256_file(path)
        if current_record["parquet_sha256"].get(relative) != source_hashes[relative]:
            raise AssertionError(f"exact author current shard changed: {relative}")
        current = pl.read_parquet(path)
        keep = membership_mask(
            current["sample_id"].to_numpy().astype(np.int64, copy=False),
            selected_ids,
        )
        current = current.filter(pl.Series("__keep", keep))
        if current.is_empty():
            continue
        expected_event_rows = author_event_count(current)
        actual_table = event_stream.take(expected_event_rows)
        actual_events = pl.from_arrow(actual_table)
        assert_event_identity(actual_events, current)
        observed_codes.update(actual_events["code"].unique().to_list())

        recreated_features, recreated_timepoints, recreated_values = (
            reconstruct_primary_arrays(actual_events, current, code_to_id)
        )
        author_features = list_matrix(current, "encoded_feature", np.int16)
        author_timepoints = list_matrix(current, "featuretime", np.int16)
        author_values = list_matrix(current, "value", np.float32)
        author_delta_time = list_matrix(current, "delta_time", np.int16)
        author_delta_value = list_matrix(current, "delta_value", np.float32)
        if not np.array_equal(recreated_features, author_features):
            raise AssertionError("MEDS failed to recreate author feature IDs")
        if not np.array_equal(recreated_timepoints, author_timepoints):
            raise AssertionError("MEDS failed to recreate author relative times")
        if not np.array_equal(recreated_values, author_values, equal_nan=True):
            raise AssertionError("MEDS failed to recreate author values/padding")
        recreated_delta_time, recreated_delta_value = reconstruct_delta_arrays(
            actual_events,
            current,
            code_to_id,
        )
        if not np.array_equal(recreated_delta_time, author_delta_time):
            mismatch = np.argwhere(recreated_delta_time != author_delta_time)[0]
            raise AssertionError(
                "MEDS failed to recreate author delta_time at "
                f"row/token={tuple(int(value) for value in mismatch)}"
            )
        finite = np.isfinite(author_delta_value)
        if not np.array_equal(np.isfinite(recreated_delta_value), finite):
            raise AssertionError("MEDS changed author delta_value missingness")
        if finite.any():
            if not np.array_equal(
                recreated_delta_value[finite], author_delta_value[finite]
            ):
                raise AssertionError("MEDS changed exact author delta_value")
            max_delta_value_abs = max(
                max_delta_value_abs,
                float(
                    np.max(
                        np.abs(
                            recreated_delta_value[finite]
                            - author_delta_value[finite]
                        )
                    )
                ),
            )

        labels = current["28-day-died"].to_numpy().astype(np.uint8)
        sample_ids = current["sample_id"].to_numpy().astype(np.int64)
        positions = np.searchsorted(index_sample_ids, sample_ids)
        if not np.array_equal(
            index_labels[positions],
            labels,
        ):
            raise AssertionError("MEDS selected label differs from author 28-day tensor")

        author_arrays = {
            "features": author_features,
            "timepoints": author_timepoints,
            "values": author_values,
            "delta_time": author_delta_time,
            "delta_value": author_delta_value,
            "labels_28_day": labels,
            "sample_id": sample_ids,
        }
        recreated_arrays = {
            "features": recreated_features,
            "timepoints": recreated_timepoints,
            "values": recreated_values,
            "delta_time": recreated_delta_time,
            "delta_value": recreated_delta_value,
        }
        for name, array in author_arrays.items():
            update_digest(digests[name], array)
            shapes[name].add((tuple(array.shape), str(array.dtype)))
        for name, array in recreated_arrays.items():
            update_digest(reconstructed_digests[name], array)
        examples += current.height
        events += expected_event_rows
        shards += 1
        print(
            f"checked {relative}: {current.height} landmarks, "
            f"{expected_event_rows} retained tokens",
            flush=True,
        )

    event_stream.assert_exhausted()
    if examples != expected_index.height:
        raise AssertionError("exhaustive sampled author example count changed")
    if events != manifest["event_count"]:
        raise AssertionError("exhaustive author event count changed")
    if observed_codes != set(expected_codes):
        raise AssertionError(
            "observed MEDS codes differ from intended author inputs: "
            f"missing={sorted(set(expected_codes) - observed_codes)}, "
            f"extra={sorted(observed_codes - set(expected_codes))}"
        )

    report = {
        "status": "exact",
        "paper": "td_icu_mortality",
        "task": TASK_ID,
        "source_entrypoint": "released load_mimic/clean/label/encode/create_final_dataframe path captured immediately before HDF5 identifier loss",
        "selected_source_configuration": manifest["source"]["configuration"],
        "resource_limited_sampling": sampling_audit,
        "retained_projection": {
            "modalities": ["structured"],
            "event_codes": list(expected_codes),
            "note_codes": [],
            "all_non_note_codes_are_intended_structured_inputs": True,
            "excluded": [
                "padding tokens (recreated transiently)",
                "next-state tensors available only as privileged TD training targets",
                "all SICdb external-domain data",
                "all identifiers and non-selected labels as predictor events",
                "all imaging and note modalities (not consumed by this paper)",
            ],
        },
        "source_boundary": {
            "author_shape_per_example": [CONTEXT_LENGTH, 5],
            "channels": [
                "encoded_feature",
                "relative_time_minutes",
                "value",
                "delta_time_minutes",
                "delta_value",
            ],
            "padding": {
                "feature/time/delta_time": -1,
                "value/delta_value": "NaN",
                "persistent_meds": "padding omitted",
            },
            "ordering": "age, gender, weight; active drug-rate tokens; historic tokens by novelty then recency",
            "delta_reconstruction": {
                "delta_time": "exact int16 equality from explicit DELTA_TIME codes",
                "delta_value": "exact float32 equality from explicit DELTA_VALUE codes",
                "maximum_absolute_difference": max_delta_value_abs,
            },
        },
        "examples": examples,
        "full_source_examples": current_record["rows"],
        "events": events,
        "source_shards": shards,
        "batch_shapes_and_dtypes": {
            name: [
                {"shape": list(shape), "dtype": dtype}
                for shape, dtype in sorted(values)
            ]
            for name, values in shapes.items()
        },
        "ordered_author_field_sha256": {
            name: digest.hexdigest() for name, digest in digests.items()
        },
        "ordered_meds_recreation_sha256": {
            name: digest.hexdigest()
            for name, digest in reconstructed_digests.items()
        },
        "source_artifact_sha256": {
            "source_build.json": sha256_file(SOURCE_BUILD_PATH),
            "current_boundary.json": sha256_file(CURRENT_BUILD_PATH),
            **source_hashes,
        },
        "source_build_status": source_record["status"],
        "manifest_sha256": manifest_hash_before,
    }
    report_path = task_dir / "audit" / "full_batch_recreation.json"
    write_json(report_path, report)
    if sha256_file(manifest_path) != manifest_hash_before:
        raise AssertionError("batch checker changed immutable manifest.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True, type=Path)
    args = parser.parse_args()
    check(args.task_dir.resolve())


if __name__ == "__main__":
    main()
