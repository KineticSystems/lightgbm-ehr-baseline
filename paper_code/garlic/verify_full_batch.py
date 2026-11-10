"""Exhaustively prove MEDS recreates GARLIC's deterministic model boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


REPOSITORY = Path(__file__).resolve().parent
WORKSPACE = REPOSITORY.parent.parent
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(WORKSPACE / "lightgbm_code" / "src"))

from main import prepare_batch  # noqa: E402
from missing_xgboost_lightgbm.meds_format import MedsTask  # noqa: E402
from cohort_source import (  # noqa: E402
    EXPECTED_COHORT_ROWS,
    EXPECTED_FEATURES,
    MAX_MODEL_TIMESTAMPS,
    cohort_rows,
    open_complete_cache,
    ordered_split_hash,
    sha256_file,
)


PARQUET_BATCH_ROWS = 65_536
MODEL_BATCH_ROWS = 32
SOURCE_ROW_STRIDE = MAX_MODEL_TIMESTAMPS * EXPECTED_FEATURES


class MedsCell(NamedTuple):
    sequence: int
    feature: int
    numeric: float
    source_time: float
    source_row: int
    event_time_us: int


def tensor_digest(digest: hashlib._Hash, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous().numpy()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())


def _row_group_position_range(
    parquet: pq.ParquetFile,
    row_group: int,
    source_row_column: int,
) -> tuple[int, int]:
    statistics = parquet.metadata.row_group(row_group).column(
        source_row_column
    ).statistics
    if statistics is None or not statistics.has_min_max:
        raise AssertionError("events.parquet lacks source_row row-group statistics")
    minimum = int(statistics.min)
    maximum = int(statistics.max)
    if minimum < 0 or maximum < minimum:
        raise AssertionError("events.parquet has an invalid source_row range")
    return minimum // SOURCE_ROW_STRIDE, maximum // SOURCE_ROW_STRIDE


def stream_meds_samples(
    event_path: Path,
    rows: list[dict],
    features: tuple[str, ...],
    code_counts: Counter,
    statistics: dict[str, int],
) -> Iterator[tuple[dict, list[MedsCell]]]:
    """Yield every sample in source order using bounded Parquet record batches."""
    parquet = pq.ParquetFile(event_path)
    source_row_column = parquet.schema_arrow.get_field_index("source_row")
    if source_row_column < 0:
        raise AssertionError("events.parquet omits source_row")
    position_ranges = [
        _row_group_position_range(parquet, group, source_row_column)
        for group in range(parquet.num_row_groups)
    ]
    for previous, current in zip(position_ranges, position_ranges[1:]):
        if current[0] < previous[1]:
            raise AssertionError(
                "Parquet row groups do not preserve source-position order"
            )

    columns = [
        "subject_id",
        "time",
        "code",
        "numeric_value",
        "sample_id",
        "source_row",
        "sequence_row",
        "source_time",
        "split",
    ]
    pending: dict[int, list[MedsCell]] = {}
    next_position = 0
    total_rows = 0
    record_batches = 0

    for row_group, (_, maximum_position) in enumerate(position_ranges):
        for batch in parquet.iter_batches(
            batch_size=PARQUET_BATCH_ROWS,
            row_groups=[row_group],
            columns=columns,
            use_threads=False,
        ):
            record_batches += 1
            total_rows += batch.num_rows
            if any(batch.column(name).null_count for name in columns):
                raise AssertionError("GARLIC MEDS events contain a null boundary field")
            subjects = batch.column("subject_id").to_numpy(zero_copy_only=False)
            event_times = (
                batch.column("time")
                .cast(pa.int64())
                .to_numpy(zero_copy_only=False)
            )
            codes = batch.column("code").to_pylist()
            numerics = batch.column("numeric_value").to_numpy(zero_copy_only=False)
            sample_ids = batch.column("sample_id").to_numpy(zero_copy_only=False)
            source_rows = batch.column("source_row").to_numpy(zero_copy_only=False)
            sequences = batch.column("sequence_row").to_numpy(zero_copy_only=False)
            source_times = batch.column("source_time").to_numpy(zero_copy_only=False)
            splits = batch.column("split").to_pylist()
            code_counts.update(codes)

            for (
                subject,
                event_time,
                code,
                numeric,
                sample_id,
                source_row,
                sequence,
                source_time,
                split,
            ) in zip(
                subjects,
                event_times,
                codes,
                numerics,
                sample_ids,
                source_rows,
                sequences,
                source_times,
                splits,
                strict=True,
            ):
                source_row = int(source_row)
                position, remainder = divmod(source_row, SOURCE_ROW_STRIDE)
                expected_sequence, feature = divmod(remainder, EXPECTED_FEATURES)
                if position < 0 or position >= len(rows):
                    raise AssertionError("MEDS source_row refers outside the cohort")
                expected = rows[position]
                if int(sample_id) != int(expected["hadm_id"]):
                    raise AssertionError("MEDS sample_id differs from source_row")
                if int(subject) != int(expected["subject_id"]):
                    raise AssertionError("MEDS subject_id differs from source cohort")
                if str(split) != str(expected["split"]):
                    raise AssertionError("MEDS event split differs from source cohort")
                if int(sequence) != expected_sequence:
                    raise AssertionError("MEDS sequence_row differs from source_row")
                if expected_sequence >= MAX_MODEL_TIMESTAMPS:
                    raise AssertionError("MEDS sequence_row exceeds released padding")
                if str(code) != features[feature]:
                    raise AssertionError("MEDS code differs from source_row vocabulary")
                pending.setdefault(position, []).append(
                    MedsCell(
                        sequence=expected_sequence,
                        feature=feature,
                        numeric=float(numeric),
                        source_time=float(source_time),
                        source_row=source_row,
                        event_time_us=int(event_time),
                    )
                )

        boundary = (
            position_ranges[row_group + 1][0]
            if row_group + 1 < len(position_ranges)
            else len(rows)
        )
        if boundary < maximum_position:
            raise AssertionError("next Parquet row group moves backward")
        while next_position < boundary:
            yield rows[next_position], pending.pop(next_position, [])
            next_position += 1

    while next_position < len(rows):
        yield rows[next_position], pending.pop(next_position, [])
        next_position += 1
    if pending:
        raise AssertionError("MEDS stream retained events outside the source cohort")
    if total_rows != parquet.metadata.num_rows:
        raise AssertionError("batch checker did not traverse every Parquet event")
    statistics.update(
        {
            "events": total_rows,
            "row_groups": parquet.num_row_groups,
            "record_batches": record_batches,
            "record_batch_rows": PARQUET_BATCH_ROWS,
        }
    )


def compare_batch(
    connection,
    batch_rows: list[dict],
    batch_events: list[list[MedsCell]],
    normal: dict[int, tuple[float, float]],
    digests: dict[str, hashlib._Hash],
) -> float:
    batch_hadm = [int(row["hadm_id"]) for row in batch_rows]
    placeholders = ",".join("?" for _ in batch_hadm)
    source_cells = connection.execute(
        f"SELECT hadm_id, sequence_row, feature_idx, raw_value, source_time, "
        f"event_time_us FROM raw_cells WHERE hadm_id IN ({placeholders}) "
        "ORDER BY hadm_id, sequence_row, feature_idx",
        batch_hadm,
    ).fetchall()
    by_hadm: dict[int, list[tuple[int, int, float, float, int]]] = {
        hadm: [] for hadm in batch_hadm
    }
    for hadm, sequence, feature, raw, source_time, event_time in source_cells:
        by_hadm[int(hadm)].append(
            (
                int(sequence),
                int(feature),
                float(raw),
                float(source_time),
                int(event_time),
            )
        )

    source_dicts = []
    meds_arr = np.zeros(
        (len(batch_rows), MAX_MODEL_TIMESTAMPS, EXPECTED_FEATURES),
        dtype=np.float32,
    )
    meds_mask = np.zeros_like(meds_arr)
    meds_time = np.zeros(
        (MAX_MODEL_TIMESTAMPS, len(batch_rows)), dtype=np.float32
    )
    meds_label = np.asarray(
        [row["label"] for row in batch_rows], dtype=np.float32
    )

    for local, (row, actual_cells) in enumerate(
        zip(batch_rows, batch_events, strict=True)
    ):
        hadm = int(row["hadm_id"])
        expected_cells = by_hadm[hadm]
        if len(actual_cells) != len(expected_cells):
            raise AssertionError("MEDS cell count differs from source mask cells")
        source_arr = np.zeros(
            (MAX_MODEL_TIMESTAMPS, EXPECTED_FEATURES), dtype=np.float64
        )
        source_mask = np.zeros_like(source_arr)
        source_time = np.zeros((MAX_MODEL_TIMESTAMPS, 1), dtype=np.float64)
        previous_source_row = -1

        for actual, expected in zip(actual_cells, expected_cells, strict=True):
            sequence, feature, raw, scaled_time, event_time = expected
            mean_value, std_value = normal[feature]
            normalized = raw - mean_value
            if std_value != 0:
                normalized /= std_value
            normalized_float32 = np.float32(normalized).item()

            if (actual.sequence, actual.feature) != (sequence, feature):
                raise AssertionError("MEDS cell order differs from source tensor order")
            if actual.source_row <= previous_source_row:
                raise AssertionError("MEDS source_row is not strictly ordered per sample")
            previous_source_row = actual.source_row
            if actual.numeric != normalized_float32:
                raise AssertionError("MEDS numeric value differs from source boundary")
            if actual.source_time != scaled_time:
                raise AssertionError("MEDS source_time differs from source boundary")
            if actual.event_time_us != event_time:
                raise AssertionError("MEDS event timestamp differs from source boundary")

            source_arr[sequence, feature] = normalized
            source_mask[sequence, feature] = 1.0
            source_time[sequence, 0] = scaled_time * 60.0
            meds_arr[local, sequence, feature] = np.float32(actual.numeric)
            meds_mask[local, sequence, feature] = 1.0
            candidate_time = np.float32(actual.source_time)
            previous_time = meds_time[sequence, local]
            if previous_time != 0 and previous_time != candidate_time:
                raise AssertionError("codes at one sequence row disagree on time")
            meds_time[sequence, local] = candidate_time

        source_dicts.append(
            {
                "arr": source_arr,
                "mask": source_mask,
                "time": source_time,
                "label": row["label"],
            }
        )

    source_arr, source_mask, source_time, source_label = prepare_batch(
        source_dicts, "cpu"
    )
    recreated = {
        "values": torch.from_numpy(meds_arr),
        "mask": torch.from_numpy(meds_mask),
        "time": torch.from_numpy(meds_time),
        "label": torch.from_numpy(meds_label),
    }
    source = {
        "values": source_arr,
        "mask": source_mask,
        "time": source_time,
        "label": source_label,
    }
    maximum_difference = 0.0
    for name in source:
        left = source[name].detach().cpu()
        right = recreated[name]
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            raise AssertionError(f"{name} dtype/shape differs")
        if not torch.equal(left, right):
            difference = float(torch.max(torch.abs(left - right)).item())
            maximum_difference = max(maximum_difference, difference)
            raise AssertionError(f"{name} differs; maximum={difference}")
        tensor_digest(digests[name], left)
    return maximum_difference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    task = MedsTask.open(args.task_dir)
    manifest_path = args.task_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha256 = sha256_file(manifest_path)
    if manifest["paper"]["key"] != "garlic":
        raise AssertionError("batch checker opened the wrong paper task")
    if manifest["task"]["modalities"] != ["structured"]:
        raise AssertionError("GARLIC is a structured-only source experiment")
    if manifest["task"]["note_codes"] != []:
        raise AssertionError("GARLIC cannot expose note codes")
    if manifest["source"]["selected_model_inputs"] != ["structured"]:
        raise AssertionError("selected GARLIC source declaration changed")

    connection, summary = open_complete_cache(args.cache_path)
    try:
        features = tuple(summary["features"])
        exported_codes = tuple(
            pq.read_table(
                args.task_dir / manifest["meds"]["codes"]["file"],
                columns=["code"],
            )["code"].to_pylist()
        )
        if exported_codes != features:
            raise AssertionError("MEDS code vocabulary/order differs from GARLIC")
        if len(features) != EXPECTED_FEATURES:
            raise AssertionError("GARLIC must retain exactly 103 channels")
        rows = cohort_rows(connection)
        if len(rows) != EXPECTED_COHORT_ROWS:
            raise AssertionError("GARLIC cache cohort changed")

        normal = {
            int(row[0]): (float(row[1]), float(row[2]))
            for row in connection.execute(
                "SELECT feature_idx, mean_value, std_value FROM normalization"
            )
        }
        if len(normal) != EXPECTED_FEATURES:
            raise AssertionError("GARLIC normalization vocabulary changed")
        expected_code_counts = {
            features[int(feature)]: int(count)
            for feature, count in connection.execute(
                "SELECT feature_idx, COUNT(*) FROM raw_cells GROUP BY feature_idx"
            )
        }

        digests = {
            name: hashlib.sha256()
            for name in ("values", "mask", "time", "label")
        }
        ordered_id_label = hashlib.sha256()
        actual_code_counts: Counter = Counter()
        stream_statistics: dict[str, int] = {}
        batches = 0
        examples = 0
        maximum_difference = 0.0
        split_batches = Counter()
        batch_rows: list[dict] = []
        batch_events: list[list[MedsCell]] = []
        event_path = args.task_dir / manifest["meds"]["events"]["file"]

        for row, cells in stream_meds_samples(
            event_path,
            rows,
            features,
            actual_code_counts,
            stream_statistics,
        ):
            batch_rows.append(row)
            batch_events.append(cells)
            hadm = int(row["hadm_id"])
            ordered_id_label.update(
                np.asarray(
                    [hadm, row["subject_id"], row["label"]], dtype=np.int64
                ).tobytes()
            )
            ordered_id_label.update(row["split"].encode() + b"\0")
            split_batches[row["split"]] += 1
            examples += 1
            if len(batch_rows) == MODEL_BATCH_ROWS:
                maximum_difference = max(
                    maximum_difference,
                    compare_batch(
                        connection,
                        batch_rows,
                        batch_events,
                        normal,
                        digests,
                    ),
                )
                batches += 1
                batch_rows = []
                batch_events = []

        if batch_rows:
            maximum_difference = max(
                maximum_difference,
                compare_batch(
                    connection,
                    batch_rows,
                    batch_events,
                    normal,
                    digests,
                ),
            )
            batches += 1

        if examples != EXPECTED_COHORT_ROWS:
            raise AssertionError("batch checker did not traverse the full cohort")
        if stream_statistics["events"] != int(summary["retained_events"]):
            raise AssertionError("MEDS retained an auxiliary or omitted a source row")
        if actual_code_counts != Counter(expected_code_counts):
            raise AssertionError("MEDS code counts differ from source mask cells")
        if ordered_split_hash(connection) != manifest["source"][
            "ordered_identifier_label_split_sha256"
        ]:
            raise AssertionError("source identifier/label/split hash changed")

        report = {
            "status": "exact",
            "manifest_sha256_before_and_after": manifest_sha256,
            "source_boundary": "main.prepare_batch retained structured projection",
            "examples": examples,
            "batches": batches,
            "batch_size": MODEL_BATCH_ROWS,
            "parquet_stream": stream_statistics,
            "split_examples": dict(split_batches),
            "vocabulary_size": len(features),
            "vocabulary_sha256": manifest["source"]["feature_order_sha256"],
            "note_codes": [],
            "retained_event_count": stream_statistics["events"],
            "per_code_counts_sha256": hashlib.sha256(
                json.dumps(expected_code_counts, sort_keys=True).encode()
            ).hexdigest(),
            "ordered_identifier_label_sha256": ordered_id_label.hexdigest(),
            "tensor_sha256": {
                name: digest.hexdigest() for name, digest in digests.items()
            },
            "tensor_shapes_and_dtypes": {
                "values": ["batch,200,103", "torch.float32"],
                "mask": ["batch,200,103", "torch.float32"],
                "time": ["200,batch", "torch.float32"],
                "label": ["batch", "torch.float32"],
            },
            "maximum_absolute_difference": maximum_difference,
            "padding": "exact zero padding through sequence row 199",
            "missingness": (
                "exact released positive-value mask; absent MEDS cells are mask zero"
            ),
            "modalities": {
                "retained": ["structured"],
                "notes": [],
                "imaging": [],
                "auxiliary_event_rows": 0,
            },
            "checks": [
                "every Parquet event streamed once in bounded record batches",
                "every source example traversed in deterministic order",
                "elementwise values, mask, timestamp, and label equality",
                "exact event order, source_row, subject, sample, and split equality",
                "exact dtype, shape, padding, and missingness equality",
                "exact ordered 103-code vocabulary and per-code row counts",
                "every non-note code is a released GARLIC structured channel",
                "no static, note, imaging, or archival-only event entered MEDS",
            ],
        }
        report_path = args.task_dir / "audit" / "full_batch_recreation.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if sha256_file(manifest_path) != manifest_sha256:
            raise AssertionError("batch checker mutated manifest.json")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
