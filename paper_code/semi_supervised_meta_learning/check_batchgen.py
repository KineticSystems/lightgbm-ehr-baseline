#!/usr/bin/env python3
"""Exhaustively exercise SSML's released BatchGen after MEDS stream parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch

import data
import utils
from check_batch import (
    BATCH_SIZE,
    build_author_components,
    direct_author_maximum,
    row_prefix_fingerprints,
    sha256_file,
    write_json,
)
from cohort_source import _group_samples, assert_source_ready, build_samples


PROCESS_CHUNK = 64


class TensorReader:
    """Released-reader interface over already exhaustively checked tensors."""

    def __init__(self, records: list[dict[str, object]]) -> None:
        self._data = list(records)
        self._current_index = 0

    def get_number_of_examples(self) -> int:
        return len(self._data)

    def random_shuffle(self, seed=None) -> None:
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_next(self) -> dict[str, object]:
        record = self._data[self._current_index]
        self._current_index = (self._current_index + 1) % len(self._data)
        return {
            "X": record["tensor64"],
            "t": record["source_cutoff_hour"],
            "y": record["boolean_value"],
            "name": record["source_filename"],
            "header": "already_checked_author_tensor",
        }


def update_array_digest(digest: hashlib._Hash, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def execute_chunk(
    records: list[dict[str, object]], seed: int
) -> tuple[dict[str, object], dict[int, int]]:
    batch_size = BATCH_SIZE if len(records) % BATCH_SIZE == 0 else len(records)
    if batch_size <= 0 or len(records) % batch_size:
        raise AssertionError("BatchGen diagnostic chunk cannot form complete batches")
    steps = len(records) // batch_size
    generator = data.BatchGen(
        reader=TensorReader(records),
        discretizer=None,
        normalizer=None,
        batch_size=batch_size,
        steps=steps,
        shuffle=True,
    )

    captured: dict[str, object] = {}
    selected_columns: list[int] = []
    original_preprocess = utils.preprocess_chunk
    original_sort = utils.sort_and_shuffle
    original_randint = random.randint

    def checked_identity(values, cutoffs, discretizer, normalizer):
        if discretizer is not None or normalizer is not None:
            raise AssertionError("tensor handoff unexpectedly received a transformer")
        if len(values) != len(cutoffs):
            raise AssertionError("BatchGen tensor/cutoff handoff differs")
        return values

    def capture_sort(values, actual_batch_size):
        if actual_batch_size != batch_size:
            raise AssertionError("released BatchGen changed its batch size")
        output = original_sort(values, actual_batch_size)
        captured["processed"] = output
        return output

    def capture_randint(begin, end):
        value = original_randint(begin, end)
        selected_columns.append(value)
        return value

    utils.preprocess_chunk = checked_identity
    utils.sort_and_shuffle = capture_sort
    random.randint = capture_randint
    try:
        random.seed(seed)
        np.random.seed(seed)
        data_all, data_aug = generator._generator()
    finally:
        utils.preprocess_chunk = original_preprocess
        utils.sort_and_shuffle = original_sort
        random.randint = original_randint

    processed_x, processed_y, processed_t, processed_names = captured["processed"]
    if (
        len(processed_x) != len(records)
        or len(processed_y) != len(records)
        or len(processed_t) != len(records)
        or len(processed_names) != len(records)
        or len(data_all) != len(records)
    ):
        raise AssertionError("released BatchGen changed the exhaustive chunk size")
    key_to_record = {
        (
            str(record["source_filename"]),
            int(record["source_cutoff_hour"]),
            int(record["boolean_value"]),
        ): record
        for record in records
    }
    if len(key_to_record) != len(records):
        raise AssertionError("BatchGen chunk has a duplicate filename/cutoff/label key")

    processed_digest = hashlib.sha256()
    window_digest = hashlib.sha256()
    augmented_digest = hashlib.sha256()
    selected_histogram = Counter()
    seen_ids: dict[int, int] = {}
    augmentation_position = 0
    windows_total = 0
    model_cells = 0
    for position, (value, label, cutoff, name) in enumerate(
        zip(processed_x, processed_y, processed_t, processed_names)
    ):
        key = (str(name), int(round(float(cutoff))), int(label))
        record = key_to_record.get(key)
        if record is None:
            raise AssertionError("BatchGen returned an unknown exhaustive source row")
        sample_id = int(record["sample_id"])
        if sample_id in seen_ids:
            raise AssertionError("BatchGen returned an exhaustive source row twice")
        seen_ids[sample_id] = 1
        expected64 = np.asarray(record["tensor64"], dtype=np.float64)
        expected32 = np.asarray(record["tensor32"], dtype=np.float32)
        actual64 = np.asarray(value, dtype=np.float64)
        if not np.array_equal(actual64, expected64):
            raise AssertionError("BatchGen tensor handoff differs from author loader")
        actual32 = torch.Tensor(actual64).cpu().numpy()
        if not np.array_equal(actual32, expected32):
            raise AssertionError("actual torch.Tensor cast differs from the audited float32 boundary")
        retained = (len(expected64) // 24) * 24
        expected_windows64 = list(np.reshape(expected64[-retained:], (-1, 24, 76)))
        expected_windows32 = list(np.reshape(expected32[-retained:], (-1, 24, 76)))
        actual_windows, actual_label, actual_true = data_all[position]
        if len(actual_windows) != len(expected_windows64):
            raise AssertionError("BatchGen 24-hour window count differs")
        if len(expected_windows32) != len(expected_windows64):
            raise AssertionError("float32/float64 expected window counts differ")
        for actual_window, wanted64, wanted32 in zip(
            actual_windows, expected_windows64, expected_windows32
        ):
            actual_window64 = np.asarray(actual_window, dtype=np.float64)
            if not np.array_equal(actual_window64, wanted64):
                raise AssertionError("BatchGen float64 window values differ")
            if not np.array_equal(torch.Tensor(actual_window64).numpy(), wanted32):
                raise AssertionError("BatchGen torch-cast window values differ")
            update_array_digest(window_digest, wanted32)
        if int(actual_label) != int(label) or int(actual_true) != int(label):
            raise AssertionError("BatchGen label propagation differs")
        for wanted64 in expected_windows64[:-1]:
            column = selected_columns[augmentation_position]
            wanted_augmented = wanted64.copy()
            wanted_augmented[:, column] = 0
            actual_augmented = np.asarray(data_aug[augmentation_position], dtype=np.float64)
            if not np.array_equal(actual_augmented, wanted_augmented):
                raise AssertionError("BatchGen random single-channel augmentation differs")
            selected_histogram[column] += 1
            update_array_digest(augmented_digest, actual_augmented.astype(np.float32))
            augmentation_position += 1
        update_array_digest(processed_digest, actual32)
        windows_total += len(expected_windows64)
        model_cells += int(actual32.size)
    if augmentation_position != len(data_aug) or len(selected_columns) != len(data_aug):
        raise AssertionError("BatchGen augmentation cardinality differs")
    if set(seen_ids) != {int(record["sample_id"]) for record in records}:
        raise AssertionError("BatchGen exhaustive chunk membership differs")
    return (
        {
            "examples": len(records),
            "batch_size": batch_size,
            "batches": steps,
            "windows_24h": windows_total,
            "augmented_windows": len(data_aug),
            "model_cells": model_cells,
            "processed_float32_sha256": processed_digest.hexdigest(),
            "window_float32_sha256": window_digest.hexdigest(),
            "augmented_float32_sha256": augmented_digest.hexdigest(),
        },
        dict(selected_histogram),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    args = parser.parse_args()

    paths = assert_source_ready(args.source_root)
    manifest_path = args.task_dir / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    stream_path = args.task_dir / "audit" / "full_batch_stream_recreation.json"
    stream = json.loads(stream_path.read_text())
    if stream.get("status") != "stream_exact":
        raise AssertionError("exhaustive MEDS/source stream phase did not pass")
    if stream.get("manifest_sha256") != manifest_sha256:
        raise AssertionError("stream phase used a different immutable manifest")
    # Phase 1 is intentionally reusable after a Phase-2-only process failure,
    # but only while its entire persistent input inventory is unchanged.  Hash
    # every manifested MEDS artifact afresh, including the event file, before
    # trusting the cached exhaustive stream reconciliation.
    for artifact, inventory in manifest["meds"].items():
        artifact_path = args.task_dir / inventory["file"]
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"manifested MEDS artifact is missing before BatchGen: {artifact}"
            )
        if sha256_file(artifact_path) != inventory["sha256"]:
            raise AssertionError(
                f"manifested MEDS artifact changed after stream phase: {artifact}"
            )
    if (
        stream["source_artifacts"]["common_events_sha256"]
        != manifest["meds"]["events"]["sha256"]
    ):
        raise AssertionError("stream report event hash differs from the manifest")
    if (
        sum(
            int(values["raw_occurrence_events"])
            for values in stream["exhaustive_recreation"].values()
        )
        != int(manifest["event_count"])
    ):
        raise AssertionError("stream report event inventory differs from the manifest")
    if (
        sum(
            int(values["examples"])
            for values in stream["exhaustive_recreation"].values()
        )
        != int(stream["batchgen_handoff"]["examples"])
    ):
        raise AssertionError("stream report sample inventory is internally inconsistent")

    samples, sample_audit = build_samples(args.source_root)
    index = pl.read_parquet(args.task_dir / manifest["meds"]["task_index"]["file"])
    expected_columns = (
        "subject_id", "sample_id", "prediction_time", "boolean_value", "split",
        "source_filename", "source_list_row", "source_cutoff_hour", "source_order",
    )
    source_index = samples.select(expected_columns).with_columns(
        pl.col("split").cast(pl.String)
    ).sort("sample_id")
    task_index = index.select(expected_columns).with_columns(
        pl.col("split").cast(pl.String)
    ).sort("sample_id")
    if not source_index.equals(
        task_index
    ):
        raise AssertionError("BatchGen phase task index differs from source membership")
    dataset_dirs, discretizer, normalizer, header = build_author_components(paths)
    if len(header.split(",")) != 76:
        raise AssertionError("released loader header changed")

    fingerprints = np.zeros((samples.height + 1, 2), dtype=np.uint64)
    split_reports = {
        split: {
            "examples": 0,
            "batches": 0,
            "full_batches": 0,
            "tail_batch_size": 0,
            "windows_24h": 0,
            "augmented_windows": 0,
            "model_cells": 0,
            "processed_chunk_digest": hashlib.sha256(),
            "window_chunk_digest": hashlib.sha256(),
            "augmented_chunk_digest": hashlib.sha256(),
            "augmentation_column_histogram": Counter(),
        }
        for split in ("train", "validation", "test")
    }
    buffers = {split: [] for split in split_reports}
    chunk_number = 0

    def flush(split: str, *, final: bool = False) -> None:
        nonlocal chunk_number
        records = buffers[split]
        while len(records) >= PROCESS_CHUNK or (final and records):
            if len(records) >= PROCESS_CHUNK:
                size = PROCESS_CHUNK
            elif len(records) >= BATCH_SIZE:
                size = (len(records) // BATCH_SIZE) * BATCH_SIZE
            else:
                size = len(records)
            selected = records[:size]
            del records[:size]
            chunk_number += 1
            values, columns = execute_chunk(selected, seed=1_000_000 + chunk_number)
            report = split_reports[split]
            report["examples"] += values["examples"]
            report["batches"] += values["batches"]
            if values["batch_size"] == BATCH_SIZE:
                report["full_batches"] += values["batches"]
            else:
                if report["tail_batch_size"]:
                    raise AssertionError(f"{split} produced more than one tail batch")
                report["tail_batch_size"] = values["batch_size"]
            report["windows_24h"] += values["windows_24h"]
            report["augmented_windows"] += values["augmented_windows"]
            report["model_cells"] += values["model_cells"]
            report["processed_chunk_digest"].update(
                bytes.fromhex(values["processed_float32_sha256"])
            )
            report["window_chunk_digest"].update(
                bytes.fromhex(values["window_float32_sha256"])
            )
            report["augmented_chunk_digest"].update(
                bytes.fromhex(values["augmented_float32_sha256"])
            )
            report["augmentation_column_histogram"].update(columns)
            if chunk_number % 1_000 == 0:
                checked = sum(value["examples"] for value in split_reports.values())
                print(f"BatchGen checked {checked:,} / {samples.height:,} examples")

    for split, filename, group in _group_samples(samples):
        maximum = group.row(-1, named=True)
        direct64, direct32 = direct_author_maximum(
            dataset_dirs[split], discretizer, normalizer, maximum
        )
        prefix_first, prefix_second = row_prefix_fingerprints(direct32)
        for row in group.iter_rows(named=True):
            cutoff = int(row["source_cutoff_hour"])
            sample_id = int(row["sample_id"])
            fingerprints[sample_id, 0] = prefix_first[cutoff]
            fingerprints[sample_id, 1] = prefix_second[cutoff]
            buffers[split].append(
                {
                    **row,
                    "tensor64": direct64[:cutoff],
                    "tensor32": direct32[:cutoff],
                }
            )
            flush(split)
    for split in split_reports:
        flush(split, final=True)

    serializable = {}
    for split, values in split_reports.items():
        ordered = index.filter(pl.col("split") == split).sort("source_order")
        ordered_ids = ordered["sample_id"].to_numpy().astype("<i8", copy=False)
        ordered_cutoffs = ordered["source_cutoff_hour"].to_numpy().astype("<i8", copy=False)
        ordered_labels = ordered["boolean_value"].to_numpy().astype(np.uint8, copy=False)
        digest = hashlib.sha256()
        digest.update(ordered_ids.tobytes())
        digest.update(ordered_cutoffs.tobytes())
        digest.update(ordered_labels.tobytes())
        digest.update(np.ascontiguousarray(fingerprints[ordered_ids]).tobytes())
        expected = stream["batchgen_handoff"]["expected_per_split"][split]
        if values["examples"] != expected["examples"]:
            raise AssertionError(f"{split} BatchGen example count differs from stream phase")
        if values["windows_24h"] != expected["windows_24h"]:
            raise AssertionError(f"{split} BatchGen window count differs from stream phase")
        if digest.hexdigest() != expected["ordered_tensor_fingerprint_sha256"]:
            raise AssertionError(f"{split} BatchGen tensor fingerprints differ from stream phase")
        if values["augmented_windows"] != values["windows_24h"] - values["examples"]:
            raise AssertionError(f"{split} BatchGen augmentation count differs")
        serializable[split] = {
            "examples": values["examples"],
            "batches": values["batches"],
            "full_batches": values["full_batches"],
            "tail_batch_size": values["tail_batch_size"],
            "windows_24h": values["windows_24h"],
            "augmented_windows": values["augmented_windows"],
            "model_cells": values["model_cells"],
            "processed_chunk_hashes_sha256": values["processed_chunk_digest"].hexdigest(),
            "window_chunk_hashes_sha256": values["window_chunk_digest"].hexdigest(),
            "augmented_chunk_hashes_sha256": values["augmented_chunk_digest"].hexdigest(),
            "augmentation_column_histogram": {
                str(column): int(count)
                for column, count in sorted(values["augmentation_column_histogram"].items())
            },
            "ordered_tensor_fingerprint_sha256": digest.hexdigest(),
        }

    if sum(values["examples"] for values in serializable.values()) != samples.height:
        raise AssertionError("released BatchGen did not cover every eligible sample")
    final = dict(stream)
    final["status"] = "exact"
    final["protocol_status_meaning"] = (
        "all persistent MEDS/source fields and every released BatchGen example passed; "
        "numeric trainer cells use only the frozen exhaustive integer-ULP rule"
    )
    final.pop("batchgen_handoff")
    final["released_batchgen"] = {
        "coverage": "all eligible samples",
        "examples": samples.height,
        "stay_groups": sum(
            values["stays"] for values in sample_audit["splits"].values()
        ),
        "nominal_batch_size": BATCH_SIZE,
        "processing_chunk_examples": PROCESS_CHUNK,
        "preprocessing_handoff": (
            "released reader/discretizer/normalizer maxima verified in the stream phase; "
            "the actual released BatchGen receives every exact landmark prefix through "
            "an instrumented identity handoff to avoid redundant 2.67-million-file reads"
        ),
        "fixed_rng": "checker seeds each processing chunk; the paper's realized RNG is unrecoverable",
        "splits": serializable,
        "semantics": [
            "actual released random reader shuffle and sort_and_shuffle ordering",
            "actual trailing full 24-hour window reshape for every sample",
            "actual label propagation for every sample",
            "actual all-but-last unlabeled-window branch for every sample",
            "actual random single-channel zeroing for every augmented window",
            "actual torch.Tensor float32 cast for every processed tensor and window",
        ],
    }
    final["checks"] = list(final["checks"]) + [
        "actual released BatchGen and torch boundary for all 2,670,178 eligible samples",
        "exact reconciliation of per-split sample/window counts and tensor fingerprints across both guarded phases",
    ]
    final["source_artifacts"] = {
        **final["source_artifacts"],
        "stream_recreation_sha256": sha256_file(stream_path),
    }
    output = args.task_dir / "audit" / "full_batch_recreation.json"
    write_json(output, final)
    if sha256_file(manifest_path) != manifest_sha256:
        raise AssertionError("BatchGen checker modified immutable manifest.json")
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
