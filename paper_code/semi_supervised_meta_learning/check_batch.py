#!/usr/bin/env python3
"""Exhaustively recreate SSML's reader, transform, and BatchGen boundary.

The common task contains the earliest attributable raw structured occurrence
stream.  This checker compares every duplicated landmark prefix with that raw
stream, applies the released discretizer/normalizer, and verifies the
``[T, 76]`` tensor and 24-hour window semantics.  Categorical values, masks,
and every nonnumeric field remain bit-exact; the twelve numeric trainer
channels use an exhaustively measured integer-ULP rule solely for the required
source-float64 to MEDS-float32 round trip.  A separate guarded process then
exercises the released ``BatchGen`` implementation over every eligible sample,
including its shuffle, sort, window, and random single-channel augmentation
behavior, without importing PyTorch into this exhaustive Parquet scan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

import preprocessing
import reader
from cohort_source import (
    ANCHOR_US,
    CONFIG_PATH,
    EPS,
    HOUR_US,
    NORMALIZER_PATH,
    SOURCE_SPLIT,
    assert_source_ready,
    load_raw_stay,
    load_transform_spec,
    reconstruct_model_tensor,
    sha256_file,
    stay_path,
)


MASK64 = (1 << 64) - 1

# Fixed after an exhaustive measurement over all 34,960 attributable stay
# maxima (41,700,264 final-trainer numeric cells, covering all 2,670,178
# eligible landmark prefixes).  This is an integer ULP bound on the final
# trainer float32 numeric channels, never a relative/absolute allclose rule.
FIXED_NUMERIC_ULP_BOUND = 44_149

# The author's training batch size, mirrored from train.py rather than
# imported: train.py loads the dataset at module scope, so importing it would
# run the trainer. check_batchgen.py needs it to drive the released BatchGen
# at the size training actually used.
BATCH_SIZE = 8


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def float32_ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.ascontiguousarray(left, dtype=np.float32)
    right = np.ascontiguousarray(right, dtype=np.float32)
    if left.shape != right.shape:
        raise AssertionError("ULP comparison received different shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise AssertionError("trainer numeric tensors must be finite")

    def ordered(bits: np.ndarray) -> np.ndarray:
        bits = bits.astype(np.uint64)
        negative = (bits & np.uint64(0x80000000)) != 0
        magnitude = bits & np.uint64(0x7FFFFFFF)
        return np.where(
            negative,
            np.uint64(0x80000000) - magnitude,
            np.uint64(0x80000000) + magnitude,
        )

    left_ordered = ordered(left.view(np.uint32))
    right_ordered = ordered(right.view(np.uint32))
    distance = np.maximum(left_ordered, right_ordered) - np.minimum(
        left_ordered, right_ordered
    )
    distance[left == right] = 0
    return distance


def build_author_components(paths: dict[str, Path]):
    dataset_dirs = {
        split: paths["decompensation"] / SOURCE_SPLIT[split]
        for split in ("train", "validation", "test")
    }
    discretizer = preprocessing.Discretizer(
        timestep=1.0,
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
        config_path=str(CONFIG_PATH),
    )
    # Derive the released header without loading the 2.67-million-row
    # listfiles into three Python object graphs.  The header is data-independent
    # once the released discretizer configuration is fixed.
    empty_row = np.asarray([["0"] + [""] * 17])
    header = discretizer.transform(empty_row, end=1.0)[1]
    continuous = [index for index, name in enumerate(header.split(",")) if "->" not in name]
    normalizer = preprocessing.Normalizer(fields=continuous)
    normalizer.load_params(str(NORMALIZER_PATH))
    return dataset_dirs, discretizer, normalizer, header


def direct_author_maximum(
    dataset_dir: Path,
    discretizer: preprocessing.Discretizer,
    normalizer: preprocessing.Normalizer,
    sample: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    # Exercise the released reader method while retaining just the already
    # cohort-verified selected row, rather than loading every list row into
    # Python tuples.  This is required to stay within the shared 7-GiB guard.
    author_reader = reader.DecompensationReader.__new__(reader.DecompensationReader)
    author_reader._dataset_dir = str(dataset_dir)
    author_reader._current_index = 0
    author_reader._listfile_header = "stay,period_length,y_true\n"
    author_reader._data = [
        (
            str(sample["source_filename"]),
            float(sample["source_cutoff_hour"]),
            int(sample["boolean_value"]),
        )
    ]
    example = author_reader.read_example(0)
    if example["name"] != sample["source_filename"]:
        raise AssertionError("source list row no longer identifies the exported stay")
    if abs(float(example["t"]) - int(sample["source_cutoff_hour"])) > EPS:
        raise AssertionError("source list row no longer identifies the exported cutoff")
    if int(example["y"]) != int(sample["boolean_value"]):
        raise AssertionError("source list row no longer identifies the exported label")
    direct64 = discretizer.transform(example["X"], end=example["t"])[0]
    direct64 = normalizer.transform(direct64)
    # NumPy and torch.Tensor both use IEEE-754 round-to-nearest here.  The
    # separate --batchgen-only process imports the actual released torch path
    # and asserts this cast for every fixed diagnostic chunk.
    direct32 = direct64.astype(np.float32)
    cutoff = int(sample["source_cutoff_hour"])
    if direct64.shape != (cutoff, 76) or direct32.shape != (cutoff, 76):
        raise AssertionError("released reader/discretizer tensor shape changed")
    return direct64, direct32


def completed_sample_frames(events_path: Path):
    columns = [
        "subject_id",
        "time",
        "code",
        "numeric_value",
        "sample_id",
        "source_row",
        "sequence_row",
        "source_time",
    ]
    carry: pl.DataFrame | None = None
    parquet = pq.ParquetFile(events_path)
    for batch in parquet.iter_batches(columns=columns, batch_size=250_000):
        frame = pl.from_arrow(batch)
        if carry is not None:
            frame = pl.concat([carry, frame], how="vertical")
        last_sample = int(frame["sample_id"][-1])
        complete = frame.filter(pl.col("sample_id") != last_sample)
        carry = frame.filter(pl.col("sample_id") == last_sample)
        for sample in complete.partition_by("sample_id", maintain_order=True):
            if sample.height:
                yield sample
    if carry is not None and carry.height:
        yield carry


def assert_raw_prefix(
    events: pl.DataFrame,
    raw: dict[str, np.ndarray],
    sample: dict[str, object],
) -> int:
    cutoff = int(sample["source_cutoff_hour"])
    count = int(np.searchsorted(raw["times"], cutoff + EPS, side="right"))
    ordered = events.sort("source_row")
    if ordered.height != count:
        raise AssertionError("common raw occurrence count differs from source prefix")
    if ordered["sample_id"].n_unique() != 1:
        raise AssertionError("common event block crosses sample IDs")
    if ordered["subject_id"].n_unique() != 1 or int(ordered["subject_id"][0]) != int(sample["subject_id"]):
        raise AssertionError("common event subject differs from task index")
    if not np.array_equal(ordered["source_row"].to_numpy(), raw["source_rows"][:count]):
        raise AssertionError("common source-row order differs from raw CSV")
    if not np.array_equal(ordered["sequence_row"].to_numpy(), raw["sequence_rows"][:count]):
        raise AssertionError("common source sequence rows differ from raw CSV")
    if not np.array_equal(ordered["source_time"].to_numpy(), raw["times"][:count]):
        raise AssertionError("common fractional source times differ from raw CSV")
    expected_time = ANCHOR_US + np.rint(raw["times"][:count] * HOUR_US).astype(np.int64)
    if not np.array_equal(ordered["time"].cast(pl.Int64).to_numpy(), expected_time):
        raise AssertionError("common absolute event times differ from fractional source times")
    if ordered["code"].to_list() != raw["codes"][:count].tolist():
        raise AssertionError("common raw value/category codes differ")
    actual_valid = ordered["numeric_value"].is_not_null().to_numpy()
    expected_valid = raw["numeric_valid"][:count]
    if not np.array_equal(actual_valid, expected_valid):
        raise AssertionError("common raw categorical/numeric validity differs")
    if not np.array_equal(
        ordered.filter(pl.col("numeric_value").is_not_null())["numeric_value"].to_numpy(),
        raw["numeric_values"][:count][expected_valid],
    ):
        raise AssertionError("common raw numeric values differ")
    prediction_us = np.datetime64(sample["prediction_time"], "us").astype(np.int64)
    if (ordered["time"].cast(pl.Int64).to_numpy() > prediction_us + 4).any():
        # The reader accepts t <= cutoff + 1e-6 hours (3.6 ms).  Exported
        # absolute times are rounded to microseconds and must remain within it.
        excess = ordered["time"].cast(pl.Int64).to_numpy() - prediction_us
        if excess.max(initial=0) > int(np.ceil(EPS * HOUR_US)):
            raise AssertionError("common raw event crosses the released cutoff tolerance")
    return count


def row_prefix_fingerprints(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.empty(len(tensor) + 1, dtype=np.uint64)
    second = np.empty(len(tensor) + 1, dtype=np.uint64)
    first[0] = second[0] = 0
    h1 = h2 = 0
    for index, row in enumerate(tensor, start=1):
        digest = hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=16).digest()
        r1 = int.from_bytes(digest[:8], "little")
        r2 = int.from_bytes(digest[8:], "little")
        h1 = (h1 * 0x9E3779B185EBCA87 + r1 + index) & MASK64
        h2 = (h2 * 0xC2B2AE3D27D4EB4F + r2 + index) & MASK64
        first[index] = h1
        second[index] = h2
    return first, second


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = assert_source_ready(args.source_root)
    manifest_path = args.task_dir / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    precision_measurement_path = (
        args.task_dir / "audit" / "source_precision_measurement.json"
    )
    if not precision_measurement_path.is_file():
        raise FileNotFoundError(
            "run the exhaustive source precision measurement before check_batch"
        )
    precision_measurement = json.loads(precision_measurement_path.read_text())
    if (
        precision_measurement.get("status") != "measurement"
        or precision_measurement.get("smallest_covering_ulp_bound")
        != FIXED_NUMERIC_ULP_BOUND
    ):
        raise AssertionError("source precision measurement does not fix the declared ULP bound")
    task = manifest["task"]
    if task["id"] != "decompensation":
        raise AssertionError("wrong SSML task")
    if task["modalities"] != ["structured"] or task["note_codes"] != []:
        raise AssertionError("SSML selected task retained a non-structured modality")

    events_path = args.task_dir / manifest["meds"]["events"]["file"]
    index = pl.read_parquet(
        args.task_dir / manifest["meds"]["task_index"]["file"]
    )
    spec = load_transform_spec()
    common_codes = tuple(
        pl.read_parquet(
            args.task_dir / manifest["meds"]["codes"]["file"], columns=["code"]
        )["code"].to_list()
    )
    if common_codes != spec["codes"] or len(common_codes) != 59:
        raise AssertionError("common vocabulary is not the raw 59-code occurrence vocabulary")

    author_dataset_dirs, discretizer, normalizer, author_header = build_author_components(paths)
    if author_header != ",".join(spec["model_codes"]):
        raise AssertionError("released loader header differs from the derived 76-channel boundary")

    index_by_id = index.sort("sample_id")
    if not np.array_equal(
        index_by_id["sample_id"].to_numpy(),
        np.arange(1, index.height + 1, dtype=np.int64),
    ):
        raise AssertionError("SSML common sample IDs are not canonical consecutive integers")
    group_maximum = {}
    for row in index.sort("source_cutoff_hour").iter_rows(named=True):
        group_maximum[(str(row["split"]), str(row["source_filename"]))] = row
    tensor_fingerprint_by_sample = np.zeros((index.height + 1, 2), dtype=np.uint64)

    split_reports = {
        split: {
            "examples": 0,
            "raw_occurrence_events": 0,
            "model_rows": 0,
            "model_cells": 0,
            "windows_24h": 0,
            "maximum_sequence_length": 0,
        }
        for split in ("train", "validation", "test")
    }
    numeric_ulp_distribution = Counter()
    per_split_numeric_ulp_distribution = {
        split: Counter() for split in ("train", "validation", "test")
    }
    numeric_ulp_cells = 0
    numeric_ulp_nonzero = 0
    maximum_numeric_ulp = 0
    maximum_numeric_absolute_difference = 0.0
    seen_samples = np.zeros(index.height + 1, dtype=bool)
    seen_sample_count = 0
    seen_groups: set[tuple[str, str]] = set()
    active_group = None
    active_raw = None
    active_direct32 = None
    active_fingerprints = None
    active_maximum_id = None
    active_maximum_verified = False

    for events in completed_sample_frames(events_path):
        sample_id = int(events["sample_id"][0])
        if not 1 <= sample_id <= index.height:
            raise AssertionError("common events reference an unknown task sample")
        if seen_samples[sample_id]:
            raise AssertionError("common events are not contiguous by sample")
        seen_samples[sample_id] = True
        seen_sample_count += 1
        sample = index_by_id.row(sample_id - 1, named=True)
        split = str(sample["split"])
        group = (split, str(sample["source_filename"]))
        if group != active_group:
            if active_group is not None and not active_maximum_verified:
                raise AssertionError("prior stay group never verified its maximum tensor")
            if group in seen_groups:
                raise AssertionError("common events revisit a noncontiguous stay group")
            seen_groups.add(group)
            maximum = group_maximum[group]
            active_raw = load_raw_stay(
                stay_path(paths, split, group[1]),
                spec,
                int(maximum["source_cutoff_hour"]),
            )
            _, active_direct32 = direct_author_maximum(
                author_dataset_dirs[split], discretizer, normalizer, maximum
            )
            active_fingerprints = row_prefix_fingerprints(active_direct32)
            active_maximum_id = int(maximum["sample_id"])
            active_maximum_verified = False
            active_group = group

        event_count = assert_raw_prefix(events, active_raw, sample)
        cutoff = int(sample["source_cutoff_hour"])
        if cutoff > len(active_direct32):
            raise AssertionError("sample cutoff exceeds the stay maximum tensor")
        if sample_id == active_maximum_id:
            reconstructed = reconstruct_model_tensor(events, spec, cutoff)
            if reconstructed.shape != active_direct32.shape:
                raise AssertionError("raw MEDS stream changed the author tensor shape")
            numeric_positions = np.asarray(spec["numeric_indices"], dtype=np.int64)
            categorical_positions = np.asarray(
                [position for position in range(59) if position not in set(numeric_positions)],
                dtype=np.int64,
            )
            if not np.array_equal(
                reconstructed[:, categorical_positions],
                active_direct32[:, categorical_positions],
            ):
                raise AssertionError("raw MEDS stream changed a categorical trainer channel")
            if not np.array_equal(reconstructed[:, 59:], active_direct32[:, 59:]):
                raise AssertionError("raw MEDS stream changed an observation-mask channel")
            numeric_ulp = float32_ulp_distance(
                reconstructed[:, numeric_positions],
                active_direct32[:, numeric_positions],
            )
            unique_ulp, unique_counts = np.unique(numeric_ulp, return_counts=True)
            distribution = {
                int(distance): int(count)
                for distance, count in zip(unique_ulp, unique_counts)
            }
            numeric_ulp_distribution.update(distribution)
            per_split_numeric_ulp_distribution[split].update(distribution)
            numeric_ulp_cells += int(numeric_ulp.size)
            numeric_ulp_nonzero += int(np.count_nonzero(numeric_ulp))
            observed_maximum = int(numeric_ulp.max(initial=0))
            maximum_numeric_ulp = max(maximum_numeric_ulp, observed_maximum)
            maximum_numeric_absolute_difference = max(
                maximum_numeric_absolute_difference,
                float(
                    np.max(
                        np.abs(
                            reconstructed[:, numeric_positions]
                            - active_direct32[:, numeric_positions]
                        ),
                        initial=0.0,
                    )
                ),
            )
            if observed_maximum > FIXED_NUMERIC_ULP_BOUND:
                raise AssertionError(
                    "numeric trainer tensor exceeds the fixed ULP bound: "
                    f"observed={observed_maximum} bound={FIXED_NUMERIC_ULP_BOUND}"
                )
            active_maximum_verified = True

        report = split_reports[split]
        report["examples"] += 1
        report["raw_occurrence_events"] += event_count
        report["model_rows"] += cutoff
        report["model_cells"] += cutoff * 76
        report["windows_24h"] += cutoff // 24
        report["maximum_sequence_length"] = max(
            report["maximum_sequence_length"], cutoff
        )
        tensor_fingerprint_by_sample[sample_id, 0] = active_fingerprints[0][cutoff]
        tensor_fingerprint_by_sample[sample_id, 1] = active_fingerprints[1][cutoff]

    if active_group is not None and not active_maximum_verified:
        raise AssertionError("final stay group never verified its maximum tensor")
    if seen_sample_count != index.height or not seen_samples[1:].all():
        missing = (np.flatnonzero(~seen_samples[1:]) + 1)[:20].tolist()
        raise AssertionError(f"not every task sample had a raw event block: {missing}")
    if sum(value["raw_occurrence_events"] for value in split_reports.values()) != manifest["event_count"]:
        raise AssertionError("exhaustive common event total differs from manifest")
    if sum(numeric_ulp_distribution.values()) != numeric_ulp_cells:
        raise AssertionError("numeric ULP histogram does not cover every measured cell")
    if numeric_ulp_distribution.get(0, 0) != numeric_ulp_cells - numeric_ulp_nonzero:
        raise AssertionError("numeric ULP nonzero count disagrees with histogram")
    expected_distribution = {
        int(distance): int(count)
        for distance, count in precision_measurement["ulp_distribution"].items()
    }
    if dict(numeric_ulp_distribution) != expected_distribution:
        raise AssertionError(
            "MEDS-derived numeric ULP distribution differs from the source measurement"
        )
    if (
        numeric_ulp_cells != int(precision_measurement["cells"])
        or numeric_ulp_nonzero != int(precision_measurement["nonzero_cells"])
        or len(seen_groups) != int(precision_measurement["groups"])
    ):
        raise AssertionError("MEDS-derived precision census has different coverage")
    for split, distribution in per_split_numeric_ulp_distribution.items():
        expected = {
            int(distance): int(count)
            for distance, count in precision_measurement["per_split"][split][
                "ulp_distribution"
            ].items()
        }
        if dict(distribution) != expected:
            raise AssertionError(f"{split} MEDS/source ULP distributions differ")
    if maximum_numeric_ulp != FIXED_NUMERIC_ULP_BOUND:
        raise AssertionError(
            "exhaustive numeric maximum no longer reproduces the smallest fixed "
            f"ULP bound: observed={maximum_numeric_ulp} "
            f"bound={FIXED_NUMERIC_ULP_BOUND}"
        )

    serializable_splits = {}
    for split, values in split_reports.items():
        if values["examples"] != manifest["splits"][split]["count"]:
            raise AssertionError(f"{split} exhaustive sample total differs")
        ordered = index.filter(pl.col("split") == split).sort("source_order")
        ordered_ids = ordered["sample_id"].to_numpy().astype("<i8", copy=False)
        ordered_cutoffs = ordered["source_cutoff_hour"].to_numpy().astype("<i8", copy=False)
        ordered_labels = ordered["boolean_value"].to_numpy().astype(np.uint8, copy=False)
        digest = hashlib.sha256()
        digest.update(ordered_ids.tobytes())
        digest.update(ordered_cutoffs.tobytes())
        digest.update(ordered_labels.tobytes())
        digest.update(
            np.ascontiguousarray(tensor_fingerprint_by_sample[ordered_ids]).tobytes()
        )
        serializable_splits[split] = {
            **values,
            "ordered_tensor_fingerprint_sha256": digest.hexdigest(),
        }

    report = {
        "status": "stream_exact",
        "paper": "semi_supervised_meta_learning",
        "task": "decompensation",
        "protocol_status_meaning": (
            "exhaustive persistent-MEDS/source-loader phase passed; canonical exact "
            "status is emitted only after the separate exhaustive released BatchGen phase"
        ),
        "common_boundary": {
            "kind": "raw pre-discretization structured occurrence stream",
            "codes": 59,
            "fractional_source_times": True,
            "within_bin_repeated_values": True,
            "source_csv_order": True,
            "derived_masks_or_imputed_values_retained": False,
        },
        "author_boundary": {
            "shape": "[T,76]",
            "numpy_dtype": "float64",
            "model_dtype": "float32",
            "value_channels": 59,
            "mask_channels": 17,
            "hourly_overwrite": "last raw value in CSV order",
            "imputation": "previous, initialized with released normal values",
            "normalization": "released decompensation normalizer",
        },
        "numeric_precision": {
            "scope": (
                "12 numeric channels of the final trainer float32 tensor, measured "
                "on every attributable stay's maximum loader tensor; every landmark "
                "is an exact prefix of one of these tensors"
            ),
            "rule": "integer distance between adjacent IEEE-754 float32 values",
            "authorized_reason": (
                "required attributable source-float64 to MEDS-float32 round-trip only; "
                "no rtol/atol allclose"
            ),
            "fixed_ulp_bound": FIXED_NUMERIC_ULP_BOUND,
            "maximum_observed_ulp": maximum_numeric_ulp,
            "maximum_absolute_difference": maximum_numeric_absolute_difference,
            "cells": numeric_ulp_cells,
            "nonzero_cells": numeric_ulp_nonzero,
            "nonzero_fraction": numeric_ulp_nonzero / numeric_ulp_cells,
            "ulp_distribution": {
                str(distance): int(count)
                for distance, count in sorted(numeric_ulp_distribution.items())
            },
            "per_split": {
                split: {
                    "maximum_observed_ulp": max(distribution, default=0),
                    "cells": sum(distribution.values()),
                    "nonzero_cells": sum(
                        count
                        for distance, count in distribution.items()
                        if distance != 0
                    ),
                    "ulp_distribution": {
                        str(distance): int(count)
                        for distance, count in sorted(distribution.items())
                    },
                }
                for split, distribution in per_split_numeric_ulp_distribution.items()
            },
        },
        "exhaustive_recreation": serializable_splits,
        "batchgen_handoff": {
            "status": "pending_exhaustive_second_guarded_process",
            "examples": index.height,
            "expected_per_split": {
                split: {
                    "examples": values["examples"],
                    "windows_24h": values["windows_24h"],
                    "ordered_tensor_fingerprint_sha256": serializable_splits[split][
                        "ordered_tensor_fingerprint_sha256"
                    ],
                }
                for split, values in split_reports.items()
            },
        },
        "checks": [
            "every common occurrence against exact raw CSV time, row, channel, and value",
            "every landmark cutoff, label, subject, and duplicated raw prefix",
            "released reader/discretizer/normalizer maximum tensor for every stay",
            "exact categorical and mask rematerialization for every stay maximum",
            "exhaustive numeric trainer float32 tensors within the fixed integer-ULP bound",
            "prefix-equivalent [T,76] tensor and 24-hour window count for every landmark under that rule",
            "all landmark tensor-prefix fingerprints handed to the exhaustive BatchGen phase",
            "structured-only 59-code common vocabulary and no retained derived masks",
        ],
        "source_artifacts": {
            "discretizer_config_sha256": sha256_file(CONFIG_PATH),
            "normalizer_sha256": sha256_file(NORMALIZER_PATH),
            "source_precision_measurement_sha256": sha256_file(
                precision_measurement_path
            ),
            "common_events_sha256": sha256_file(events_path),
        },
        "manifest_sha256": manifest_sha256,
    }
    write_json(args.task_dir / "audit" / "full_batch_stream_recreation.json", report)
    if sha256_file(manifest_path) != manifest_sha256:
        raise AssertionError("batch checker modified immutable manifest.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
