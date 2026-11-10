"""Recreate every retained DDL-CXR prediction-loader field from MEDS."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from cohort_source import (
    EPS,
    HOURS,
    author_preprocessors,
    code_vocabulary,
    configuration,
    iter_raw_rows,
    load_prediction_pickles,
    load_source_rows,
    variable_names,
)


# Fixed after an exhaustive 7,486-example measurement. This is an integer ULP
# bound on the final float32 model tensor, not an rtol/atol comparison.
FIXED_NUMERIC_ULP_BOUND = 1_669
CONTINUOUS_POSITIONS = np.asarray(
    [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61], dtype=np.int64
)


def float32_ulp_distance(left, right):
    left = np.ascontiguousarray(left, dtype=np.float32)
    right = np.ascontiguousarray(right, dtype=np.float32)
    if left.shape != right.shape:
        raise AssertionError("ULP comparison received different shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise AssertionError("trainer tensors must be finite")

    def ordered(bits):
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


def iter_meds_samples(path: Path):
    parquet = pq.ParquetFile(path)
    columns = [
        "subject_id",
        "sample_id",
        "time",
        "split",
        "source_row",
        "sequence_row",
        "source_time",
        "code",
        "numeric_value",
    ]
    current_sample = None
    current_records = []
    expected_source_row = 0
    for row_group in range(parquet.num_row_groups):
        records = (
            parquet.read_row_group(row_group, columns=columns)
            .sort_by([("source_row", "ascending")])
            .to_pylist()
        )
        for record in records:
            source_row = int(record["source_row"])
            if source_row != expected_source_row:
                raise AssertionError("MEDS source_row is not exact and contiguous")
            expected_source_row += 1
            sample_id = int(record["sample_id"])
            if current_sample is None:
                current_sample = sample_id
            elif sample_id != current_sample:
                yield current_sample, current_records
                current_sample = sample_id
                current_records = []
            current_records.append(record)
    if current_sample is not None:
        yield current_sample, current_records


def expected_event_records(sample: dict[str, object]):
    config = configuration()
    categorical = config["is_categorical_channel"]
    possible = config["possible_values"]
    variables = variable_names()
    for sequence_row, source_time, source_values in iter_raw_rows(sample):
        if source_time < EPS:
            continue
        event_time = sample["intime"] + timedelta(hours=source_time)
        for variable, value in zip(variables, source_values[1:]):
            if value == "":
                continue
            if categorical[variable]:
                rendered = str(float(value)) if not isinstance(value, str) else value
                if rendered not in possible[variable]:
                    raise AssertionError("source category is outside the released vocabulary")
                code = f"{variable}//{rendered}"
                numeric = None
            else:
                code = variable
                numeric = float(np.float32(float(value)))
            yield {
                "subject_id": int(sample["subject_id"]),
                "sample_id": int(sample["stay_id"]),
                "time": event_time,
                "split": str(sample["split"]),
                "sequence_row": sequence_row,
                "source_time": source_time,
                "code": code,
                "numeric_value": numeric,
            }


def assert_event_projection(
    sample: dict[str, object], actual: list[dict[str, object]]
) -> None:
    expected = list(expected_event_records(sample))
    if len(actual) != len(expected):
        raise AssertionError("MEDS event count differs from the source projection")
    for found, wanted in zip(actual, expected):
        for name in (
            "subject_id",
            "sample_id",
            "time",
            "split",
            "sequence_row",
            "source_time",
            "code",
        ):
            if found[name] != wanted[name]:
                raise AssertionError(f"MEDS event field differs: {name}")
        found_numeric = found["numeric_value"]
        wanted_numeric = wanted["numeric_value"]
        if wanted_numeric is None:
            if found_numeric is not None:
                raise AssertionError("categorical occurrence acquired a numeric value")
        elif found_numeric != wanted_numeric:
            raise AssertionError("numeric event differs after the required float32 encoding")


def reconstruct_rows(records: list[dict[str, object]]) -> list[list[object]]:
    variables = variable_names()
    positions = {name: index for index, name in enumerate(variables)}
    result = []
    current_sequence = None
    current = None
    for record in records:
        sequence = int(record["sequence_row"])
        if sequence != current_sequence:
            if current is not None:
                result.append(current)
            current_sequence = sequence
            current = [float(record["source_time"])] + [""] * len(variables)
        elif float(record["source_time"]) != current[0]:
            raise AssertionError("one source sequence row has inconsistent times")
        code = str(record["code"])
        if "//" in code:
            variable, value = code.split("//", 1)
        else:
            variable = code
            value = record["numeric_value"]
            if value is None:
                raise AssertionError("numeric source code became occurrence-only")
        position = positions.get(variable)
        if position is None:
            raise AssertionError("MEDS contains an auxiliary code")
        if current[position + 1] != "":
            raise AssertionError("MEDS repeats a source variable within one raw row")
        current[position + 1] = value
    if current is not None:
        result.append(current)
    if not result:
        raise AssertionError("MEDS sample has no reconstructable source rows")
    return result


def update_digest(digest, array, dtype):
    value = np.ascontiguousarray(array, dtype=dtype)
    digest.update(struct.pack("<I", value.ndim))
    digest.update(struct.pack("<" + "Q" * value.ndim, *value.shape))
    digest.update(value.tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--measure-only", action="store_true")
    args = parser.parse_args()

    task_dir = args.task_dir.resolve()
    manifest = json.loads((task_dir / "manifest.json").read_text())
    source = manifest["source"]
    if source["selected_experiment"] != "Table 1 DDL-CXR mortality AUROC":
        raise AssertionError("batch checker received the wrong experiment")
    if source["selected_model_inputs"] != [
        "structured",
        "last_cxr_image",
        "generated_cxr_latent",
    ]:
        raise AssertionError("source input declaration differs")
    if source["retained_projection"] != ["structured"]:
        raise AssertionError("wrong retained source projection")
    if manifest["task"]["modalities"] != ["structured"]:
        raise AssertionError("imaging or notes entered the persistent task")
    if manifest["task"]["note_codes"]:
        raise AssertionError("DDL-CXR must have an empty note-code exception list")

    vocabulary = tuple(
        pq.read_table(task_dir / "metadata" / "codes.parquet")["code"].to_pylist()
    )
    if vocabulary != code_vocabulary():
        raise AssertionError("MEDS vocabulary differs from the released EHR vocabulary")

    rows = load_source_rows(args.work_root.resolve())
    pickles = load_prediction_pickles(args.work_root.resolve())
    for split, values in pickles.items():
        expected_count = sum(row["split"] == split for row in rows)
        if len(values) != expected_count:
            raise AssertionError("released EHR pickle has the wrong split size")

    event_samples = iter(iter_meds_samples(task_dir / "data" / "events.parquet"))
    discretizer, normalizer = author_preprocessors()
    source_tensor_digest = hashlib.sha256()
    recreated_tensor_digest = hashlib.sha256()
    source_mask_digest = hashlib.sha256()
    recreated_mask_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    event_digest = hashlib.sha256()
    maximum_float64_difference = 0.0
    maximum_numeric_ulp = 0
    numeric_distribution: Counter[int] = Counter()
    per_split_distribution = {name: Counter() for name in ("train", "validation", "test")}
    mask_summary = {
        name: {"observed": 0, "cells": 0}
        for name in ("train", "validation", "test")
    }
    batch_shapes = set()
    batch_tensors = []
    examples = 0
    events = 0

    for sample in rows:
        try:
            sample_id, records = next(event_samples)
        except StopIteration as error:
            raise AssertionError("MEDS ended before the source cohort") from error
        if sample_id != int(sample["stay_id"]):
            raise AssertionError("MEDS sample order differs from the source exporter")
        assert_event_projection(sample, records)
        reconstructed_rows = reconstruct_rows(records)
        recreated = normalizer.transform(
            discretizer.transform(np.asarray(reconstructed_rows, dtype=object), end=48)[0]
        )
        source_tensor = np.asarray(
            pickles[str(sample["split"])][str(sample["source_index"])].numpy(),
            dtype=np.float64,
        )
        if source_tensor.shape != (48, 79) or recreated.shape != (48, 79):
            raise AssertionError("prediction loader tensor does not have shape [48,79]")
        difference = float(np.max(np.abs(source_tensor - recreated)))
        maximum_float64_difference = max(maximum_float64_difference, difference)

        source_masks = source_tensor[:, 62:]
        recreated_masks = recreated[:, 62:]
        if not np.array_equal(source_masks, recreated_masks):
            raise AssertionError("observation masks differ")
        if not np.isin(source_masks, [0.0, 1.0]).all():
            raise AssertionError("source masks are not binary")
        noncontinuous = np.setdiff1d(np.arange(62), CONTINUOUS_POSITIONS)
        if not np.array_equal(
            source_tensor[:, noncontinuous].astype(np.float32),
            recreated[:, noncontinuous].astype(np.float32),
        ):
            raise AssertionError("categorical/static source channels differ")
        distances = float32_ulp_distance(
            source_tensor[:, CONTINUOUS_POSITIONS],
            recreated[:, CONTINUOUS_POSITIONS],
        )
        local_maximum = int(distances.max(initial=0))
        maximum_numeric_ulp = max(maximum_numeric_ulp, local_maximum)
        values, counts = np.unique(distances, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            numeric_distribution[int(value)] += int(count)
            per_split_distribution[str(sample["split"])][int(value)] += int(count)

        update_digest(source_tensor_digest, source_tensor, np.float64)
        update_digest(recreated_tensor_digest, recreated, np.float64)
        update_digest(source_mask_digest, source_masks, np.uint8)
        update_digest(recreated_mask_digest, recreated_masks, np.uint8)
        update_digest(label_digest, [sample["label"]], np.int8)
        for record in records:
            event_digest.update(
                json.dumps(record, sort_keys=True, default=str).encode("utf-8") + b"\n"
            )
        split = str(sample["split"])
        mask_summary[split]["observed"] += int(source_masks.sum())
        mask_summary[split]["cells"] += int(source_masks.size)
        examples += 1
        events += len(records)
        batch_tensors.append(source_tensor)
        if len(batch_tensors) == args.batch_size:
            batch_shapes.add(tuple(np.stack(batch_tensors).shape))
            batch_tensors = []
    if batch_tensors:
        batch_shapes.add(tuple(np.stack(batch_tensors).shape))
    try:
        extra_sample = next(event_samples)
    except StopIteration:
        extra_sample = None
    if extra_sample is not None:
        raise AssertionError("MEDS contains samples absent from the source cohort")

    if not args.measure_only:
        if FIXED_NUMERIC_ULP_BOUND is None:
            raise AssertionError("numeric ULP bound has not been fixed from measurement")
        if maximum_numeric_ulp > FIXED_NUMERIC_ULP_BOUND:
            raise AssertionError(
                f"numeric tensor exceeds fixed ULP bound: {maximum_numeric_ulp}"
            )

    report = {
        "status": "measurement" if args.measure_only else "exact",
        "selected_experiment": source["selected_experiment"],
        "selected_launcher": source["selected_launcher"],
        "selected_model_inputs": source["selected_model_inputs"],
        "retained_projection": ["structured"],
        "examples": examples,
        "events": events,
        "batch_size": args.batch_size,
        "author_loader": {
            "per_example_shape": [48, 79],
            "pickle_dtype": "float64",
            "expanded_value_channels": 62,
            "observation_mask_channels": 17,
            "sequence_length": 48,
            "collated_shapes": [list(shape) for shape in sorted(batch_shapes)],
            "padding": "none",
        },
        "maximum_pre_model_float64_difference_from_meds_float32": maximum_float64_difference,
        "numeric_precision": {
            "scope": "13 continuous channels after the final released normalizer",
            "rule": "integer distance between adjacent IEEE-754 float32 values",
            "fixed_ulp_bound": FIXED_NUMERIC_ULP_BOUND,
            "maximum_observed_ulp": maximum_numeric_ulp,
            "cells": sum(numeric_distribution.values()),
            "nonzero_cells": sum(
                count for distance, count in numeric_distribution.items() if distance
            ),
            "ulp_distribution": {
                str(distance): count
                for distance, count in sorted(numeric_distribution.items())
            },
            "per_split": {
                split: {
                    "maximum_observed_ulp": max(distribution, default=0),
                    "cells": sum(distribution.values()),
                    "ulp_distribution": {
                        str(distance): count
                        for distance, count in sorted(distribution.items())
                    },
                }
                for split, distribution in per_split_distribution.items()
            },
        },
        "event_vocabulary": {
            "declared_count": len(vocabulary),
            "ordered_code_sha256": hashlib.sha256(
                "\n".join(vocabulary).encode("utf-8")
            ).hexdigest(),
            "note_codes": [],
            "all_non_note_codes_are_released_structured_inputs": True,
            "imaging_or_auxiliary_codes": [],
        },
        "digests": {
            "source_model_tensor_sha256": source_tensor_digest.hexdigest(),
            "recreated_model_tensor_sha256": recreated_tensor_digest.hexdigest(),
            "source_mask_sha256": source_mask_digest.hexdigest(),
            "recreated_mask_sha256": recreated_mask_digest.hexdigest(),
            "ordered_label_sha256": label_digest.hexdigest(),
            "meds_event_projection_sha256": event_digest.hexdigest(),
        },
        "mask_missingness": {
            split: {
                **summary,
                "observed_fraction": summary["observed"] / summary["cells"],
            }
            for split, summary in mask_summary.items()
        },
        "checks": [
            "exhaustive exact source event projection after MEDS float32 encoding",
            "exhaustive reconstruction through released EHR_Pickler discretizer and normalizer",
            "exact 17-channel masks and categorical/static channels",
            "continuous channels within the fixed authorized final-float32 ULP bound",
            "exact labels, shapes, dtypes, sequence lengths, and no padding",
            "exact released vocabulary and empty note-code exception list",
            "no imaging, text, identifier, label, or auxiliary events",
        ],
    }
    if not args.measure_only:
        output = task_dir / "audit" / "full_batch_recreation.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
