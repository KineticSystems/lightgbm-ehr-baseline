"""Exhaustively recreate the released torchmimic IHM loader and collator."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torchmimic.utils import pad_colalte

from cohort_source import (
    HOURS,
    NORMALIZER_SHA256,
    PRODUCER_COMMIT,
    STUDY_COMMIT,
    TORCHMIMIC_COMMIT,
    author_datasets,
    author_preprocessors,
    code_vocabulary,
    configuration,
    load_source_rows,
    sha256,
    torchmimic_artifacts,
    variable_names,
    write_json,
)


# MEDS numeric_value is float32, while the released episode CSV is parsed to
# float64 before normalization.  This fixed bound is populated only after an
# exhaustive --measure-only traversal and then enforced at the model boundary.
FIXED_NUMERIC_ULP_BOUND = 10_214


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
                raise AssertionError("MEDS source_row sequence is not exact and contiguous")
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


def canonical_event_digest(digest, sequence_row, source_time, code, numeric_value):
    digest.update(np.asarray([sequence_row], dtype=np.int32).tobytes())
    digest.update(np.asarray([source_time], dtype=np.float64).tobytes())
    encoded = code.encode()
    digest.update(np.asarray([len(encoded)], dtype=np.int32).tobytes())
    digest.update(encoded)
    if numeric_value is None:
        digest.update(b"NULL")
    else:
        digest.update(np.asarray([numeric_value], dtype=np.float32).tobytes())


def expected_events(raw, header, config):
    result = []
    for sequence_row, values in enumerate(raw):
        source_time = float(values[0])
        for variable, value in zip(header[1:], values[1:]):
            if value == "":
                continue
            if config["is_categorical_channel"][variable]:
                code = f"{variable}//{value}"
                numeric = None
            else:
                code = variable
                numeric = float(np.float32(float(value)))
            result.append((sequence_row, source_time, code, numeric))
    return result


def recreate_item(records, discretizer, normalizer, config):
    variables = tuple(config["id_to_channel"])
    variable_index = {name: position for position, name in enumerate(variables)}
    sequence_rows = [int(row["sequence_row"]) for row in records]
    length = max(sequence_rows) + 1
    raw = np.full((length, len(variables) + 1), "", dtype=object)
    times = np.full(length, np.nan, dtype=np.float64)
    for record in records:
        sequence_row = int(record["sequence_row"])
        source_time = float(record["source_time"])
        if np.isnan(times[sequence_row]):
            times[sequence_row] = source_time
            raw[sequence_row, 0] = repr(source_time)
        elif times[sequence_row] != source_time:
            raise AssertionError("one source row has inconsistent source_time values")
        code = str(record["code"])
        if "//" in code:
            variable, value = code.split("//", 1)
            if not config["is_categorical_channel"].get(variable, False):
                raise AssertionError("occurrence code is not a released category")
        else:
            variable = code
            value = str(np.float32(record["numeric_value"]))
            if config["is_categorical_channel"].get(variable, False):
                raise AssertionError("categorical variable was encoded numerically")
        column = variable_index.get(variable)
        if column is None:
            raise AssertionError("MEDS event is absent from the author declaration")
        if raw[sequence_row, column + 1] != "":
            raise AssertionError("MEDS duplicated one source variable/time cell")
        raw[sequence_row, column + 1] = value
    if not np.isfinite(times).all():
        raise AssertionError("MEDS lost an entire source timeseries row")
    header = ["Hours"] + list(variables)
    data = discretizer.transform(raw, header=header, end=HOURS)[0]
    return normalizer.transform(data)


def update_array_digest(digest, value):
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    array = np.ascontiguousarray(array)
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--measure-only", action="store_true")
    args = parser.parse_args()

    manifest_path = args.task_dir / "manifest.json"
    manifest_hash_before = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    task = manifest["task"]
    source = manifest["source"]
    if task.get("modalities") != ["structured"] or task.get("note_codes") != []:
        raise AssertionError("selected IHM common task must contain structured data only")
    expected_source = {
        "study_commit": STUDY_COMMIT,
        "producer_commit": PRODUCER_COMMIT,
        "torchmimic_commit": TORCHMIMIC_COMMIT,
        "selected_experiment": "Table 1 Ours, IHM, 100% labels",
        "selected_model_inputs": ["structured"],
        "common_profile": "fixed_numeric",
        "selection_tier": 1,
        "retained_projection": ["structured"],
        "unused_modality_restriction": False,
    }
    if any(source.get(key) != value for key, value in expected_source.items()):
        raise AssertionError("batch checker selected the wrong paper experiment")
    expected_preprocessing = {
        "timestep_hours": 1.0,
        "observation_hours": 48.0,
        "imputation": "previous then released normal value",
        "start_time": "zero",
        "store_masks": True,
        "value_channels": 59,
        "mask_channels": 17,
        "normalized_continuous_channels": 12,
        "batch_padding": "none for fixed 48-bin IHM",
    }
    if source.get("author_preprocessing") != expected_preprocessing:
        raise AssertionError("released loader preprocessing declaration changed")

    config = configuration()
    variables = variable_names()
    if len(variables) != 17 or tuple(config["id_to_channel"]) != variables:
        raise AssertionError("released structured variable order changed")
    expected_vocabulary = code_vocabulary()
    codes_path = args.task_dir / manifest["meds"]["codes"]["file"]
    actual_vocabulary = tuple(pq.read_table(codes_path)["code"].to_pylist())
    if actual_vocabulary != expected_vocabulary:
        raise AssertionError("MEDS vocabulary differs from released channel/category order")
    if any(
        token in code.lower()
        for code in actual_vocabulary
        for token in ("note", "image", "cxr", "radiology", "modality", "pair")
    ):
        raise AssertionError("a note/imaging/modality code entered the common task")

    source_rows = load_source_rows(args.source_root.resolve())
    rows_by_split = {
        split: [row for row in source_rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    datasets = author_datasets(args.source_root.resolve())
    discretizer, normalizer, output_header = author_preprocessors()
    if len(output_header) != 76:
        raise AssertionError("released loader must emit exactly 76 channels")
    numeric_positions = tuple(
        position for position, name in enumerate(output_header[:59]) if "->" not in name
    )
    categorical_positions = tuple(
        position for position in range(59) if position not in numeric_positions
    )
    if len(numeric_positions) != 12 or len(categorical_positions) != 47:
        raise AssertionError("released 12 numeric / 47 categorical partition changed")

    events_path = args.task_dir / manifest["meds"]["events"]["file"]
    event_samples = iter_meds_samples(events_path)
    actual_event_codes = set()
    source_event_digest = hashlib.sha256()
    meds_event_digest = hashlib.sha256()
    source_tensor_digest = hashlib.sha256()
    recreated_tensor_digest = hashlib.sha256()
    source_mask_digest = hashlib.sha256()
    recreated_mask_digest = hashlib.sha256()
    expanded_mask_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    per_split = {
        split: {
            "source_tensor": hashlib.sha256(),
            "recreated_tensor": hashlib.sha256(),
            "mask": hashlib.sha256(),
            "labels": hashlib.sha256(),
        }
        for split in ("train", "validation", "test")
    }
    mask_summary = {
        split: {"observed": 0, "cells": 0}
        for split in ("train", "validation", "test")
    }
    numeric_ulp_distribution = Counter()
    per_split_ulp_distribution = {
        split: Counter() for split in ("train", "validation", "test")
    }
    maximum_numeric_ulp = 0
    maximum_float32_absolute_difference = 0.0
    maximum_float64_precast_difference = 0.0
    sample_count = 0
    batch_count = 0
    event_count = 0
    observed_shapes = set()

    for split in ("train", "validation", "test"):
        dataset = datasets[split]
        rows = rows_by_split[split]
        if len(dataset) != len(rows):
            raise AssertionError(f"{split} exact loader/listfile count differs")
        for batch_start in range(0, len(rows), args.batch_size):
            source_items = []
            recreated_items = []
            for offset, row in enumerate(
                rows[batch_start : batch_start + args.batch_size], start=batch_start
            ):
                if dataset.names[offset] != row["source_file"]:
                    raise AssertionError("IHMDataset changed the exact listfile row order")
                raw_item = dataset.reader.read_example(offset)
                if (
                    raw_item["name"] != row["source_file"]
                    or int(raw_item["y"]) != int(row["label"])
                    or float(raw_item["t"]) != HOURS
                ):
                    raise AssertionError("IHM reader changed name, label, or cutoff")
                expected = expected_events(raw_item["X"], raw_item["header"], config)
                try:
                    actual_sample_id, actual = next(event_samples)
                except StopIteration as error:
                    raise AssertionError("MEDS omitted a released event sample") from error
                if actual_sample_id != int(row["stay_id"]):
                    raise AssertionError("MEDS event sample order differs from source")
                if len(expected) != len(actual):
                    raise AssertionError("MEDS changed the observed source cell count")
                for expected_row, actual_row in zip(expected, actual):
                    sequence_row, source_time, code, numeric = expected_row
                    expected_time = row["intime"] + timedelta(
                        hours=min(HOURS, max(0.0, source_time))
                    )
                    if (
                        int(actual_row["subject_id"]) != int(row["subject_id"])
                        or actual_row["time"] != expected_time
                        or actual_row["split"] != split
                    ):
                        raise AssertionError(
                            "MEDS changed an event identifier, timestamp, or split"
                        )
                    comparable = (
                        int(actual_row["sequence_row"]),
                        float(actual_row["source_time"]),
                        str(actual_row["code"]),
                        None
                        if actual_row["numeric_value"] is None
                        else float(np.float32(actual_row["numeric_value"])),
                    )
                    if comparable != expected_row:
                        raise AssertionError("MEDS changed an observed source event")
                    actual_event_codes.add(str(actual_row["code"]))
                    canonical_event_digest(source_event_digest, *expected_row)
                    canonical_event_digest(meds_event_digest, *comparable)
                event_count += len(actual)

                source_item = dataset[offset]
                if source_item[0].dtype != torch.float32:
                    raise AssertionError("IHMDataset __getitem__ no longer returns float32")
                precursor = dataset.data[offset]
                if precursor.dtype != np.float64 or precursor.shape != (48, 76):
                    raise AssertionError("unexpected eager IHMDataset precursor")
                if not torch.equal(source_item[0], torch.Tensor(precursor)):
                    raise AssertionError("__getitem__ is not the exact precursor float32 cast")
                recreated_data = recreate_item(actual, discretizer, normalizer, config)
                if recreated_data.dtype != np.float64 or recreated_data.shape != (48, 76):
                    raise AssertionError("recreated loader precursor has wrong dtype/shape")
                recreated_x = torch.Tensor(recreated_data)
                recreated_expanded_mask = dataset.expand_mask(recreated_data[:, 59:])
                recreated_item = (
                    recreated_x,
                    int(row["label"]),
                    48,
                    recreated_expanded_mask,
                )
                if int(source_item[1]) != int(row["label"]) or source_item[2] != 48:
                    raise AssertionError("IHMDataset changed label or sequence length")
                if source_item[3].shape != (48, 59) or not torch.equal(
                    source_item[3], recreated_expanded_mask
                ):
                    raise AssertionError("IHMDataset expanded mask recreation differs")
                if not np.array_equal(precursor[:, 59:], recreated_data[:, 59:]):
                    raise AssertionError("MEDS failed to recreate the exact 17 masks")
                if not torch.equal(
                    source_item[0][:, categorical_positions],
                    recreated_x[:, categorical_positions],
                ):
                    raise AssertionError("MEDS changed a categorical value channel")
                maximum_float64_precast_difference = max(
                    maximum_float64_precast_difference,
                    float(np.max(np.abs(precursor - recreated_data), initial=0.0)),
                )
                source_items.append(source_item)
                recreated_items.append(recreated_item)
                sample_count += 1

            source_batch = pad_colalte(source_items)
            recreated_batch = pad_colalte(recreated_items)
            if source_batch[0].dtype != torch.float32 or recreated_batch[0].dtype != torch.float32:
                raise AssertionError("collated IHM tensor dtype changed")
            if source_batch[0].shape != recreated_batch[0].shape:
                raise AssertionError("collated tensor shape differs")
            if not torch.equal(source_batch[1], recreated_batch[1]):
                raise AssertionError("collated float32 labels differ")
            if source_batch[2] != recreated_batch[2] or set(source_batch[2]) != {48}:
                raise AssertionError("collated lengths/padding semantics differ")
            if len(source_batch[3]) != len(recreated_batch[3]) or any(
                not torch.equal(left, right)
                for left, right in zip(source_batch[3], recreated_batch[3])
            ):
                raise AssertionError("collated expanded mask tuple differs")
            source_masks = source_batch[0][:, :, 59:].numpy().astype(np.uint8)
            recreated_masks = recreated_batch[0][:, :, 59:].numpy().astype(np.uint8)
            if not np.array_equal(source_masks, recreated_masks):
                raise AssertionError("collated observation masks differ")
            numeric_ulp = float32_ulp_distance(
                source_batch[0][:, :, numeric_positions].numpy(),
                recreated_batch[0][:, :, numeric_positions].numpy(),
            )
            unique, counts = np.unique(numeric_ulp, return_counts=True)
            distribution = {
                int(distance): int(count)
                for distance, count in zip(unique, counts)
            }
            numeric_ulp_distribution.update(distribution)
            per_split_ulp_distribution[split].update(distribution)
            maximum_numeric_ulp = max(
                maximum_numeric_ulp, int(numeric_ulp.max(initial=0))
            )
            maximum_float32_absolute_difference = max(
                maximum_float32_absolute_difference,
                float(
                    np.max(
                        np.abs(
                            source_batch[0][:, :, numeric_positions].numpy()
                            - recreated_batch[0][:, :, numeric_positions].numpy()
                        ),
                        initial=0.0,
                    )
                ),
            )
            observed_shapes.add(tuple(source_batch[0].shape[1:]))
            mask_summary[split]["observed"] += int(source_masks.sum())
            mask_summary[split]["cells"] += int(source_masks.size)
            update_array_digest(source_tensor_digest, source_batch[0])
            update_array_digest(recreated_tensor_digest, recreated_batch[0])
            update_array_digest(source_mask_digest, source_masks)
            update_array_digest(recreated_mask_digest, recreated_masks)
            for mask in source_batch[3]:
                update_array_digest(expanded_mask_digest, mask)
            update_array_digest(label_digest, source_batch[1])
            update_array_digest(per_split[split]["source_tensor"], source_batch[0])
            update_array_digest(
                per_split[split]["recreated_tensor"], recreated_batch[0]
            )
            update_array_digest(per_split[split]["mask"], source_masks)
            update_array_digest(per_split[split]["labels"], source_batch[1])
            batch_count += 1

    try:
        next(event_samples)
    except StopIteration:
        pass
    else:
        raise AssertionError("MEDS contains a sample absent from source listfiles")
    if sample_count != 21_139 or event_count != manifest["event_count"]:
        raise AssertionError("exhaustive batch traversal changed sample/event counts")
    if observed_shapes != {(48, 76)}:
        raise AssertionError("IHM loader introduced padding or another tensor shape")
    if actual_event_codes != set(expected_vocabulary):
        raise AssertionError("actual MEDS event-code set is not the released vocabulary")
    if source_event_digest.hexdigest() != meds_event_digest.hexdigest():
        raise AssertionError("canonical source and MEDS event hashes differ")
    if source_mask_digest.hexdigest() != recreated_mask_digest.hexdigest():
        raise AssertionError("source and recreated mask hashes differ")
    if not args.measure_only and maximum_numeric_ulp != FIXED_NUMERIC_ULP_BOUND:
        raise AssertionError(
            "numeric loader mismatch changed from the exhaustively measured "
            f"{FIXED_NUMERIC_ULP_BOUND}-ULP boundary: {maximum_numeric_ulp}"
        )

    mapping = []
    for variable_index, variable in enumerate(config["id_to_channel"]):
        width = (
            len(config["possible_values"][variable])
            if config["is_categorical_channel"][variable]
            else 1
        )
        mapping.extend([variable_index] * width)
    if len(mapping) != 59 or set(mapping) != set(range(17)):
        raise AssertionError("released 59-to-17 mask mapping changed")
    artifacts = torchmimic_artifacts()
    if sha256(artifacts["normalizer"]) != NORMALIZER_SHA256:
        raise AssertionError("actual dependency normalizer changed")
    result = {
        "status": "measured" if args.measure_only else "exact",
        "selected_experiment": "Table 1 Ours, IHM, 100% labels",
        "selected_model_inputs": ["structured"],
        "retained_modalities": ["structured"],
        "note_codes": [],
        "samples": sample_count,
        "batches": batch_count,
        "events": event_count,
        "tensor_shape": [48, 76],
        "source_precursor_dtype": "float64",
        "getitem_and_collate_dtype": "torch.float32",
        "label_dtype": "torch.float32",
        "sequence_lengths": [48],
        "padding_added": False,
        "value_channels": 59,
        "numeric_value_channels": 12,
        "categorical_value_channels": 47,
        "mask_channels": 17,
        "expanded_returned_mask_shape": [48, 59],
        "source_tensor_sha256": source_tensor_digest.hexdigest(),
        "recreated_tensor_sha256": recreated_tensor_digest.hexdigest(),
        "source_mask_sha256": source_mask_digest.hexdigest(),
        "recreated_mask_sha256": recreated_mask_digest.hexdigest(),
        "expanded_mask_sha256": expanded_mask_digest.hexdigest(),
        "label_sha256": label_digest.hexdigest(),
        "source_event_sha256": source_event_digest.hexdigest(),
        "meds_event_sha256": meds_event_digest.hexdigest(),
        "vocabulary_sha256": hashlib.sha256(
            "\n".join(expected_vocabulary).encode()
        ).hexdigest(),
        "value_to_mask_index_sha256": hashlib.sha256(
            np.asarray(mapping, dtype=np.int16).tobytes()
        ).hexdigest(),
        "mask_summary": mask_summary,
        "maximum_float64_precast_absolute_difference": maximum_float64_precast_difference,
        "maximum_float32_absolute_difference": maximum_float32_absolute_difference,
        "maximum_numeric_ulp": maximum_numeric_ulp,
        "fixed_numeric_ulp_bound": FIXED_NUMERIC_ULP_BOUND,
        "numeric_ulp_distribution": dict(sorted(numeric_ulp_distribution.items())),
        "per_split_numeric_ulp_distribution": {
            split: dict(sorted(values.items()))
            for split, values in per_split_ulp_distribution.items()
        },
        "per_split_hashes": {
            split: {name: digest.hexdigest() for name, digest in values.items()}
            for split, values in per_split.items()
        },
        "normalizer_sha256": sha256(artifacts["normalizer"]),
        "manifest_sha256": manifest_hash_before,
    }
    report_name = (
        "full_batch_measurement.json"
        if args.measure_only
        else "full_batch_recreation.json"
    )
    write_json(args.task_dir / "audit" / report_name, result)
    if sha256(manifest_path) != manifest_hash_before:
        raise AssertionError("batch checker modified immutable manifest.json")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
