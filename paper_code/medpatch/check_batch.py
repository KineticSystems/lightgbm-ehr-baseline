"""Recreate every MedPatch author-loader/collate field from the MEDS task."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


REPOSITORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY))

# Fixed from an exhaustive 26,226-example measurement. This is intentionally an
# ULP rule on the final trainer float32 tensor, not an absolute/relative allclose.
FIXED_NUMERIC_ULP_BOUND = 4_016

from export_source import NOTE_CODE  # noqa: E402
from cohort_source import (  # noqa: E402
    HOURS,
    author_datasets,
    author_preprocessors,
    code_vocabulary,
    configuration,
    load_source_rows,
    variable_names,
)


def iter_meds_samples(path: Path):
    """Yield source-order sample events with one Parquet row group in memory."""
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
            # The note channel is verified separately by note_events(); the
            # structured reconstruction must see exactly the released
            # channel/category events, so skip note rows here while still
            # consuming their source_row to keep the sequence check exact.
            if record["code"] == NOTE_CODE:
                continue
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


def note_events(path: Path) -> dict[int, list[dict]]:
    """Collect every exported note event, keyed by sample, in source order."""
    parquet = pq.ParquetFile(path)
    collected: dict[int, list[dict]] = {}
    for row_group in range(parquet.num_row_groups):
        table = parquet.read_row_group(
            row_group,
            columns=["sample_id", "source_row", "code", "numeric_value", "text_value"],
        ).sort_by([("source_row", "ascending")])
        for record in table.to_pylist():
            if record["code"] != NOTE_CODE:
                continue
            if record["numeric_value"] is not None:
                raise AssertionError("a note event carries a numeric value")
            if not record["text_value"]:
                raise AssertionError("a note event carries empty text")
            collected.setdefault(int(record["sample_id"]), []).append(record)
    return collected


def released_radiology_text(work_root: Path, note_root: Path) -> dict[int, str]:
    """Rebuild the released first-48-hour radiology projection independently.

    This deliberately does not import the exporter's `radiology_notes` module.
    It re-derives the projection from the released `datasets/DataFusion.py`
    `data_pairs='partial'` steps so the batch check is an independent
    reconstruction rather than a restatement of the exporter.
    """
    import pandas as pd

    stays = pd.read_csv(
        work_root / "subjects" / "all_stays.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "intime"],
    )
    stays["intime"] = pd.to_datetime(stays["intime"], errors="raise")

    note_root = Path(note_root)
    for candidate in (
        note_root / "note" / "radiology.csv",
        note_root / "note" / "radiology.csv.gz",
        note_root / "radiology.csv",
        note_root / "radiology.csv.gz",
    ):
        if candidate.is_file():
            radiology_path = candidate
            break
    else:
        raise FileNotFoundError(f"no radiology note table under {note_root}")

    reports = pd.read_csv(
        radiology_path, usecols=["subject_id", "hadm_id", "charttime", "text"]
    )
    reports["charttime"] = pd.to_datetime(reports["charttime"], errors="raise")

    merged = stays.merge(reports, how="inner", on=["subject_id", "hadm_id"])
    window_end = merged["intime"] + pd.DateOffset(hours=48)
    merged = merged.loc[
        (merged["charttime"] >= merged["intime"])
        & (merged["charttime"] <= window_end)
    ]
    merged["text"] = merged["text"].fillna("").astype(str)
    return {
        int(stay_id): " ".join(group["text"].tolist())
        for stay_id, group in merged.groupby("stay_id", sort=False)
    }


def float32_ulp_distance(left, right):
    left = np.ascontiguousarray(left, dtype=np.float32)
    right = np.ascontiguousarray(right, dtype=np.float32)
    if left.shape != right.shape:
        raise AssertionError("ULP comparison received different shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise AssertionError("trainer numeric tensors must be finite")

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
    encoded = code.encode("utf-8")
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
        code = record["code"]
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
            raise AssertionError("MEDS code is absent from the author input declaration")
        if raw[sequence_row, column + 1] != "":
            raise AssertionError("MEDS duplicated one source variable/time cell")
        raw[sequence_row, column + 1] = value
    if not np.isfinite(times).all():
        raise AssertionError("MEDS lost an entire source timeseries row")
    header = ["Hours"] + list(variables)
    data = discretizer.transform(raw, header=header, end=HOURS)[0]
    data = normalizer.transform(data)
    return data


def update_array_digest(digest, array):
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--note-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--measure-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((args.task_dir / "manifest.json").read_text())
    task = manifest["task"]
    source = manifest["source"]
    if task.get("modalities") != ["structured", "notes"] or task.get(
        "note_codes"
    ) != [NOTE_CODE]:
        raise AssertionError(
            "MedPatch common task must contain structured EHR plus the "
            "released radiology-report note channel"
        )
    if source.get("selected_model_inputs") != ["structured", "notes", "imaging"]:
        raise AssertionError("batch check is not using the selected trimodal task")
    if source.get("retained_projection") != ["structured", "notes"]:
        raise AssertionError("batch check must retain exactly structured and notes")
    if source.get("common_profile") != "fixed_numeric_notes":
        raise AssertionError("batch check is not using the prespecified numeric profile")
    if source.get("selection_tier") != 2:
        raise AssertionError("batch check is not using the priority-tier-2 experiment")
    if source.get("selected_launcher") != (
        "scripts/mortality/MedPatch/Confidence-Patching.sh"
    ):
        raise AssertionError("batch check selected the wrong released launcher")
    if source.get("author_preprocessing") != {
        "timestep_hours": 1.0,
        "observation_hours": 48.0,
        "imputation": "previous",
        "start_time": "zero",
        "store_masks": True,
        "value_channels": 59,
        "mask_channels": 17,
        "normalized_continuous_channels": 12,
    }:
        raise AssertionError("batch check selected the wrong EHR preprocessing config")

    expected_vocabulary = code_vocabulary() + (NOTE_CODE,)
    codes_path = args.task_dir / manifest["meds"]["codes"]["file"]
    actual_vocabulary = tuple(pq.read_table(codes_path)["code"].to_pylist())
    if actual_vocabulary != expected_vocabulary:
        raise AssertionError("MEDS vocabulary differs from the released channel/category order")
    structured_vocabulary = tuple(
        code for code in actual_vocabulary if code != NOTE_CODE
    )
    if len(structured_vocabulary) != 59 or len(set(structured_vocabulary)) != 59:
        raise AssertionError("MedPatch structured vocabulary must contain 59 unique codes")

    events_path = args.task_dir / manifest["meds"]["events"]["file"]
    event_samples = iter_meds_samples(events_path)
    actual_event_codes = set()
    if any(
        token in code.lower()
        for code in structured_vocabulary
        for token in ("cxr", "image", "radiology", "note", "pair", "modality")
    ):
        raise AssertionError("imaging/text/presence data entered the structured task")
    source_rows = load_source_rows(args.source_root)
    rows_by_split = {
        split: [row for row in source_rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    loader_module, datasets, output_header = author_datasets(args.source_root)
    recreated_discretizer, recreated_normalizer, recreated_header = (
        author_preprocessors()
    )
    if recreated_header != output_header:
        raise AssertionError("author discretizer returned inconsistent headers")
    if len(output_header) != 76:
        raise AssertionError("author loader does not return the audited 76 channels")
    numeric_positions = tuple(
        position
        for position, name in enumerate(output_header[:59])
        if "->" not in name
    )
    categorical_positions = tuple(
        position for position in range(59) if position not in numeric_positions
    )
    if len(numeric_positions) != 12 or len(categorical_positions) != 47:
        raise AssertionError("unexpected numeric/categorical trainer channel partition")
    config = configuration()
    variables = variable_names()
    if tuple(config["id_to_channel"]) != variables or len(variables) != 17:
        raise AssertionError("author structured variable declaration changed")

    source_tensor_digest = hashlib.sha256()
    recreated_tensor_digest = hashlib.sha256()
    source_mask_digest = hashlib.sha256()
    recreated_mask_digest = hashlib.sha256()
    source_event_digest = hashlib.sha256()
    meds_event_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    split_digests = {
        split: {
            "source_model": hashlib.sha256(),
            "recreated_model": hashlib.sha256(),
            "source_mask": hashlib.sha256(),
            "recreated_mask": hashlib.sha256(),
            "labels": hashlib.sha256(),
        }
        for split in ("train", "validation", "test")
    }
    mask_summary = {
        split: {"observed": 0, "cells": 0}
        for split in ("train", "validation", "test")
    }
    maximum_loader_difference = 0.0
    numeric_ulp_distribution = Counter()
    per_split_numeric_ulp_distribution = {
        split: Counter() for split in ("train", "validation", "test")
    }
    numeric_ulp_cells = 0
    numeric_ulp_nonzero = 0
    maximum_numeric_ulp = 0
    maximum_model_float32_absolute_difference = 0.0
    sample_count = 0
    batch_count = 0
    observed_shapes = set()

    for split in ("train", "validation", "test"):
        dataset = datasets[split]
        rows = rows_by_split[split]
        if len(dataset) != len(rows):
            raise AssertionError(f"{split} exact loader/listfile row count differs")
        for batch_start in range(0, len(rows), args.batch_size):
            source_items = []
            recreated_items = []
            for offset, row in enumerate(
                rows[batch_start : batch_start + args.batch_size], start=batch_start
            ):
                if dataset.names[offset] != row["source_file"]:
                    raise AssertionError("author loader source order differs from listfile")
                raw_item = dataset.read_by_file_name(row["source_file"], None)
                raw = raw_item["X"]
                raw_header = raw_item["header"]
                expected = expected_events(raw, raw_header, config)
                try:
                    actual_sample_id, actual = next(event_samples)
                except StopIteration as error:
                    raise AssertionError("MEDS omitted a released event sample") from error
                if actual_sample_id != int(row["stay_id"]):
                    raise AssertionError("MEDS event sample order/cohort differs from source")
                if len(expected) != len(actual):
                    raise AssertionError("MEDS changed the number of observed source cells")
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
                        actual_row["code"],
                        None
                        if actual_row["numeric_value"] is None
                        else float(np.float32(actual_row["numeric_value"])),
                    )
                    if comparable != expected_row:
                        raise AssertionError("MEDS changed an observed source event")
                    actual_event_codes.add(actual_row["code"])
                    canonical_event_digest(
                        source_event_digest,
                        sequence_row,
                        source_time,
                        code,
                        numeric,
                    )
                    canonical_event_digest(
                        meds_event_digest,
                        *comparable,
                    )

                source_item = dataset[offset]
                recreated_data = recreate_item(
                    actual, recreated_discretizer, recreated_normalizer, config
                )
                recreated_item = (recreated_data, np.int32(row["label"]))
                if source_item[0].dtype != np.float64 or recreated_data.dtype != np.float64:
                    raise AssertionError("unexpected author/recreated loader dtype")
                if source_item[0].shape != (48, 76) or recreated_data.shape != (48, 76):
                    raise AssertionError("unexpected author/recreated loader shape")
                if int(source_item[1]) != int(row["label"]):
                    raise AssertionError("author loader changed the mortality label")
                if not np.array_equal(source_item[0][:, 59:], recreated_data[:, 59:]):
                    raise AssertionError("MEDS did not exactly recreate the 17 masks")
                if not np.array_equal(
                    source_item[0][:, categorical_positions].astype(np.float32),
                    recreated_data[:, categorical_positions].astype(np.float32),
                ):
                    raise AssertionError("MEDS changed a categorical trainer channel")
                maximum_loader_difference = max(
                    maximum_loader_difference,
                    float(np.max(np.abs(source_item[0] - recreated_data), initial=0.0)),
                )
                source_items.append(source_item)
                recreated_items.append(recreated_item)
                sample_count += 1

            source_batch = loader_module.my_collate(source_items)
            recreated_batch = loader_module.my_collate(recreated_items)
            if source_batch[0].dtype != recreated_batch[0].dtype:
                raise AssertionError("collated loader dtype differs")
            if source_batch[0].shape != recreated_batch[0].shape:
                raise AssertionError("collated loader shape differs")
            if not np.array_equal(source_batch[1], recreated_batch[1]):
                raise AssertionError("collated labels differ")
            if source_batch[2] != recreated_batch[2] or set(source_batch[2]) != {48}:
                raise AssertionError("collated sequence lengths/padding differ")
            source_model = source_batch[0].astype(np.float32)
            recreated_model = recreated_batch[0].astype(np.float32)
            source_masks = source_batch[0][:, :, 59:].astype(np.uint8)
            recreated_masks = recreated_batch[0][:, :, 59:].astype(np.uint8)
            if not np.array_equal(source_masks, recreated_masks):
                raise AssertionError("collated masks differ")
            if not np.array_equal(
                source_model[:, :, categorical_positions],
                recreated_model[:, :, categorical_positions],
            ):
                raise AssertionError("collated categorical trainer channels differ")
            numeric_ulp = float32_ulp_distance(
                source_model[:, :, numeric_positions],
                recreated_model[:, :, numeric_positions],
            )
            unique_ulp, unique_counts = np.unique(numeric_ulp, return_counts=True)
            batch_distribution = {
                int(distance): int(count)
                for distance, count in zip(unique_ulp, unique_counts)
            }
            numeric_ulp_distribution.update(batch_distribution)
            per_split_numeric_ulp_distribution[split].update(batch_distribution)
            numeric_ulp_cells += int(numeric_ulp.size)
            numeric_ulp_nonzero += int(np.count_nonzero(numeric_ulp))
            batch_maximum_ulp = int(numeric_ulp.max(initial=0))
            maximum_numeric_ulp = max(maximum_numeric_ulp, batch_maximum_ulp)
            maximum_model_float32_absolute_difference = max(
                maximum_model_float32_absolute_difference,
                float(
                    np.max(
                        np.abs(
                            source_model[:, :, numeric_positions]
                            - recreated_model[:, :, numeric_positions]
                        ),
                        initial=0.0,
                    )
                ),
            )
            if (
                not args.measure_only
                and FIXED_NUMERIC_ULP_BOUND is not None
                and batch_maximum_ulp > FIXED_NUMERIC_ULP_BOUND
            ):
                raise AssertionError(
                    "numeric trainer tensor exceeds authorized ULP bound: "
                    f"observed={batch_maximum_ulp} bound={FIXED_NUMERIC_ULP_BOUND}"
                )
            update_array_digest(source_tensor_digest, source_model)
            update_array_digest(recreated_tensor_digest, recreated_model)
            update_array_digest(source_mask_digest, source_masks)
            update_array_digest(recreated_mask_digest, recreated_masks)
            update_array_digest(label_digest, source_batch[1].astype(np.int32))
            update_array_digest(split_digests[split]["source_model"], source_model)
            update_array_digest(
                split_digests[split]["recreated_model"], recreated_model
            )
            update_array_digest(split_digests[split]["source_mask"], source_masks)
            update_array_digest(
                split_digests[split]["recreated_mask"], recreated_masks
            )
            update_array_digest(
                split_digests[split]["labels"], source_batch[1].astype(np.int32)
            )
            mask_summary[split]["observed"] += int(source_masks.sum())
            mask_summary[split]["cells"] += int(source_masks.size)
            observed_shapes.add(tuple(source_batch[0].shape))
            batch_count += 1

    try:
        extra_sample = next(event_samples)
    except StopIteration:
        extra_sample = None
    if extra_sample is not None:
        raise AssertionError("MEDS added an event sample absent from the source cohort")
    if not actual_event_codes.issubset(expected_vocabulary):
        raise AssertionError("MEDS contains an auxiliary or undeclared event")
    if sum(numeric_ulp_distribution.values()) != numeric_ulp_cells:
        raise AssertionError("numeric ULP histogram does not cover every numeric cell")
    if numeric_ulp_distribution.get(0, 0) != numeric_ulp_cells - numeric_ulp_nonzero:
        raise AssertionError("numeric ULP nonzero count disagrees with histogram")

    digests = {
        "source_model_tensor_sha256": source_tensor_digest.hexdigest(),
        "recreated_model_tensor_sha256": recreated_tensor_digest.hexdigest(),
        "source_mask_sha256": source_mask_digest.hexdigest(),
        "recreated_mask_sha256": recreated_mask_digest.hexdigest(),
        "source_event_projection_sha256": source_event_digest.hexdigest(),
        "meds_event_projection_sha256": meds_event_digest.hexdigest(),
        "ordered_label_sha256": label_digest.hexdigest(),
    }
    if digests["source_mask_sha256"] != digests["recreated_mask_sha256"]:
        raise AssertionError("mask hashes differ")
    if digests["source_event_projection_sha256"] != digests[
        "meds_event_projection_sha256"
    ]:
        raise AssertionError("event projection hashes differ")
    per_split_digests = {}
    for split, values in split_digests.items():
        per_split_digests[split] = {
            name + "_sha256": digest.hexdigest()
            for name, digest in values.items()
        }
        if per_split_digests[split]["source_mask_sha256"] != per_split_digests[
            split
        ]["recreated_mask_sha256"]:
            raise AssertionError(f"{split} mask hashes differ")

    if not args.measure_only:
        if FIXED_NUMERIC_ULP_BOUND is None:
            raise AssertionError("numeric ULP bound has not been fixed from measurement")
        if maximum_numeric_ulp != FIXED_NUMERIC_ULP_BOUND:
            raise AssertionError(
                "exhaustive numeric maximum no longer reproduces the smallest fixed "
                f"ULP bound: observed={maximum_numeric_ulp} "
                f"bound={FIXED_NUMERIC_ULP_BOUND}"
            )

    # Independently reconstruct the released radiology projection and assert
    # that the exported note channel reproduces it exactly, both in membership
    # and in the authors' space-joined text.
    expected_notes = released_radiology_text(args.source_root.parent, args.note_root)
    exported_notes = note_events(events_path)
    cohort_stays = {int(row["stay_id"]) for row in source_rows}
    expected_in_cohort = {
        stay_id: text
        for stay_id, text in expected_notes.items()
        if stay_id in cohort_stays and text
    }
    for stay_id, records in exported_notes.items():
        if stay_id not in cohort_stays:
            raise AssertionError("a note event references a sample outside the cohort")
        if len(records) != 1:
            raise AssertionError(
                "the released BioBERT branch receives one joined string per stay, "
                "so each sample must carry at most one note event"
            )
    if set(exported_notes) != set(expected_in_cohort):
        missing = len(set(expected_in_cohort) - set(exported_notes))
        extra = len(set(exported_notes) - set(expected_in_cohort))
        raise AssertionError(
            f"note membership differs from the released projection "
            f"({missing} missing, {extra} unexpected)"
        )
    for stay_id, text in expected_in_cohort.items():
        if exported_notes[stay_id][0]["text_value"] != text:
            raise AssertionError(
                f"exported note text differs from the released join for stay {stay_id}"
            )
    note_report = {
        "released_construction": (
            "datasets/DataFusion.py loadmetadata, data_pairs=partial: inner "
            "merge on (subject_id, hadm_id), charttime within [intime, "
            "intime+48h], groupby then ' '.join(text), left-joined onto every "
            "EHR stay"
        ),
        "independent_reconstruction": True,
        "cohort_samples": len(cohort_stays),
        "samples_with_report": len(expected_in_cohort),
        "samples_without_report": len(cohort_stays) - len(expected_in_cohort),
        "note_events": sum(len(v) for v in exported_notes.values()),
        "text_sha256": hashlib.sha256(
            "\n".join(
                exported_notes[stay_id][0]["text_value"]
                for stay_id in sorted(exported_notes)
            ).encode("utf-8")
        ).hexdigest(),
        "exact_text_match": True,
    }

    report = {
        "status": "measurement" if args.measure_only else "exact",
        "note_channel": note_report,
        "protocol_status_meaning": (
            "measurement only"
            if args.measure_only
            else "exact under the declared numeric ULP rule; every other field is exact"
        ),
        "selected_experiment": source["selected_experiment"],
        "selected_launcher": source["selected_launcher"],
        "selected_model_inputs": ["structured", "notes", "imaging"],
        "retained_projection": ["structured", "notes"],
        "examples": sample_count,
        "batches": batch_count,
        "batch_size": args.batch_size,
        "author_loader": {
            "per_example_shape": [48, 76],
            "dtype": "float64",
            "value_channels": 59,
            "observation_mask_channels": 17,
            "sequence_length": 48,
            "collated_shapes": [list(shape) for shape in sorted(observed_shapes)],
            "actual_model_dtype": "float32",
        },
        "maximum_pre_model_float64_difference_from_meds_float32": maximum_loader_difference,
        "numeric_precision": {
            "scope": "12 numeric channels of the final trainer float32 tensor",
            "rule": "integer distance between adjacent IEEE-754 float32 values",
            "authorized_reason": (
                "user-authorized tolerance solely for attributable source float64 "
                "to MEDS float32 round-trip; no rtol/atol allclose"
            ),
            "fixed_ulp_bound": FIXED_NUMERIC_ULP_BOUND,
            "maximum_observed_ulp": maximum_numeric_ulp,
            "maximum_absolute_difference": maximum_model_float32_absolute_difference,
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
        "event_vocabulary": {
            "declared_count": len(actual_vocabulary),
            "observed_count": len(actual_event_codes),
            "ordered_code_sha256": hashlib.sha256(
                "\n".join(actual_vocabulary).encode("utf-8")
            ).hexdigest(),
            "author_output_header_sha256": hashlib.sha256(
                "\n".join(output_header).encode("utf-8")
            ).hexdigest(),
            "note_codes": [NOTE_CODE],
            "all_non_note_codes_are_released_structured_inputs": True,
            "imaging_or_auxiliary_codes": [],
        },
        "digests": digests,
        "per_split_digests": per_split_digests,
        "mask_missingness": {
            split: {
                **summary,
                "observed_fraction": summary["observed"] / summary["cells"],
            }
            for split, summary in mask_summary.items()
        },
        "checks": [
            "exact observed source time/category/value projection after required MEDS float32 encoding",
            "exact exhaustive 17-channel observation masks",
            "exhaustive numeric trainer float32 tensors within the fixed authorized ULP bound",
            "exact exhaustive categorical trainer float32 channels",
            "exact labels, shapes, dtypes, sequence lengths, and no padding",
            "exact event identifiers, timestamps, split, source order, and sample order",
            "exact released 59-code vocabulary and empty note-code exception list",
            "no imaging, text, availability, identifier, label, or auxiliary events",
        ],
    }
    if not args.measure_only:
        report_path = args.task_dir / "audit" / "full_batch_recreation.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
