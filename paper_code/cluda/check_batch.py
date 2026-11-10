"""Exhaustively recreate CLUDA's released loader/collate from its MEDS task."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data._utils.collate import default_collate


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from cohort_source import (  # noqa: E402
    CODES,
    HISTORY_HOURS,
    MIN_HOURS,
    SOURCE_AGE_GROUP,
    STATIC_FEATURES,
    TARGET_AGE_GROUP,
    TASK_ID,
    TS_FEATURES,
    load_author_source,
    require_source,
    sample_id,
    selected_splits,
    sha256,
    validate_age_scaling,
    validate_upstream_files,
    write_json,
)
from utils import dataset as released_dataset  # noqa: E402


def author_dataset(arrays, age_group: int):
    released_name = {"train": "train", "validation": "val", "test": "test"}[
        arrays.source_split
    ]
    payload = {
        f"timeseries_{released_name}.npy": arrays.sequence,
        f"timeseries_{released_name}_mask.npy": arrays.sequence_mask,
        f"static_{released_name}.npy": arrays.static,
        f"mortality_{released_name}.npy": arrays.mortality,
    }

    def fake_load(path, allow_pickle=True):
        name = Path(path).name
        if name not in payload:
            raise AssertionError(f"released loader requested unexpected artifact: {name}")
        if not allow_pickle:
            raise AssertionError("released ragged arrays require allow_pickle=True")
        return payload[name]

    with patch.object(released_dataset.np, "load", side_effect=fake_load):
        result = released_dataset.ICUDataset(
            "/instrumented/miiv_fullstays",
            split_type=released_name,
            age_group=age_group,
            stay_hours=HISTORY_HOURS,
            min_hours=MIN_HOURS,
            task="mortality",
            is_full_subset=True,
            is_cuda=False,
        )
    return result


def iter_event_groups(path: Path):
    current_id = None
    current: dict[str, list] | None = None
    seen: set[int] = set()
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
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=250_000):
        values = {name: batch.column(name).to_pylist() for name in columns}
        values["time"] = (
            batch.column("time")
            .to_numpy(zero_copy_only=False)
            .astype("datetime64[us]")
            .astype(np.int64, copy=False)
        )
        for row_index, sid_value in enumerate(values["sample_id"]):
            sid = int(sid_value)
            if current_id != sid:
                if current is not None:
                    if current_id in seen:
                        raise AssertionError("a CLUDA sample's events are non-contiguous")
                    seen.add(int(current_id))
                    yield int(current_id), current
                current_id = sid
                current = {name: [] for name in columns}
            assert current is not None
            for name in columns:
                current[name].append(values[name][row_index])
    if current is not None:
        if current_id in seen:
            raise AssertionError("a CLUDA sample's events are non-contiguous")
        yield int(current_id), current


def reconstruct(
    rows: dict[str, list], index_row: pd.Series, source_charttime: np.ndarray
) -> dict:
    end_index = int(index_row["stay_hour"])
    start_index = max(0, end_index - HISTORY_HOURS)
    width = end_index - start_index
    pad = HISTORY_HOURS - width
    sequence = np.zeros((HISTORY_HOURS, len(TS_FEATURES)), dtype=np.float32)
    intended_mask = np.zeros((HISTORY_HOURS, len(TS_FEATURES)), dtype=np.int64)
    static = np.zeros(len(STATIC_FEATURES), dtype=np.float32)
    static_seen = np.zeros(len(STATIC_FEATURES), dtype=bool)
    carry_seen = np.zeros(len(TS_FEATURES), dtype=bool)
    observed_seen: set[tuple[int, int]] = set()
    value_updates: dict[tuple[int, int], np.float32] = {}
    expected_subject_id = int(index_row["subject_id"])
    source_charttime = np.asarray(source_charttime, dtype=np.int64)
    if len(source_charttime) < end_index:
        raise AssertionError("released source clock is shorter than the selected window")
    source_start_hour = int(source_charttime[start_index])
    observation_start_us = int(
        np.datetime64(index_row["observation_start"], "us").astype(np.int64)
    )
    hour_us = 3_600_000_000
    persisted_source_rows = np.asarray(rows["source_row"], dtype=np.int64)
    if len(persisted_source_rows) > 1 and not np.all(np.diff(persisted_source_rows) > 0):
        raise AssertionError("CLUDA events changed canonical source-row order")

    for row, (code, value) in enumerate(zip(rows["code"], rows["numeric_value"])):
        if int(rows["subject_id"][row]) != expected_subject_id:
            raise AssertionError("MEDS event subject differs from its CLUDA task index row")
        event_time_us = int(rows["time"][row])
        source_row = int(rows["source_row"][row])
        sequence_row = int(rows["sequence_row"][row])
        source_time = float(rows["source_time"][row])
        if code.startswith("STATIC/"):
            feature = code.split("/", 1)[1]
            feature_index = STATIC_FEATURES.index(feature)
            if (
                static_seen[feature_index]
                or sequence_row != -1
                or source_time != source_start_hour
                or source_row != feature_index
                or event_time_us != observation_start_us
            ):
                raise AssertionError("static event duplication/position changed")
            static[feature_index] = np.float32(value)
            static_seen[feature_index] = True
        elif code.startswith("CARRY/"):
            if end_index < HISTORY_HOURS:
                raise AssertionError("an early defective window exposed the discarded true mask")
            feature = code.split("/", 1)[1]
            feature_index = TS_FEATURES.index(feature)
            if (
                carry_seen[feature_index]
                or sequence_row != pad
                or source_time != source_start_hour
                or source_row != 100 + feature_index
                or event_time_us != observation_start_us
            ):
                raise AssertionError("window-boundary carry encoding changed")
            sequence[pad, feature_index] = np.float32(value)
            carry_seen[feature_index] = True
        elif code.startswith("VALUE/"):
            if end_index >= HISTORY_HOURS:
                raise AssertionError("a normal window used the early-defect value encoding")
            feature = code.split("/", 1)[1]
            feature_index = TS_FEATURES.index(feature)
            key = (sequence_row, feature_index)
            if key in value_updates or not pad <= sequence_row < HISTORY_HOURS:
                raise AssertionError("early-window value-run encoding changed")
            source_index = start_index + sequence_row - pad
            expected_source_time = int(source_charttime[source_index])
            if source_time != expected_source_time:
                raise AssertionError("early-window value source coordinate changed")
            expected_source_row = 500 + source_index * len(TS_FEATURES) + feature_index
            expected_time_us = observation_start_us + (
                expected_source_time - source_start_hour
            ) * hour_us
            if source_row != expected_source_row or event_time_us != expected_time_us:
                raise AssertionError("early-window event time/source-row coordinate changed")
            value_updates[key] = np.float32(value)
        elif code.startswith("TS/"):
            if end_index < HISTORY_HOURS:
                raise AssertionError("an early defective window exposed a true-mask occurrence")
            feature = code.split("/", 1)[1]
            feature_index = TS_FEATURES.index(feature)
            key = (sequence_row, feature_index)
            if key in observed_seen or not pad <= sequence_row < HISTORY_HOURS:
                raise AssertionError("hourly observation encoding changed")
            source_index = start_index + sequence_row - pad
            expected_source_time = int(source_charttime[source_index])
            if source_time != expected_source_time:
                raise AssertionError("released hourly source coordinate changed")
            expected_source_row = 1_000 + source_index * len(TS_FEATURES) + feature_index
            expected_time_us = observation_start_us + (
                expected_source_time - source_start_hour
            ) * hour_us
            if source_row != expected_source_row or event_time_us != expected_time_us:
                raise AssertionError("hourly event time/source-row coordinate changed")
            sequence[sequence_row, feature_index] = np.float32(value)
            intended_mask[sequence_row, feature_index] = 1
            observed_seen.add(key)
        else:
            raise AssertionError(f"unknown retained CLUDA code: {code}")
    if not static_seen.all():
        raise AssertionError("MEDS omitted a released static input")

    if end_index < HISTORY_HOURS:
        current = np.zeros(len(TS_FEATURES), dtype=np.float32)
        for row in range(pad, HISTORY_HOURS):
            for feature_index in range(len(TS_FEATURES)):
                key = (row, feature_index)
                if key in value_updates:
                    value = value_updates[key]
                    if value == current[feature_index]:
                        raise AssertionError("early-window value run contains a redundant update")
                    current[feature_index] = value
            sequence[row] = current
    else:
        for row in range(pad + 1, HISTORY_HOURS):
            missing = intended_mask[row] == 0
            sequence[row, missing] = sequence[row - 1, missing]
    # Code-faithful defect in ICUDataset.get_subsequence(): every endpoint
    # before 48 replaces the mask by the feature tensor and then torch.long()
    # truncates values toward zero.
    if end_index < HISTORY_HOURS:
        loader_mask = sequence.astype(np.int64)
    else:
        loader_mask = intended_mask
    expected_event_count = (
        len(STATIC_FEATURES)
        + int(carry_seen.sum())
        + len(observed_seen)
        + len(value_updates)
    )
    if expected_event_count != len(rows["code"]):
        raise AssertionError("MEDS contains redundant CLUDA boundary events")
    return {
        "sequence": torch.from_numpy(sequence),
        "sequence_mask": torch.from_numpy(loader_mask),
        "static": torch.from_numpy(static),
        "label": torch.tensor([float(bool(index_row["boolean_value"]))], dtype=torch.float32),
        "patient_id": int(index_row["loader_patient_id"]),
        "stay_hour": end_index,
    }


def compare_collated(author_rows: list[dict], meds_rows: list[dict], hashes: dict) -> None:
    author_batch = default_collate(author_rows)
    meds_batch = default_collate(meds_rows)
    expected_dtypes = {
        "sequence": torch.float32,
        "sequence_mask": torch.int64,
        "static": torch.float32,
        "label": torch.float32,
        "patient_id": torch.int64,
        "stay_hour": torch.int64,
    }
    for name, dtype in expected_dtypes.items():
        left = author_batch[name]
        right = meds_batch[name]
        if left.dtype != dtype or right.dtype != dtype:
            raise AssertionError(f"CLUDA collate dtype changed: {name}")
        if left.shape != right.shape or not torch.equal(left, right):
            raise AssertionError(f"MEDS failed exact released collate equality: {name}")
        hashes[name].update(left.detach().cpu().numpy().tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--allow-synthetic", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    source_manifest = require_source(args.source_root, allow_synthetic=args.allow_synthetic)
    upstream_files = validate_upstream_files()
    age_scaling = validate_age_scaling(
        args.source_root, allow_synthetic=args.allow_synthetic
    )
    manifest_path = args.task_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_manifest_sha256 = sha256(args.source_root / "source_manifest.json")
    if manifest["source"].get("source_manifest_sha256") != source_manifest_sha256:
        raise AssertionError("MEDS task is not sealed to the exact CLUDA source manifest")
    if manifest["source"].get("source_reconstruction_status") != source_manifest["status"]:
        raise AssertionError("MEDS task changed the CLUDA source reconstruction status")
    if manifest["task"]["id"] != TASK_ID:
        raise AssertionError("batch check selected the wrong CLUDA task")
    if not args.allow_synthetic and manifest["source"].get("upstream_file_sha256") != upstream_files:
        raise AssertionError("MEDS task is not sealed to the executed CLUDA upstream files")
    if manifest["task"]["modalities"] != ["structured"] or manifest["task"]["note_codes"]:
        raise AssertionError("CLUDA batch must contain structured inputs only")
    codes = pq.read_table(args.task_dir / manifest["meds"]["codes"]["file"])["code"].to_pylist()
    if codes != list(CODES):
        raise AssertionError("CLUDA batch vocabulary/order changed")

    source = load_author_source(args.source_root, allow_synthetic=args.allow_synthetic)
    expected_selected = selected_splits(source)
    author = {
        "train": author_dataset(source["train"], SOURCE_AGE_GROUP),
        "validation": author_dataset(source["validation"], SOURCE_AGE_GROUP),
        "test": author_dataset(source["test"], TARGET_AGE_GROUP),
    }
    for split, dataset in author.items():
        if len(dataset.sequence) != len(expected_selected[split].sequence):
            raise AssertionError("released loader age/length membership differs from source projection")

    index_path = args.task_dir / manifest["meds"]["task_index"]["file"]
    index = pq.read_table(index_path).to_pandas().set_index("sample_id", verify_integrity=True)
    lookup: dict[int, tuple[str, int]] = {}
    for split in ("train", "validation", "test"):
        arrays = expected_selected[split]
        dataset = author[split]
        for position, (patient_id, end_index) in enumerate(dataset.stay_dict):
            stay_id = int(arrays.stay_ids[int(patient_id)])
            sid = sample_id(stay_id, int(end_index))
            if sid in lookup:
                raise AssertionError("released task sample ID is not unique")
            lookup[sid] = (split, position)
    if set(lookup) != set(index.index.astype(int)):
        raise AssertionError("MEDS task index differs from released loader traversal")
    hashes = {name: hashlib.sha256() for name in (
        "sequence", "sequence_mask", "static", "label", "patient_id", "stay_hour"
    )}
    author_buffer: list[dict] = []
    meds_buffer: list[dict] = []
    checked = 0
    early_bug = 0
    event_path = args.task_dir / manifest["meds"]["events"]["file"]
    for sid, rows in iter_event_groups(event_path):
        try:
            split, position = lookup.pop(sid)
        except KeyError as error:
            raise AssertionError(f"duplicate or unknown MEDS sample: {sid}") from error
        index_row = index.loc[sid]
        if index_row["split"] != split:
            raise AssertionError("MEDS split differs from released loader")
        author_row = author[split][position]
        loader_patient_id = int(author_row["patient_id"])
        selected_arrays = expected_selected[split]
        source_charttime = selected_arrays.source_charttime[loader_patient_id]
        end_index = int(author_row["stay_hour"])
        start_index = max(0, end_index - HISTORY_HOURS)
        expected_observation_start = selected_arrays.intimes[loader_patient_id] + np.timedelta64(
            int(source_charttime[start_index]), "h"
        )
        expected_prediction_time = selected_arrays.intimes[loader_patient_id] + np.timedelta64(
            int(source_charttime[end_index - 1]) + 1, "h"
        )
        if (
            np.datetime64(index_row["observation_start"], "us")
            != expected_observation_start
            or np.datetime64(index_row["prediction_time"], "us")
            != expected_prediction_time
        ):
            raise AssertionError("MEDS landmark changed CLUDA's absolute source clock")
        meds_row = reconstruct(rows, index_row, source_charttime)
        if int(author_row["stay_hour"]) < HISTORY_HOURS:
            early_bug += 1
        author_buffer.append(author_row)
        meds_buffer.append(meds_row)
        checked += 1
        if len(author_buffer) == 512:
            compare_collated(author_buffer, meds_buffer, hashes)
            author_buffer.clear()
            meds_buffer.clear()
    if author_buffer:
        compare_collated(author_buffer, meds_buffer, hashes)
    if lookup:
        raise AssertionError(f"MEDS omitted {len(lookup)} released loader samples")
    if checked != len(index):
        raise AssertionError("batch traversal did not cover every CLUDA sample")

    report = {
        "status": (
            "synthetic_exact_loader_projection" if args.allow_synthetic else
            "exact_structured_source_boundary"
        ),
        "selected_source_configuration": manifest["source"]["selected_source_configuration"],
        "retained_projection": {
            "modalities": ["structured"],
            "codes": codes,
            "note_codes": [],
            "time_series_shape": [HISTORY_HOURS, len(TS_FEATURES)],
            "static_shape": [len(STATIC_FEATURES)],
            "all_samples_checked": checked,
            "early_mask_bug_samples": early_bug,
        },
        "collate": {
            "implementation": "torch.utils.data.default_collate over released ICUDataset.__getitem__",
            "batch_size": 512,
            "exhaustive": True,
            "ordered_source_sha256": {name: digest.hexdigest() for name, digest in hashes.items()},
            "dtypes": {
                "sequence": "torch.float32",
                "sequence_mask": "torch.int64",
                "static": "torch.float32",
                "label": "torch.float32",
                "patient_id": "torch.int64",
                "stay_hour": "torch.int64",
            },
        },
        "source_manifest_sha256": source_manifest_sha256,
        "upstream_file_sha256": upstream_files,
        "age_scaling_reconstruction": age_scaling,
        "manifest_sha256": sha256(manifest_path),
        "checks": [
            "released ICUDataset invoked with exact mortality/age/history/min-hour configuration",
            "every released prediction window and default-collated field compared elementwise",
            "short-window integer-cast feature-as-mask defect preserved",
            "exact dynamic/static vocabulary and structured-only modality projection",
            "all values, masks, static fields, labels, identifiers, shapes, and dtypes matched",
            "all persisted event subjects, timestamps, source rows, and sequence coordinates matched",
            "absolute RICU charttime and sequence-relative released-loader coordinates both matched",
            "executed producer/loader and feature/static-tail evidence match pinned upstream files",
            "training age scaling agrees with the released loader constants",
        ],
        "source_status": source_manifest["status"],
        "fidelity_qualifications": {
            "ricu": "released code did not pin RICU; the export uses time-compatible RICU 0.5.2",
            "split_rng": (
                "the released script's 37-name-draw pre-split state does not reproduce the "
                "paper artifacts; the export fail-closed advances to the uniquely matching "
                "72-draw paper-evaluated state, equivalent to one additional released "
                "time-series loader invocation"
            ),
            "static_tail": "mechanical identifier-renamed counterpart of the released AUMC tail",
            "preserved_loader_defects": True,
        },
    }
    output = args.task_dir / "audit" / "full_batch_recreation.json"
    write_json(output, report)
    print(json.dumps({"status": report["status"], "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
