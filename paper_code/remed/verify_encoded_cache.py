#!/usr/bin/env python3
"""Verify a complete released REMed event-vector cache against source HDF5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


EXPECTED_STAYS = 32_449
EXPECTED_EVENTS = 129_548_126
EXPECTED_MAX_EVENTS = 266_766
EXPECTED_WIDTH = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def equal_attribute(left, right) -> bool:
    left_array, right_array = np.asarray(left), np.asarray(right)
    try:
        return np.array_equal(left_array, right_array, equal_nan=True)
    except TypeError:
        return np.array_equal(left_array, right_array)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-hdf", type=Path, required=True)
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--encoded-hdf", type=Path, required=True)
    parser.add_argument("--stage-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    stage_report = json.loads(args.stage_report.read_text())
    if stage_report.get("status") != "terminal_source_stage":
        raise AssertionError(stage_report.get("status"))
    for path in (args.source_hdf, args.cohort_csv, args.encoded_hdf):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    cohort = pd.read_csv(
        args.cohort_csv,
        usecols=["stay_id", "split_2020"],
        dtype={"stay_id": "int64", "split_2020": "string"},
    )
    split_by_key = dict(
        zip(cohort["stay_id"].astype(str), cohort["split_2020"].astype(str))
    )
    if len(split_by_key) != EXPECTED_STAYS:
        raise AssertionError(len(split_by_key))

    per_split_longest = {
        split: {"events": -1, "stay_id": None} for split in ("train", "valid", "test")
    }
    event_total = 0
    global_longest = {"events": -1, "stay_id": None}
    zero_rows = 0
    nonfinite_values = 0

    with h5py.File(args.source_hdf, "r") as source, h5py.File(
        args.encoded_hdf, "r"
    ) as encoded:
        source_ehr, encoded_ehr = source["ehr"], encoded["ehr"]
        source_keys = list(source_ehr.keys())
        encoded_keys = list(encoded_ehr.keys())
        if len(source_keys) != EXPECTED_STAYS or set(source_keys) != set(encoded_keys):
            raise AssertionError((len(source_keys), len(encoded_keys)))
        if set(source_keys) != set(split_by_key):
            raise AssertionError("cohort/source HDF stay membership mismatch")

        for index, key in enumerate(source_keys, start=1):
            source_group = source_ehr[key]
            encoded_group = encoded_ehr[key]
            source_time = source_group["time"][()]
            encoded_time = encoded_group["time"][()]
            if not np.array_equal(source_time, encoded_time):
                raise AssertionError(f"time mismatch for stay {key}")
            events = len(source_time)
            dataset = encoded_group["encoded"]
            if dataset.shape != (events, EXPECTED_WIDTH) or dataset.dtype != np.dtype("int16"):
                raise AssertionError((key, dataset.shape, dataset.dtype, events))
            if set(source_group.attrs) != set(encoded_group.attrs):
                raise AssertionError(f"attribute names differ for stay {key}")
            for name in source_group.attrs:
                if not equal_attribute(source_group.attrs[name], encoded_group.attrs[name]):
                    raise AssertionError(f"attribute {name} differs for stay {key}")

            values = np.ascontiguousarray(dataset[()])
            bits = values.view(np.uint16)
            zero_rows += int(np.all(bits == 0, axis=1).sum())
            nonfinite_values += int(((bits & 0x7F80) == 0x7F80).sum())
            event_total += events
            if events > global_longest["events"]:
                global_longest = {"events": events, "stay_id": key}
            split = split_by_key[key]
            if split not in per_split_longest:
                raise AssertionError((key, split))
            if events > per_split_longest[split]["events"]:
                per_split_longest[split] = {"events": events, "stay_id": key}
            if index % 500 == 0 or index == EXPECTED_STAYS:
                print(json.dumps({"events": event_total, "stays_verified": index}))

    if event_total != EXPECTED_EVENTS:
        raise AssertionError((event_total, EXPECTED_EVENTS))
    if global_longest["events"] != EXPECTED_MAX_EVENTS:
        raise AssertionError(global_longest)
    if nonfinite_values:
        raise AssertionError({"nonfinite_bfloat16_values": nonfinite_values})

    report = {
        "all_zero_event_rows": zero_rows,
        "encoded_hdf": {
            "bytes": args.encoded_hdf.stat().st_size,
            "path": str(args.encoded_hdf),
            "sha256": sha256(args.encoded_hdf),
        },
        "events": event_total,
        "global_longest": global_longest,
        "nonfinite_bfloat16_values": nonfinite_values,
        "per_split_longest": per_split_longest,
        "source_hdf": {
            "bytes": args.source_hdf.stat().st_size,
            "path": str(args.source_hdf),
            "sha256": sha256(args.source_hdf),
        },
        "stage_report_sha256": sha256(args.stage_report),
        "status": "passed",
        "stays": EXPECTED_STAYS,
        "test_metric_accesses": 0,
        "vector_width": EXPECTED_WIDTH,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
