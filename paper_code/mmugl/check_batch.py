"""Exhaustively recreate MMUGL's retained source boundary from MEDS."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from cohort_source import (
    NOTE_CODE,
    build_samples,
    iter_note_rows,
    iter_structured_rows,
    load_author_boundary,
    split_summary,
    structured_event_count,
)


def update_value(digest, value):
    if value is None:
        digest.update(b"N")
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S" + len(encoded).to_bytes(8, "little") + encoded)
    elif isinstance(value, float):
        digest.update(b"F" + np.asarray([value], dtype=np.float64).tobytes())
    else:
        digest.update(b"I" + int(value).to_bytes(8, "little", signed=True))


def actual_event_rows(path):
    parquet = pq.ParquetFile(path)
    previous_max = -1
    columns = [
        "subject_id",
        "time",
        "code",
        "numeric_value",
        "sample_id",
        "source_row",
        "sequence_row",
        "source_time",
        "text_value",
    ]
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=columns)
        source_rows = table["source_row"].to_numpy().astype(np.int64)
        order = np.argsort(source_rows, kind="stable")
        if len(order) and int(source_rows[order[0]]) <= previous_max:
            raise AssertionError("MEDS Parquet source-row ranges are not ordered")
        if len(order):
            previous_max = int(source_rows[order[-1]])
        table = table.take(order)
        values = table.to_pydict()
        for position in range(table.num_rows):
            timestamp = values["time"][position]
            time_us = int(np.datetime64(timestamp, "us").astype(np.int64))
            yield {
                "subject_id": int(values["subject_id"][position]),
                "time_us": time_us,
                "code": values["code"][position],
                "numeric_value": values["numeric_value"][position],
                "sample_id": int(values["sample_id"][position]),
                "source_row": int(values["source_row"][position]),
                "sequence_row": int(values["sequence_row"][position]),
                "source_time": values["source_time"][position],
                "text_value": values["text_value"][position],
            }


def expected_event_rows(boundary, samples, note_stats):
    yield from iter_structured_rows(samples)
    yield from iter_note_rows(
        boundary["paths"]["NOTEEVENTS.csv.gz"],
        samples,
        structured_event_count(samples),
        note_stats,
    )


def compare_events(boundary, samples, path):
    note_stats = {}
    digest = hashlib.sha256()
    count = 0
    structured = 0
    notes = 0
    for position, (expected, actual) in enumerate(
        itertools.zip_longest(
            expected_event_rows(boundary, samples, note_stats),
            actual_event_rows(path),
        )
    ):
        if expected is None or actual is None:
            raise AssertionError("MEDS and source have different event counts")
        expected_full = {**expected, "numeric_value": None}
        for field in (
            "subject_id",
            "time_us",
            "code",
            "numeric_value",
            "sample_id",
            "source_row",
            "sequence_row",
            "source_time",
            "text_value",
        ):
            left = expected_full[field]
            right = actual[field]
            if left != right:
                raise AssertionError(
                    f"event {position} field {field} differs: {left!r} != {right!r}"
                )
            update_value(digest, right)
        count += 1
        if actual["code"] == NOTE_CODE:
            notes += 1
        else:
            structured += 1
    return {
        "events": count,
        "structured_events": structured,
        "note_events": notes,
        "ordered_event_field_sha256": digest.hexdigest(),
        "raw_note_projection": note_stats,
    }


def compare_index(samples, task_dir, manifest):
    path = task_dir / manifest["meds"]["task_index"]["file"]
    rows = sorted(pq.read_table(path).to_pylist(), key=lambda row: row["source_position"])
    if len(rows) != len(samples):
        raise AssertionError("MEDS changed the number of MMUGL transitions")
    digest = hashlib.sha256()
    for expected, actual in zip(samples, rows):
        fields = {
            "subject_id": expected["subject_id"],
            "sample_id": expected["sample_id"],
            "boolean_value": expected["boolean_value"],
            "split": expected["split"],
            "source_position": expected["source_position"],
            "source_patient_position": expected["source_patient_position"],
            "source_target_visit_index": expected["source_target_visit_index"],
            "target_hadm_id": expected["target_hadm_id"],
        }
        for field, value in fields.items():
            if actual[field] != value:
                raise AssertionError(f"task index changed {field}")
            update_value(digest, value)
        for field, value in (
            ("prediction_time", expected["prediction_time_us"]),
            ("observation_start", expected["observation_start_us"]),
            ("actual_observation_start", expected["actual_observation_start_us"]),
        ):
            actual_us = int(np.datetime64(actual[field], "us").astype(np.int64))
            if actual_us != value:
                raise AssertionError(f"task index changed {field}")
            update_value(digest, actual_us)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.task_dir / "manifest.json").read_text())
    if manifest["task"].get("modalities") != ["structured", "notes"]:
        raise AssertionError("MMUGL must retain its structured/raw-note projection")
    if manifest["task"].get("note_codes") != [NOTE_CODE]:
        raise AssertionError("MMUGL note-code exception list differs")
    if manifest["source"].get("selected_model_inputs") != ["structured", "notes"]:
        raise AssertionError("MMUGL selected source experiment differs")
    if manifest["source"].get("selection_tier") != 2:
        raise AssertionError("MMUGL selected modality tier differs")

    boundary = load_author_boundary(
        args.mimic_root, args.mapping_root, include_raw_hashes=False
    )
    samples, observation_hours = build_samples(boundary)
    if manifest["task"]["observation_window_hours"] != observation_hours:
        raise AssertionError("MMUGL maximum observation window differs")
    if manifest["task"].get("temporal_profile") != "multi_visit_untimed":
        raise AssertionError("MMUGL must declare the multi-visit untimed profile")
    if "temporal_grid_mode" in manifest["task"]:
        raise AssertionError("the cohort-fitted absolute grid is no longer part of the format")

    codes_path = args.task_dir / manifest["meds"]["codes"]["file"]
    codes = pq.read_table(codes_path)["code"].to_pylist()
    if codes != list(boundary["codes"]):
        raise AssertionError("MEDS event vocabulary differs from released code lists")
    if set(codes) - {NOTE_CODE} != set(boundary["codes"]) - {NOTE_CODE}:
        raise AssertionError("a non-note event is not an intended structured input")

    index_hash = compare_index(samples, args.task_dir, manifest)
    events_path = args.task_dir / manifest["meds"]["events"]["file"]
    event_report = compare_events(boundary, samples, events_path)
    if event_report["events"] != manifest["event_count"]:
        raise AssertionError("manifest event count differs from exhaustive recreation")

    report = {
        "status": "exact",
        "comparison": (
            "all retained transition identifiers, labels, split membership, ordered "
            "prior visits, ICD/ATC occurrence sets, admission times, and eligible raw "
            "note texts from the released CGL loader boundary"
        ),
        "samples": len(samples),
        "splits": split_summary(samples),
        **event_report,
        "ordered_task_index_sha256": index_hash,
        "event_vocabulary_sha256": manifest["meds"]["codes"]["sha256"],
        "retained_modalities": ["structured", "notes"],
        "note_codes": [NOTE_CODE],
        "imaging_retained": False,
        "author_boundary_assertions": [
            "released patient and visit order",
            "HADM_ID captured immediately before author drop",
            "ICD and ATC strings before stochastic shuffling",
            "earliest-42-visit truncation",
            "every training transition and final validation/test transition",
            "exact target-prefix label logic",
            "raw non-error note eligibility and source-row order",
        ],
        "full_model_parity_limitation": (
            "CUI token/negation and UMLS graph tensors are intentionally not common-task "
            "events and cannot be byte-recreated because the paper omitted its licensed "
            "UMLS release and QuickUMLS-index provenance"
        ),
    }
    output = args.task_dir / "audit" / "full_batch_recreation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
