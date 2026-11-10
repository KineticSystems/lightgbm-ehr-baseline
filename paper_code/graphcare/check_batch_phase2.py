#!/usr/bin/env python3
"""Exhaustively recreate the released PyHealth loader/collate fields (phase 2)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from export_phase2 import TASK_SPECS
from cohort_phase2 import (
    DATASETS,
    MODALITY_KEYS,
    PREFIX_BY_KEY,
    dataset_cache_dir,
    event_time,
    hash_values,
    load_phase2_source,
    meds_codes,
    phase2_event_rows,
    sha256_file,
)


PAPER_ROOT = Path(__file__).resolve().parent


def update_digest(digest, value) -> None:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def check_collate(expected_batch, recreated_batch) -> None:
    from pyhealth.datasets.utils import collate_fn_dict

    if collate_fn_dict(expected_batch) != collate_fn_dict(recreated_batch):
        raise AssertionError(
            "MEDS reconstruction differs after author collate_fn_dict"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_SPECS), required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    spec = TASK_SPECS[args.task]
    dataset = spec["dataset"]
    shards = DATASETS[dataset]["shards"]

    manifest = json.loads((args.task_dir / "manifest.json").read_text())
    for record in manifest["meds"].values():
        path = args.task_dir / record["file"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise AssertionError(f"missing or modified MEDS file: {path}")
    if manifest["task"]["id"] != args.task:
        raise AssertionError("the exported task id is not the selected task")
    if manifest["task"]["modalities"] != ["structured"]:
        raise AssertionError("unexpected model-input modality declaration")
    if manifest["task"]["note_codes"] != []:
        raise AssertionError("GraphCare has no permitted note codes")
    if manifest["source"]["retained_projection"] != [
        "conditions",
        "procedures",
        "drugs",
    ]:
        raise AssertionError("the retained author projection is incomplete")
    if manifest["source"]["source_boundary"] != (
        f"{DATASETS[dataset]['class']}.set_task({spec['task_function']}) "
        "three aligned cumulative categorical visit histories"
    ):
        raise AssertionError("the declared source boundary is not the selected task")

    source = load_phase2_source(
        PAPER_ROOT, dataset, dataset_cache_dir(PAPER_ROOT, dataset), shards,
        spec["outcome"],
    )
    records = source.records
    max_history = source.max_history

    source_by_sample = {int(record.sample["visit_id"]): record for record in records}
    if len(source_by_sample) != len(records):
        raise AssertionError("current visit IDs are not unique")

    expected_codes = meds_codes(PAPER_ROOT)
    actual_codes = tuple(
        pl.read_parquet(args.task_dir / manifest["meds"]["codes"]["file"])[
            "code"
        ].to_list()
    )
    if actual_codes != expected_codes:
        raise AssertionError("event vocabulary differs from released code maps")

    index = pl.read_parquet(
        args.task_dir / manifest["meds"]["task_index"]["file"]
    ).sort("source_index")
    if index["source_index"].to_list() != list(range(len(records))):
        raise AssertionError("source example order is not reconstructable")
    if index["sample_id"].to_list() != [
        int(record.sample["visit_id"]) for record in records
    ]:
        raise AssertionError("current visit IDs differ from author loader")
    if index["target_visit_id"].to_list() != [
        int(record.target_visit_id) for record in records
    ]:
        raise AssertionError("target visit IDs differ from author loader")
    if index["boolean_value"].to_list() != [
        bool(record.sample["label"]) for record in records
    ]:
        raise AssertionError("labels differ from author loader")
    if index["current_visit_id"].to_list() != [
        int(record.sample["visit_id"]) for record in records
    ]:
        raise AssertionError("current_visit_id differs from author loader")

    # The shared writer stores every sample's rows contiguously and in
    # ascending `source_row`, so the parquet is walked in file order rather
    # than materialised and re-sorted: a MIMIC-IV task holds 16.4M events,
    # which does not fit the 7 GiB per-process budget as one frame.  Any
    # deviation from that layout still fails, because each sample is popped
    # exactly once and its rows are compared element by element in order.
    event_columns = (
        "subject_id",
        "time",
        "code",
        "numeric_value",
        "sample_id",
        "source_row",
        "sequence_row",
        "source_time",
        "visit_id",
        "split",
    )
    event_file = pq.ParquetFile(args.task_dir / manifest["meds"]["events"]["file"])
    observed_codes = set()
    event_total = 0

    expected_digest = hashlib.sha256()
    actual_digest = hashlib.sha256()
    modality_counts = {key: 0 for key in MODALITY_KEYS}
    checked_samples = 0
    expected_collate_batch = []
    recreated_collate_batch = []

    current_sample_id = None
    current_rows = []

    def verify_one(sample_id, actual_rows) -> None:
        nonlocal checked_samples, expected_collate_batch, recreated_collate_batch
        record = source_by_sample.pop(sample_id, None)
        if record is None:
            raise AssertionError(f"unknown or duplicated sample events: {sample_id}")
        expected_rows = list(phase2_event_rows(record, max_history))
        if len(actual_rows) != len(expected_rows):
            raise AssertionError(f"event count differs for sample {sample_id}")
        history_length = len(record.sample["conditions"])
        padding = max_history - history_length
        reconstructed = {
            key: [[] for _ in range(history_length)] for key in MODALITY_KEYS
        }

        for actual, expected in zip(actual_rows, expected_rows):
            (
                subject_id,
                time,
                code,
                numeric_value,
                actual_sample_id,
                source_row,
                sequence_row,
                source_time,
                visit_id,
                split,
            ) = actual
            (
                expected_source_row,
                expected_sequence,
                expected_time,
                expected_code,
                expected_visit,
            ) = expected
            if numeric_value is not None:
                raise AssertionError("occurrence event has a numeric value")
            if int(subject_id) != int(record.sample["patient_id"]):
                raise AssertionError("event patient differs from author sample")
            if int(actual_sample_id) != sample_id or split != record.split:
                raise AssertionError("event identity/split differs from author sample")
            if (
                int(source_row) != expected_source_row
                or int(sequence_row) != expected_sequence
                or float(source_time) != expected_time
                or code != expected_code
                or int(visit_id) != expected_visit
                or time != event_time(expected_sequence)
            ):
                raise AssertionError("ordered MEDS event differs from author history")
            update_digest(
                expected_digest,
                (
                    sample_id,
                    expected_source_row,
                    expected_sequence,
                    expected_time,
                    expected_code,
                    expected_visit,
                ),
            )
            update_digest(
                actual_digest,
                (
                    sample_id,
                    int(source_row),
                    int(sequence_row),
                    float(source_time),
                    code,
                    int(visit_id),
                ),
            )
            for key, prefix in PREFIX_BY_KEY.items():
                if code.startswith(prefix):
                    visit_offset = int(sequence_row) - padding
                    if not 0 <= visit_offset < history_length:
                        raise AssertionError(
                            "event is outside the author history length"
                        )
                    reconstructed[key][visit_offset].append(code[len(prefix) :])
                    modality_counts[key] += 1
                    break
            else:
                raise AssertionError(
                    "non-note event is not an intended structured code"
                )

        for key in MODALITY_KEYS:
            if reconstructed[key] != record.sample[key]:
                raise AssertionError(f"recreated {key} list differs for {sample_id}")
        recreated_sample = {
            "visit_id": record.sample["visit_id"],
            "patient_id": record.sample["patient_id"],
            "conditions": reconstructed["conditions"],
            "procedures": reconstructed["procedures"],
            "drugs": reconstructed["drugs"],
            "label": int(record.sample["label"]),
        }
        if recreated_sample != record.sample:
            raise AssertionError("recreated PyHealth sample dictionary differs")
        expected_collate_batch.append(record.sample)
        recreated_collate_batch.append(recreated_sample)
        if len(expected_collate_batch) == 64:
            check_collate(expected_collate_batch, recreated_collate_batch)
            expected_collate_batch = []
            recreated_collate_batch = []
        checked_samples += 1

    for batch in event_file.iter_batches(
        batch_size=250_000, columns=list(event_columns)
    ):
        event_total += batch.num_rows
        columns = [batch.column(name).to_pylist() for name in event_columns]
        observed_codes.update(columns[2])
        for row in zip(*columns):
            if row[3] is not None:
                raise AssertionError(
                    "categorical occurrence events gained numeric values"
                )
            if row[8] is None:
                raise AssertionError("a multi-visit event lost its encounter identity")
            sample_id = int(row[4])
            if current_sample_id is None:
                current_sample_id = sample_id
            elif sample_id != current_sample_id:
                verify_one(current_sample_id, current_rows)
                current_sample_id = sample_id
                current_rows = []
            current_rows.append(row)
        del columns
    if current_sample_id is not None:
        verify_one(current_sample_id, current_rows)
    if event_total != manifest["event_count"]:
        raise AssertionError("event count differs from immutable manifest")
    if not observed_codes.issubset(expected_codes):
        raise AssertionError("events contain undeclared codes")
    if expected_collate_batch:
        check_collate(expected_collate_batch, recreated_collate_batch)
    if source_by_sample:
        raise AssertionError("some author samples have no MEDS events")
    if checked_samples != len(records):
        raise AssertionError("not every author sample was checked")
    if actual_digest.hexdigest() != expected_digest.hexdigest():
        raise AssertionError("full ordered event digest differs")

    report = {
        "status": "exact",
        "paper": "graphcare",
        "task": args.task,
        "selected_cell": spec["published_cell"],
        "source_dataset": (
            f"PyHealth 1.1.2 {DATASETS[dataset]['class']} with the released "
            f"GraphCare {spec['task_function']}"
        ),
        "released_entry_point": spec["load_dataset_call"],
        "source_collate": "pyhealth.datasets.utils.collate_fn_dict",
        "source_fields": {
            "conditions": "ragged list[list[str]]",
            "procedures": "ragged list[list[str]]",
            "drugs": "ragged list[list[str]]",
            "label": "scalar Python int",
            "patient_id": "scalar string MIMIC subject_id",
            "visit_id": "scalar string current MIMIC hadm_id",
        },
        "checked_samples": checked_samples,
        "checked_events": event_total,
        "ordered_event_sha256": actual_digest.hexdigest(),
        "ordered_current_visit_sha256": hash_values(
            record.sample["visit_id"] for record in records
        ),
        "ordered_target_visit_sha256": hash_values(
            record.target_visit_id for record in records
        ),
        "ordered_label_sha256": hash_values(
            int(record.sample["label"]) for record in records
        ),
        "ordered_vocabulary_sha256": hash_values(expected_codes),
        "vocabulary_size": len(expected_codes),
        "retained_event_counts": modality_counts,
        "maximum_history": max_history,
        "missingness_and_padding": {
            "numeric_missing_values": (
                "not applicable; all inputs are categorical occurrences"
            ),
            "observation_masks": "absent from the author PyHealth boundary",
            "padding": (
                "right-aligned visit ordinal exactly preserves the released "
                "whole-cohort maximum visit padding position"
            ),
        },
        "timestamp_fidelity": (
            "the source boundary exposes visit order, not real event "
            "timestamps; MEDS time is exactly the audited right-aligned visit "
            "ordinal and every event carries its encounter's visit_id"
        ),
        "model_input_audit": {
            "declared_modalities": ["structured"],
            "note_codes": [],
            "every_non_note_code": (
                "one of CONDITION//<CCS-CM>, PROCEDURE//<CCS-PROC>, or "
                "DRUG//<ATC-3>"
            ),
            "excluded": [
                "knowledge-graph nodes/edges/embeddings/attention",
                "patient and admission identifiers as features",
                "real encounter time gaps",
                "notes",
                "imaging",
            ],
        },
    }
    output = args.task_dir / "audit" / "full_batch_recreation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": str(output),
                "samples": checked_samples,
                "events": event_total,
                "event_sha256": actual_digest.hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
