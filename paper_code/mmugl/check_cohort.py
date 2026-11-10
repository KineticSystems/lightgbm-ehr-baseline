"""Assert MMUGL's paper-specific CGL heart-failure cohort evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cohort_source import ICD_PREFIX, NOTE_CODE, PUBLISHED_AUROC, SPLITS, source_paths


PAPER_PATIENTS = {"train": 6155, "validation": 125, "test": 1221}
RELEASED_CODE_PATIENTS = {"train": 6156, "validation": 125, "test": 1221}


def hash_ints(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.task_dir / "manifest.json").read_text())
    source = manifest["source"]
    if source.get("selected_paper_cell") != (
        "Table 2, CGL split, MMUGL heart-failure AuROC"
    ):
        raise AssertionError("MMUGL selected paper cell differs")
    if source.get("selected_model_inputs") != ["structured", "notes"]:
        raise AssertionError("MMUGL selected model inputs differ")
    if source.get("selection_tier") != 2:
        raise AssertionError("MMUGL modality-priority tier differs")
    if manifest["task"].get("published_auroc") != PUBLISHED_AUROC:
        raise AssertionError("MMUGL published AUROC differs")
    if manifest["task"].get("note_codes") != [NOTE_CODE]:
        raise AssertionError("MMUGL raw-note declaration differs")
    if manifest["task"].get("temporal_profile") != "multi_visit_untimed":
        raise AssertionError("MMUGL must declare the multi-visit untimed profile")
    if "temporal_grid_mode" in manifest["task"]:
        raise AssertionError("the cohort-fitted absolute grid is no longer part of the format")

    paths = source_paths(args.mimic_root, args.mapping_root)
    with paths["cgl_patient_splits.pkl"].open("rb") as handle:
        released = pickle.load(handle)
    released_ids = {
        split: {int(value) for value in released[key]}
        for split, key in (("train", "train"), ("validation", "val"), ("test", "test"))
    }
    if {split: len(ids) for split, ids in released_ids.items()} != {
        "train": 6000,
        "validation": 125,
        "test": 1221,
    }:
        raise AssertionError("released CGL split artifact counts differ")
    for position, left in enumerate(SPLITS):
        for right in SPLITS[position + 1 :]:
            if released_ids[left] & released_ids[right]:
                raise AssertionError("released CGL split has patient overlap")

    index_path = args.task_dir / manifest["meds"]["task_index"]["file"]
    index = pq.read_table(index_path)
    index_rows = index.to_pylist()
    subjects = {
        split: {
            int(row["subject_id"]) for row in index_rows if row["split"] == split
        }
        for split in SPLITS
    }
    counts = {split: len(subjects[split]) for split in SPLITS}
    if counts != RELEASED_CODE_PATIENTS:
        raise AssertionError(f"MMUGL released-code patient counts differ: {counts}")
    if subjects["validation"] != released_ids["validation"]:
        raise AssertionError("MMUGL validation identifiers differ from released split")
    if subjects["test"] != released_ids["test"]:
        raise AssertionError("MMUGL test identifiers differ from released split")
    if not released_ids["train"].issubset(subjects["train"]):
        raise AssertionError("MMUGL lost a released training patient")
    if len(subjects["train"] - released_ids["train"]) != 156:
        raise AssertionError("MMUGL released all-diagnosis expansion is not 156 patients")
    if any(subjects[left] & subjects[right] for left in SPLITS for right in SPLITS if left < right):
        raise AssertionError("MMUGL exported cross-split patient overlap")

    split_rows = {}
    for split in SPLITS:
        selected = [row for row in index_rows if row["split"] == split]
        positives = sum(bool(row["boolean_value"]) for row in selected)
        split_rows[split] = {
            "samples": len(selected),
            "patients": len(subjects[split]),
            "positive": positives,
            "prevalence": positives / len(selected),
        }
    if split_rows["validation"]["samples"] != 125 or split_rows["test"]["samples"] != 1221:
        raise AssertionError("validation/test must contain one final transition per patient")
    if round(100 * split_rows["test"]["prevalence"]) != 37:
        raise AssertionError("MMUGL test prevalence does not reproduce paper Table 14")

    test_samples = {
        int(row["sample_id"]): bool(row["boolean_value"])
        for row in index_rows
        if row["split"] == "test"
    }
    history_hf = {sample_id: False for sample_id in test_samples}
    events_path = args.task_dir / manifest["meds"]["events"]["file"]
    parquet = pq.ParquetFile(events_path)
    prefix = ICD_PREFIX + "428"
    for batch in parquet.iter_batches(columns=["sample_id", "code"], batch_size=250_000):
        codes = batch["code"]
        selected = pc.starts_with(codes, pattern=prefix)
        if pc.any(selected).as_py():
            for sample_id in batch.filter(selected)["sample_id"].to_pylist():
                sample_id = int(sample_id)
                if sample_id in history_hf:
                    history_hf[sample_id] = True
    with_history = [
        test_samples[sample_id] for sample_id, value in history_hf.items() if value
    ]
    without_history = [
        test_samples[sample_id] for sample_id, value in history_hf.items() if not value
    ]
    if len(with_history) != 425 or len(without_history) != 796:
        raise AssertionError("MMUGL historical-heart-failure subgroups differ from Table 14")
    if round(100 * sum(with_history) / len(with_history)) != 74:
        raise AssertionError("MMUGL with-history prevalence differs from Table 14")
    if round(100 * sum(without_history) / len(without_history)) != 17:
        raise AssertionError("MMUGL without-history prevalence differs from Table 14")

    ordered = sorted(index_rows, key=lambda row: row["source_position"])
    report = {
        "status": "exact",
        "paper_evidence": {
            "table_2_test_auroc": PUBLISHED_AUROC,
            "table_5_patients": PAPER_PATIENTS,
            "table_14_test_patients": 1221,
            "table_14_test_prevalence_percent": 37,
            "table_14_with_history_patients": 425,
            "table_14_without_history_patients": 796,
        },
        "paper_source_discrepancy": {
            "field": "Table 5 training patients",
            "paper": PAPER_PATIENTS["train"],
            "released_code": RELEASED_CODE_PATIENTS["train"],
            "difference": 1,
            "interpretation": (
                "the exact released get_splits/CglDatasetDownstream path includes all "
                "6,000 artifact training IDs plus 156 additional multi-visit patients; "
                "the paper itself warns that post-publication code can cause slight "
                "count deviations"
            ),
        },
        "splits": split_rows,
        "released_split_artifact_patients": {
            split: len(values) for split, values in released_ids.items()
        },
        "expanded_training_patients": len(subjects["train"] - released_ids["train"]),
        "cross_split_subject_overlap": {
            f"{left}_{right}": len(subjects[left] & subjects[right])
            for position, left in enumerate(SPLITS)
            for right in SPLITS[position + 1 :]
        },
        "test_history_subgroups": {
            "with_heart_failure_history": {
                "patients": len(with_history),
                "positive": sum(with_history),
                "prevalence": sum(with_history) / len(with_history),
            },
            "without_heart_failure_history": {
                "patients": len(without_history),
                "positive": sum(without_history),
                "prevalence": sum(without_history) / len(without_history),
            },
        },
        "ordered_sample_id_sha256": hash_ints([row["sample_id"] for row in ordered]),
        "ordered_label_sha256": hash_ints([int(row["boolean_value"]) for row in ordered]),
        "checks": [
            "paper Table 5 patient counts, including the documented one-patient released-code discrepancy",
            "released validation/test identifier sets",
            "released 6,000-patient training set plus code-faithful 156-patient expansion",
            "exact split membership and label marginals",
            "paper Table 14 test prevalence and prior-heart-failure subgroup sizes",
            "no note-availability cohort restriction",
            "structured-plus-notes selection and note-code declaration",
        ],
    }
    output = args.task_dir / "audit" / "cohort_reconstruction.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
