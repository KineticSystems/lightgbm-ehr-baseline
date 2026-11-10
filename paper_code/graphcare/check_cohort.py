#!/usr/bin/env python3
"""Paper-specific cohort reconstruction for GraphCare."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from cohort_source import (
    EXPECTED_VOCABULARY_SIZES,
    hash_values,
    load_author_source,
    meds_codes,
    paper_cohort_statistics,
    sample_time_bounds,
    sha256_file,
    source_artifact_hashes,
    source_summary,
)


PAPER_ROOT = Path(__file__).resolve().parent


def read_manifest(task_dir: Path):
    manifest_path = task_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["meds"].values():
        path = task_dir / record["file"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise AssertionError(f"missing or modified MEDS file: {path}")
    return manifest


def assert_series(frame: pl.DataFrame, name: str, expected) -> None:
    actual = frame[name].to_list()
    if actual != list(expected):
        raise AssertionError(f"task index {name} differs from author source")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_manifest(args.task_dir)
    task = manifest["task"]
    if task["id"] != "subsequent_visit_mortality":
        raise AssertionError("the selected task is not subsequent-visit mortality")
    if task["published_auroc"] != 0.703:
        raise AssertionError("the selected Table 2 AUROC is not 0.703")
    if task["modalities"] != ["structured"] or task["note_codes"] != []:
        raise AssertionError("GraphCare must be a structured-only export")
    source_metadata = manifest["source"]
    if source_metadata["selection_tier"] != 1:
        raise AssertionError("the selected structured experiment is not tier 1")
    if source_metadata["retained_projection"] != [
        "conditions",
        "procedures",
        "drugs",
    ]:
        raise AssertionError("the structured projection is incomplete")

    source = load_author_source(PAPER_ROOT, args.mimic_root)
    expected_summary = source_summary(source, PAPER_ROOT)
    if source_metadata["author_source_summary"] != expected_summary:
        raise AssertionError("manifest source summary differs from author loader")
    expected_artifacts = source_artifact_hashes(PAPER_ROOT, args.mimic_root)
    if source_metadata["source_artifacts"] != expected_artifacts:
        raise AssertionError("source artifact hashes changed")

    index_path = args.task_dir / manifest["meds"]["task_index"]["file"]
    index = pl.read_parquet(index_path).sort("source_index")
    records = source.records
    if index.height != len(records):
        raise AssertionError("MEDS sample count differs from author loader")
    assert_series(index, "source_index", range(len(records)))
    assert_series(
        index, "subject_id", (int(record.sample["patient_id"]) for record in records)
    )
    assert_series(
        index, "sample_id", (int(record.sample["visit_id"]) for record in records)
    )
    assert_series(
        index,
        "current_visit_id",
        (int(record.sample["visit_id"]) for record in records),
    )
    assert_series(
        index,
        "target_visit_id",
        (int(record.target_visit_id) for record in records),
    )
    assert_series(index, "split", (record.split for record in records))
    assert_series(
        index, "source_split_position", (record.split_position for record in records)
    )
    assert_series(
        index,
        "history_length",
        (len(record.sample["conditions"]) for record in records),
    )
    assert_series(
        index, "boolean_value", (bool(record.sample["label"]) for record in records)
    )
    observation_start, prediction_time = sample_time_bounds(source.max_history)
    assert_series(index, "observation_start", [observation_start] * len(records))
    assert_series(index, "prediction_time", [prediction_time] * len(records))

    code_path = args.task_dir / manifest["meds"]["codes"]["file"]
    actual_codes = tuple(pl.read_parquet(code_path)["code"].to_list())
    expected_codes = meds_codes(PAPER_ROOT)
    if actual_codes != expected_codes:
        raise AssertionError("MEDS vocabulary differs from released vocabularies")

    memberships = {
        split: {
            int(record.sample["patient_id"])
            for record in records
            if record.split == split
        }
        for split in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": len(memberships["train"] & memberships["validation"]),
        "train_test": len(memberships["train"] & memberships["test"]),
        "validation_test": len(
            memberships["validation"] & memberships["test"]
        ),
    }
    if any(overlaps.values()):
        raise AssertionError("released patient split unexpectedly overlaps")
    if manifest["cross_split_subject_overlap"] != {
        **overlaps,
        "subjects_in_multiple_splits": 0,
    }:
        raise AssertionError("manifest overlap audit differs")

    paper_stats = paper_cohort_statistics(source.base_dataset)
    if paper_stats["patients"] != 35_707 or paper_stats["visits"] != 44_399:
        raise AssertionError("Table 1 MIMIC-III cohort counts were not recreated")
    if round(paper_stats["visits_per_patient"], 2) != 1.24:
        raise AssertionError("Table 1 visit average was not recreated")
    expected_code_averages = {
        "conditions": 12.89,
        "procedures": 4.54,
        "drugs": 33.71,
    }
    actual_code_averages = paper_stats["code_occurrences_per_patient"]
    code_average_mismatches = {}
    for key, expected in expected_code_averages.items():
        if round(actual_code_averages[key], 2) != expected:
            code_average_mismatches[key] = {
                "published": expected,
                "reconstructed": actual_code_averages[key],
                "reconstructed_rounded": round(actual_code_averages[key], 2),
            }
    approved_historical_qualification = {
        "conditions": {
            "published": 12.89,
            "reconstructed": 12.922312151678943,
            "reconstructed_rounded": 12.92,
        }
    }
    if code_average_mismatches != approved_historical_qualification:
        raise AssertionError(
            "unexpected Table 1 code occurrences/patient discrepancy: "
            + json.dumps(code_average_mismatches, sort_keys=True)
        )
    if expected_summary["vocabulary_sizes"] != EXPECTED_VOCABULARY_SIZES:
        raise AssertionError("Appendix C.1 vocabulary sizes were not recreated")

    split_report = {}
    for split in ("train", "validation", "test"):
        selected = sorted(
            (record for record in records if record.split == split),
            key=lambda record: record.split_position,
        )
        positives = sum(int(record.sample["label"]) for record in selected)
        split_report[split] = {
            "samples": len(selected),
            "patients": len({record.sample["patient_id"] for record in selected}),
            "positive": positives,
            "prevalence": positives / len(selected),
            "ordered_current_visit_id_sha256": hash_values(
                record.sample["visit_id"] for record in selected
            ),
            "ordered_target_visit_id_sha256": hash_values(
                record.target_visit_id for record in selected
            ),
            "ordered_label_sha256": hash_values(
                int(record.sample["label"]) for record in selected
            ),
        }

    report = {
        "status": "exact_source_with_historical_crosswalk_mismatch",
        "paper": "graphcare",
        "task": "subsequent_visit_mortality",
        "selected_cell": "Table 2 GraphCare/BAT MIMIC-III mortality AUROC 0.703",
        "positive_class": "death at discharge from the subsequent admission",
        "source_release": "MIMIC-III 1.4",
        "source_boundary": (
            "PyHealth 1.1.2 MIMIC3Dataset.set_task with the released "
            "mortality_prediction_mimic3_fn"
        ),
        "cohort_identifier_set_sha256": hash_values(
            sorted(int(record.sample["visit_id"]) for record in records)
        ),
        "current_target_pair_sha256": hash_values(
            (record.sample["visit_id"], record.target_visit_id) for record in records
        ),
        "ordered_vocabulary_sha256": hash_values(expected_codes),
        "vocabulary_sizes": expected_summary["vocabulary_sizes"],
        "paper_table_1": {
            "expected": {
                "patients": 35_707,
                "visits": 44_399,
                "visits_per_patient": 1.24,
                "conditions_per_patient": 12.89,
                "procedures_per_patient": 4.54,
                "drugs_per_patient": 33.71,
            },
            "reconstructed": paper_stats,
            "approved_historical_qualification": {
                "status": "accepted_known_mismatch",
                "metric": "conditions_per_patient",
                "published": 12.89,
                "reconstructed": 12.922312151678943,
                "reconstructed_rounded": 12.92,
                "absolute_difference": (
                    actual_code_averages["conditions"]
                    - expected_code_averages["conditions"]
                ),
                "scope": (
                    "only this exact discrepancy is accepted; all other "
                    "Table 1 assertions remain exact"
                ),
                "reason": (
                    "GraphCare used mutable unversioned PyHealth crosswalks, "
                    "and the paper-time ICD9CM-to-CCSCM bytes were not released"
                ),
            },
        },
        "selected_task_source_counts": {
            "samples": len(records),
            "patients": len({record.sample["patient_id"] for record in records}),
            "maximum_cumulative_history": source.max_history,
            "splits": split_report,
            "cross_split_patient_overlap": overlaps,
        },
        "modality_projection": {
            "retained": ["conditions", "procedures", "drugs"],
            "notes": "not used by the selected method",
            "imaging": "not used by the selected method",
            "knowledge_graph": (
                "not retained; complete proposed-model KG artifacts are unreleased"
            ),
            "cohort_restricted_by_unused_modality": False,
        },
        "source_artifact_sha256": expected_artifacts,
        "residual_uncertainty": {
            "severity": "explicit_historical_source_qualification",
            "detail": (
                "the audited current unversioned PyHealth crosswalk payload is "
                "exact for this export, but the paper-time bytes were not "
                "released; this produces the documented 12.89 versus 12.9223 "
                "conditions/patient discrepancy"
            ),
        },
    }
    output = args.task_dir / "audit" / "cohort_reconstruction.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": str(output),
                "qualification": approved_historical_qualification,
                **split_report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
