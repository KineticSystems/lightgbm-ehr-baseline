#!/usr/bin/env python3
"""Paper-specific cohort reconstruction for the GraphCare phase-2 tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from export_phase2 import TASK_SPECS
from cohort_phase2 import (
    DATASETS,
    EXPECTED_VOCABULARY_SIZES,
    dataset_cache_dir,
    hash_values,
    load_phase2_source,
    meds_codes,
    parse_cache_summary,
    phase2_artifact_hashes,
    phase2_event_rows,
    phase2_source_summary,
    readmission_defect_report,
    sample_time_bounds,
    sha256_file,
    visit_order_report,
)


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
PRIMARY_TASK_DIR = (
    WORKSPACE_ROOT / "paper_tasks" / "graphcare" / "subsequent_visit_mortality"
)

# Table 1 of the paper, "Statistics of pre-processed EHR datasets".
PAPER_TABLE_1 = {
    "mimic3": {
        "patients": 35_707,
        "visits": 44_399,
        "visits_per_patient": 1.24,
        "conditions_per_patient": 12.89,
        "procedures_per_patient": 4.54,
        "drugs_per_patient": 33.71,
    },
    "mimic4": {
        "patients": 123_488,
        "visits": 232_263,
        "visits_per_patient": 1.88,
        "conditions_per_patient": 21.74,
        "procedures_per_patient": 4.70,
        "drugs_per_patient": 43.89,
    },
}

# Only these exact Table 1 discrepancies are accepted, and only at their exact
# observed values.  They are recorded, not repaired.  See
# `paper_export_notes/graphcare.md` (Phase 2) for the adjudication: PyHealth
# 1.1.2 downloads unversioned ICD->CCS crosswalks and the paper-time bytes were
# never released, so the currently served payload maps slightly fewer diagnosis
# and procedure codes than GraphCare saw in 2023.  MIMIC-III is ICD-9 only and
# barely moves; MIMIC-IV is roughly half ICD-10 and moves more, which also
# drops a handful of visits below the released "all three modalities non-empty"
# eligibility test.  The NDC->ATC drug channel reproduces Table 1 exactly on
# both releases, which is what identifies the ICD crosswalks as the cause.
APPROVED_TABLE_1_QUALIFICATIONS = {
    "mimic3": {
        "conditions_per_patient": {
            "published": 12.89,
            "reconstructed_rounded": 12.92,
        }
    },
    "mimic4": {
        "patients": {"published": 123_488, "reconstructed_rounded": 123_478},
        "visits": {"published": 232_263, "reconstructed_rounded": 232_247},
        "conditions_per_patient": {
            "published": 21.74,
            "reconstructed_rounded": 20.23,
        },
        "procedures_per_patient": {
            "published": 4.70,
            "reconstructed_rounded": 4.39,
        },
    },
}


def read_manifest(task_dir: Path):
    manifest = json.loads((task_dir / "manifest.json").read_text())
    for record in manifest["meds"].values():
        path = task_dir / record["file"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise AssertionError(f"missing or modified MEDS file: {path}")
    return manifest


def assert_series(frame: pl.DataFrame, name: str, expected) -> None:
    actual = frame[name].to_list()
    if actual != list(expected):
        raise AssertionError(f"task index {name} differs from author source")


def _ordered_event_digest(task_dir: Path, manifest) -> str:
    """Hash the exported events in the released within-sample order."""
    digest = hashlib.sha256()
    events = (
        pl.scan_parquet(task_dir / manifest["meds"]["events"]["file"])
        .select("sample_id", "source_row", "sequence_row", "source_time", "code")
        .sort(["sample_id", "source_row"])
        .collect(streaming=True)
    )
    for row in events.iter_rows(named=False):
        payload = json.dumps(
            [int(row[0]), int(row[1]), int(row[2]), float(row[3]), row[4]],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def sealed_primary_crosscheck(source, task_dir: Path) -> dict:
    """Prove the sharded parse reproduces the sealed phase-1 MIMIC-III task.

    Phase 1 built MIMIC-III with one unsharded `MIMIC3Dataset` call.  The
    MIMIC-III mortality and readmission task functions select identical
    samples, identical cumulative histories and identical patient splits (they
    differ only in the label expression), so the sealed task is an exact
    external check on this module's sharded execution of the same parser.
    """
    manifest = read_manifest(PRIMARY_TASK_DIR)
    index = pl.read_parquet(
        PRIMARY_TASK_DIR / manifest["meds"]["task_index"]["file"]
    ).sort("source_index")
    records = source.records
    if index.height != len(records):
        raise AssertionError("sealed primary sample count differs")
    for name, expected in (
        ("source_index", [record.source_index for record in records]),
        ("subject_id", [int(record.sample["patient_id"]) for record in records]),
        ("sample_id", [int(record.sample["visit_id"]) for record in records]),
        (
            "current_visit_id",
            [int(record.sample["visit_id"]) for record in records],
        ),
        (
            "target_visit_id",
            [int(record.target_visit_id) for record in records],
        ),
        ("split", [record.split for record in records]),
        (
            "source_split_position",
            [record.split_position for record in records],
        ),
        (
            "history_length",
            [len(record.sample["conditions"]) for record in records],
        ),
    ):
        if index[name].to_list() != expected:
            raise AssertionError(f"sealed primary {name} differs from sharded parse")

    exported_digest = _ordered_event_digest(PRIMARY_TASK_DIR, manifest)
    digest = hashlib.sha256()
    total = 0
    # The shared writer stores events sorted by (sample_id, time, source_row,
    # code); `_ordered_event_digest` reads them back in that order, so the
    # reconstruction walks samples by ascending sample_id as well.
    for record in sorted(records, key=lambda item: int(item.sample["visit_id"])):
        sample_id = int(record.sample["visit_id"])
        for source_row, sequence_row, source_time, code, _visit in phase2_event_rows(
            record, source.max_history
        ):
            payload = json.dumps(
                [sample_id, source_row, sequence_row, source_time, code],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
            total += 1
    if total != manifest["event_count"]:
        raise AssertionError("sealed primary event count differs from sharded parse")
    if digest.hexdigest() != exported_digest:
        raise AssertionError("sealed primary event stream differs from sharded parse")
    return {
        "task": str(PRIMARY_TASK_DIR.relative_to(WORKSPACE_ROOT)),
        "samples": index.height,
        "events": manifest["event_count"],
        "ordered_event_sha256": exported_digest,
        "max_history": source.max_history,
        "result": (
            "the sharded parse reproduces the sealed unsharded phase-1 cohort, "
            "split, cumulative histories and full ordered event stream exactly"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_SPECS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    spec = TASK_SPECS[args.task]
    dataset = spec["dataset"]
    shards = DATASETS[dataset]["shards"]

    manifest = read_manifest(args.task_dir)
    task = manifest["task"]
    if task["id"] != args.task:
        raise AssertionError("the exported task id is not the selected task")
    if task["published_auroc"] != spec["published_auroc"]:
        raise AssertionError("the transcribed Table 2 AUROC differs")
    if task["published_cell"] != spec["published_cell"]:
        raise AssertionError("the transcribed Table 2 cell differs")
    if task["modalities"] != ["structured"] or task["note_codes"] != []:
        raise AssertionError("GraphCare must be a structured-only export")
    if task["temporal_profile"] != "multi_visit_untimed":
        raise AssertionError("the released boundary is a visit sequence without time")
    source_metadata = manifest["source"]
    if source_metadata["selection_tier"] != 1:
        raise AssertionError("the selected structured experiment is not tier 1")
    if source_metadata["retained_projection"] != [
        "conditions",
        "procedures",
        "drugs",
    ]:
        raise AssertionError("the structured projection is incomplete")
    if source_metadata["mimic_release"] != DATASETS[dataset]["release"]:
        raise AssertionError("the exported MIMIC release differs from the released code")

    cache_dir = dataset_cache_dir(PAPER_ROOT, dataset)
    source = load_phase2_source(
        PAPER_ROOT, dataset, cache_dir, shards, spec["outcome"]
    )
    expected_summary = phase2_source_summary(source, PAPER_ROOT)
    if source_metadata["author_source_summary"] != expected_summary:
        raise AssertionError("manifest source summary differs from author loader")
    expected_artifacts = phase2_artifact_hashes(PAPER_ROOT, dataset, args.root)
    if source_metadata["source_artifacts"] != expected_artifacts:
        raise AssertionError("source artifact hashes changed")

    index = pl.read_parquet(
        args.task_dir / manifest["meds"]["task_index"]["file"]
    ).sort("source_index")
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
        index, "target_visit_id", (int(record.target_visit_id) for record in records)
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

    expected_codes = meds_codes(PAPER_ROOT)
    actual_codes = tuple(
        pl.read_parquet(args.task_dir / manifest["meds"]["codes"]["file"])[
            "code"
        ].to_list()
    )
    if actual_codes != expected_codes:
        raise AssertionError("MEDS vocabulary differs from released vocabularies")
    if expected_summary["vocabulary_sizes"] != EXPECTED_VOCABULARY_SIZES:
        raise AssertionError("Appendix C.1 vocabulary sizes were not recreated")

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
        "validation_test": len(memberships["validation"] & memberships["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError("released patient split unexpectedly overlaps")
    if manifest["cross_split_subject_overlap"] != {
        **overlaps,
        "subjects_in_multiple_splits": 0,
    }:
        raise AssertionError("manifest overlap audit differs")

    paper_stats = parse_cache_summary(dataset, cache_dir, shards)
    expected_table = PAPER_TABLE_1[dataset]
    observed_table = {
        "patients": paper_stats["patients"],
        "visits": paper_stats["visits"],
        "visits_per_patient": paper_stats["visits_per_patient"],
        **{
            f"{key}_per_patient": paper_stats["code_occurrences_per_patient"][key]
            for key in ("conditions", "procedures", "drugs")
        },
    }
    mismatches = {}
    for name, published in expected_table.items():
        actual = observed_table[name]
        rounded = actual if isinstance(actual, int) else round(actual, 2)
        if rounded != published:
            mismatches[name] = {
                "published": published,
                "reconstructed": actual,
                "reconstructed_rounded": rounded,
            }
    approved = APPROVED_TABLE_1_QUALIFICATIONS[dataset]
    if set(mismatches) != set(approved):
        raise AssertionError(
            "unexpected Table 1 discrepancy: " + json.dumps(mismatches, sort_keys=True)
        )
    for name, observed in mismatches.items():
        if (
            observed["published"] != approved[name]["published"]
            or observed["reconstructed_rounded"]
            != approved[name]["reconstructed_rounded"]
        ):
            raise AssertionError(f"Table 1 qualification changed for {name}")

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

    status = "exact_source_with_historical_crosswalk_mismatch" if mismatches else "exact"
    report = {
        "status": status,
        "paper": "graphcare",
        "task": args.task,
        "selected_cell": spec["published_cell"],
        "published_auroc": spec["published_auroc"],
        "positive_class": spec["positive_class"],
        "label_definition": task["label_definition"],
        "source_release": DATASETS[dataset]["release"],
        "source_boundary": source_metadata["source_boundary"],
        "released_entry_point": spec["load_dataset_call"],
        "source_execution": source_metadata["source_execution"],
        "cohort_identifier_set_sha256": hash_values(
            sorted(int(record.sample["visit_id"]) for record in records)
        ),
        "current_target_pair_sha256": hash_values(
            (record.sample["visit_id"], record.target_visit_id) for record in records
        ),
        "ordered_vocabulary_sha256": hash_values(expected_codes),
        "vocabulary_sizes": expected_summary["vocabulary_sizes"],
        "paper_table_1": {
            "expected": expected_table,
            "reconstructed": paper_stats,
            "observed": observed_table,
            "accepted_known_mismatches": mismatches,
            "qualification_reason": (
                "GraphCare used mutable unversioned PyHealth crosswalks and "
                "the paper-time bytes were not released; the currently served "
                "ICD->CCS payload maps slightly fewer diagnosis/procedure "
                "codes, while the NDC->ATC drug channel reproduces Table 1 "
                "exactly. Only these exact discrepancies are accepted; every "
                "other Table 1 assertion is strict"
            )
            if mismatches
            else None,
        },
        "selected_task_source_counts": {
            "samples": len(records),
            "patients": len({record.sample["patient_id"] for record in records}),
            "maximum_cumulative_history": source.max_history,
            "splits": split_report,
            "cross_split_patient_overlap": overlaps,
        },
        "released_visit_order": visit_order_report(source),
        "modality_projection": {
            "retained": ["conditions", "procedures", "drugs"],
            "notes": "not used by the selected method",
            "imaging": "not used by the selected method",
            "knowledge_graph": (
                "not retained; complete proposed-model KG artifacts are unreleased"
            ),
            "cohort_restricted_by_unused_modality": False,
        },
        "disclosed_source_defects": source_metadata["disclosed_source_defects"],
        "source_artifact_sha256": expected_artifacts,
        "residual_uncertainty": {
            "severity": "explicit_historical_source_qualification",
            "detail": (
                "the audited current unversioned PyHealth crosswalk payload is "
                "exact for this export, but the paper-time bytes were not "
                "released"
            ),
        },
    }
    if spec["outcome"] == "readmission":
        report["released_readmission_label_audit"] = readmission_defect_report(source)
    if dataset == "mimic3":
        report["sealed_primary_crosscheck"] = sealed_primary_crosscheck(
            source, args.task_dir
        )

    output = args.task_dir / "audit" / "cohort_reconstruction.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": str(output),
                "status": status,
                "table_1_qualification": mismatches,
                **split_report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
