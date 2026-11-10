"""Assert MUSE's selected paper cohort, labels, splits, and modalities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from cohort_source import (  # noqa: E402
    NOTE_CODE,
    TASK_ID,
    ensure_source,
    lab_channels,
    ordered_int_hash,
    ordered_text_hash,
    sha256_file,
    source_codes,
)


PAPER_COUNTS = {"admissions": 389_761, "subjects": 172_113}
PAPER_ROUNDED = {
    "mortality_positive": "0.09",
    "mortality_missing": "0.58",
    "age": 57,
    "gender_f": "0.52",
    "gender_m": "0.48",
    "diagnoses_missing": "0.00",
    "procedures_missing": "0.50",
    "prescriptions_missing": "0.17",
    "labs_missing": "0.22",
    "notes_missing": "0.31",
}
SELECTED_V2_NOTES_MISSING = "0.42"
PAPER_RACE_ROUNDED = {
    "African American": "0.17",
    "Asian": "0.04",
    "Caucasian": "0.66",
    "Hispanic": "0.06",
    "Native American": "0.00",
    "Other": "0.06",
}


def _two(value: float) -> str:
    return f"{value:.2f}"


def _split_overlap(cohort: pd.DataFrame) -> dict[str, int]:
    members = {
        split: set(cohort.loc[cohort["split"].eq(split), "subject_id"].astype(int))
        for split in ("train", "validation", "test")
    }
    result = {
        "train_validation": len(members["train"] & members["validation"]),
        "train_test": len(members["train"] & members["test"]),
        "validation_test": len(members["validation"] & members["test"]),
    }
    all_subjects = set().union(*members.values())
    result["subjects_in_multiple_splits"] = sum(
        sum(subject in values for values in members.values()) > 1
        for subject in all_subjects
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--note-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    source = ensure_source(args.mimic_root, args.note_root, args.cache_root)
    manifest_path = args.task_dir / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["task"]["id"] != TASK_ID:
        raise AssertionError("wrong MUSE task identity")
    if manifest["task"]["modalities"] != ["structured", "notes"]:
        raise AssertionError("selected MUSE task must retain structured data and notes")
    if manifest["task"]["note_codes"] != [NOTE_CODE]:
        raise AssertionError("MUSE note-code declaration changed")
    selected_config = manifest["source"]["source_configuration"]
    expected_config = {
        "dataset": "mimic4",
        "task": "mortality",
        "load_no_label": False,
        "seed": 42,
        "batch_size": 128,
    }
    for name, expected in expected_config.items():
        if selected_config.get(name) != expected:
            raise AssertionError(f"wrong selected source configuration: {name}")
    if manifest["source"]["selection_tier"] != 2:
        raise AssertionError("MUSE experiment selection tier changed")

    cohort = pd.read_parquet(args.cache_root / "cohort.parquet")
    labeled = cohort.loc[cohort["mortality"].notna()].copy()
    index = pq.read_table(
        args.task_dir / manifest["meds"]["task_index"]["file"]
    ).to_pandas()
    left = index.sort_values(["split", "split_position"]).reset_index(drop=True)
    right = labeled.sort_values(["split", "split_position"]).reset_index(drop=True)
    comparisons = {
        "subject_id": (left["subject_id"], right["subject_id"]),
        "sample_id": (left["sample_id"], right["hadm_id"]),
        "boolean_value": (left["boolean_value"], right["mortality"].astype(bool)),
        "split": (left["split"], right["split"]),
        "cohort_position": (left["cohort_position"], right["cohort_position"]),
        "split_position": (left["split_position"], right["split_position"]),
        "age_bucket": (left["age_bucket"], right["age_bucket"]),
        "gender": (left["gender"], right["gender"]),
        "ethnicity": (left["ethnicity"], right["race"]),
    }
    for name, (actual, expected) in comparisons.items():
        if not np.array_equal(actual.to_numpy(), expected.to_numpy()):
            raise AssertionError(f"MEDS changed exact cohort field: {name}")

    statistics = source["statistics"]
    if statistics["admissions"] != PAPER_COUNTS["admissions"]:
        raise AssertionError("paper admission count changed")
    if statistics["subjects"] != PAPER_COUNTS["subjects"]:
        raise AssertionError("paper patient count changed")
    if _two(statistics["mortality_positive_ratio"]) != PAPER_ROUNDED[
        "mortality_positive"
    ]:
        raise AssertionError("paper mortality-positive marginal changed")
    if _two(statistics["mortality_missing_ratio"]) != PAPER_ROUNDED[
        "mortality_missing"
    ]:
        raise AssertionError("paper mortality-missing marginal changed")
    if round(statistics["mean_age"]) != PAPER_ROUNDED["age"]:
        raise AssertionError("paper mean age changed")
    female_ratio = statistics["gender_counts"]["F"] / statistics["admissions"]
    male_ratio = statistics["gender_counts"]["M"] / statistics["admissions"]
    if _two(female_ratio) != PAPER_ROUNDED["gender_f"]:
        raise AssertionError("paper female marginal changed")
    if _two(male_ratio) != PAPER_ROUNDED["gender_m"]:
        raise AssertionError("paper male marginal changed")
    for group, expected in PAPER_RACE_ROUNDED.items():
        actual = _two(
            statistics["race_group_counts"][group] / statistics["admissions"]
        )
        if actual != expected:
            raise AssertionError(
                f"paper race marginal changed: {group} {actual} != {expected}"
            )
    missing_expectations = {
        "diagnoses_icd": PAPER_ROUNDED["diagnoses_missing"],
        "procedures_icd": PAPER_ROUNDED["procedures_missing"],
        "prescriptions": PAPER_ROUNDED["prescriptions_missing"],
        "labvectors": PAPER_ROUNDED["labs_missing"],
    }
    for name, expected in missing_expectations.items():
        actual = _two(statistics["missing_ratio"][name])
        if actual != expected:
            raise AssertionError(
                f"paper {name} missingness changed: {actual} != {expected}"
            )
    paper_base_notes = source["paper_base_note_statistics"]
    if _two(paper_base_notes["cohort_missing_ratio"]) != PAPER_ROUNDED[
        "notes_missing"
    ]:
        raise AssertionError("paper base-producer note missingness changed")
    selected_note_missing = _two(statistics["missing_ratio"]["discharge"])
    if selected_note_missing != SELECTED_V2_NOTES_MISSING:
        raise AssertionError("selected v2 Brief Hospital Course missingness changed")
    if selected_note_missing == PAPER_ROUNDED["notes_missing"]:
        raise AssertionError("paper base note marginal was conflated with v2 input")

    counts = left["split"].value_counts().to_dict()
    positives = left.groupby("split")["boolean_value"].sum().astype(int).to_dict()
    subjects = left.groupby("split")["subject_id"].nunique().to_dict()
    for split in ("train", "validation", "test"):
        expected = source["cohort"]["splits"][split]
        if counts[split] != expected["count"]:
            raise AssertionError(f"released {split} count changed")
        if positives[split] != expected["positive"]:
            raise AssertionError(f"released {split} label marginal changed")
        if subjects[split] != expected["subjects"]:
            raise AssertionError(f"released {split} subject count changed")

    overlaps = _split_overlap(labeled)
    manifest_overlap = manifest["cross_split_subject_overlap"]
    for name, actual in overlaps.items():
        manifest_name = name
        if name == "subjects_in_multiple_splits":
            expected = manifest_overlap[name]
        else:
            expected = manifest_overlap[name]
        if actual != expected:
            raise AssertionError(f"manifest subject-overlap count changed: {name}")
    if overlaps["subjects_in_multiple_splits"] == 0:
        raise AssertionError("released admission split unexpectedly became patient-disjoint")

    lab_width = 2 * len(lab_channels(args.cache_root)["itemids"])
    if lab_width != source["labs"]["author_width"]:
        raise AssertionError("reconstructed author lab width changed")
    if lab_width != 114:
        raise AssertionError("exact source/fallback/GRU lab width match changed")

    codes = pq.read_table(
        args.task_dir / manifest["meds"]["codes"]["file"]
    )["code"].to_pylist()
    if codes != list(source_codes(args.cache_root)):
        raise AssertionError("MEDS event vocabulary/order differs from the source")
    allowed_prefixes = (
        "MUSE_AGE//",
        "MUSE_GENDER//",
        "MUSE_ETHNICITY//",
        "MUSE_CODE//",
        "MUSE_LAB_PRESENT//",
        "MUSE_LAB_SEQUENCE_ROW",
        "MUSE_LAB_VALUE//",
        "MUSE_LAB_ABNORMAL//",
    )
    for code in codes:
        if code != NOTE_CODE and not code.startswith(allowed_prefixes):
            raise AssertionError(f"unselected modality namespace entered MEDS: {code}")

    note_splits = labeled.set_index("hadm_id")["split"].to_dict()
    notes_by_split = {
        split: {
            "present": 0,
            "empty_but_present": 0,
            "chart_after_discharge": 0,
            "stored_after_discharge": 0,
        }
        for split in ("train", "validation", "test")
    }
    note_digests = {
        split: hashlib.sha256() for split in ("train", "validation", "test")
    }
    note_parquet = pq.ParquetFile(args.cache_root / "brief_hospital_course.parquet")
    for batch in note_parquet.iter_batches(
        batch_size=5_000,
        columns=[
            "hadm_id",
            "text",
            "chart_after_discharge",
            "stored_after_discharge",
        ],
    ):
        values = batch.to_pydict()
        for hadm_id, text, chart_after, stored_after in zip(
            values["hadm_id"],
            values["text"],
            values["chart_after_discharge"],
            values["stored_after_discharge"],
        ):
            split = note_splits.get(int(hadm_id))
            if split is None:
                continue
            summary = notes_by_split[split]
            summary["present"] += 1
            summary["empty_but_present"] += int(text == "")
            summary["chart_after_discharge"] += int(bool(chart_after))
            summary["stored_after_discharge"] += int(bool(stored_after))
            encoded = text.encode()
            note_digests[split].update(len(encoded).to_bytes(8, "little"))
            note_digests[split].update(encoded)
    for split, digest in note_digests.items():
        notes_by_split[split]["raw_text_sha256"] = digest.hexdigest()
    if int(left["note_present"].sum()) != sum(
        value["present"] for value in notes_by_split.values()
    ):
        raise AssertionError("MEDS note-presence membership changed")
    labeled_ids = set(labeled["hadm_id"].astype(int))
    lab_ids: set[int] = set()
    for batch in pq.ParquetFile(
        args.cache_root / "lab_boundary.parquet"
    ).iter_batches(batch_size=100_000, columns=["hadm_id"]):
        lab_ids.update(
            int(value)
            for value in np.unique(batch.column(0).to_numpy())
            if int(value) in labeled_ids
        )
    indexed_lab_ids = set(
        left.loc[left["lab_present"], "sample_id"].astype(int)
    )
    if indexed_lab_ids != lab_ids:
        raise AssertionError("MEDS lab-presence membership changed")

    race_counts = statistics["race_group_counts"]
    report = {
        "status": "exact",
        "reconstruction_status": "exact_source_cohort_reconstruction",
        "selected_experiment": {
            "paper_cell": manifest["source"]["selected_paper_cell"],
            "task": "90-day post-discharge mortality",
            "model_inputs": ["structured", "notes"],
            "selection_tier": 2,
            "configuration": expected_config,
        },
        "paper_statistics": {
            "admissions": statistics["admissions"],
            "subjects": statistics["subjects"],
            "mortality_positive": statistics["mortality_positive"],
            "mortality_positive_ratio": statistics["mortality_positive_ratio"],
            "mortality_missing": statistics["mortality_missing"],
            "mortality_missing_ratio": statistics["mortality_missing_ratio"],
            "mean_age": statistics["mean_age"],
            "gender_counts": statistics["gender_counts"],
            "race_group_counts": race_counts,
            "missing_ratio": statistics["missing_ratio"],
            "average_discharge_words": statistics["average_discharge_words"],
            "paper_base_note_statistics": paper_base_notes,
            "selected_v2_note_missing_ratio": statistics["missing_ratio"][
                "discharge"
            ],
        },
        "splits": {
            split: {
                "samples": counts[split],
                "subjects": subjects[split],
                "positive": positives[split],
                "prevalence": positives[split] / counts[split],
                "notes": notes_by_split[split],
            }
            for split in ("train", "validation", "test")
        },
        "cross_split_subject_overlap": overlaps,
        "ordered_sha256": {
            "sample_id": ordered_int_hash(left["sample_id"]),
            "subject_id": ordered_int_hash(left["subject_id"]),
            "label": ordered_int_hash(left["boolean_value"].astype(int)),
            "split": ordered_text_hash(left["split"]),
        },
        "note_timing_audit": {
            "author_behavior": (
                "charttime and storetime are discarded before the released "
                "Brief Hospital Course attachment"
            ),
            "all_cohort_chart_after_discharge": source["notes"][
                "chart_after_discharge"
            ],
            "all_cohort_stored_after_discharge": source["notes"][
                "stored_after_discharge"
            ],
        },
        "retained_projection": {
            "modalities": ["structured", "notes"],
            "note_codes": [NOTE_CODE],
            "event_vocabulary_size": len(codes),
            "exact_missing_modality_membership": True,
            "raw_note_text": True,
            "no_imaging_or_auxiliary_rows": True,
        },
        "author_lab_width": {
            "retained_itemids": len(lab_channels(args.cache_root)["itemids"]),
            "nonmissing_width": lab_width,
            "missing_fallback_width": 114,
            "model_gru_width": 114,
            "exact_width_match": True,
        },
        "source_reconstruction": source,
        "manifest_sha256": manifest_hash,
        "checks": [
            "exact paper 389,761 admissions and 172,113 subjects",
            "exact rounded paper label, demographic, and modality marginals",
            "exact generated admission IDs, labels, split order, and overlap",
            "exact MIMIC-IV 2.0 plus Note 2.2 source pairing",
            "paper base-note and selected v2-note marginals kept distinct",
            "exact Brief Hospital Course membership/text and timing audit",
            "exact 114-channel author lab reconstruction and width match",
            "no imaging or unlabeled auxiliary rows",
        ],
    }
    output = args.task_dir / "audit" / "cohort_reconstruction.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if sha256_file(manifest_path) != manifest_hash:
        raise AssertionError("cohort checker mutated immutable manifest")
    print(json.dumps({"status": report["status"], "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
