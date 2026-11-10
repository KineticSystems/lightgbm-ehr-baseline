"""Assert MUSE's released 15-day readmission cohort, labels, splits, modalities."""

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

from check_cohort import PAPER_COUNTS, PAPER_RACE_ROUNDED, _split_overlap, _two  # noqa: E402
from cohort_source import (  # noqa: E402
    NOTE_CODE,
    lab_channels,
    ordered_int_hash,
    ordered_text_hash,
    sha256_file,
    source_codes,
)
from cohort_readmission import (  # noqa: E402
    AUTHOR_TASK,
    LABEL_COLUMN,
    PAPER_READMISSION_MISSING,
    PAPER_READMISSION_POSITIVE,
    PUBLISHED_AUROC,
    PUBLISHED_AUROC_MUSE_PLUS,
    TASK_ID,
    load_source,
    readmission_cohort,
)


PAPER_ROUNDED = {
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--note-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    source = load_source(args.mimic_root, args.note_root, args.cache_root)
    manifest_path = args.task_dir / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["task"]["id"] != TASK_ID:
        raise AssertionError("wrong MUSE task identity")
    if manifest["task"]["modalities"] != ["structured", "notes"]:
        raise AssertionError("selected MUSE task must retain structured data and notes")
    if manifest["task"]["note_codes"] != [NOTE_CODE]:
        raise AssertionError("MUSE note-code declaration changed")
    if manifest["task"]["published_auroc"] != PUBLISHED_AUROC:
        raise AssertionError("transcribed MUSE readmission comparator changed")
    if manifest["task"]["published_auroc_muse_plus"] != PUBLISHED_AUROC_MUSE_PLUS:
        raise AssertionError("transcribed MUSE+ readmission cell changed")
    selected_config = manifest["source"]["source_configuration"]
    expected_config = {
        "dataset": "mimic4",
        "task": AUTHOR_TASK,
        "load_no_label": False,
        "seed": 42,
        "batch_size": 128,
    }
    for name, expected in expected_config.items():
        if selected_config.get(name) != expected:
            raise AssertionError(f"wrong selected source configuration: {name}")
    if selected_config.get("prediction_cutoff") != "hospital discharge":
        raise AssertionError("released readmission prediction time changed")
    if manifest["source"]["selection_tier"] != 2:
        raise AssertionError("MUSE experiment selection tier changed")

    labeled, detail = readmission_cohort(args.cache_root)
    summary = detail["summary"]
    replay = summary["mortality_split_replay_matches_cached_author_files"]
    if not all(replay.values()):
        raise AssertionError("released split RNG stream was not replayed exactly")

    index = pq.read_table(
        args.task_dir / manifest["meds"]["task_index"]["file"]
    ).to_pandas()
    left = index.sort_values(["split", "split_position"]).reset_index(drop=True)
    right = labeled.sort_values(["split", "split_position"]).reset_index(drop=True)
    comparisons = {
        "subject_id": (left["subject_id"], right["subject_id"]),
        "sample_id": (left["sample_id"], right["hadm_id"]),
        "boolean_value": (left["boolean_value"], right[LABEL_COLUMN].astype(bool)),
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
    if int(left["prediction_time"].ne(right["dischtime"]).sum()):
        raise AssertionError("MEDS prediction time is not the released discharge time")

    statistics = source["statistics"]
    if statistics["admissions"] != PAPER_COUNTS["admissions"]:
        raise AssertionError("paper admission count changed")
    if statistics["subjects"] != PAPER_COUNTS["subjects"]:
        raise AssertionError("paper patient count changed")
    if summary["admissions"] != PAPER_COUNTS["admissions"]:
        raise AssertionError("readmission cohort is not the released common cohort")
    if summary["subjects"] != PAPER_COUNTS["subjects"]:
        raise AssertionError("readmission cohort lost released subjects")
    if _two(summary["positive_ratio_labeled"]) != PAPER_READMISSION_POSITIVE:
        raise AssertionError("paper readmission-positive marginal changed")
    if _two(summary["missing_ratio_all_admissions"]) != PAPER_READMISSION_MISSING:
        raise AssertionError("paper readmission-missing marginal changed")
    if round(statistics["mean_age"]) != PAPER_ROUNDED["age"]:
        raise AssertionError("paper mean age changed")
    female_ratio = statistics["gender_counts"]["F"] / statistics["admissions"]
    male_ratio = statistics["gender_counts"]["M"] / statistics["admissions"]
    if _two(female_ratio) != PAPER_ROUNDED["gender_f"]:
        raise AssertionError("paper female marginal changed")
    if _two(male_ratio) != PAPER_ROUNDED["gender_m"]:
        raise AssertionError("paper male marginal changed")
    for group, expected in PAPER_RACE_ROUNDED.items():
        actual = _two(statistics["race_group_counts"][group] / statistics["admissions"])
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
    if _two(paper_base_notes["cohort_missing_ratio"]) != PAPER_ROUNDED["notes_missing"]:
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
        expected = summary["splits"][split]
        if counts[split] != expected["count"]:
            raise AssertionError(f"released {split} count changed")
        if positives[split] != expected["positive"]:
            raise AssertionError(f"released {split} label marginal changed")
        if subjects[split] != expected["subjects"]:
            raise AssertionError(f"released {split} subject count changed")
    if sum(counts.values()) != summary["labeled"]:
        raise AssertionError("released 70/10/20 slicing did not cover the labeled set")
    for split, ratio in (("train", 0.7), ("validation", 0.1), ("test", 0.2)):
        if abs(counts[split] / summary["labeled"] - ratio) > 1e-4:
            raise AssertionError(f"released {split} ratio changed")

    overlaps = _split_overlap(labeled)
    manifest_overlap = manifest["cross_split_subject_overlap"]
    for name, actual in overlaps.items():
        if actual != manifest_overlap[name]:
            raise AssertionError(f"manifest subject-overlap count changed: {name}")
    if overlaps["subjects_in_multiple_splits"] == 0:
        raise AssertionError("released admission split unexpectedly became patient-disjoint")

    # This must be a different task from the sealed mortality export, not a
    # relabeling of the same admissions.
    cohort_frame = pd.read_parquet(args.cache_root / "cohort.parquet")
    mortality_ids = set(
        cohort_frame.loc[cohort_frame["mortality"].notna(), "hadm_id"].astype(int)
    )
    readmission_ids = set(labeled["hadm_id"].astype(int))
    if readmission_ids == mortality_ids:
        raise AssertionError("readmission cohort collapsed onto the mortality cohort")
    task_relationship = {
        "mortality_labeled": len(mortality_ids),
        "readmission_labeled": len(readmission_ids),
        "shared_admissions": len(mortality_ids & readmission_ids),
        "readmission_only": len(readmission_ids - mortality_ids),
        "mortality_only": len(mortality_ids - readmission_ids),
    }

    lab_width = 2 * len(lab_channels(args.cache_root)["itemids"])
    if lab_width != source["labs"]["author_width"]:
        raise AssertionError("reconstructed author lab width changed")
    if lab_width != 114:
        raise AssertionError("exact source/fallback/GRU lab width match changed")

    codes = pq.read_table(args.task_dir / manifest["meds"]["codes"]["file"])[
        "code"
    ].to_pylist()
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
    note_digests = {split: hashlib.sha256() for split in ("train", "validation", "test")}
    note_parquet = pq.ParquetFile(args.cache_root / "brief_hospital_course.parquet")
    for batch in note_parquet.iter_batches(
        batch_size=5_000,
        columns=["hadm_id", "text", "chart_after_discharge", "stored_after_discharge"],
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
            entry = notes_by_split[split]
            entry["present"] += 1
            entry["empty_but_present"] += int(text == "")
            entry["chart_after_discharge"] += int(bool(chart_after))
            entry["stored_after_discharge"] += int(bool(stored_after))
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
    for batch in pq.ParquetFile(args.cache_root / "lab_boundary.parquet").iter_batches(
        batch_size=100_000, columns=["hadm_id"]
    ):
        lab_ids.update(
            int(value)
            for value in np.unique(batch.column(0).to_numpy())
            if int(value) in labeled_ids
        )
    indexed_lab_ids = set(left.loc[left["lab_present"], "sample_id"].astype(int))
    if indexed_lab_ids != lab_ids:
        raise AssertionError("MEDS lab-presence membership changed")

    split_files = manifest["source"]["readmission_split_files"]
    for name, record in split_files.items():
        path = args.cache_root / record["file"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise AssertionError(f"released readmission split file changed: {name}")

    report = {
        "status": "exact_source_with_recorded_environment_substitution",
        "reconstruction_status": "exact_source_cohort_reconstruction",
        "selected_experiment": {
            "paper_cell": manifest["source"]["selected_paper_cell"],
            "task": "15-day readmission after hospital discharge",
            "model_inputs": ["structured", "notes"],
            "selection_tier": 2,
            "configuration": expected_config,
            "published_cells": manifest["source"]["published_cells"],
            "comparator_adjudication": manifest["source"]["comparator_adjudication"],
        },
        "released_label_definition": manifest["source"]["released_label_definition"],
        "paper_statistics": {
            "admissions": statistics["admissions"],
            "subjects": statistics["subjects"],
            "readmission_positive": summary["positive"],
            "readmission_positive_ratio_labeled": summary["positive_ratio_labeled"],
            "readmission_missing": summary["missing_label"],
            "readmission_missing_ratio_all_admissions": summary[
                "missing_ratio_all_admissions"
            ],
            "paper_table_4_readmission_positive": PAPER_READMISSION_POSITIVE,
            "paper_table_4_readmission_missing": PAPER_READMISSION_MISSING,
            "mean_age": statistics["mean_age"],
            "gender_counts": statistics["gender_counts"],
            "race_group_counts": statistics["race_group_counts"],
            "missing_ratio": statistics["missing_ratio"],
            "average_discharge_words": statistics["average_discharge_words"],
            "paper_base_note_statistics": paper_base_notes,
            "selected_v2_note_missing_ratio": statistics["missing_ratio"]["discharge"],
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
        "split_reconstruction": {
            "released_stream": (
                "set_seed(42) once, then shuffle(mortality labeled ids), then "
                "shuffle(readmission labeled ids) - data_split_mimic4.main"
            ),
            "mortality_shuffle_length": summary["mortality_shuffle_length"],
            "mortality_split_replay_matches_cached_author_files": replay,
            "readmission_split_files": split_files,
        },
        "cross_split_subject_overlap": overlaps,
        "task_relationship_to_sealed_mortality_task": task_relationship,
        "released_label_defects_preserved": summary["label_defects"],
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
            "all_cohort_chart_after_discharge": source["notes"]["chart_after_discharge"],
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
        "recorded_deviations": [
            {
                "kind": "environment_substitution",
                "detail": (
                    "paper_code/muse/.venv (CPython 3.8.20) is unusable - its "
                    "interpreter under /tmp/medpatch_uv_python was deleted. The "
                    "paper's own paper_code/muse/.export_assets/gpu_venv "
                    "(CPython 3.10.20) was used instead; it carries the same "
                    "NumPy 1.24.4 / pandas 2.0.3 / Polars 1.8.2 / PyArrow 17.0.0 "
                    "pins recorded for the sealed export."
                ),
                "evidence": (
                    "the released seed-42 mortality split regenerated under "
                    "CPython 3.10.20 reproduces the cached CPython 3.8.20 "
                    "author split id files exactly for train, validation and "
                    "test"
                ),
            },
            {
                "kind": "adapter_revision_drift",
                "detail": source["adapter_revision_drift"]["note"],
                "sha256_recorded_at_cache_build": source["adapter_revision_drift"][
                    "sha256_recorded_at_cache_build"
                ],
                "sha256_current": source["adapter_revision_drift"]["sha256_current"],
            },
        ],
        "source_reconstruction": source,
        "manifest_sha256": manifest_hash,
        "checks": [
            "exact paper 389,761 admissions and 172,113 subjects",
            "exact rounded paper readmission positive (0.25) and missing (0.41) marginals",
            "exact rounded paper demographic and modality marginals",
            "released RNG stream replayed: mortality split reproduces the cached author files",
            "exact generated readmission admission IDs, labels, split order, and overlap",
            "prediction time equals the released discharge time for every sample",
            "exact MIMIC-IV 2.0 plus Note 2.2 source pairing, re-hashed",
            "paper base-note and selected v2-note marginals kept distinct",
            "exact Brief Hospital Course membership/text and timing audit",
            "exact 114-channel author lab reconstruction and width match",
            "readmission cohort proven distinct from the sealed mortality cohort",
            "released readmission label defects quantified, not repaired",
            "no imaging or unlabeled auxiliary rows",
        ],
    }
    output = args.task_dir / "audit" / "cohort_reconstruction.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if sha256_file(manifest_path) != manifest_hash:
        raise AssertionError("cohort checker mutated immutable manifest")
    print(json.dumps({"status": report["status"], "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
