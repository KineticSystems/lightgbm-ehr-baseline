"""Assert the paper, released-artifact, and construction-path MedPatch cohort facts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPOSITORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY / "mimic4extract"))

from mimic3benchmark import mimic3csv  # noqa: E402
from export_source import NOTE_CODE  # noqa: E402
from cohort_source import (  # noqa: E402
    CONFIG_PATH,
    ITEMID_MAP_PATH,
    NORMALIZER_PATH,
    TEST_MEMBERSHIP_PATH,
    VALIDATION_MEMBERSHIP_PATH,
    code_vocabulary,
    load_source_rows,
    sha256,
    split_members,
)


PAPER_TOTAL = 26_226
# Table 1, RR / BioBERT row: the released first-48-hour radiology projection
# covers this many of the PAPER_TOTAL mortality examples.
PAPER_RR_TOTAL = 22_626
PAPER_SPLIT_PERCENT = {"train": 70, "validation": 10, "test": 20}
PAPER_PREVALENCE_PERCENT = {"train": 12.5, "validation": 11.9, "test": 12.4}
EXACT_RELEASED_SPLIT = {
    "train": {"rows": 18_845, "positive": 2_357},
    "validation": {"rows": 2_138, "positive": 254},
    "test": {"rows": 5_243, "positive": 651},
}
EXACT_MIMIC_IV_1_0_VALIDATION_COUNTS = {
    "n_events": 294_769_993,
    "empty_hadm": 16_442_087,
    "no_hadm_in_stay": 48_894_460,
    "no_icustay": 19_764_220,
    "recovered": 19_764_220,
    "could_not_recover": 0,
    "icustay_missing_in_stays": 7_413_571,
}
EXACT_MIMIC_IV_1_0_RETAINED_TABLE_COUNTS = {
    "outputevents": 3_624_585,
    "chartevents": 239_861_587,
    "labevents": 51_283_821,
}
AUDITED_ARTIFACT_SHA256 = {
    "discretizer": "fe920aae287359c357ec6731e631dc27e1f24b60541156b9ecf3a1ab7c82cff3",
    "normalizer": "47f898f711f44347c082f299af5359dc89464dbb9dedafe548d66d8efdbe4341",
    "test_membership": "9fe512f5feade04158a08eb2ff7aca90476e9d54ab5a45db658abc1dd3ce5b68",
    "validation_membership": "563917e13e65ab1974152ecda546a398e30b8f7cea378b51dc7fe4bff41b8594",
    "itemid_map": "faa58ab758ccf9e187b7656fdeace6910d70c0ccfd1e3c22c4744a4a1a39b270",
}
AUDITED_UPSTREAM_COMMIT = "f33c5ed61c0ea00c28051bb2b7b50ed50485b90b"
MIMIC_IV_1_0_MANIFEST_SHA256 = (
    "fa9fe89ecfb5b156fb5ed27bd420476d1c22b1e45fd26ae66aef0368c8bca06b"
)


def source_construction(raw_root: Path):
    patients = mimic3csv.read_patients_table(raw_root / "core" / "patients.csv")
    admissions = mimic3csv.read_admissions_table(raw_root / "core" / "admissions.csv")
    stays = mimic3csv.read_icustays_table(raw_root / "icu" / "icustays.csv")

    def count(frame):
        return {
            "stays": int(frame.stay_id.nunique()),
            "admissions": int(frame.hadm_id.nunique()),
            "subjects": int(frame.subject_id.nunique()),
        }

    stages = {"raw_icustays": count(stays)}
    stays = mimic3csv.remove_icustays_with_transfers(stays)
    stages["after_transfer_filter"] = count(stays)
    stays = mimic3csv.merge_on_subject_admission(stays, admissions)
    stays = mimic3csv.merge_on_subject(stays, patients)
    stays = mimic3csv.filter_admissions_on_nb_icustays(stays)
    stages["after_single_stay_per_admission"] = count(stays)
    stays = mimic3csv.add_age_to_icustays(stays)
    stays = mimic3csv.add_inunit_mortality_to_icustays(stays)
    stays = mimic3csv.add_inhospital_mortality_to_icustays(stays)
    stays = mimic3csv.filter_icustays_on_age(stays)
    stages["after_adult_filter"] = count(stays)
    return stays, stages


def validation_counts(log_path: Path) -> dict[str, int]:
    text = log_path.read_text()
    result = {}
    for name in EXACT_MIMIC_IV_1_0_VALIDATION_COUNTS:
        match = re.search(rf"^{name}: ([0-9]+)$", text, flags=re.MULTILINE)
        if match is None:
            raise AssertionError(f"validation log omitted {name}")
        result[name] = int(match.group(1))
    return result


def retained_table_counts(log_path: Path) -> dict[str, int]:
    text = log_path.read_text()
    result = {}
    for table in ("outputevents", "chartevents", "labevents"):
        match = re.search(
            rf"^retained_event_rows {table}: ([0-9]+)$", text, flags=re.MULTILINE
        )
        if match is None:
            raise AssertionError(f"extractor log omitted retained {table} count")
        result[table] = int(match.group(1))
    if result != EXACT_MIMIC_IV_1_0_RETAINED_TABLE_COUNTS:
        raise AssertionError(f"exact MIMIC-IV 1.0 table-retention counts changed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--note-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    work_root = args.work_root.resolve()
    source_root = work_root / "in-hospital-mortality"
    manifest_path = args.task_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    task = manifest["task"]
    source = manifest["source"]
    expected_task = {
        "id": "mortality_48h",
        "published_auroc": 0.876,
        "observation_window_hours": 48.0,
        "positive_class": "death during the hospital admission",
    }
    if any(task.get(name) != value for name, value in expected_task.items()):
        raise AssertionError("manifest differs from the selected paper task/cell")
    if task.get("modalities") != ["structured", "notes"] or task.get(
        "note_codes"
    ) != [NOTE_CODE]:
        raise AssertionError(
            "selected MedPatch experiment must retain structured EHR plus the "
            "released radiology-report note channel"
        )
    if source.get("selected_experiment") != (
        "Table 3 trimodal EHR+CXR+RR mortality (MedPatch)"
    ):
        raise AssertionError("MedPatch selected the wrong paper experiment")
    if source.get("selected_launcher") != (
        "scripts/mortality/MedPatch/Confidence-Patching.sh"
    ):
        raise AssertionError("MedPatch selected the wrong released launcher")
    if source.get("common_profile") != "fixed_numeric_notes":
        raise AssertionError("MedPatch must use the prespecified fixed_numeric profile")
    if source.get("selection_tier") != 2:
        raise AssertionError(
            "MedPatch must select the structured-plus-notes multimodal tier"
        )
    if source.get("selected_model_inputs") != ["structured", "notes", "imaging"]:
        raise AssertionError("MedPatch selected the wrong model-input declaration")
    if source.get("retained_projection") != ["structured", "notes"]:
        raise AssertionError("MedPatch must retain exactly structured and notes")
    if source.get("released_split_assignment_unit") != "subject_id":
        raise AssertionError("MedPatch must preserve the released patient split")
    if source.get("unused_modality_restriction") is not False:
        raise AssertionError("an unused imaging/note restriction changed the EHR cohort")

    rows = load_source_rows(source_root)
    if len(rows) != PAPER_TOTAL:
        raise AssertionError(f"paper reports {PAPER_TOTAL} EHR examples, got {len(rows)}")
    index_path = args.task_dir / manifest["meds"]["task_index"]["file"]
    index_rows = pq.read_table(index_path).to_pylist()
    index_by_stay = {int(row["sample_id"]): row for row in index_rows}
    if len(index_by_stay) != len(index_rows):
        raise AssertionError("MEDS sample/stay identifiers are not unique")
    for row in rows:
        record = index_by_stay.get(int(row["stay_id"]))
        if record is None:
            raise AssertionError("MEDS omitted a released mortality stay")
        expected = {
            "subject_id": int(row["subject_id"]),
            "hadm_id": int(row["hadm_id"]),
            "boolean_value": bool(row["label"]),
            "split": str(row["split"]),
            "source_file": str(row["source_file"]),
            "source_position": int(row["source_position"]),
        }
        if any(record[name] != value for name, value in expected.items()):
            raise AssertionError("MEDS changed a released cohort identifier/label/split")
        if float(row["los_days"]) * 24.0 < 48.0 - 1e-6:
            raise AssertionError("mortality cohort contains a stay shorter than 48 hours")
    if len(rows) != len(index_by_stay):
        raise AssertionError("MEDS added a stay absent from the source listfiles")

    test_members = split_members(TEST_MEMBERSHIP_PATH)
    validation_members = split_members(VALIDATION_MEMBERSHIP_PATH)
    if test_members & validation_members:
        raise AssertionError("released test and validation subject sets overlap")
    for row in rows:
        subject = int(row["subject_id"])
        expected_split = (
            "test"
            if subject in test_members
            else "validation"
            if subject in validation_members
            else "train"
        )
        if row["split"] != expected_split:
            raise AssertionError("source listfile violates released patient membership")

    split_summary = {}
    subject_sets = {}
    for split in PAPER_SPLIT_PERCENT:
        selected = [row for row in rows if row["split"] == split]
        subjects = {int(row["subject_id"]) for row in selected}
        positive = sum(int(row["label"]) for row in selected)
        prevalence = 100.0 * positive / len(selected)
        ratio = 100.0 * len(selected) / len(rows)
        if {
            "rows": len(selected),
            "positive": positive,
        } != EXACT_RELEASED_SPLIT[split]:
            raise AssertionError(f"{split} released count/label marginal changed")
        if round(prevalence, 1) != PAPER_PREVALENCE_PERCENT[split]:
            raise AssertionError(f"{split} mortality prevalence differs from Table B2")
        split_summary[split] = {
            "rows": len(selected),
            "subjects": len(subjects),
            "positive": positive,
            "prevalence_percent": prevalence,
            "fraction_percent": ratio,
            "ordered_stay_id_sha256": hashlib.sha256(
                np.asarray(
                    [int(row["stay_id"]) for row in selected], dtype=np.int64
                ).tobytes()
            ).hexdigest(),
            "ordered_label_sha256": hashlib.sha256(
                np.asarray(
                    [int(row["label"]) for row in selected], dtype=np.uint8
                ).tobytes()
            ).hexdigest(),
        }
        subject_sets[split] = subjects
    overlaps = {
        "train_validation": len(subject_sets["train"] & subject_sets["validation"]),
        "train_test": len(subject_sets["train"] & subject_sets["test"]),
        "validation_test": len(subject_sets["validation"] & subject_sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"released patient splits overlap: {overlaps}")

    reconstructed_stays, construction_stages = source_construction(
        work_root / "raw_mimiciv_1_0"
    )
    all_stays = pd.read_csv(work_root / "subjects" / "all_stays.csv")
    actual_ids = set(all_stays.stay_id.astype(int))
    reconstructed_ids = set(reconstructed_stays.stay_id.astype(int))
    if actual_ids != reconstructed_ids:
        raise AssertionError("all_stays identifiers differ from released extraction functions")
    actual_labels = dict(
        zip(all_stays.stay_id.astype(int), all_stays.mortality.astype(int))
    )
    expected_labels = dict(
        zip(
            reconstructed_stays.stay_id.astype(int),
            reconstructed_stays.mortality.astype(int),
        )
    )
    if actual_labels != expected_labels:
        raise AssertionError("all_stays mortality labels differ from construction path")
    if construction_stages != {
        "raw_icustays": {"stays": 76540, "admissions": 69211, "subjects": 53150},
        "after_transfer_filter": {"stays": 70913, "admissions": 64672, "subjects": 50339},
        "after_single_stay_per_admission": {
            "stays": 59372,
            "admissions": 59372,
            "subjects": 47046,
        },
        "after_adult_filter": {
            "stays": 59372,
            "admissions": 59372,
            "subjects": 47046,
        },
    }:
        raise AssertionError(f"released raw cohort stage counts changed: {construction_stages}")

    validated = validation_counts(work_root / "logs" / "validate_events.log")
    if validated != EXACT_MIMIC_IV_1_0_VALIDATION_COUNTS:
        raise AssertionError(f"exact MIMIC-IV 1.0 validation counts differ: {validated}")
    retained_counts = retained_table_counts(work_root / "logs" / "extract_subjects.log")
    if sum(retained_counts.values()) != validated["n_events"]:
        raise AssertionError("retained table counts do not sum to validated source events")
    build_report = json.loads((work_root / "source_build.json").read_text())
    if source.get("source_build") != build_report:
        raise AssertionError("manifest source build provenance differs")
    artifact_paths = {
        "discretizer": CONFIG_PATH,
        "normalizer": NORMALIZER_PATH,
        "test_membership": TEST_MEMBERSHIP_PATH,
        "validation_membership": VALIDATION_MEMBERSHIP_PATH,
        "itemid_map": ITEMID_MAP_PATH,
    }
    actual_artifact_hashes = {
        name: sha256(path) for name, path in artifact_paths.items()
    }
    if actual_artifact_hashes != AUDITED_ARTIFACT_SHA256:
        raise AssertionError(
            f"released artifact hashes differ from the frozen audit: {actual_artifact_hashes}"
        )
    for name in (
        "discretizer",
        "normalizer",
        "itemid_map",
        "test_membership",
        "validation_membership",
    ):
        if source["source_artifacts"][name] != AUDITED_ARTIFACT_SHA256[name]:
            raise AssertionError(f"manifest does not pin the audited {name}")
    if build_report["source_files"]["itemid_map"] != AUDITED_ARTIFACT_SHA256[
        "itemid_map"
    ]:
        raise AssertionError("source build does not pin the audited ITEMID map")
    if build_report["raw_manifest_sha256"] != MIMIC_IV_1_0_MANIFEST_SHA256:
        raise AssertionError("source build does not pin exact MIMIC-IV 1.0")
    upstream = json.loads((REPOSITORY / "UPSTREAM.json").read_text())
    if upstream["repositories"][0]["commit"] != AUDITED_UPSTREAM_COMMIT:
        raise AssertionError("released upstream commit differs from the frozen audit")
    if source["upstream"]["commit"] != AUDITED_UPSTREAM_COMMIT:
        raise AssertionError("manifest does not pin the audited upstream commit")
    launcher_path = REPOSITORY / "scripts" / "mortality" / "Unimodal" / "EHR.sh"
    if source["source_artifacts"]["selected_launcher"] != sha256(launcher_path):
        raise AssertionError("selected unimodal EHR launcher changed")
    launcher = launcher_path.read_text()
    if "--modalities EHR" not in launcher or "--fusion_type unimodal_ehr" not in launcher:
        raise AssertionError("selected launcher is not the exact structured-only experiment")
    metadata_path = REPOSITORY / "datasets" / "DataFusion.py"
    if source["source_artifacts"]["metadata_loader"] != sha256(metadata_path):
        raise AssertionError("released metadata loader changed")
    metadata_source = metadata_path.read_text()
    for marker in (
        "elif args.data_pairs == 'partial':",
        "rr_ehr_merged_icustays",
        "how='left'",
    ):
        if marker not in metadata_source:
            raise AssertionError(
                f"metadata loader lacks the partial-pairing marker {marker!r}"
            )

    # The trimodal cell attaches radiology reports by left join, so note
    # availability is not a cohort filter. Pin the released coverage against
    # the paper's own Table 1 RR sample size.
    from check_batch import released_radiology_text

    cohort_stays = {int(row["stay_id"]) for row in rows}
    released_notes = released_radiology_text(args.work_root, args.note_root)
    with_report = sum(
        1 for stay_id in cohort_stays if released_notes.get(stay_id)
    )
    if with_report != PAPER_RR_TOTAL:
        raise AssertionError(
            f"reconstructed radiology coverage {with_report} does not match the "
            f"paper's Table 1 RR sample size {PAPER_RR_TOTAL}"
        )

    sorted_ids = np.sort(
        np.asarray([int(row["stay_id"]) for row in rows], dtype=np.int64)
    )
    report = {
        "status": "exact",
        "radiology_note_coverage": {
            "cohort_samples": len(cohort_stays),
            "samples_with_report": with_report,
            "paper_table_1_rr_examples": PAPER_RR_TOTAL,
            "note_availability_is_a_cohort_filter": False,
        },
        "paper_evidence": {
            "table_1_ehr_examples": PAPER_TOTAL,
            "table_b2_prevalence_percent": PAPER_PREVALENCE_PERCENT,
            "nominal_split_percent": PAPER_SPLIT_PERCENT,
            "task": "death during admission from first 48 ICU hours",
            "modalities": ["structured"],
        },
        "cohort_rows": len(rows),
        "sorted_stay_id_sha256": hashlib.sha256(sorted_ids.tobytes()).hexdigest(),
        "splits": split_summary,
        "cross_split_subject_overlap": overlaps,
        "paper_nominal_split_percent": PAPER_SPLIT_PERCENT,
        "released_sample_fraction_note": (
            "The fixed patient artifacts reconstruct 71.9/8.2/20.0 percent of "
            "eligible stay samples, not the paper caption's nominal 70/10/20."
        ),
        "construction_stages": construction_stages,
        "released_validation_counts": validated,
        "retained_event_rows_by_source_table": retained_counts,
        "released_membership_artifacts": {
            "test_sha256": sha256(TEST_MEMBERSHIP_PATH),
            "validation_sha256": sha256(VALIDATION_MEMBERSHIP_PATH),
        },
        "audited_artifact_sha256": AUDITED_ARTIFACT_SHA256,
        "upstream_commit": AUDITED_UPSTREAM_COMMIT,
        "mimic_iv_1_0_manifest_sha256": MIMIC_IV_1_0_MANIFEST_SHA256,
        "code_vocabulary_size": len(code_vocabulary()),
        "retained_projection": ["structured"],
        "checks": [
            "paper total/prevalence and nominal split statement versus exact released counts",
            "exact stay, admission, subject, label, source order, and split membership",
            "released patient membership and zero patient overlap",
            "raw ICU filters and all_stays identifier/label reconstruction",
            "exact MIMIC-IV 1.0 event-validation counts",
            "exact retained rows from outputevents, chartevents, and labevents",
            "all EHR mortality listfile samples without CXR or note availability restriction",
        ],
    }
    report_path = args.task_dir / "audit" / "cohort_reconstruction.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
