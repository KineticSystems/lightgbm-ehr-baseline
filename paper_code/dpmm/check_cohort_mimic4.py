"""Assert DPMM's exact MIMIC-IV partial-cohort EHR task."""

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

from cohort_source_mimic4 import (  # noqa: E402
    EXPECTED_COUNTS,
    EXPECTED_TOTAL,
    HOURS,
    IHM_TASK_ID,
    PUBLISHED_IHM_TOTAL_POSITIVES,
    PUBLISHED_LOADER_IHM_POSITIVES,
    PUBLISHED_LOADER_READM_POSITIVES,
    PUBLISHED_PAIRED_CXR,
    PUBLISHED_PAIRED_CXR_TOTAL,
    PUBLISHED_PAIRED_IHM_POSITIVES,
    PUBLISHED_PAIRED_READM_POSITIVES,
    PUBLISHED_READM_TOTAL_POSITIVES,
    READM_TASK_ID,
    assign_split,
    ensure_source,
    load_source,
    ordered_text_hash,
    released_candidates,
    released_diagnosis_stays,
    released_readmission_labels,
    sha256,
    source_codes,
)


TASKS = {
    IHM_TASK_ID: {
        "author_task": "in-hospital-mortality",
        "labels_set": "mortality",
        "label_column": "mortality_label",
        "published_total": PUBLISHED_IHM_TOTAL_POSITIVES,
        "published_loader": PUBLISHED_LOADER_IHM_POSITIVES,
        "published_paired_positives": PUBLISHED_PAIRED_IHM_POSITIVES,
        "name": "MIMIC-IV in-hospital mortality",
    },
    READM_TASK_ID: {
        "author_task": "readmission",
        "labels_set": "readm",
        "label_column": "readmission_label",
        "published_total": PUBLISHED_READM_TOTAL_POSITIVES,
        "published_loader": PUBLISHED_LOADER_READM_POSITIVES,
        "published_paired_positives": PUBLISHED_PAIRED_READM_POSITIVES,
        "name": "MIMIC-IV 30-day readmission",
    },
}


def ordered_int_hash(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def released_cxr_pairing(cxr_root: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """Reproduce `dataset_mimic4/fusion.py:loadmetadata` exactly, transiently.

    This never enters the exported task; it only proves which stays the paper's
    imaging branch matched, and therefore that `data_ratio=1.0` keeps every
    unpaired stay in the partial cohort.
    """
    metadata = pd.read_csv(
        cxr_root / "mimic-cxr-2.0.0-metadata.csv.gz",
        usecols=[
            "dicom_id",
            "subject_id",
            "study_id",
            "ViewPosition",
            "StudyDate",
            "StudyTime",
        ],
    )
    columns = ["subject_id", "stay_id", "intime", "outtime"]
    merged = metadata.merge(cohort[columns], how="inner", on="subject_id")
    merged["StudyTime"] = merged["StudyTime"].apply(lambda x: f"{int(float(x)):06}")
    merged["StudyDateTime"] = pd.to_datetime(
        merged["StudyDate"].astype(str) + " " + merged["StudyTime"].astype(str),
        format="%Y%m%d %H%M%S",
    )
    merged["intime"] = pd.to_datetime(merged["intime"])
    merged["outtime"] = pd.to_datetime(merged["outtime"])
    end_time = merged["intime"] + pd.DateOffset(hours=HOURS)
    during = merged.loc[
        (merged["StudyDateTime"] >= merged["intime"])
        & (merged["StudyDateTime"] <= end_time)
    ]
    ap_views = during[during["ViewPosition"] == "AP"]
    selected = (
        ap_views.sort_values("StudyDateTime").groupby("stay_id", as_index=False).tail(1)
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cxr-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    ensure_source(args.mimic_root, args.cache_root)
    cohort, source_manifest = load_source(args.cache_root)

    manifest_path = args.task_dir / "manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    task_id = manifest["task"]["id"]
    if task_id not in TASKS:
        raise AssertionError("wrong DPMM MIMIC-IV task identity")
    spec = TASKS[task_id]
    if manifest["task"]["modalities"] != ["structured"]:
        raise AssertionError("DPMM MIMIC-IV tasks retain structured inputs only")
    if manifest["task"]["note_codes"] != []:
        raise AssertionError("DPMM MIMIC-IV has no note channel")
    if manifest["task"]["observation_window_hours"] != 48.0:
        raise AssertionError("released MIMIC-IV n_hours is 48")
    configuration = manifest["source"]["source_configuration"]
    expected_config = {
        "task": spec["author_task"],
        "labels_set": spec["labels_set"],
        "data_pairs": "partial_ehr_cxr",
        "data_ratio": 1.0,
    }
    for key, expected in expected_config.items():
        if configuration.get(key) != expected:
            raise AssertionError(f"wrong selected source configuration: {key}")
    if manifest["source"]["mimic_iv_version"] != "2.2":
        raise AssertionError("the declared MIMIC-IV release changed")

    # Independent re-run of the released cohort and label producers.
    all_stays, candidates = released_candidates(args.mimic_root)
    candidates = candidates.copy()
    candidates["split"] = assign_split(candidates)
    diagnosis_stays = released_diagnosis_stays(args.mimic_root, all_stays)
    without_diagnosis = candidates.loc[
        ~candidates["stay_id"].isin(diagnosis_stays)
    ]
    if set(cohort["stay_id"]) & set(without_diagnosis["stay_id"]):
        raise AssertionError(
            "a stay with no released ICD diagnosis row entered the cohort; the "
            "released assemble_episodic_data inner merge excludes it"
        )
    labels, measurement = released_readmission_labels(all_stays)
    recomputed = candidates.merge(
        labels[["stay_id", "readmission_30d"]], on="stay_id", how="left", validate="1:1"
    )
    recomputed["readmission_label"] = np.where(
        recomputed["mortality"].astype(int) == 1,
        1,
        recomputed["readmission_30d"].astype(int),
    )
    recomputed["mortality_label"] = recomputed["mortality"].astype(int)
    recomputed = recomputed.set_index("stay_id")
    if not set(cohort["stay_id"]).issubset(set(recomputed.index)):
        raise AssertionError("the exported cohort is not a subset of released candidates")
    aligned = recomputed.loc[cohort["stay_id"].to_numpy()]
    for column in ("mortality_label", "readmission_label"):
        if not np.array_equal(
            aligned[column].to_numpy(np.int64), cohort[column].to_numpy(np.int64)
        ):
            raise AssertionError(f"cached labels drifted from the released producer: {column}")
    if not np.array_equal(
        aligned["split"].to_numpy(), cohort["split"].to_numpy()
    ):
        raise AssertionError("cached split drifted from the released resources")
    dropped = len(candidates) - len(cohort)
    dropped_no_diagnosis = int(
        (~candidates["stay_id"].isin(diagnosis_stays)).sum()
    )
    dropped_by_split = {
        split: int(
            (candidates["split"] == split).sum()
            - (cohort["split"] == split).sum()
        )
        for split in ("train", "validation", "test")
    }
    if measurement != source_manifest["readmission_defect_measurement"]:
        raise AssertionError("released readmission defect measurement changed")
    if measurement["subjects_reordered_by_admittime_sort"] or measurement[
        "stays_changed_by_index_misuse"
    ]:
        raise AssertionError(
            "the audited readmission index defect is no longer latent; a corrected "
            "variant would be required"
        )

    index = pq.read_table(args.task_dir / manifest["meds"]["task_index"]["file"]).to_pandas()
    exported = index.sort_values("sample_id").reset_index(drop=True)
    expected_frame = cohort.sort_values("stay_id").reset_index(drop=True)
    comparisons = {
        "subject_id": (exported["subject_id"], expected_frame["subject_id"]),
        "sample_id": (exported["sample_id"], expected_frame["stay_id"]),
        "boolean_value": (
            exported["boolean_value"],
            expected_frame[spec["label_column"]].astype(bool),
        ),
        "split": (exported["split"], expected_frame["split"]),
        "source_name": (exported["source_name"], expected_frame["source_name"]),
        "hadm_id": (exported["hadm_id"], expected_frame["hadm_id"]),
        "episode_number": (
            exported["episode_number"],
            expected_frame["episode_number"],
        ),
    }
    for name, (actual, expected) in comparisons.items():
        if not np.array_equal(actual.to_numpy(), expected.to_numpy()):
            raise AssertionError(f"MEDS changed exact cohort field: {name}")

    counts = exported["split"].value_counts().to_dict()
    positives = exported.groupby("split")["boolean_value"].sum().astype(int).to_dict()
    if counts != EXPECTED_COUNTS or len(exported) != EXPECTED_TOTAL:
        raise AssertionError(f"published split counts differ: {counts}")
    for split in ("validation", "test"):
        if positives[split] != spec["published_loader"][split]:
            raise AssertionError(
                f"complete released {split} positives must match the published "
                f"diagnostic exactly: {positives}"
            )
    if positives["train"] < spec["published_loader"]["train"]:
        raise AssertionError(
            "released training positives cannot be below the drop-last diagnostic"
        )

    subject_sets = {
        split: set(exported.loc[exported["split"] == split, "subject_id"].astype(int))
        for split in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": len(subject_sets["train"] & subject_sets["validation"]),
        "train_test": len(subject_sets["train"] & subject_sets["test"]),
        "validation_test": len(subject_sets["validation"] & subject_sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"released patient splits overlap: {overlaps}")

    paired = released_cxr_pairing(args.cxr_root, cohort)
    paired_stays = set(int(value) for value in paired["stay_id"])
    paired_frame = exported.loc[exported["sample_id"].isin(paired_stays)]
    paired_counts = paired_frame["split"].value_counts().to_dict()
    paired_positives = int(paired_frame["boolean_value"].sum())
    if paired_counts != PUBLISHED_PAIRED_CXR:
        raise AssertionError(
            f"released CXR pairing does not reproduce the published matched cohort: "
            f"{paired_counts}"
        )
    if len(paired_frame) != PUBLISHED_PAIRED_CXR_TOTAL:
        raise AssertionError("published matched-cohort total differs")

    codes = pq.read_table(args.task_dir / manifest["meds"]["codes"]["file"])[
        "code"
    ].to_pylist()
    if codes != list(source_codes()):
        raise AssertionError("MEDS code vocabulary/order differs from the release")
    if any(
        "image" in code.lower() or "cxr" in code.lower() or "note" in code.lower()
        for code in codes
    ):
        raise AssertionError("a non-structured code entered the DPMM MIMIC-IV task")

    events = pq.ParquetFile(args.task_dir / manifest["meds"]["events"]["file"])
    observed_codes = set()
    for batch in events.iter_batches(columns=["code"], batch_size=250_000):
        observed_codes.update(batch.column(0).to_pylist())
    absent = [code for code in codes if code not in observed_codes]

    report = {
        "status": "exact_source_with_published_drop_last_count",
        "selected_experiment": {
            "paper_cell": manifest["source"]["selected_paper_cell"],
            "task": spec["name"],
            "source_model_inputs": ["structured", "chest radiograph"],
            "retained_model_inputs": ["structured"],
            "configuration": expected_config,
            "selection_tier": 4,
            "observation_window_hours": 48.0,
            "prediction_time": "ICU admission + 48 hours",
            "mimic_iv_version": "2.2",
        },
        "label_definition": manifest["source"]["label_definition"],
        "released_label_producer": manifest["source"]["label_producer"],
        "partial_cohort_definition": {
            "released_rule": (
                "MIMIC_CXR_EHR.__len__ for data_pairs=partial_ehr_cxr returns "
                "len(ehr_files_paired) + int(data_ratio * len(ehr_files_unpaired)); "
                "with data_ratio=1.0 that is every stay in the released EHR list "
                "files, so the chest radiograph never restricts membership"
            ),
            "ehr_stays": len(exported),
            "cxr_paired_stays": len(paired_frame),
            "cxr_paired_by_split": paired_counts,
            "published_cxr_paired_by_split": PUBLISHED_PAIRED_CXR,
            "cxr_paired_positives": paired_positives,
            "published_cxr_paired_positives": spec["published_paired_positives"],
            "imaging_retained": False,
        },
        "cohort_construction": {
            "released_candidates_with_los_at_least_48h": len(candidates),
            "dropped_total": dropped,
            "dropped_for_no_released_diagnosis_row": dropped_no_diagnosis,
            "dropped_for_no_retained_measurement": dropped - dropped_no_diagnosis,
            "dropped_by_split": dropped_by_split,
            "released_rule": (
                "assemble_episodic_data ends in an inner merge with the ICD "
                "diagnosis label matrix, so a stay whose admission has no "
                "diagnosis row gets an empty episode label CSV and is skipped "
                "by both list-file builders at their label_df.shape[0] == 0 "
                "guard"
            ),
        },
        "splits": {
            split: {
                "samples": counts[split],
                "subjects": len(subject_sets[split]),
                "positive": positives[split],
                "prevalence": positives[split] / counts[split],
                "cxr_paired": paired_counts[split],
            }
            for split in ("train", "validation", "test")
        },
        "total": {
            "samples": len(exported),
            "subjects": int(exported["subject_id"].nunique()),
            "positive": int(exported["boolean_value"].sum()),
        },
        "paper_statistic_comparison": {
            "published_supplement_total_positive": spec["published_total"],
            "released_source_reconstruction_total_positive": int(
                exported["boolean_value"].sum()
            ),
            "published_drop_last_loader_positive": spec["published_loader"],
            "released_source_reconstruction_positive": positives,
            "interpretation": (
                "The released fusion diagnostic sums the training DataLoader, which "
                "uses shuffle=True and drop_last=True at batch size 32, so it omits "
                "a random remainder of 16 training examples; validation and test are "
                "complete and match exactly. The exact source labels are retained "
                "rather than altered to reproduce the incomplete count."
            ),
        },
        "audited_label_defect": {
            "finding": (
                "paper_audits/dpmm.md, Critical - readmission labels are scrambled: "
                "the MIMIC-IV copy of create_readmission_30d.py has the same index "
                "misuse as the MIMIC-III copy"
            ),
            "measurement": measurement,
            "conclusion": (
                "fully latent on this cohort: no subject reorders under the "
                "ADMITTIME/DISCHTIME sort, no released label changes, and there is "
                "no negative inter-admission gap. The faithful released labels are "
                "exported; no corrected variant is produced."
            ),
        },
        "cross_split_subject_overlap": overlaps,
        "ordered_sha256": {
            "sample_id": ordered_int_hash(exported["sample_id"]),
            "subject_id": ordered_int_hash(exported["subject_id"]),
            "label": ordered_int_hash(exported["boolean_value"].astype(int)),
            "source_name": ordered_text_hash(exported["source_name"]),
            "split": ordered_text_hash(exported["split"]),
        },
        "retained_projection": {
            "modalities": ["structured"],
            "note_codes": [],
            "codes": codes,
            "codes_absent_from_this_release": absent,
            "no_imaging_or_auxiliary_rows": True,
        },
        "source_reconstruction": source_manifest,
        "manifest_sha256": manifest_hash,
        "checks": [
            "exact 25,071 partial-EHR cohort with the released 48-hour window",
            "exact 18,064/2,035/4,972 split counts",
            "released cohort, split and labels independently recomputed from "
            "MIMIC-IV 2.2",
            "complete validation and test positives match the released diagnostic",
            "released CXR pairing reproduces the published 4,287/465/1,179 matched "
            "cohort without restricting membership",
            "audited readmission index defect measured fully latent",
            "released subject resources produce zero overlap",
            "exact identifiers, episode numbers, labels and 59-code vocabulary",
        ],
    }
    output = args.task_dir / "audit" / "cohort_reconstruction.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if sha256(manifest_path) != manifest_hash:
        raise AssertionError("cohort checker mutated immutable manifest")
    print(json.dumps({"status": report["status"], "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
