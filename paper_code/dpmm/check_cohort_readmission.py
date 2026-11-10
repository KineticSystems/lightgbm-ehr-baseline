"""Assert DPMM's exact MIMIC-III partial EHR-note readmission cohort."""

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
    ensure_source,
    load_source,
    ordered_text_hash,
    sha256,
    source_codes,
)
from cohort_readmission import (  # noqa: E402
    EXPECTED_COUNTS,
    EXPECTED_DEFECT_MEASUREMENT,
    EXPECTED_PAIRED,
    EXPECTED_PAIRED_POSITIVES,
    EXPECTED_POSITIVES,
    PUBLISHED_PAIRED_POSITIVES,
    PUBLISHED_PAIRED_TOTAL,
    PUBLISHED_TOTAL_POSITIVES,
    TASK_ID,
    ensure_readmission_labels,
    readmission_cohort,
    released_readmission_labels,
)


NEGATIVE_GAP_STAY_ID = 263279


def ordered_int_hash(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--readmission-cache", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    ensure_source(args.mimic_root, args.cache_root)
    readm_manifest = ensure_readmission_labels(
        args.mimic_root, args.readmission_cache
    )
    base_cohort, _, notes, source_manifest = load_source(args.cache_root)

    manifest_path = args.task_dir / "manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["task"]["id"] != TASK_ID:
        raise AssertionError("wrong DPMM task identity")
    if manifest["task"]["modalities"] != ["structured", "notes"]:
        raise AssertionError("selected DPMM task must retain EHR and raw notes")
    if manifest["task"]["note_codes"] != [NOTE_CODE]:
        raise AssertionError("DPMM note-code declaration changed")
    if manifest["task"]["observation_window_hours"] != 48.0:
        raise AssertionError("released readmission n_hours is 48")
    selected = manifest["source"]["source_configuration"]
    expected_config = {
        "task": "readmission",
        "labels_set": "readm",
        "data_pairs": "partial_ehr_note",
        "data_ratio": 1.0,
    }
    for key, expected in expected_config.items():
        if selected.get(key) != expected:
            raise AssertionError(f"wrong selected source configuration: {key}")

    # Independent re-run of the released labeler on the exact raw release.
    recomputed, measurement = released_readmission_labels(args.mimic_root)
    if measurement != EXPECTED_DEFECT_MEASUREMENT:
        raise AssertionError(
            f"released readmission defect measurement changed: {measurement}"
        )
    cached_labels = pd.read_parquet(
        args.readmission_cache / readm_manifest["files"]["readmission_labels"]
    )
    left = recomputed.sort_values("stay_id").reset_index(drop=True)
    right = cached_labels.sort_values("stay_id").reset_index(drop=True)
    for column in ("stay_id", "readmission_30d", "index_corrected", "bounded"):
        if not np.array_equal(
            left[column].to_numpy(np.int64), right[column].to_numpy(np.int64)
        ):
            raise AssertionError(f"cached readmission labels drifted: {column}")

    negative = recomputed.loc[recomputed["next_admission_gap_hours"] < 0]
    if len(negative) != 1 or int(negative["stay_id"].iloc[0]) != NEGATIVE_GAP_STAY_ID:
        raise AssertionError("the audited negative-gap readmission case changed")
    if int(negative["readmission_30d"].iloc[0]) != 1 or int(
        negative["bounded"].iloc[0]
    ) != 0:
        raise AssertionError("the negative-gap stay is no longer a released positive")

    cohort = readmission_cohort(base_cohort, cached_labels)
    index = pq.read_table(
        args.task_dir / manifest["meds"]["task_index"]["file"]
    ).to_pandas()
    exported = index.sort_values(["split", "fusion_position"]).reset_index(drop=True)
    expected_frame = cohort.sort_values(["split", "fusion_position"]).reset_index(
        drop=True
    )
    comparisons = {
        "subject_id": (exported["subject_id"], expected_frame["SUBJECT_ID"]),
        "sample_id": (exported["sample_id"], expected_frame["stay_id"]),
        "boolean_value": (
            exported["boolean_value"],
            expected_frame["readm_label"].astype(bool),
        ),
        "split": (exported["split"], expected_frame["split"]),
        "source_name": (exported["source_name"], expected_frame["source_name"]),
        "hadm_id": (exported["hadm_id"], expected_frame["HADM_ID"]),
        "fusion_position": (
            exported["fusion_position"],
            expected_frame["fusion_position"],
        ),
        "note_present": (exported["note_present"], expected_frame["note_present"]),
    }
    for name, (actual, expected) in comparisons.items():
        if not np.array_equal(actual.to_numpy(), expected.to_numpy()):
            raise AssertionError(f"MEDS changed exact cohort field: {name}")

    counts = exported["split"].value_counts().to_dict()
    positives = exported.groupby("split")["boolean_value"].sum().astype(int).to_dict()
    paired = exported.loc[exported["note_present"]].groupby("split").size().to_dict()
    paired_positives = (
        exported.loc[exported["note_present"]]
        .groupby("split")["boolean_value"]
        .sum()
        .astype(int)
        .to_dict()
    )
    if counts != EXPECTED_COUNTS:
        raise AssertionError(f"published split counts differ: {counts}")
    if positives != EXPECTED_POSITIVES:
        raise AssertionError(f"released readmission positive counts differ: {positives}")
    if paired != EXPECTED_PAIRED:
        raise AssertionError(f"published paired-note counts differ: {paired}")
    if paired_positives != EXPECTED_PAIRED_POSITIVES:
        raise AssertionError(
            f"released paired readmission positives differ: {paired_positives}"
        )
    if len(exported) != 21139:
        raise AssertionError("published total cohort count differs")
    if PUBLISHED_PAIRED_POSITIVES["validation"] != paired_positives["validation"] or (
        PUBLISHED_PAIRED_POSITIVES["test"] != paired_positives["test"]
    ):
        raise AssertionError(
            "the complete released validation/test paired positives must match the "
            "published diagnostic exactly"
        )

    # The readmission task must be the same event stream with a different label.
    ihm_positive = int(cohort["ihm_label"].sum())
    if ihm_positive != 2797:
        raise AssertionError("the shared IHM label set changed")
    if int((cohort["ihm_label"] != cohort["readm_label"]).sum()) == 0:
        raise AssertionError("readmission labels are identical to the IHM labels")
    override_only = int(
        ((cohort["MORTALITY"].astype(int) == 1) & (cohort["readmission_30d"] == 0)).sum()
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

    sampled = pd.read_parquet(args.cache_root / "sampled_note_lists.parquet")
    sampled_counts = sampled.groupby("split").size().to_dict()
    expected_sampled = {"train": 4894, "validation": 1074, "test": 1079}
    if sampled_counts != expected_sampled:
        raise AssertionError(
            f"released one-third note sampling changed: {sampled_counts}"
        )
    if len(notes) != 5 * sum(EXPECTED_PAIRED.values()):
        raise AssertionError("every paired sample must have exactly five retained notes")

    codes = pq.read_table(args.task_dir / manifest["meds"]["codes"]["file"])[
        "code"
    ].to_pylist()
    if codes != list(source_codes()):
        raise AssertionError("MEDS code vocabulary/order differs from the release")
    if any("image" in code.lower() or "cxr" in code.lower() for code in codes):
        raise AssertionError("an imaging code entered the DPMM task")

    report = {
        "status": "exact_source_with_published_drop_last_count",
        "selected_experiment": {
            "paper_cell": manifest["source"]["selected_paper_cell"],
            "task": "MIMIC-III 30-day readmission",
            "model_inputs": ["structured", "notes"],
            "configuration": expected_config,
            "selection_tier": 2,
            "observation_window_hours": 48.0,
            "prediction_time": "ICU admission + 48 hours",
        },
        "label_definition": readm_manifest["label_definition"],
        "released_label_producer": readm_manifest["producer"],
        "released_label_producer_sha256": readm_manifest["producer_sha256"],
        "splits": {
            split: {
                "samples": counts[split],
                "subjects": len(subject_sets[split]),
                "positive": positives[split],
                "prevalence": positives[split] / counts[split],
                "paired_notes": paired[split],
                "paired_positive": paired_positives[split],
                "sampled_for_note_availability": sampled_counts[split],
            }
            for split in ("train", "validation", "test")
        },
        "total": {
            "samples": len(exported),
            "subjects": int(exported["subject_id"].nunique()),
            "positive": int(exported["boolean_value"].sum()),
            "paired_notes": int(exported["note_present"].sum()),
            "note_rows": len(notes),
        },
        "audited_label_defect": {
            "finding": (
                "paper_audits/dpmm.md, Critical - readmission labels are scrambled: "
                "create_readmission_30d.py iterates index labels but indexes the "
                "next stay positionally with .iloc"
            ),
            "measurement": measurement,
            "conclusion": (
                "latent on this cohort: the ADMITTIME/DISCHTIME sort reorders no "
                "subject because the benchmark keeps one ICU stay per admission, so "
                "no released label changes. The faithful released labels are "
                "exported; no corrected variant is produced."
            ),
            "remaining_definitional_defects": {
                "missing_lower_bound": {
                    "stays_changed": measurement["stays_changed_by_missing_lower_bound"],
                    "stay_id": NEGATIVE_GAP_STAY_ID,
                    "gap_hours": float(
                        negative["next_admission_gap_hours"].iloc[0]
                    ),
                },
                "whole_day_flooring": "pandas Timedelta.days < 30 rather than <= 30 days",
                "in_hospital_death_override_positives": override_only,
            },
        },
        "paper_statistic_comparison": {
            "published_supplement_total_positive": PUBLISHED_TOTAL_POSITIVES,
            "released_source_reconstruction_total_positive": int(
                exported["boolean_value"].sum()
            ),
            "total_delta": int(exported["boolean_value"].sum())
            - PUBLISHED_TOTAL_POSITIVES,
            "released_comment_paired_positive": PUBLISHED_PAIRED_POSITIVES,
            "released_comment_paired_total": PUBLISHED_PAIRED_TOTAL,
            "released_source_reconstruction_paired_positive": EXPECTED_PAIRED_POSITIVES,
            "released_source_reconstruction_paired_total": sum(
                EXPECTED_PAIRED_POSITIVES.values()
            ),
            "paired_delta": sum(EXPECTED_PAIRED_POSITIVES.values())
            - PUBLISHED_PAIRED_TOTAL,
            "interpretation": (
                "The released fusion diagnostic sums the training DataLoader, which "
                "uses shuffle=True and drop_last=True. At batch size 16 it omits a "
                "random four paired training examples, one of which is positive; "
                "validation and test match exactly. The exact source labels are "
                "retained rather than altered to reproduce the incomplete count."
            ),
        },
        "shared_event_stream": {
            "base_task": "mortality_48h_partial_ehr_note",
            "identical_cohort_split_and_events": True,
            "ihm_positive": ihm_positive,
            "stays_with_different_label": int(
                (cohort["ihm_label"] != cohort["readm_label"]).sum()
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
        "source_reconstruction": source_manifest,
        "readmission_label_reconstruction": readm_manifest,
        "retained_projection": {
            "modalities": ["structured", "notes"],
            "note_codes": [NOTE_CODE],
            "codes": codes,
            "no_imaging_or_auxiliary_rows": True,
        },
        "manifest_sha256": manifest_hash,
        "checks": [
            "exact 21,139 partial-EHR cohort with the released 48-hour window",
            "exact 14,681/3,222/3,236 split counts",
            "released readmission labels independently recomputed from MIMIC-III 1.4",
            "released 2,794/640/556 positive counts",
            "exact 3,652/815/806 paired-note counts and 693/167/139 paired positives",
            "released paired diagnostic 692/167/139 is a drop-last loader observation",
            "published supplement total 3,987 positives is the same drop-last "
            "observation of the exact 3,990 released labels",
            "audited index-misuse defect measured latent (0 reordered, 0 changed)",
            "one negative-gap stay preserved as a released positive",
            "released subject resources produce zero overlap",
            "released one-third note prefilter counts are exact",
            "exact identifiers, fusion positions, labels, and modality presence",
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
