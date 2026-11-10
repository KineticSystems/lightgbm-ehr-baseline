"""Export a MERGE structured/raw-note projection to MEDS Parquet.

The paper's two MIMIC-IV binary experiments (Table 2 IHM and Table 2 LOS)
share one released producer, cohort, window, split, structured branch, and
note stream; only the released label constructor differs.  ``--task``
selects which released task script is transcribed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPOSITORY.parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from cohort_source import (  # noqa: E402
    FEATURE_NAMES,
    HOURS,
    LOS_HOURS,
    LOS_PUBLISHED_AUROC,
    LOS_TASK_ID,
    NOTE_CODE,
    PUBLISHED_AUROC,
    SEED,
    TASK_ID,
    TASKS,
    UPSTREAM_COMMIT,
    array_sha256,
    ensure_note_cache,
    event_batches,
    reconstruct_author_boundary,
    retained_notes,
    samples_frame,
    scaler_summary,
)


TASK_METADATA = {
    "ihm": {
        "task_id": TASK_ID,
        "name": "In-hospital mortality from the first 48 ICU hours",
        "published_auroc": PUBLISHED_AUROC,
        "paper_cell": "Table 2, MIMIC-IV IHM, MERGE, AUROC 85.40 +/- 0.24",
        "task_script": "mimiciv/data_preprocess/create_ihm_task.py",
        "restrict_hours": HOURS,
        "trainer": "mimiciv/train_mimiciv_ihm.sh",
        "positive_class": (
            "ADMISSIONS.hospital_expire_flag == 1 (hospital death)"
        ),
        "author_scaling": "create_ihm_task.py::fit_and_apply_scaler",
        "dataset_name": (
            "MERGE MIMIC-IV IHM structured and radiology-note projection"
        ),
    },
    "los": {
        "task_id": LOS_TASK_ID,
        "name": (
            "ICU stay under 96 hours with hospital survival, from the first "
            "48 ICU hours"
        ),
        "published_auroc": LOS_PUBLISHED_AUROC,
        "paper_cell": "Table 2, MIMIC-IV LOS, MERGE, AUROC 81.88 +/- 0.18",
        "task_script": "mimiciv/data_preprocess/create_los_task.py",
        # create_los_task.py hard-codes the 48-hour observation window; it has
        # no --restrict_hours option.
        "restrict_hours": HOURS,
        "trainer": "mimiciv/train_mimiciv_los.sh",
        "positive_class": (
            "(ICUSTAYS.outtime - ICUSTAYS.intime) < 96 hours AND "
            "ADMISSIONS.hospital_expire_flag == 0 (released label)"
        ),
        "author_scaling": "create_los_task.py::fit_and_apply_scaler",
        "dataset_name": (
            "MERGE MIMIC-IV LOS structured and radiology-note projection"
        ),
    },
}


def split_summary(records: list[dict]) -> dict[str, dict[str, object]]:
    result = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in records if row["split"] == split]
        positives = sum(int(row["label"]) for row in selected)
        result[split] = {
            "count": len(selected),
            "subjects": len({int(row["subject_id"]) for row in selected}),
            "admissions": len({int(row["hadm_id"]) for row in selected}),
            "positive": positives,
            "prevalence": positives / len(selected),
            "minimum_irregular_rows": min(row["irregular_rows"] for row in selected),
            "maximum_irregular_rows": max(row["irregular_rows"] for row in selected),
        }
    return result


def text_sha256(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def note_summary(notes, records: list[dict]) -> dict[str, object]:
    split_by_stay = {int(row["stay_id"]): row["split"] for row in records}
    result = {}
    for split in ("train", "validation", "test"):
        selected = notes.loc[
            notes["stay_id"].map(split_by_stay).eq(split)
        ]
        result[split] = {
            "notes": len(selected),
            "stays_with_notes": int(selected["stay_id"].nunique()),
            "raw_text_sha256": text_sha256(selected["text"]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--note-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, default="ihm")
    args = parser.parse_args()
    selected = TASK_METADATA[args.task]

    note_cache = ensure_note_cache(
        args.mimic_root, args.note_root, args.cache_root
    )
    boundary = reconstruct_author_boundary(
        args.mimic_root, args.cache_root, task=args.task
    )
    records = boundary["records"]
    prefilter = boundary["prefilter_records"]
    retained_ids = {int(row["stay_id"]) for row in records}
    notes = retained_notes(args.cache_root, retained_ids)
    if not set(notes["stay_id"].astype(int)).issubset(retained_ids):
        raise AssertionError("note projection contains a non-retained stay")
    record_by_stay = {int(row["stay_id"]): row for row in records}
    note_cutoffs = notes["stay_id"].map(
        lambda value: record_by_stay[int(value)]["intime"] + np.timedelta64(48, "h")
    )
    retained_unavailable = notes["storetime"] >= note_cutoffs
    feature_hash = hashlib.sha256(
        json.dumps(list(FEATURE_NAMES), separators=(",", ":")).encode()
    ).hexdigest()
    source = {
        "upstream_commit": UPSTREAM_COMMIT,
        "selected_paper_cell": selected["paper_cell"],
        "selected_model_inputs": ["structured", "notes", "CXR"],
        "selection_tier": 3,
        "retained_modalities": ["structured", "notes"],
        "omitted_modalities": ["CXR"],
        "omitted_modality_cohort_effect": (
            "none: the structured pivot contains every LOS>=2 candidate stay; "
            "the released include_missing union with notes/CXR is therefore identical; "
            "note absence remains an empty document rather than a cohort filter"
        ),
        "source_configuration": {
            "task_script": selected["task_script"],
            "launcher": "mimiciv/data_preprocess/preprocess_mimic.sh",
            "trainer": selected["trainer"],
            "restrict_hours": selected["restrict_hours"],
            "include_notes": True,
            "include_cxr": True,
            "include_missing": True,
            "standardize_features": True,
            "seed": SEED,
        },
        "structured_capture_boundary": (
            "float32 labs_vitals [48,30] branch after released half-open cutoff, "
            "train-only interpolation/fallback, StandardScaler, >12 irregular-"
            "timestamp filter, and Dataset conversion; before model computation"
        ),
        "note_capture_boundary": (
            "raw MIMIC-IV-Note 2.2 radiology text and charttime after the "
            "released add_time_delta_notes_vectorized mapper and the selected "
            "half-open [0,48) ICU-hour filter; before unpinned BioBERT encoding"
        ),
        "positive_class": selected["positive_class"],
        "split_policy": (
            "np.union1d-sorted eligible stay IDs, NumPy RandomState seed 42 shuffle, "
            "then released 70/15/15 stay slicing"
        ),
        "prefilter": {
            "count": len(prefilter),
            "split_counts": {
                split: sum(row["split"] == split for row in prefilter)
                for split in ("train", "validation", "test")
            },
            "cross_split_subject_overlap": boundary["prefilter_subject_overlap"],
        },
        "postfilter": {
            "rule": "len(ts_tt) > 12",
            "count": len(records),
            "splits": split_summary(records),
            "cross_split_subject_overlap": boundary["postfilter_subject_overlap"],
            "cross_split_admission_overlap": boundary[
                "postfilter_admission_overlap"
            ],
        },
        "feature_names": list(FEATURE_NAMES),
        "feature_names_sha256": feature_hash,
        "potassium_omission": (
            "preserved: released first-match item 50833 has zero observations"
        ),
        "tensor_shape": [HOURS, len(FEATURE_NAMES)],
        "tensor_dtype": "float32",
        "structured_mask": "all true at the released model boundary",
        "ordered_example_id_sha256": boundary["ordered_id_sha256"],
        "ordered_label_sha256": boundary["ordered_label_sha256"],
        "structured_boundary_sha256": boundary["boundary_sha256"],
        "train_fallback_means_sha256": array_sha256(
            boundary["global_means"].to_numpy(np.float64)
        ),
        "irregular_scaler": scaler_summary(boundary["irregular_scaler"]),
        "imputed_scaler": scaler_summary(boundary["imputed_scaler"]),
        "source_cache": boundary["cache_manifest"],
        "note_source_cache": note_cache,
        "notes": {
            "code": NOTE_CODE,
            "candidate_notes": note_cache["note_rows"],
            "candidate_stays_with_notes": note_cache["stays_with_notes"],
            "retained_notes": len(notes),
            "retained_stays_with_notes": int(notes["stay_id"].nunique()),
            "retained_stays_without_notes": len(records)
            - int(notes["stay_id"].nunique()),
            "notes_removed_by_structured_loader_filter": note_cache["note_rows"]
            - len(notes),
            "raw_text_sha256": text_sha256(notes["text"]),
            "note_id_sha256": hashlib.sha256(
                "\n".join(notes["note_id"].astype(str)).encode()
            ).hexdigest(),
            "stored_at_or_after_cutoff_rows": int(retained_unavailable.sum()),
            "stored_at_or_after_cutoff_stays": int(
                notes.loc[retained_unavailable, "stay_id"].nunique()
            ),
            "splits": note_summary(notes, records),
        },
        "batch_recreation": {
            "author_imputation": "preprocess_imputed_time_series.py::impute_stays",
            "author_scaling": selected["author_scaling"],
            "author_dataset": "train_mimiciv_multimodal.py::MultimodalMIMICIVDataset",
            "author_collate": "train_mimiciv_multimodal.py::collate_multimodal",
            "actual_model_modality_order": ["cxr", "labs_vitals", "notes"],
            "retained_author_branches": ["labs_vitals", "notes"],
            "retained_branch_indices": {"labs_vitals": 1, "notes": 2},
            "raw_note_mapper": (
                "mimiciv/data_preprocess/preprocess_notes.py::"
                "add_time_delta_notes_vectorized"
            ),
            "released_batch_sizes": {
                "train": 512,
                "validation": 512,
                "test": 512,
            },
            "mask_dtype": "torch.bool",
            "mask_value": True,
        },
    }
    if args.task == "los":
        source["released_label"] = {
            "constructor": (
                "mimiciv/data_preprocess/create_los_task.py::get_stay_list"
            ),
            "expression": (
                "label = 1 if ((outtime - intime).total_seconds()/3600 < 96) "
                "and (ADMISSIONS.hospital_expire_flag == 0) else 0"
            ),
            "horizon_hours": LOS_HOURS,
            "prediction_time": (
                "ICU intime + 48 hours; the released observation window is the "
                "half-open [0,48) ICU-hour interval and create_los_task.py "
                "hard-codes it (no --restrict_hours option)"
            ),
            "eligible_cohort": (
                "ICUSTAYS.los >= 2 days, so every retained stay is still in the "
                "ICU at the prediction time and the endpoint is strictly future"
            ),
            "paper_prose": (
                "Appendix B.1 states the LOS endpoint as 'leave the ICU alive "
                "in less than 96 hours'; the released code additionally "
                "requires eventual hospital survival. Preserved as released "
                "and disclosed, not repaired (paper_audits/merge.md)."
            ),
        }
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "merge",
            "title": (
                "Massively Multimodal Foundation Models: A Framework for "
                "Capturing Interactions with Specialized Mixture-of-Experts"
            ),
        },
        task={
            "id": selected["task_id"],
            "name": selected["name"],
            "primary_metric": "AUROC",
            "published_auroc": selected["published_auroc"],
            "observation_window_hours": float(HOURS),
        },
        codes=FEATURE_NAMES + (NOTE_CODE,),
        note_codes=(NOTE_CODE,),
        events=event_batches(records, notes),
        samples=samples_frame(records),
        source=source,
        dataset_name=selected["dataset_name"],
        dataset_version="MIMIC-IV 3.1 / MIMIC-IV-Note 2.2 / MERGE released HEAD",
    )
    print(
        json.dumps(
            {
                "task": str(args.output_dir),
                "samples": len(records),
                "events": manifest["event_count"],
                "splits": manifest["splits"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
