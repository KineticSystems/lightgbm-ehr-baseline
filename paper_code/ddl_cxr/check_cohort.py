"""Verify DDL-CXR's selected prediction cohort against released evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from cohort_source import COMMITTED_COHORT_ROOT, SOURCE_SPLITS, sha256


EXPECTED = {
    "train": {"rows": 5142, "subjects": 4538, "positive": 746, "negative": 4396},
    "validate": {"rows": 861, "subjects": 757, "positive": 124, "negative": 737},
    "test": {"rows": 1483, "subjects": 1324, "positive": 219, "negative": 1264},
}
COMPARE_COLUMNS = (
    "index",
    "subject_id",
    "stay_id",
    "study_id",
    "dicom_id",
    "mortality_inhospital",
    "Gender",
    "Age",
    "cha",
)


def ordered_hash(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    text = frame.loc[:, columns].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--cxr-root", type=Path, required=True)
    args = parser.parse_args()

    task_dir = args.task_dir.resolve()
    manifest = json.loads((task_dir / "manifest.json").read_text())
    source = manifest["source"]
    if source["selected_experiment"] != "Table 1 DDL-CXR mortality AUROC":
        raise AssertionError("wrong prespecified DDL-CXR experiment")
    if source["selection_tier"] != 4:
        raise AssertionError("DDL-CXR proposed-method mortality must be tier 4")
    if source["selected_model_inputs"] != [
        "structured",
        "last_cxr_image",
        "generated_cxr_latent",
    ]:
        raise AssertionError("wrong source model-input declaration")
    if source["retained_projection"] != ["structured"]:
        raise AssertionError("wrong retained projection")
    if manifest["task"]["modalities"] != ["structured"]:
        raise AssertionError("task retained a prohibited modality")
    if manifest["task"]["note_codes"]:
        raise AssertionError("DDL-CXR does not consume notes")

    source_root = args.work_root.resolve() / "prediction_cohort"
    task_index = pq.read_table(task_dir / "audit" / "task_index.parquet").to_pandas()
    task_summaries = {}
    cohort_digests = {}
    split_subjects = {}
    split_stays = {}
    all_selected_images = 0
    for source_split in SOURCE_SPLITS:
        common_split = "validation" if source_split == "validate" else source_split
        reconstructed = pd.read_csv(source_root / f"{source_split}_pred.csv")
        committed = pd.read_csv(COMMITTED_COHORT_ROOT / f"{source_split}_pred.csv")
        missing = set(COMPARE_COLUMNS) - set(reconstructed.columns)
        if missing:
            raise AssertionError(f"reconstructed cohort lacks columns: {sorted(missing)}")
        pd.testing.assert_frame_equal(
            reconstructed.loc[:, COMPARE_COLUMNS],
            committed.loc[:, COMPARE_COLUMNS],
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
        expected = EXPECTED[source_split]
        actual = {
            "rows": len(reconstructed),
            "subjects": int(reconstructed["subject_id"].nunique()),
            "positive": int(reconstructed["mortality_inhospital"].sum()),
        }
        actual["negative"] = actual["rows"] - actual["positive"]
        if actual != expected:
            raise AssertionError(f"{source_split} paper counts differ: {actual}")

        selected = task_index.loc[task_index["split"] == common_split]
        expected_task = reconstructed.loc[:, ["subject_id", "stay_id", "mortality_inhospital"]].copy()
        expected_task.columns = ["subject_id", "sample_id", "boolean_value"]
        expected_task["boolean_value"] = expected_task["boolean_value"].astype(bool)
        actual_task = selected.loc[:, ["subject_id", "sample_id", "boolean_value"]]
        actual_task = actual_task.sort_values("sample_id").reset_index(drop=True)
        expected_task = expected_task.sort_values("sample_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(actual_task, expected_task, check_dtype=False)

        for row in reconstructed.itertuples(index=False):
            image = (
                args.cxr_root
                / "files"
                / f"p{str(int(row.subject_id))[:2]}"
                / f"p{int(row.subject_id)}"
                / f"s{int(row.study_id)}"
                / f"{row.dicom_id}.jpg"
            )
            if not image.is_file():
                raise FileNotFoundError(image)
            all_selected_images += 1

        split_subjects[common_split] = set(reconstructed["subject_id"].astype(int))
        split_stays[common_split] = set(reconstructed["stay_id"].astype(int))
        task_summaries[common_split] = actual
        cohort_digests[common_split] = {
            "ordered_selected_metadata_sha256": ordered_hash(
                reconstructed, COMPARE_COLUMNS
            ),
            "source_csv_sha256": sha256(source_root / f"{source_split}_pred.csv"),
            "committed_csv_sha256": sha256(
                COMMITTED_COHORT_ROOT / f"{source_split}_pred.csv"
            ),
        }

    overlap = {}
    common_splits = ("train", "validation", "test")
    for index, left in enumerate(common_splits):
        for right in common_splits[index + 1 :]:
            overlap[f"{left}_{right}_subjects"] = len(
                split_subjects[left] & split_subjects[right]
            )
            overlap[f"{left}_{right}_stays"] = len(
                split_stays[left] & split_stays[right]
            )
    if any(overlap.values()):
        raise AssertionError(f"released patient split unexpectedly overlaps: {overlap}")

    train = pd.read_csv(source_root / "train_pred.csv")
    paper_train_statistics = {
        "age_mean": float(train["Age"].mean()),
        "age_sample_std": float(train["Age"].std()),
        "men_percent": float(100.0 * train["Gender"].mean()),
        "mortality_percent": float(100.0 * train["mortality_inhospital"].mean()),
    }
    if round(paper_train_statistics["age_mean"], 1) != 63.9:
        raise AssertionError("Table 9 training age mean differs")
    if round(paper_train_statistics["age_sample_std"], 1) != 16.6:
        raise AssertionError("Table 9 training age standard deviation differs")
    if round(paper_train_statistics["men_percent"], 2) != 54.98:
        raise AssertionError("Table 9 training gender prevalence differs")
    if round(paper_train_statistics["mortality_percent"]) != 15:
        raise AssertionError("Table 9 training mortality prevalence differs")

    report = {
        "status": "exact",
        "selected_experiment": source["selected_experiment"],
        "paper_cells": [
            "Table 1 DDL-CXR mortality AUROC 0.822 +/- 0.009",
            "Table 6 prediction split counts",
            "Table 7 mortality label marginals",
            "Table 9 training cohort age, gender, and mortality summaries",
        ],
        "source_model_inputs": source["selected_model_inputs"],
        "retained_projection": ["structured"],
        "unused_imaging_cohort_restriction_preserved": True,
        "selected_image_files_verified": all_selected_images,
        "splits": task_summaries,
        "ordered_cohort_digests": cohort_digests,
        "split_overlap": overlap,
        "paper_train_statistics": paper_train_statistics,
        "checks": [
            "exact reconstructed-versus-committed row order and cohort metadata",
            "exact Table 6 sample counts and Table 7 label marginals",
            "Table 9 rounded training-cohort statistics",
            "exact MEDS identifiers, labels, and released split membership",
            "zero released subject and ICU-stay overlap",
            "every selected CXR identifier resolves in the verified 2.0.0 payload",
            "structured-only retained projection with no note or imaging events",
        ],
    }
    output = task_dir / "audit" / "cohort_reconstruction.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
