"""Verify DiPro paper/source cohort facts independently against MEDS."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from export_source import CODES  # noqa: E402
from cohort_source import TASK_ID, UPSTREAM_COMMIT  # noqa: E402


EXPECTED_SPLITS = {
    "train": {"count": 1889, "positive": 316, "subjects": 1795},
    "validation": {"count": 285, "positive": 39, "subjects": 264},
    "test": {"count": 546, "positive": 110, "subjects": 520},
}
EXPECTED_CXR_LENGTHS = {
    "train": {2: 1367, 3: 465, 4: 56, 5: 1},
    "validation": {2: 199, 3: 76, 4: 10, 5: 0},
    "test": {2: 409, 3: 120, 4: 16, 5: 1},
}
SPLIT_FILES = (
    ("train", "train.csv"),
    ("validation", "val.csv"),
    ("test", "test.csv"),
)


def hash_rows(rows: list[tuple[int, int, int]]) -> str:
    values = np.asarray(rows, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.task_dir / "manifest.json").read_text())
    if manifest.get("paper", {}).get("key") != "dipro":
        raise AssertionError("task is not DiPro")
    task = manifest.get("task", {})
    if task.get("id") != TASK_ID or task.get("published_auroc") != 0.897:
        raise AssertionError("DiPro selected the wrong binary AUROC task")
    if task.get("modalities") != ["structured"] or task.get("note_codes") != []:
        raise AssertionError("DiPro common-task projection must be structured-only")
    source = manifest.get("source", {})
    if source.get("upstream_commit") != UPSTREAM_COMMIT:
        raise AssertionError("DiPro upstream commit changed")
    if source.get("selection_tier") != 4:
        raise AssertionError("DiPro selected the wrong modality-priority tier")
    if source.get("selected_model_inputs") != [
        "structured",
        "longitudinal_regional_cxr_images",
    ]:
        raise AssertionError("DiPro selected model-input declaration changed")
    if source.get("retained_modalities") != ["structured"]:
        raise AssertionError("DiPro retained projection changed")
    if source.get("omitted_modalities") != ["imaging"]:
        raise AssertionError("DiPro omitted-imaging declaration changed")
    cohort_summary = json.loads(
        (args.cache_root / "cohort_summary.json").read_text()
    )
    if source.get("raw_source_artifacts") != cohort_summary.get("source_hashes"):
        raise AssertionError("DiPro exact-release source inventory changed")

    index_path = args.task_dir / manifest["meds"]["task_index"]["file"]
    index = pq.read_table(index_path).to_pandas()
    actual_by_id = index.set_index("sample_id", drop=False)
    if actual_by_id.index.duplicated().any():
        raise AssertionError("DiPro MEDS duplicated a stay")

    source_rows = []
    split_summaries = {}
    subject_members = {}
    for split, filename in SPLIT_FILES:
        frame = pd.read_csv(args.cache_root / "split" / "mortality_los" / filename)
        frame["source_split_row"] = np.arange(len(frame), dtype=np.int64)
        frame["cxr_count"] = frame["cxr_list"].map(lambda value: len(ast.literal_eval(value)))
        actual_summary = {
            "count": len(frame),
            "positive": int(frame["mortality_inhospital"].sum()),
            "subjects": int(frame["subject_id"].nunique()),
        }
        if actual_summary != EXPECTED_SPLITS[split]:
            raise AssertionError(f"DiPro {split} paper/source counts changed: {actual_summary}")
        distribution = {
            length: int((frame["cxr_count"] == length).sum())
            for length in (2, 3, 4, 5)
        }
        if distribution != EXPECTED_CXR_LENGTHS[split]:
            raise AssertionError(
                f"DiPro {split} published CXR-length distribution changed: {distribution}"
            )
        manifest_summary = manifest["splits"][split]
        for key in ("count", "positive", "subjects"):
            if manifest_summary[key] != actual_summary[key]:
                raise AssertionError(f"DiPro {split} manifest summary changed")
        for row in frame.itertuples(index=False):
            stay_id = int(row.stay_id)
            if stay_id not in actual_by_id.index:
                raise AssertionError("MEDS omitted a released DiPro stay")
            actual = actual_by_id.loc[stay_id]
            expected = {
                "subject_id": int(row.subject_id),
                "sample_id": stay_id,
                "boolean_value": bool(int(row.mortality_inhospital)),
                "split": split,
                "source_split_row": int(row.source_split_row),
            }
            if any(actual[name] != value for name, value in expected.items()):
                raise AssertionError("MEDS changed a released DiPro cohort row")
            source_rows.append(
                (int(row.subject_id), stay_id, int(row.mortality_inhospital))
            )
        subject_members[split] = set(frame["subject_id"].astype(int))
        split_summaries[split] = {
            **actual_summary,
            "prevalence": actual_summary["positive"] / actual_summary["count"],
            "cxr_sequence_length_counts": distribution,
            "ordered_subject_stay_label_sha256": hash_rows(
                list(
                    zip(
                        frame["subject_id"].astype(int),
                        frame["stay_id"].astype(int),
                        frame["mortality_inhospital"].astype(int),
                    )
                )
            ),
        }

    if len(index) != 2720 or len(source_rows) != 2720:
        raise AssertionError("DiPro total cohort count changed")
    if int(index["boolean_value"].sum()) != 465:
        raise AssertionError("DiPro total mortality marginal changed")
    overlaps = {
        "train_validation": len(subject_members["train"] & subject_members["validation"]),
        "train_test": len(subject_members["train"] & subject_members["test"]),
        "validation_test": len(subject_members["validation"] & subject_members["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"released subject split is no longer disjoint: {overlaps}")

    codes = pq.read_table(
        args.task_dir / manifest["meds"]["codes"]["file"]
    )["code"].to_pylist()
    if codes != list(CODES):
        raise AssertionError("DiPro MEDS vocabulary/order changed")
    forbidden = ("CXR", "IMAGE", "BBOX", "CHEXPERT", "PROGRESSION", "AUX")
    if any(any(token in code.upper() for token in forbidden) for code in codes):
        raise AssertionError("DiPro retained an imaging or auxiliary code")

    report = {
        "status": "exact",
        "paper_evidence": {
            "selected_cell": (
                "Table 2 / Table 12 longitudinal DiPro mortality, "
                "AUROC 0.897 +/- 0.002"
            ),
            "reported_split_counts": {name: item["count"] for name, item in EXPECTED_SPLITS.items()},
            "reported_total_label_counts": {"negative": 2255, "positive": 465},
            "checkpoint_derived_train_label_counts": {"negative": 1573, "positive": 316},
            "reported_cxr_sequence_length_counts": EXPECTED_CXR_LENGTHS,
            "positive_class": "mortality_inhospital == 1",
        },
        "source_configuration": {
            "model_inputs": ["structured", "longitudinal_regional_cxr_images"],
            "selection_tier": 4,
            "retained_projection": ["structured"],
            "omitted_modalities": ["imaging"],
            "imaging_retained": False,
            "cohort_restriction": (
                "same common Table-2 cohort requiring at least two eligible CXRs; "
                "the selected source model uses those CXRs, but the common task "
                "retains only its structured projection"
            ),
        },
        "cohort": {
            "count": len(index),
            "positive": int(index["boolean_value"].sum()),
            "negative": int(len(index) - index["boolean_value"].sum()),
            "splits": split_summaries,
            "cross_split_subject_overlap": overlaps,
            "ordered_source_rows_sha256": hash_rows(source_rows),
        },
        "meds_boundary": {
            "code_count": len(codes),
            "note_codes": [],
            "imaging_or_auxiliary_code_count": 0,
            "event_count": manifest["event_count"],
        },
    }
    output = args.task_dir / "audit" / "cohort_reconstruction.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
