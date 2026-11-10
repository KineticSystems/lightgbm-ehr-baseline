"""Check OC-Distill cohort counts, labels, and membership independently."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


REPOSITORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY))

from export_source import author_split, load_cohort, sha256  # noqa: E402


EXPECTED_SOURCE_ROWS = {"train": 16861, "test": 3055}
EXPECTED_SPLITS = {
    "train": {"rows": 13522, "subjects": 11664, "positive": 1808},
    "validation": {"rows": 3339, "subjects": 2916, "positive": 426},
    "test": {"rows": 3055, "subjects": 2631, "positive": 357},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--auxiliary-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.task_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["task"].get("modalities") != ["structured"]:
        raise AssertionError("OC-Distill must remain structured-only")
    if manifest["task"].get("note_codes") != []:
        raise AssertionError("OC-Distill MEDS must not contain notes")

    rows, source, retained_frames, _ = load_cohort(
        args.cohort_root,
        args.mimic_root,
        args.auxiliary_root,
        args.cache_root,
        reuse_cache=True,
    )
    author_split(rows)
    actual_source_rows = {
        split: sum(row["source_split"] == split for row in rows)
        for split in EXPECTED_SOURCE_ROWS
    }
    if actual_source_rows != EXPECTED_SOURCE_ROWS:
        raise AssertionError(
            f"OC-Distill paper cohort counts differ: {actual_source_rows}"
        )
    if {
        split: len(retained_frames[split]) for split in EXPECTED_SOURCE_ROWS
    } != EXPECTED_SOURCE_ROWS:
        raise AssertionError("OC-Distill source loader counts differ")

    index_path = args.task_dir / manifest["meds"]["task_index"]["file"]
    index = pq.read_table(index_path)
    index_by_stay = {
        int(row["sample_id"]): row for row in index.to_pylist()
    }
    if len(index_by_stay) != len(index):
        raise AssertionError("OC-Distill MEDS stay identifiers are not unique")

    for row in rows:
        record = index_by_stay.get(int(row["stay_id"]))
        if record is None:
            raise AssertionError("MEDS omitted an OC-Distill stay")
        expected = {
            "subject_id": int(row["subject_id"]),
            "hadm_id": int(row["hadm_id"]),
            "boolean_value": bool(row["label"]),
            "split": row["split"],
            "source_split": row["source_split"],
            "source_position": int(row["source_position"]),
            "retained_position": int(row["retained_position"]),
        }
        if any(record[name] != value for name, value in expected.items()):
            raise AssertionError("MEDS changed an OC-Distill cohort row or label")
    if len(rows) != len(index_by_stay):
        raise AssertionError("MEDS contains an extra OC-Distill stay")

    split_summary = {}
    subjects_by_split = {}
    for split, expected in EXPECTED_SPLITS.items():
        selected = [row for row in rows if row["split"] == split]
        subjects = {int(row["subject_id"]) for row in selected}
        actual = {
            "rows": len(selected),
            "subjects": len(subjects),
            "positive": sum(int(row["label"]) for row in selected),
        }
        if actual != expected:
            raise AssertionError(f"OC-Distill {split} summary differs: {actual}")
        split_summary[split] = {
            **actual,
            "prevalence": actual["positive"] / actual["rows"],
        }
        subjects_by_split[split] = subjects
    overlaps = {
        "train_validation": len(
            subjects_by_split["train"] & subjects_by_split["validation"]
        ),
        "train_test": len(subjects_by_split["train"] & subjects_by_split["test"]),
        "validation_test": len(
            subjects_by_split["validation"] & subjects_by_split["test"]
        ),
    }
    if any(overlaps.values()):
        raise AssertionError(f"OC-Distill split subjects overlap: {overlaps}")

    source_inventory = {
        f"mortality_{split}_48h.npz": {
            "sha256": source[split]["sha256"],
            "rows": len(source[split]["y"]),
        }
        for split in EXPECTED_SOURCE_ROWS
    }
    if source_inventory != manifest["source"]["reference_cohort_artifacts"]:
        raise AssertionError("OC-Distill source cohort artifact inventory differs")
    stay_ids = np.sort(
        np.asarray([int(row["stay_id"]) for row in rows], dtype=np.int64)
    )
    stay_hash = hashlib.sha256(stay_ids.tobytes()).hexdigest()

    report = {
        "status": "exact",
        "paper_evidence": {
            "location": "main-text 48-hour MIMIC-III experiment",
            "reported_train_side_stays": 16861,
            "reported_test_stays": 3055,
        },
        "source_cohort_rows": actual_source_rows,
        "cohort_rows": len(rows),
        "sorted_stay_id_sha256": stay_hash,
        "splits": split_summary,
        "cross_split_subject_overlap": overlaps,
        "source_artifacts": source_inventory,
        "checks": [
            "paper-reported train-side and test cohort sizes",
            "exact stay, admission, and subject identifiers",
            "exact labels and released seed-0 split membership",
            "artifact-derived counts and prevalence",
            "patient-disjoint train, validation, and test membership",
        ],
    }
    report_path = args.task_dir / "audit" / "cohort_reconstruction.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
