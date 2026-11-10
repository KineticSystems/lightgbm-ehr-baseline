"""Check FuseMoE's selected paper cohort independently of batch recreation."""

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

from export_source import COHORT_ID_SHA256, load_source, sha256  # noqa: E402


EXPECTED_SPLITS = {
    "train": {"rows": 6139, "subjects": 5274, "positive": 974},
    "validation": {"rows": 1315, "subjects": 1268, "positive": 226},
    "test": {"rows": 1316, "subjects": 1258, "positive": 224},
}
EXPECTED_SUBJECT_OVERLAP = {
    "train_validation": 323,
    "train_test": 319,
    "validation_test": 86,
}
EXPECTED_ADMISSION_OVERLAP = {
    "train_validation": 140,
    "train_test": 119,
    "validation_test": 26,
}


def overlap(rows, key):
    members = {
        split: {int(row[key]) for row in rows if row["split"] == split}
        for split in EXPECTED_SPLITS
    }
    return {
        f"{left}_{right}": len(members[left] & members[right])
        for index, left in enumerate(members)
        for right in tuple(members)[index + 1 :]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.task_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = manifest["source"]
    if source.get("selected_paper_cell") != (
        "Table 4, 48-IHM, vital signs + clinical notes, Laplace gating"
    ):
        raise AssertionError("FuseMoE selected the wrong paper experiment")
    if source.get("selected_model_inputs") != ["structured", "notes"]:
        raise AssertionError("FuseMoE selected the wrong model inputs")
    if source.get("selection_tier") != 2:
        raise AssertionError("FuseMoE selected the wrong modality tier")

    split_data, inventory = load_source(args.source_root)
    index_path = args.task_dir / manifest["meds"]["task_index"]["file"]
    index = pq.read_table(index_path)
    index_rows = index.to_pylist()
    by_source_position = {
        (row["source_split"], int(row["source_position"])): row
        for row in index_rows
    }
    if len(by_source_position) != len(index_rows):
        raise AssertionError("MEDS source positions are not unique")

    expected_rows = []
    for split, stays in split_data:
        for position, stay in enumerate(stays):
            record = by_source_position.get((split, position))
            if record is None:
                raise AssertionError("MEDS omitted a selected FuseMoE stay")
            expected = {
                "sample_id": int(stay["stay_id"]),
                "subject_id": int(stay["name"]),
                "hadm_id": int(stay["hadm_id"]),
                "boolean_value": bool(stay["label"]),
            }
            if any(record[name] != value for name, value in expected.items()):
                raise AssertionError("MEDS changed a FuseMoE cohort row or label")
            if record["split"] != split or int(record["text_missing"]) != 0:
                raise AssertionError("MEDS changed a FuseMoE split or note eligibility")
            expected_rows.append(
                {
                    "split": split,
                    "sample_id": expected["sample_id"],
                    "subject_id": expected["subject_id"],
                    "hadm_id": expected["hadm_id"],
                    "label": int(expected["boolean_value"]),
                }
            )
    if len(expected_rows) != len(index_rows):
        raise AssertionError("MEDS contains an extra FuseMoE cohort row")

    stay_ids = np.sort(
        np.asarray([row["sample_id"] for row in expected_rows], dtype=np.int64)
    )
    cohort_hash = hashlib.sha256(stay_ids.tobytes()).hexdigest()
    if len(stay_ids) != 8770 or cohort_hash != COHORT_ID_SHA256:
        raise AssertionError("FuseMoE does not match the paper's 8,770-stay cohort")

    split_summary = {}
    for split, expected in EXPECTED_SPLITS.items():
        selected = [row for row in expected_rows if row["split"] == split]
        actual = {
            "rows": len(selected),
            "subjects": len({row["subject_id"] for row in selected}),
            "positive": sum(row["label"] for row in selected),
        }
        if actual != expected:
            raise AssertionError(f"FuseMoE {split} summary differs: {actual}")
        split_summary[split] = {
            **actual,
            "prevalence": actual["positive"] / actual["rows"],
        }

    subject_overlap = overlap(expected_rows, "subject_id")
    admission_overlap = overlap(expected_rows, "hadm_id")
    if subject_overlap != EXPECTED_SUBJECT_OVERLAP:
        raise AssertionError("FuseMoE subject overlap differs")
    if admission_overlap != EXPECTED_ADMISSION_OVERLAP:
        raise AssertionError("FuseMoE admission overlap differs")
    if inventory != source["source_artifacts"]:
        raise AssertionError("FuseMoE source artifact inventory differs")

    report = {
        "status": "exact",
        "paper_evidence": {
            "location": "Table 4 initial-analysis population",
            "reported_stays": 8770,
            "selected_cell": source["selected_paper_cell"],
        },
        "cohort_rule": "text_missing == 0 and cxr_missing == 0; ECG ignored",
        "cohort_rows": len(expected_rows),
        "sorted_stay_id_sha256": cohort_hash,
        "splits": split_summary,
        "cross_split_subject_overlap": subject_overlap,
        "cross_split_admission_overlap": admission_overlap,
        "source_artifacts": inventory,
        "checks": [
            "exact stay identifiers",
            "exact subject and admission identifiers",
            "exact labels and split membership",
            "paper-reported cohort size",
            "artifact-derived counts and prevalence",
            "unused ECG availability does not restrict the cohort",
        ],
    }
    report_path = args.task_dir / "audit" / "cohort_reconstruction.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
