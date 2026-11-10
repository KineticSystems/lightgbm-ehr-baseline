"""Assert every attributable cohort fact for the selected full-label IHM task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from prepare_cohort_source import canonical_stays, official_checksums
from cohort_source import (
    EXPECTED_SPLITS,
    FROZEN_AUDIT_REPORTED_ORDERED_LABEL_SHA256,
    FROZEN_AUDIT_REPORTED_ORDERED_NAME_SHA256,
    LISTFILE_SHA256,
    ORDERED_LABEL_SHA256,
    ORDERED_NAME_SHA256,
    PRODUCER_COMMIT,
    STUDY_COMMIT,
    TORCHMIMIC_COMMIT,
    hash_lines,
    import_producer,
    listfile_paths,
    load_list_rows,
    load_source_rows,
    sha256,
    verify_pinned_sources,
    write_json,
)


def membership(path: Path, split_value: int) -> set[int]:
    frame = pd.read_csv(path, header=None, names=["subject_id", "split"])
    return set(frame.loc[frame.split.eq(split_value), "subject_id"].astype(int))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--list-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.task_dir / "manifest.json"
    manifest_hash_before = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    task = manifest["task"]
    source = manifest["source"]
    expected_task = {
        "id": "mortality_48h",
        "published_auroc": 0.856,
        "observation_window_hours": 48.0,
        "positive_class": "death during the hospitalization",
    }
    if any(task.get(key) != value for key, value in expected_task.items()):
        raise AssertionError("manifest differs from the prespecified Table 1 IHM cell")
    if task.get("modalities") != ["structured"] or task.get("note_codes") != []:
        raise AssertionError("selected downstream experiment is structured-only")
    expected_source = {
        "study_commit": STUDY_COMMIT,
        "producer_commit": PRODUCER_COMMIT,
        "torchmimic_commit": TORCHMIMIC_COMMIT,
        "mimic_release": "MIMIC-III 1.4",
        "source_protocol": "single released split",
        "selected_experiment": "Table 1 Ours, IHM, 100% labels",
        "selected_model_inputs": ["structured"],
        "common_profile": "fixed_numeric",
        "selection_tier": 1,
        "retained_projection": ["structured"],
        "unused_modality_restriction": False,
    }
    if any(source.get(key) != value for key, value in expected_source.items()):
        raise AssertionError("manifest selected the wrong experiment/cohort projection")

    source_rows = load_source_rows(args.source_root.resolve())
    public_rows = load_list_rows(args.list_root.resolve())
    if len(source_rows) != len(public_rows) or len(source_rows) != 21_139:
        raise AssertionError("full released IHM cohort must contain 21,139 stays")
    public_by_key = {
        (str(row["split"]), int(row["source_position"])): row for row in public_rows
    }
    index_path = args.task_dir / manifest["meds"]["task_index"]["file"]
    index_rows = pq.read_table(index_path).to_pylist()
    if len(index_rows) != len(source_rows):
        raise AssertionError("MEDS task index changed the released cohort size")
    index_by_key = {
        (str(row["split"]), int(row["source_position"])): row for row in index_rows
    }
    if len(index_by_key) != len(index_rows):
        raise AssertionError("MEDS source positions are not unique within split")
    for key, public in public_by_key.items():
        actual = index_by_key.get(key)
        if actual is None:
            raise AssertionError("MEDS omitted an ordered author list row")
        expected = {
            "subject_id": int(public["subject_id"]),
            "sample_id": int(public["stay_id"]),
            "boolean_value": bool(public["label"]),
            "source_file": str(public["source_file"]),
            "source_outtime": public["outtime"].to_pydatetime(),
        }
        for field, value in expected.items():
            if actual[field] != value:
                raise AssertionError(f"MEDS changed author cohort field {field}")
    if set(index_by_key) != set(public_by_key):
        raise AssertionError("MEDS added a row absent from the author listfiles")

    split_summary = {}
    subjects_by_split = {}
    stays_by_split = {}
    for split, path in listfile_paths(args.list_root.resolve()).items():
        selected = [row for row in public_rows if row["split"] == split]
        names = [str(row["source_file"]) for row in selected]
        labels = [str(row["label"]) for row in selected]
        subjects = {int(row["subject_id"]) for row in selected}
        stays = {int(row["stay_id"]) for row in selected}
        actual = {
            "rows": len(selected),
            "subjects": len(subjects),
            "positive": sum(int(value) for value in labels),
        }
        if actual != EXPECTED_SPLITS[split]:
            raise AssertionError(f"{split} list count/label marginal changed: {actual}")
        if sha256(path) != LISTFILE_SHA256[split]:
            raise AssertionError(f"{split} public artifact hash changed")
        if hash_lines(names) != ORDERED_NAME_SHA256[split]:
            raise AssertionError(f"{split} ordered filename hash changed")
        if hash_lines(labels) != ORDERED_LABEL_SHA256[split]:
            raise AssertionError(f"{split} ordered label hash changed")
        split_summary[split] = {
            **actual,
            "prevalence": actual["positive"] / actual["rows"],
            "listfile_sha256": sha256(path),
            "ordered_filename_sha256": hash_lines(names),
            "ordered_label_sha256": hash_lines(labels),
            "ordered_stay_id_sha256": hashlib.sha256(
                "\n".join(str(row["stay_id"]) for row in selected).encode()
            ).hexdigest(),
        }
        subjects_by_split[split] = subjects
        stays_by_split[split] = stays
    overlaps = {}
    for position, left in enumerate(("train", "validation", "test")):
        for right in ("train", "validation", "test")[position + 1 :]:
            overlaps[f"{left}_{right}"] = {
                "subjects": len(subjects_by_split[left] & subjects_by_split[right]),
                "stays": len(stays_by_split[left] & stays_by_split[right]),
            }
    if any(value for pair in overlaps.values() for value in pair.values()):
        raise AssertionError(f"released author splits unexpectedly overlap: {overlaps}")
    if len(set().union(*stays_by_split.values())) != 21_139:
        raise AssertionError("released ICU-stay identifiers are not globally unique")
    if len(set().union(*subjects_by_split.values())) != 18_094:
        raise AssertionError("released cohort must contain 18,094 unique subjects")

    producer_root = args.producer_root.resolve()
    producer_hashes = verify_pinned_sources(producer_root)
    mimic3csv, _, _ = import_producer(producer_root)
    stays, construction_stages = canonical_stays(args.mimic_root.resolve(), mimic3csv)
    stays = stays.set_index("ICUSTAY_ID", verify_integrity=True)
    for row in public_rows:
        stay = stays.loc[int(row["stay_id"])]
        if (
            int(stay.SUBJECT_ID) != int(row["subject_id"])
            or int(stay.MORTALITY) != int(row["label"])
            or pd.Timestamp(stay.INTIME) != pd.Timestamp(row["intime"])
            or pd.Timestamp(stay.OUTTIME) != pd.Timestamp(row["outtime"])
            or float(stay.LOS) * 24.0 < 48.0 - 1e-6
        ):
            raise AssertionError("raw producer reconstruction disagrees with author row")

    producer_test = membership(
        producer_root / "mimic3benchmark" / "resources" / "testset.csv", 1
    )
    validation = membership(
        producer_root / "mimic3models" / "resources" / "valset.csv", 1
    )
    producer_train = membership(
        producer_root / "mimic3models" / "resources" / "valset.csv", 0
    )
    if not subjects_by_split["train"].issubset(producer_train):
        raise AssertionError("author train subjects violate released validation membership")
    if not subjects_by_split["validation"].issubset(validation):
        raise AssertionError("author validation subjects violate released membership")
    if not subjects_by_split["test"].issubset(producer_test):
        raise AssertionError("author test subjects violate producer test membership")
    if (subjects_by_split["train"] | subjects_by_split["validation"]) & producer_test:
        raise AssertionError("producer test patients entered author train/validation")

    source_files = 0
    source_rows_count = 0
    observed_cells = 0
    for row in source_rows:
        path = Path(row["timeseries_path"])
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header[0] != "Hours" or len(header) != 18:
                raise AssertionError("source timeseries header differs from 17-variable task")
            sample_rows = 0
            for values in reader:
                hour = float(values[0])
                if not (-1e-6 < hour < 48.0 + 1e-6):
                    raise AssertionError("source timeseries contains an ineligible timestamp")
                sample_rows += 1
                observed_cells += sum(value != "" for value in values[1:])
            if sample_rows == 0:
                raise AssertionError("source cohort contains an empty first-48-hour stay")
            source_rows_count += sample_rows
            source_files += 1

    raw_manifest, raw_hashes = official_checksums(args.mimic_root.resolve())
    codes_path = args.task_dir / manifest["meds"]["codes"]["file"]
    codes = tuple(pq.read_table(codes_path)["code"].to_pylist())
    if len(codes) != 59 or len(set(codes)) != 59:
        raise AssertionError("MEDS must declare the exact 59 structured value codes")
    if any(
        token in code.lower()
        for code in codes
        for token in ("note", "image", "cxr", "radiology", "modality", "pair")
    ):
        raise AssertionError("notes/imaging/modality flags entered the common cohort")

    report = {
        "status": "exact",
        "selected_experiment": "Table 1 Ours, IHM, 100% labels",
        "selection_tier": 1,
        "selected_model_inputs": ["structured"],
        "retained_modalities": ["structured"],
        "unused_modality_restriction": False,
        "mimic_release": "MIMIC-III 1.4",
        "split_summary": split_summary,
        "cross_split_overlap": overlaps,
        "unique_stays": 21_139,
        "unique_subjects": 18_094,
        "source_timeseries": {
            "files": source_files,
            "rows": source_rows_count,
            "observed_cells": observed_cells,
        },
        "producer_construction_stages": construction_stages,
        "producer_source_sha256": producer_hashes,
        "official_raw_sha256": raw_hashes,
        "official_manifest_sha256": sha256(raw_manifest),
        "task_index_sha256": sha256(index_path),
        "frozen_audit_hash_definition_discrepancy": {
            "description": (
                "the frozen report's displayed ordered hashes do not equal SHA-256 "
                "over its stated newline-joined field-value convention; exact full-file "
                "SHA-256, parsed rows, and independently recomputed ordered hashes pass"
            ),
            "reported_filename_sha256": FROZEN_AUDIT_REPORTED_ORDERED_NAME_SHA256,
            "reported_label_sha256": FROZEN_AUDIT_REPORTED_ORDERED_LABEL_SHA256,
        },
        "cohort_id_sha256": hashlib.sha256(
            "\n".join(
                f"{row['split']}:{row['source_position']}:{row['stay_id']}"
                for row in public_rows
            ).encode()
        ).hexdigest(),
        "manifest_sha256": manifest_hash_before,
    }
    report_path = args.task_dir / "audit" / "cohort_reconstruction.json"
    write_json(report_path, report)
    if sha256(manifest_path) != manifest_hash_before:
        raise AssertionError("cohort checker modified immutable manifest.json")
    print(json.dumps(split_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
