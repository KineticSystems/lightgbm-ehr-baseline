"""Independently assert GARLIC's paper, MIMIC, and attributable cohort facts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPOSITORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY))

from cohort_source import (  # noqa: E402
    EXPECTED_COHORT_ROWS,
    EXPECTED_FEATURES,
    EXPECTED_POSITIVE,
    PUBLISHED_COHORT_ROWS,
    PUBLISHED_POSITIVE,
    cohort_rows,
    open_complete_cache,
    ordered_split_hash,
    selected_id_hash,
    sha256_file,
)


WARP_ROOT = (
    REPOSITORY
    / ".export_assets"
    / "external_artifacts"
    / "warpformer"
    / "377d64db2b146c7acf4197c970a236dca3ba0663"
)
WARP_SPLITS = {
    "train": ("train_adm_idx.p", 47_180),
    "validation": ("val_adm_idx.p", 5_897),
    "test": ("test_adm_idx.p", 5_899),
}
WARP_HASHES = {
    "train": "e5ae9623f9b43ad6a70427d19eef458906d62de359607c3615dc4d686ea30e18",
    "validation": "04f9c88030750db6a44f5f3c40517c46b566925c5318ad3547a6d6ab1a233fdd",
    "test": "f7af8b8a5f9adce34d89862bc81e31a2bce890d052a1b54b183f552b34ac4a81",
}


def safe_pickle_integer_list(path: Path) -> list[int]:
    allowed = {
        "PROTO",
        "EMPTY_LIST",
        "BINPUT",
        "LONG_BINPUT",
        "MEMOIZE",
        "MARK",
        "BININT",
        "BININT1",
        "BININT2",
        "LONG1",
        "APPENDS",
        "APPEND",
        "STOP",
    }
    values: list[int] = []
    with path.open("rb") as handle:
        for opcode, argument, _ in pickletools.genops(handle):
            if opcode.name not in allowed:
                raise AssertionError(f"unexpected opcode in attributable split: {opcode.name}")
            if opcode.name in {"BININT", "BININT1", "BININT2", "LONG1"}:
                values.append(int(argument))
    return values


def int64_hash(values: list[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.task_dir / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["paper"]["key"] != "garlic":
        raise AssertionError("cohort checker opened the wrong task")
    if manifest["task"]["name"] != "MIMIC-III in-hospital mortality":
        raise AssertionError("GARLIC selected endpoint changed")
    if manifest["task"]["modalities"] != ["structured"]:
        raise AssertionError("GARLIC selected experiment is structured-only")
    if manifest["task"]["note_codes"] != []:
        raise AssertionError("GARLIC cannot expose notes")
    if manifest["source"]["selection_tier"] != 1:
        raise AssertionError("GARLIC must use the available tier-1 experiment")
    if manifest["source"]["omitted_released_fields"] != ["static", "extended_static"]:
        raise AssertionError("GARLIC static audit-only fields changed")
    if manifest["source"]["omitted_modalities"] != []:
        raise AssertionError("GARLIC has no unselected cohort-defining modality")
    expected_count_exception = {
        "published_count": PUBLISHED_COHORT_ROWS,
        "reconstructed_count": EXPECTED_COHORT_ROWS,
        "difference": EXPECTED_COHORT_ROWS - PUBLISHED_COHORT_ROWS,
        "paper_count_match": False,
    }
    if any(
        manifest["source"]["cohort_selection"].get(key) != value
        for key, value in expected_count_exception.items()
    ):
        raise AssertionError("manifest paper/source count exception changed")

    connection, summary = open_complete_cache(args.cache_path)
    try:
        rows = cohort_rows(connection)
        if any(
            summary.get(key) != value
            for key, value in {
                "published_cohort_rows": PUBLISHED_COHORT_ROWS,
                "cohort_rows": EXPECTED_COHORT_ROWS,
                "paper_count_difference": EXPECTED_COHORT_ROWS
                - PUBLISHED_COHORT_ROWS,
                "paper_count_match": False,
                "published_positive": PUBLISHED_POSITIVE,
                "positive": EXPECTED_POSITIVE,
                "paper_positive_difference": EXPECTED_POSITIVE
                - PUBLISHED_POSITIVE,
                "paper_positive_match": False,
            }.items()
        ):
            raise AssertionError("source cache paper-count exception changed")
        if len(rows) != EXPECTED_COHORT_ROWS:
            raise AssertionError("released GARLIC 49,375-example cohort was not rebuilt")
        if sum(row["label"] for row in rows) != EXPECTED_POSITIVE:
            raise AssertionError("released GARLIC 5,742-death cohort was not rebuilt")
        if len(summary["features"]) != EXPECTED_FEATURES:
            raise AssertionError("paper-reported 103 structured signals were not rebuilt")
        if selected_id_hash(connection) != manifest["source"]["cohort_selection"][
            "sorted_hadm_id_sha256"
        ]:
            raise AssertionError("manifest cohort identifier hash differs")
        if ordered_split_hash(connection) != manifest["source"][
            "ordered_identifier_label_split_sha256"
        ]:
            raise AssertionError("manifest ordered membership hash differs")

        index = pq.read_table(args.task_dir / manifest["meds"]["task_index"]["file"])
        indexed = {int(row["sample_id"]): row for row in index.to_pylist()}
        if len(indexed) != len(rows):
            raise AssertionError("MEDS HADM_ID identifiers are not one-to-one")
        for row in rows:
            actual = indexed.get(row["hadm_id"])
            if actual is None:
                raise AssertionError("MEDS omitted a GARLIC admission")
            expected = {
                "subject_id": row["subject_id"],
                "hadm_id": row["hadm_id"],
                "boolean_value": bool(row["label"]),
                "split": row["split"],
                "source_position": row["source_position"],
            }
            if any(actual[name] != value for name, value in expected.items()):
                raise AssertionError("MEDS changed a GARLIC cohort row, label, or split")

        admissions = pd.read_csv(
            args.mimic_root / "ADMISSIONS.csv.gz",
            usecols=["SUBJECT_ID", "HADM_ID", "HOSPITAL_EXPIRE_FLAG"],
        ).drop_duplicates()
        raw = {
            int(row.HADM_ID): (int(row.SUBJECT_ID), int(row.HOSPITAL_EXPIRE_FLAG))
            for row in admissions.itertuples(index=False)
        }
        for row in rows:
            if raw.get(row["hadm_id"]) != (row["subject_id"], row["label"]):
                raise AssertionError("GARLIC HADM_ID label/subject mapping differs from ADMISSIONS")

        split_summary = {}
        subjects = {}
        admissions_by_split = {}
        for split in ("train", "validation", "test"):
            selected = [row for row in rows if row["split"] == split]
            subject_set = {row["subject_id"] for row in selected}
            hadm_set = {row["hadm_id"] for row in selected}
            actual = {
                "count": len(selected),
                "positive": sum(row["label"] for row in selected),
                "subjects": len(subject_set),
                "prevalence": sum(row["label"] for row in selected) / len(selected),
                "ordered_hadm_label_sha256": hashlib.sha256(
                    b"".join(
                        np.asarray([row["hadm_id"], row["label"]], dtype=np.int64).tobytes()
                        for row in selected
                    )
                ).hexdigest(),
            }
            cached = summary["splits"][split]
            if {key: actual[key] for key in ("count", "positive", "subjects")} != cached:
                raise AssertionError(f"GARLIC {split} source summary changed")
            if manifest["splits"][split] != cached:
                raise AssertionError(f"GARLIC {split} manifest summary changed")
            split_summary[split] = actual
            subjects[split] = subject_set
            admissions_by_split[split] = hadm_set
        subject_overlap = {
            "train_validation": len(subjects["train"] & subjects["validation"]),
            "train_test": len(subjects["train"] & subjects["test"]),
            "validation_test": len(subjects["validation"] & subjects["test"]),
        }
        if subject_overlap != summary["cross_split_subject_overlap"]:
            raise AssertionError("released patient overlap audit changed")
        admission_overlap = {
            "train_validation": len(admissions_by_split["train"] & admissions_by_split["validation"]),
            "train_test": len(admissions_by_split["train"] & admissions_by_split["test"]),
            "validation_test": len(admissions_by_split["validation"] & admissions_by_split["test"]),
        }
        if any(admission_overlap.values()):
            raise AssertionError("one GARLIC admission entered multiple splits")

        warp_sets: dict[str, set[int]] = {}
        warp_summary = {}
        for split, (filename, expected_count) in WARP_SPLITS.items():
            path = WARP_ROOT / "preprocess" / "adm_id" / filename
            if sha256_file(path) != WARP_HASHES[split]:
                raise AssertionError("attributable Warpformer split artifact changed")
            values = safe_pickle_integer_list(path)
            if len(values) != expected_count or len(set(values)) != expected_count:
                raise AssertionError("attributable Warpformer split count/uniqueness changed")
            warp_sets[split] = set(values)
            warp_summary[split] = {
                "count": len(values),
                "ordered_id_sha256": int64_hash(values),
            }
        if (
            warp_sets["train"] & warp_sets["validation"]
            or warp_sets["train"] & warp_sets["test"]
            or warp_sets["validation"] & warp_sets["test"]
        ):
            raise AssertionError("attributable Warpformer split artifacts now overlap")
        warp_union = set().union(*warp_sets.values())
        if len(warp_union) != 58_976:
            raise AssertionError("attributable Warpformer union changed")
        garlic_ids = {row["hadm_id"] for row in rows}
        warp_diagnostic = {
            "union_count": len(warp_union),
            "garlic_ids_in_warpformer_union": len(garlic_ids & warp_union),
            "garlic_ids_outside_warpformer_union": len(garlic_ids - warp_union),
            "warpformer_ids_outside_garlic": len(warp_union - garlic_ids),
        }
        readme = (WARP_ROOT / "README.md").read_text()
        compact_readme = readme.replace(",", "").replace(" ", "")
        for count in ("39449", "4939", "4970"):
            if count not in compact_readme:
                raise AssertionError("Warpformer README post-filter mortality count disappeared")

        raw_inventory = {
            name: {
                "sha256": sha256_file(args.mimic_root / name),
                "bytes": (args.mimic_root / name).stat().st_size,
            }
            for name in summary["raw_artifacts"]
        }
        if raw_inventory != summary["raw_artifacts"]:
            raise AssertionError("MIMIC-III 1.4 source artifact inventory changed")
        if raw_inventory != manifest["source"]["raw_source_artifacts"]:
            raise AssertionError("manifest source artifact inventory changed")

        report = {
            "status": "exact",
            "manifest_sha256_before_and_after": manifest_sha256,
            "paper_evidence": {
                "dataset_table": {
                    "admissions": PUBLISHED_COHORT_ROWS,
                    "deaths": PUBLISHED_POSITIVE,
                    "signals": EXPECTED_FEATURES,
                },
                "main_result": {
                    "endpoint": "in-hospital mortality",
                    "auroc": 0.9009,
                    "model_inputs": ["structured time series"],
                },
                "paper_code_window_conflict_preserved": (
                    "paper says all ICU-stay data and LOS >=24h; selected executable "
                    "path uses first 48h from earliest retained event and no LOS filter"
                ),
            },
            "cohort_rows": len(rows),
            "published_count": PUBLISHED_COHORT_ROWS,
            "reconstructed_count": EXPECTED_COHORT_ROWS,
            "difference": EXPECTED_COHORT_ROWS - PUBLISHED_COHORT_ROWS,
            "paper_count_match": False,
            "published_positive": PUBLISHED_POSITIVE,
            "reconstructed_positive": EXPECTED_POSITIVE,
            "positive_difference": EXPECTED_POSITIVE - PUBLISHED_POSITIVE,
            "paper_positive_match": False,
            "status_meaning": (
                "exact equality to the reproducible pinned released pipeline; "
                "not agreement with the paper's 49,380 count"
            ),
            "protocol_exception": (
                "user explicitly authorized the reproducible 49,375-admission "
                "released cohort; no rows were invented or substituted"
            ),
            "positive": sum(row["label"] for row in rows),
            "prevalence": sum(row["label"] for row in rows) / len(rows),
            "sorted_hadm_id_sha256": selected_id_hash(connection),
            "ordered_identifier_label_split_sha256": ordered_split_hash(connection),
            "splits": split_summary,
            "cross_split_subject_overlap": subject_overlap,
            "cross_split_admission_overlap": admission_overlap,
            "retained_projection": {
                "structured_codes": EXPECTED_FEATURES,
                "notes": 0,
                "static_model_fields": 0,
                "imaging": 0,
            },
            "mimic_admissions_mapping": {
                "rows_checked": len(rows),
                "positive_definition": "HOSPITAL_EXPIRE_FLAG == 1",
                "status": "exact",
            },
            "attributable_warpformer_artifact": {
                "commit": "377d64db2b146c7acf4197c970a236dca3ba0663",
                "splits": warp_summary,
                "published_post_filter_counts": [39_449, 4_939, 4_970],
                "diagnostic_not_substituted": warp_diagnostic,
            },
            "checks": [
                "paper count recorded as 49,380 and released reconstruction asserted as 49,375",
                "paper/reconstruction difference asserted as -5 without row substitution",
                "paper-reported 5,747 versus released reconstruction 5,742 deaths",
                "paper-reported 103 signals",
                "exact HADM_ID, SUBJECT_ID, label, order, and released split membership",
                "exact per-split counts, deaths, prevalence, and patient/admission overlap",
                "independent raw ADMISSIONS HOSPITAL_EXPIRE_FLAG mapping",
                "exact structured-only selected experiment and retained modality projection",
                "all versioned MIMIC-III 1.4 artifact hashes",
                "all attributable Warpformer split counts, disjointness, hashes, and README counts",
            ],
        }
        report_path = args.task_dir / "audit" / "cohort_reconstruction.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if sha256_file(manifest_path) != manifest_sha256:
            raise AssertionError("cohort checker mutated manifest.json")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
