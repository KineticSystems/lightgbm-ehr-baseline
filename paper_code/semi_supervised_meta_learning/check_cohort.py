"""Cohort, endpoint, split, and provenance checks for the SSML export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from cohort_source import (
    COMMON_SPLIT,
    SPLIT_MAP,
    assert_source_ready,
    build_samples,
    listfile_path,
    load_transform_spec,
    sha256_file,
)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def assert_frame_equal(expected: pl.DataFrame, actual: pl.DataFrame) -> None:
    columns = (
        "subject_id", "sample_id", "observation_start", "prediction_time",
        "boolean_value", "split", "source_filename", "source_list_row",
        "source_cutoff_hour", "source_order", "episode_number",
    )
    expected = expected.sort("source_order").select(columns)
    actual = actual.sort("source_order").select(columns)
    if expected.shape != actual.shape:
        raise AssertionError(f"MEDS/source cohort shape differs: {actual.shape} != {expected.shape}")
    for column in columns:
        if not np.array_equal(expected[column].to_numpy(), actual[column].to_numpy()):
            raise AssertionError(f"MEDS/source cohort differs in {column}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = assert_source_ready(args.source_root)
    expected, expected_audit = build_samples(args.source_root)
    spec = load_transform_spec()
    manifest = json.loads((args.task_dir / "manifest.json").read_text())
    index = pl.read_parquet(args.task_dir / manifest["meds"]["task_index"]["file"])
    labels = pl.read_parquet(args.task_dir / manifest["meds"]["labels"]["file"])
    codes = pl.read_parquet(args.task_dir / manifest["meds"]["codes"]["file"])
    assert_frame_equal(expected, index)
    expected_labels = index.select(
        "subject_id", "sample_id", "prediction_time", "boolean_value"
    ).sort("sample_id")
    if not expected_labels.equals(labels.sort("sample_id")):
        raise AssertionError("MEDS labels differ from the exact deterministic cohort")
    if tuple(codes["code"].to_list()) != spec["codes"]:
        raise AssertionError("MEDS vocabulary differs from exact released [T,76] order")
    task = manifest["task"]
    if task["modalities"] != ["structured"] or task["note_codes"] != []:
        raise AssertionError("structured-only selected task retained another modality")
    if task["published_auroc"] != 0.906 or not task["source_task"].startswith("Table 4"):
        raise AssertionError("manifest does not identify the selected paper cell")
    if index.filter(
        (pl.col("source_cutoff_hour") < 24) | (pl.col("source_cutoff_hour") >= 2000)
    ).height:
        raise AssertionError("released read_chunk cutoff filter was not preserved")

    preparation = json.loads(paths["preparation"].read_text())
    train_validation_inventory = {
        key: preparation["listfiles"]["train"][key]
        + preparation["listfiles"]["val"][key]
        for key in ("rows", "positive", "negative")
    }
    if train_validation_inventory != {
        "rows": 2_908_414,
        "positive": 61_013,
        "negative": 2_847_401,
    }:
        raise AssertionError("published M3B train+validation inventory changed")
    test_inventory = {
        key: preparation["listfiles"]["test"][key]
        for key in ("rows", "positive", "negative")
    }
    if test_inventory != {
        "rows": 523_208,
        "positive": 9_683,
        "negative": 513_525,
    }:
        raise AssertionError("published M3B test inventory changed")
    summaries = {}
    for split in ("train", "validation", "test"):
        selected = index.filter(pl.col("split") == split)
        positive = int(selected["boolean_value"].sum())
        summaries[split] = {
            "landmarks": selected.height,
            "positive": positive,
            "negative": selected.height - positive,
            "subjects": selected["subject_id"].n_unique(),
            "stays": selected["source_filename"].n_unique(),
        }
        expected_split = expected_audit["splits"][split]
        for key in ("landmarks", "positive", "negative", "subjects", "stays"):
            if summaries[split][key] != expected_split[key]:
                raise AssertionError(f"{split} {key} differs from source reconstruction")
        source_split = "val" if split == "validation" else split
        prep = preparation["listfiles"][source_split]
        if selected.height != prep["eligible_24_to_2000_rows"] or positive != prep["eligible_positive"]:
            raise AssertionError(f"{split} eligible inventory differs from source preparation")
        if manifest["splits"][split] != {
            "count": selected.height,
            "subjects": selected["subject_id"].n_unique(),
            "positive": positive,
        }:
            raise AssertionError(f"manifest {split} summary differs from cohort")

    memberships = {
        split: set(index.filter(pl.col("split") == split)["subject_id"].unique().to_list())
        for split in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": len(memberships["train"] & memberships["validation"]),
        "train_test": len(memberships["train"] & memberships["test"]),
        "validation_test": len(memberships["validation"] & memberships["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"fixed M3B subject splits overlap: {overlaps}")
    parquet_rows = pq.ParquetFile(
        args.task_dir / manifest["meds"]["events"]["file"]
    ).metadata.num_rows
    if parquet_rows != manifest["event_count"]:
        raise AssertionError("event Parquet row count differs from manifest")
    if manifest["source"]["event_count_precomputed"] != parquet_rows:
        raise AssertionError("event pre-count differs from Parquet")
    if manifest["source"]["paper_run_evaluation"] != {
        "exact_membership_recoverable": False,
        "heldout_test_exercised": False,
        "reason": "the released entrypoint freshly and without a seed shuffles 800 source rows for both train and validation, then skips out-of-range cutoffs without replacement",
    }:
        raise AssertionError("unrecoverable paper-run membership defect was not preserved")

    listfile_hashes = {
        COMMON_SPLIT[split]: sha256_file(listfile_path(paths, split)) for split in SPLIT_MAP
    }
    if listfile_hashes != manifest["source"]["listfile_sha256"]:
        raise AssertionError("manifest listfile hashes changed")
    report = {
        "status": "exact",
        "paper": "semi_supervised_meta_learning",
        "task": "decompensation",
        "canonical_export": {
            "kind": "complete deterministic attributable benchmark after exact released cutoff filter",
            "splits": summaries,
            "subject_overlap": overlaps,
            "ordered_eligible_tuple_sha256": expected_audit["ordered_eligible_tuple_sha256"],
            "listfile_sha256": listfile_hashes,
            "event_count": parquet_rows,
        },
        "author_boundary": {
            "shape": "[T,76]",
            "value_channels": 59,
            "mask_channels": 17,
            "imputation": "previous, initialized with released normal values",
            "normalization": "released decompensation normalizer",
        },
        "paper_run_evaluation_defects": {
            "exact_800_row_membership_recoverable": False,
            "skipped_rows_replaced": False,
            "heldout_test_exercised": False,
        },
        "selected_result": {"table": 4, "method": "SSML-TDV", "auroc": 0.906},
        "modalities": ["structured"],
    }
    write_json(args.task_dir / "audit" / "cohort_reconstruction.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
