"""Paper-specific cohort and rolling-landmark reconstruction assertions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from cohort_source import (
    COMMON_SPLIT,
    SPLIT_MAP,
    assert_source_ready,
    build_samples,
    inspect_hdf,
    sha256_file,
    source_filenames,
)
from missing_xgboost_lightgbm.landmark_sampling import select_uniform_landmarks


M3B_PUBLISHED = {
    "train_plus_validation": {"negative": 2_847_401, "positive": 61_013},
    "test": {"negative": 513_525, "positive": 9_683},
}
M3B_STAYS = {"train_plus_validation": 35_621, "test": 6_281}
M3B_STATISTICS_SHA256 = (
    "07e6a56654c6388db92dc31b2d40d96baa2e94647c4a36e1daddc8dffc302aae"
)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def ordered_strings_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def m3b_base_decomp_counts(path: Path) -> dict[str, int]:
    """Count the published M3B decomp masks before TLS derives new horizons."""
    positive = 0
    negative = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            encoded = row["decompensation task (masks;labels)"].split(";")
            if encoded == ["", ""]:
                continue
            if len(encoded) % 2:
                raise AssertionError("malformed M3B decomp mask/label field")
            length = len(encoded) // 2
            masks = encoded[:length]
            labels = encoded[length:]
            for mask, label in zip(masks, labels):
                if mask == "1":
                    if label == "1":
                        positive += 1
                    elif label == "0":
                        negative += 1
                    else:
                        raise AssertionError("eligible M3B decomp label is not binary")
                elif mask != "0":
                    raise AssertionError("M3B decomp mask is not binary")
    return {"positive": positive, "negative": negative}


def assert_frame_equal(expected: pl.DataFrame, actual: pl.DataFrame) -> None:
    columns = (
        "subject_id",
        "sample_id",
        "boolean_value",
        "split",
        "source_filename",
        "source_stay_order",
        "author_target_row",
        "source_cutoff_hour",
        "source_label_row",
        "source_order",
        "paper_encoded_stay_id",
        "episode_number",
        "observation_start",
        "prediction_time",
    )
    expected = expected.sort("source_order").select(columns)
    actual = actual.sort("source_order").select(columns)
    if expected.shape != actual.shape:
        raise AssertionError(
            f"MEDS/source cohort shape differs: {actual.shape} != {expected.shape}"
        )
    for column in columns:
        left = expected[column].to_numpy()
        right = actual[column].to_numpy()
        if not np.array_equal(left, right):
            raise AssertionError(f"MEDS/source cohort differs in {column}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = assert_source_ready(args.source_root)
    preparation = json.loads(paths["preparation"].read_text())
    if (
        preparation["source_artifacts"]["m3b_statistics"]["sha256"]
        != M3B_STATISTICS_SHA256
    ):
        raise AssertionError("attributable M3B statistics artifact changed")
    metadata = inspect_hdf(args.source_root)
    full_cohort, expected_audit = build_samples(args.source_root)
    expected, sampling_audit = select_uniform_landmarks(full_cohort)
    manifest = json.loads((args.task_dir / "manifest.json").read_text())
    index = pl.read_parquet(args.task_dir / manifest["meds"]["task_index"]["file"])
    labels = pl.read_parquet(args.task_dir / manifest["meds"]["labels"]["file"])
    codes = pl.read_parquet(args.task_dir / manifest["meds"]["codes"]["file"])
    assert_frame_equal(expected, index)
    if manifest["source"].get("sampling") != sampling_audit:
        raise AssertionError("manifest sampling record differs from deterministic contract")
    expected_labels = index.select(
        "subject_id", "sample_id", "prediction_time", "boolean_value"
    ).sort("sample_id")
    if not expected_labels.equals(labels.sort("sample_id")):
        raise AssertionError("MEDS label table differs from exact rolling cohort")
    if codes["code"].to_list() != list(metadata["columns"]):
        raise AssertionError("MEDS vocabulary differs from authoritative HDF order")
    task = manifest["task"]
    if task["modalities"] != ["structured"] or task["note_codes"] != []:
        raise AssertionError("selected structured-only task retained another modality")
    if task["source_task"] != "decomp_24Hours" or task["published_auroc"] != 0.916:
        raise AssertionError("manifest does not identify the prespecified paper cell")
    if not np.array_equal(
        index["author_target_row"].to_numpy() - 1,
        index["source_cutoff_hour"].to_numpy(),
    ):
        raise AssertionError("released one-hour target shift was not preserved")

    exported_summaries = {}
    for split in ("train", "validation", "test"):
        selected = index.filter(pl.col("split") == split)
        exported_summaries[split] = {
            "landmarks": selected.height,
            "positive": int(selected["boolean_value"].sum()),
            "negative": selected.height - int(selected["boolean_value"].sum()),
            "stays": selected["source_filename"].n_unique(),
            "subjects": selected["subject_id"].n_unique(),
        }
        if manifest["splits"][split] != {
            "count": selected.height,
            "subjects": selected["subject_id"].n_unique(),
            "positive": int(selected["boolean_value"].sum()),
        }:
            raise AssertionError(f"manifest {split} summary differs from subsample")
    full_summaries = {}
    for split in ("train", "validation", "test"):
        selected = full_cohort.filter(pl.col("split") == split)
        full_summaries[split] = {
            "landmarks": selected.height,
            "positive": int(selected["boolean_value"].sum()),
            "negative": selected.height - int(selected["boolean_value"].sum()),
            "stays": selected["source_filename"].n_unique(),
            "subjects": selected["subject_id"].n_unique(),
        }
    m3b_base = {
        split: m3b_base_decomp_counts(
            paths["multitask"]
            / f"{'val' if split == 'validation' else split}_listfile.csv"
        )
        for split in ("train", "validation", "test")
    }
    m3b_base_train_validation = {
        key: m3b_base["train"][key] + m3b_base["validation"][key]
        for key in ("positive", "negative")
    }
    if (
        m3b_base_train_validation["positive"]
        != M3B_PUBLISHED["train_plus_validation"]["positive"]
    ):
        raise AssertionError("train+validation positive count differs from M3B")
    if (
        m3b_base_train_validation["negative"]
        != M3B_PUBLISHED["train_plus_validation"]["negative"]
    ):
        raise AssertionError("train+validation negative count differs from M3B")
    if m3b_base["test"]["positive"] != M3B_PUBLISHED["test"]["positive"]:
        raise AssertionError("test positive count differs from M3B")
    if m3b_base["test"]["negative"] != M3B_PUBLISHED["test"]["negative"]:
        raise AssertionError("test negative count differs from M3B")
    author_hdf = preparation["source_artifacts"]["standard_scaled_hdf"]["splits"]
    for split, reconstructed in full_summaries.items():
        source_split = "val" if split == "validation" else split
        for key in ("landmarks", "positive", "negative"):
            if reconstructed[key] != author_hdf[source_split][key]:
                raise AssertionError(
                    f"{split} exact author-HDF {key} differs from preparation"
                )
    memberships = {
        split: set(
            full_cohort.filter(pl.col("split") == split)["subject_id"].unique().to_list()
        )
        for split in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": len(memberships["train"] & memberships["validation"]),
        "train_test": len(memberships["train"] & memberships["test"]),
        "validation_test": len(memberships["validation"] & memberships["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"released subject split overlap changed: {overlaps}")

    ordered_stays = []
    listfile_hashes = {}
    listfile_stays = {}
    for split in SPLIT_MAP:
        names = source_filenames(paths["multitask"], split)
        listfile_stays[COMMON_SPLIT[split]] = len(names)
        ordered_stays.extend(f"{COMMON_SPLIT[split]}|{name}" for name in names)
        listfile_hashes[COMMON_SPLIT[split]] = sha256_file(
            paths["multitask"] / f"{'val' if split == 'val' else split}_listfile.csv"
        )
    if (
        listfile_stays["train"] + listfile_stays["validation"]
        != M3B_STAYS["train_plus_validation"]
    ):
        raise AssertionError("train+validation listfile stay count differs from M3B")
    if listfile_stays["test"] != M3B_STAYS["test"]:
        raise AssertionError("test listfile stay count differs from M3B")
    encoded_collisions = (
        full_cohort.select("source_filename", "paper_encoded_stay_id")
        .unique()
        .group_by("paper_encoded_stay_id")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    report = {
        "status": "exact",
        "selected_experiment": {
            "task": "decomp_24Hours",
            "paper_cell": "Appendix Table 9, FTT decompensation AUROC 0.916 ± 0.001",
            "source_configuration": "MIMIC structured-only, full-data decompensation",
            "retained_modalities": ["structured"],
            "unused_modality_cohort_filter": False,
        },
        "paper_and_attributable_facts": {
            "m3b_statistics_sha256": M3B_STATISTICS_SHA256,
            "m3b_published_landmark_counts": M3B_PUBLISHED,
            "reconstructed_m3b_base_decomp": {
                "train_plus_validation": m3b_base_train_validation,
                "test": m3b_base["test"],
            },
            "m3b_stay_counts": M3B_STAYS,
            "reconstructed_listfile_stays": listfile_stays,
            "reconstructed_author_hdf_decomp_24hours": full_summaries,
        },
        "exact_split_membership": {
            "ordered_listfile_identity_sha256": ordered_strings_hash(ordered_stays),
            "listfile_sha256": listfile_hashes,
            "subject_overlap": overlaps,
            "paper_encoded_stay_id_collisions": encoded_collisions,
        },
        "exact_landmarks": {
            "ordered_target_tuple_sha256": expected_audit[
                "ordered_target_tuple_sha256"
            ],
            "count": expected_audit["landmarks"],
            "one_hour_source_cutoff_shift_preserved": True,
        },
        "resource_limited_export": {
            "sampling": sampling_audit,
            "exported_splits": exported_summaries,
            "interpretation": "uniform unweighted landmark subsample; paper AUROC used the full released evaluation set",
        },
        "author_channels": {
            "count": len(metadata["columns"]),
            "ordered_sha256": ordered_strings_hash(list(metadata["columns"])),
            "note_codes": [],
        },
        "assertions": [
            "exact M3B train/validation/test listfile membership",
            "exact TLS decomp_24Hours eligible landmarks and labels",
            "exact deterministic resource-limited landmark subsample",
            "published M3B base-decomp train+validation and test marginals",
            "distinct exact TLS-derived decomp_24Hours marginals",
            "exact subject-disjoint released partitions",
            "exact structured-only modality projection",
            "faithful one-hour source-target misalignment",
        ],
    }
    output = args.task_dir / "audit" / "cohort_reconstruction.json"
    write_json(output, report)
    print(json.dumps(report["paper_and_attributable_facts"], indent=2))


if __name__ == "__main__":
    main()
