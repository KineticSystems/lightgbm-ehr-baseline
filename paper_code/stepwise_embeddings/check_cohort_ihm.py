"""Paper-specific cohort assertions for the phase-2 `ihm` task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from cohort_ihm import (
    AUTHOR_MAXLEN,
    AUTHOR_TARGET_ROW,
    AUTHOR_TASK,
    build_samples,
)
from cohort_source import (
    COMMON_SPLIT,
    SPLIT_MAP,
    assert_source_ready,
    inspect_hdf,
    sha256_file,
    source_filenames,
)


# mimic3-benchmarks @ ea0314c7 statistics.md, "In-hospital mortality prediction".
# Its "Train" column is the released train partition before the validation
# subject split, so it is compared against train+validation here.
M3B_PUBLISHED = {
    "train_plus_validation": {"negative": 15_480, "positive": 2_423},
    "test": {"negative": 2_862, "positive": 374},
}
M3B_STAYS = {"train_plus_validation": 35_621, "test": 6_281}
M3B_STATISTICS_SHA256 = (
    "07e6a56654c6388db92dc31b2d40d96baa2e94647c4a36e1daddc8dffc302aae"
)
PUBLISHED_AUROC = 0.858
PAPER_CELL = (
    "Appendix Table 9, MIMIC-III Mortality, '+ FTT embedding "
    "(Gorishniy et al., 2021)' AUROC 85.8 +/- 0.2"
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


def m3b_base_ihm(path: Path) -> dict[str, object]:
    """Read the released M3B `pos;mask;label` in-hospital-mortality field."""
    positive = 0
    negative = 0
    masked_out = 0
    labels: dict[str, int] = {}
    positions = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            encoded = row["in-hospital mortality task (pos;mask;label)"].split(";")
            if len(encoded) != 3:
                raise AssertionError("malformed M3B ihm pos;mask;label field")
            position, mask, label = (int(value) for value in encoded)
            if mask not in (0, 1) or label not in (0, 1):
                raise AssertionError("M3B ihm mask/label is not binary")
            if mask == 0:
                if position != 0:
                    raise AssertionError("masked M3B ihm row does not use position 0")
                masked_out += 1
                continue
            positions.add(position)
            labels[row["filename"]] = label
            if label == 1:
                positive += 1
            else:
                negative += 1
    if positions != {AUTHOR_TARGET_ROW}:
        raise AssertionError(f"M3B ihm positions changed: {sorted(positions)}")
    return {
        "positive": positive,
        "negative": negative,
        "eligible": positive + negative,
        "masked_out": masked_out,
        "labels": labels,
    }


def assert_frame_equal(expected: pl.DataFrame, actual: pl.DataFrame) -> None:
    columns = (
        "subject_id",
        "sample_id",
        "boolean_value",
        "split",
        "source_filename",
        "source_stay_order",
        "author_target_row",
        "prefix_rows",
        "source_label_row",
        "source_order",
        "paper_encoded_stay_id",
        "episode_number",
        "author_stay_rows",
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
        if not np.array_equal(
            expected[column].to_numpy(), actual[column].to_numpy()
        ):
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
    expected, expected_audit = build_samples(args.source_root)

    manifest = json.loads((args.task_dir / "manifest.json").read_text())
    index = pl.read_parquet(args.task_dir / manifest["meds"]["task_index"]["file"])
    labels = pl.read_parquet(args.task_dir / manifest["meds"]["labels"]["file"])
    codes = pl.read_parquet(args.task_dir / manifest["meds"]["codes"]["file"])
    assert_frame_equal(expected, index)
    expected_labels = index.select(
        "subject_id", "sample_id", "prediction_time", "boolean_value"
    ).sort("sample_id")
    if not expected_labels.equals(labels.sort("sample_id")):
        raise AssertionError("MEDS label table differs from the exact cohort")
    if codes["code"].to_list() != list(metadata["columns"]):
        raise AssertionError("MEDS vocabulary differs from authoritative HDF order")

    task = manifest["task"]
    if task["modalities"] != ["structured"] or task["note_codes"] != []:
        raise AssertionError("selected structured-only task retained another modality")
    if task["source_task"] != AUTHOR_TASK or task["published_auroc"] != PUBLISHED_AUROC:
        raise AssertionError("manifest does not identify the prespecified paper cell")
    if task["author_maxlen"] != AUTHOR_MAXLEN:
        raise AssertionError("manifest does not bind the released MAXLEN=48")
    if (index["author_target_row"] != AUTHOR_TARGET_ROW).any():
        raise AssertionError("an exported target is not at released ihm_position 47")
    if (index["prefix_rows"] != AUTHOR_MAXLEN).any():
        raise AssertionError("an exported prefix is not the released 48-row window")
    if (index["author_stay_rows"] <= AUTHOR_TARGET_ROW).any():
        raise AssertionError("an eligible stay is shorter than the released window")
    if index["sample_id"].n_unique() != index.height:
        raise AssertionError("duplicate sample identifiers")
    if index["source_filename"].n_unique() != index.height:
        raise AssertionError("the one-target-per-stay task duplicated a stay")

    exported = {}
    for split in ("train", "validation", "test"):
        selected = index.filter(pl.col("split") == split)
        positive = int(selected["boolean_value"].sum())
        exported[split] = {
            "stays": selected.height,
            "positive": positive,
            "negative": selected.height - positive,
            "subjects": selected["subject_id"].n_unique(),
            "prevalence": round(positive / selected.height, 6),
        }
        if manifest["splits"][split] != {
            "count": selected.height,
            "subjects": selected["subject_id"].n_unique(),
            "positive": positive,
        }:
            raise AssertionError(f"manifest {split} summary differs from the cohort")

    # Independent reconstruction from the M3B multitask listfiles.
    m3b = {}
    listfile_hashes = {}
    listfile_stays = {}
    ordered_stays: list[str] = []
    for split in SPLIT_MAP:
        common = COMMON_SPLIT[split]
        listfile = paths["multitask"] / f"{'val' if split == 'val' else split}_listfile.csv"
        m3b[common] = m3b_base_ihm(listfile)
        listfile_hashes[common] = sha256_file(listfile)
        names = source_filenames(paths["multitask"], split)
        listfile_stays[common] = len(names)
        ordered_stays.extend(f"{common}|{name}" for name in names)
        # exact per-stay label identity between M3B and the export
        selected = index.filter(pl.col("split") == common)
        exported_labels = dict(
            zip(
                selected["source_filename"].to_list(),
                (selected["boolean_value"].to_numpy().astype(int)).tolist(),
            )
        )
        if exported_labels != m3b[common]["labels"]:
            raise AssertionError(
                f"{common} exported ihm labels differ from the M3B listfile field"
            )
    m3b_train_validation = {
        key: m3b["train"][key] + m3b["validation"][key]
        for key in ("positive", "negative", "eligible")
    }
    if (
        m3b_train_validation["positive"]
        != M3B_PUBLISHED["train_plus_validation"]["positive"]
        or m3b_train_validation["negative"]
        != M3B_PUBLISHED["train_plus_validation"]["negative"]
    ):
        raise AssertionError("train+validation ihm marginals differ from M3B statistics")
    if (
        m3b["test"]["positive"] != M3B_PUBLISHED["test"]["positive"]
        or m3b["test"]["negative"] != M3B_PUBLISHED["test"]["negative"]
    ):
        raise AssertionError("test ihm marginals differ from M3B statistics")
    if (
        listfile_stays["train"] + listfile_stays["validation"]
        != M3B_STAYS["train_plus_validation"]
        or listfile_stays["test"] != M3B_STAYS["test"]
    ):
        raise AssertionError("listfile stay counts differ from M3B")

    memberships = {
        split: set(index.filter(pl.col("split") == split)["subject_id"].to_list())
        for split in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": len(memberships["train"] & memberships["validation"]),
        "train_test": len(memberships["train"] & memberships["test"]),
        "validation_test": len(memberships["validation"] & memberships["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"released subject split overlap changed: {overlaps}")
    # Released defect (audit finding): TLS's `extract_raw_data` parses only the
    # last character of the episode token, so `episode_digit * 1e7 + subject_id`
    # collides for episodes 2/12 etc.  Recorded, not repaired.  The export keys
    # every sample on the original M3B filename, so no colliding stay is merged.
    collision_groups = (
        index.select("source_filename", "paper_encoded_stay_id")
        .unique()
        .group_by("paper_encoded_stay_id")
        .len()
        .filter(pl.col("len") > 1)
    )
    encoded_collisions = collision_groups.height
    colliding_stays = (
        int(collision_groups["len"].sum()) if encoded_collisions else 0
    )

    report = {
        "status": "exact",
        "selected_experiment": {
            "task": AUTHOR_TASK,
            "paper_cell": PAPER_CELL,
            "published_auroc": PUBLISHED_AUROC,
            "source_configuration": "MIMIC structured-only, full-data ihm with MAXLEN=48",
            "retained_modalities": ["structured"],
            "unused_modality_cohort_filter": False,
            "phase": "phase 2 second published binary outcome",
        },
        "paper_and_attributable_facts": {
            "m3b_statistics_sha256": M3B_STATISTICS_SHA256,
            "m3b_published_ihm_counts": M3B_PUBLISHED,
            "reconstructed_m3b_base_ihm": {
                "train_plus_validation": m3b_train_validation,
                "train": {
                    key: m3b["train"][key]
                    for key in ("positive", "negative", "eligible", "masked_out")
                },
                "validation": {
                    key: m3b["validation"][key]
                    for key in ("positive", "negative", "eligible", "masked_out")
                },
                "test": {
                    key: m3b["test"][key]
                    for key in ("positive", "negative", "eligible", "masked_out")
                },
            },
            "m3b_stay_counts": M3B_STAYS,
            "reconstructed_listfile_stays": listfile_stays,
            "reconstructed_author_hdf_ihm": expected_audit["splits"],
        },
        "exact_split_membership": {
            "ordered_listfile_identity_sha256": ordered_strings_hash(ordered_stays),
            "listfile_sha256": listfile_hashes,
            "subject_overlap": overlaps,
            "paper_encoded_stay_id_collisions": encoded_collisions,
            "stays_in_colliding_surrogate_groups": colliding_stays,
            "collision_effect_on_export": (
                "none: every sample is keyed on its original M3B "
                "<subject>_episode<n>_timeseries.csv filename and its own HDF window"
            ),
        },
        "exact_targets": {
            "ordered_target_tuple_sha256": expected_audit["ordered_target_tuple_sha256"],
            "count": expected_audit["samples"],
            "author_target_row": AUTHOR_TARGET_ROW,
            "author_maxlen": AUTHOR_MAXLEN,
            "one_target_per_eligible_stay": True,
            "per_stay_label_identity_with_m3b_listfile": True,
        },
        "exported_splits": exported,
        "author_channels": {
            "count": len(metadata["columns"]),
            "ordered_sha256": ordered_strings_hash(list(metadata["columns"])),
            "note_codes": [],
        },
        "released_defects_preserved": [
            "get_one_hot fits the categorical dummy vocabulary on concatenated "
            "train+validation+test covariates (held-out category presence sets the "
            "released feature dimensionality)",
            "impute_sample_1h copies the first irregular measurement back to grid "
            "row 0 and repeats it until its true timestamp",
            "TLS aliases masked_labels_ihm to masked_labels_los and clears LOS "
            "before writing the ihm target (corrupts the released LOS task only)",
            "TLS extract_raw_data derives its surrogate stay id from only the last "
            "digit of the episode token, so episode 2/12 pairs collide",
        ],
        "defects_absent_for_this_task": [
            "the one-hour decompensation target shift does not apply: TLS writes "
            "the ihm target directly at author row 47"
        ],
        "assertions": [
            "exact M3B train/validation/test listfile membership and hashes",
            "exact per-stay M3B ihm label identity with the exported labels",
            "published M3B in-hospital-mortality train+validation and test marginals",
            "exact TLS author-HDF ihm targets, one per eligible stay at row 47",
            "released MAXLEN=48 prefix bound on every sample",
            "exact subject-disjoint released partitions",
            "exact structured-only modality projection and 42-channel vocabulary",
            "released TLS surrogate stay-id collisions measured and reported",
        ],
    }
    output = args.task_dir / "audit" / "cohort_reconstruction.json"
    write_json(output, report)
    print(json.dumps(report["paper_and_attributable_facts"], indent=2))


if __name__ == "__main__":
    main()
