"""Cohort, endpoint, split, and provenance checks for the SSML ihm export."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from cohort_source import COMMON_SPLIT, SPLIT_MAP, load_transform_spec, sha256_file
from cohort_source_ihm import (
    IHM_POSITION,
    RELEASED_LABEL_DEFINITION,
    TASK_ID,
    WINDOW_HOURS,
    assert_source_ready,
    build_samples,
    listfile_path,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
STEPWISE_IHM_INDEX = (
    WORKSPACE_ROOT / "paper_tasks" / "stepwise_embeddings" / "ihm" / "audit" / "task_index.parquet"
)
# mimic3-benchmarks statistics.md, "# In-hospital mortality prediction".  Its
# "Train" column is the full pre-split train partition, i.e. this export's
# train + validation.
PUBLISHED_M3B_IHM = {
    "train_plus_validation": {"negative": 15_480, "positive": 2_423},
    "test": {"negative": 2_862, "positive": 374},
}
PUBLISHED_AUROC = 0.869
PUBLISHED_BASELINE_AUROC = 0.858
PUBLISHED_BASELINE_NAME = "MAML"
INDEX_COLUMNS = (
    "subject_id", "sample_id", "observation_start", "prediction_time",
    "boolean_value", "split", "source_filename", "source_list_row",
    "source_cutoff_hour", "ihm_position", "source_order", "episode_number",
)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def assert_frame_equal(expected: pl.DataFrame, actual: pl.DataFrame) -> None:
    expected = expected.sort("source_order").select(INDEX_COLUMNS)
    actual = actual.sort("source_order").select(INDEX_COLUMNS)
    if expected.shape != actual.shape:
        raise AssertionError(
            f"MEDS/source cohort shape differs: {actual.shape} != {expected.shape}"
        )
    for column in INDEX_COLUMNS:
        if not np.array_equal(expected[column].to_numpy(), actual[column].to_numpy()):
            raise AssertionError(f"MEDS/source cohort differs in {column}")


def published_ihm_statistics(source_root: Path) -> tuple[dict[str, dict[str, int]], str]:
    path = Path(source_root).resolve() / "mimic3-benchmarks" / "statistics.md"
    text = path.read_text()
    section = text.split("# In-hospital mortality prediction", 1)[1].split("#", 1)[0]
    values = {}
    for name in ("Negative", "Positive"):
        match = re.search(rf"^\|\s*{name}\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", section, re.M)
        if match is None:
            raise AssertionError("released statistics.md ihm table changed")
        values[name.lower()] = (int(match.group(1)), int(match.group(2)))
    published = {
        "train_plus_validation": {
            "negative": values["negative"][0],
            "positive": values["positive"][0],
        },
        "test": {
            "negative": values["negative"][1],
            "positive": values["positive"][1],
        },
    }
    if published != PUBLISHED_M3B_IHM:
        raise AssertionError(f"published M3B ihm inventory changed: {published}")
    return published, sha256_file(path)


def cross_export_check(index: pl.DataFrame) -> dict[str, object]:
    """Compare with the same M3B stays exported through the stepwise TLS loader."""

    if not STEPWISE_IHM_INDEX.is_file():
        return {"available": False, "reason": f"missing {STEPWISE_IHM_INDEX}"}
    other = pl.read_parquet(
        STEPWISE_IHM_INDEX, columns=["split", "source_filename", "boolean_value"]
    )
    report: dict[str, object] = {
        "available": True,
        "task": "paper_tasks/stepwise_embeddings/ihm",
        "boundary": "TLS loader HDF grid (a different released boundary over the same M3B stays)",
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        mine = index.filter(pl.col("split") == split)
        theirs = other.filter(pl.col("split") == split)
        my_pairs = set(
            zip(mine["source_filename"].to_list(), mine["boolean_value"].to_list())
        )
        their_pairs = set(
            zip(theirs["source_filename"].to_list(), theirs["boolean_value"].to_list())
        )
        if my_pairs != their_pairs:
            raise AssertionError(
                f"{split} stay/label set differs from the stepwise_embeddings ihm export: "
                f"{len(my_pairs - their_pairs)} only here, "
                f"{len(their_pairs - my_pairs)} only there"
            )
        report["splits"][split] = {
            "stays": len(my_pairs),
            "identical_stay_and_label_set": True,
        }
    return report


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
        raise AssertionError("MEDS labels differ from the released eligible cohort")
    if tuple(codes["code"].to_list()) != spec["codes"]:
        raise AssertionError("MEDS vocabulary differs from the released 59-value order")

    task = manifest["task"]
    if task["id"] != TASK_ID:
        raise AssertionError("wrong SSML task")
    if task["modalities"] != ["structured"] or task["note_codes"] != []:
        raise AssertionError("structured-only selected task retained another modality")
    if task["published_auroc"] != PUBLISHED_AUROC:
        raise AssertionError("manifest does not carry the Table 4 SSML-TDV ihm AUROC")
    if not str(task["source_task"]).startswith("Table 4"):
        raise AssertionError("manifest does not identify the selected paper cell")
    baseline = task["published_baseline"]
    if (
        baseline["name"] != PUBLISHED_BASELINE_NAME
        or baseline["auroc"] != PUBLISHED_BASELINE_AUROC
        or "canonical best baseline" not in baseline["rule"]
    ):
        raise AssertionError("manifest does not carry the canonical published baseline")
    if task["author_target_row"] != IHM_POSITION:
        raise AssertionError("manifest does not carry the released ihm prediction row")
    if float(task["observation_window_hours"]) != float(WINDOW_HOURS):
        raise AssertionError("manifest window differs from the released ihm window")
    if task["label_definition"] != RELEASED_LABEL_DEFINITION:
        raise AssertionError("manifest label definition is not the released one")
    if task["temporal_profile"] != "single_visit_timed":
        raise AssertionError("ihm is a single timed ICU stay")

    # Every retained row is the released pos=47/mask=1 landmark at 48 hours.
    if index.filter(pl.col("ihm_position") != IHM_POSITION).height:
        raise AssertionError("a sample is not at the released ihm prediction position")
    if index.filter(pl.col("source_cutoff_hour") != WINDOW_HOURS).height:
        raise AssertionError("a sample does not use the released 48-hour window")
    widths = index.select(
        (
            (pl.col("prediction_time") - pl.col("observation_start"))
            .dt.total_microseconds()
            .cast(pl.Int64)
        ).unique()
    ).to_series().to_list()
    if widths != [WINDOW_HOURS * 3_600_000_000]:
        raise AssertionError(f"sample windows are not exactly 48 hours: {widths}")

    published, statistics_sha256 = published_ihm_statistics(args.source_root)

    summaries = {}
    for split in ("train", "validation", "test"):
        selected = index.filter(pl.col("split") == split)
        positive = int(selected["boolean_value"].sum())
        summaries[split] = {
            "samples": selected.height,
            "positive": positive,
            "negative": selected.height - positive,
            "subjects": selected["subject_id"].n_unique(),
            "stays": selected["source_filename"].n_unique(),
            "prevalence": round(positive / selected.height, 6),
        }
        expected_split = expected_audit["splits"][split]
        for key in ("samples", "positive", "negative", "subjects", "stays"):
            if summaries[split][key] != expected_split[key]:
                raise AssertionError(f"{split} {key} differs from source reconstruction")
        if manifest["splits"][split] != {
            "count": selected.height,
            "subjects": selected["subject_id"].n_unique(),
            "positive": positive,
        }:
            raise AssertionError(f"manifest {split} summary differs from cohort")

    reconstructed_published = {
        "train_plus_validation": {
            "negative": summaries["train"]["negative"] + summaries["validation"]["negative"],
            "positive": summaries["train"]["positive"] + summaries["validation"]["positive"],
        },
        "test": {
            "negative": summaries["test"]["negative"],
            "positive": summaries["test"]["positive"],
        },
    }
    if reconstructed_published != published:
        raise AssertionError(
            f"exported ihm marginals differ from published M3B: {reconstructed_published}"
        )

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

    events_path = args.task_dir / manifest["meds"]["events"]["file"]
    parquet = pq.ParquetFile(events_path)
    if parquet.metadata.num_rows != manifest["event_count"]:
        raise AssertionError("event Parquet row count differs from manifest")
    if manifest["source"]["event_count_precomputed"] != parquet.metadata.num_rows:
        raise AssertionError("event pre-count differs from Parquet")

    # Cheap independent restatement of the parity rule: no exported occurrence
    # may precede the observation start or postdate the 48-hour prediction time.
    lower = index["observation_start"].min()
    upper = index["prediction_time"].max()
    minimum = maximum = None
    for batch in parquet.iter_batches(columns=["time"], batch_size=1_000_000):
        column = pl.from_arrow(batch)["time"]
        low, high = column.min(), column.max()
        minimum = low if minimum is None else min(minimum, low)
        maximum = high if maximum is None else max(maximum, high)
    if minimum < lower or maximum > upper:
        raise AssertionError("an exported occurrence falls outside the released window")

    if manifest["source"]["listfile_sha256"] != {
        COMMON_SPLIT[split]: sha256_file(listfile_path(paths, split))
        for split in SPLIT_MAP
    }:
        raise AssertionError("manifest listfile hashes changed")
    defect = manifest["source"]["released_ihm_training_path"]
    if defect.get("ihm_trainer_released") is not False or not defect.get("reader_returns_ihm"):
        raise AssertionError("unreleased ihm-trainer defect was not preserved")

    cross_export = cross_export_check(index)

    report = {
        "status": "exact_source_with_unreleased_ihm_trainer",
        "paper": "semi_supervised_meta_learning",
        "task": TASK_ID,
        "canonical_export": {
            "kind": (
                "every released multitask listfile row whose in-hospital mortality "
                "mask is 1, at the released pos=47 (48-hour) prediction point"
            ),
            "splits": summaries,
            "subject_overlap": overlaps,
            "ordered_eligible_tuple_sha256": expected_audit["ordered_eligible_tuple_sha256"],
            "listfile_sha256": manifest["source"]["listfile_sha256"],
            "event_count": parquet.metadata.num_rows,
            "earliest_event": str(minimum),
            "latest_event": str(maximum),
            "window_hours": float(WINDOW_HOURS),
        },
        "released_label_definition": RELEASED_LABEL_DEFINITION,
        "released_ihm_training_path_defect": defect,
        "published_m3b_inventory": {
            "statistics_md_sha256": statistics_sha256,
            "published": published,
            "reconstructed": reconstructed_published,
        },
        "cross_export_identity": cross_export,
        "author_boundary": {
            "shape": "[48,76]",
            "value_channels": 59,
            "mask_channels": 17,
            "prediction_row": IHM_POSITION,
            "imputation": "previous, initialized with released normal values",
            "normalization": "released SSML normalizer state (decomp_ts1.0)",
        },
        "selected_result": {
            "table": 4,
            "column": "In-hospital Mortality (AUCROC)",
            "method": "SSML-TDV",
            "auroc": PUBLISHED_AUROC,
            "baseline": {
                "method": PUBLISHED_BASELINE_NAME,
                "auroc": PUBLISHED_BASELINE_AUROC,
            },
        },
        "modalities": ["structured"],
    }
    write_json(args.task_dir / "audit" / "cohort_reconstruction.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
