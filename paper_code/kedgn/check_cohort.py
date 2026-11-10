"""Assert KEDGN's paper-specific MIMIC-III cohort facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from cohort_source import (
    EXPECTED_BLACKLIST_REMOVALS,
    EXPECTED_DUPLICATE_ROWS_COLLAPSED,
    EXPECTED_EVENT_COUNT,
    EXPECTED_MAX_LENGTH,
    EXPECTED_MAX_READER_LENGTH,
    EXPECTED_MISSING_RATIO,
    EXPECTED_POSITIVES,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_POSITIVES,
    EXPECTED_SPLIT_SUBJECTS,
    EXPECTED_TOTAL,
    SPLITS,
    assert_mimic_release,
    blacklist,
    code_vocabulary,
    export_code_vocabulary,
    iter_source_examples,
    load_build_report,
    load_cohort,
    ordered_text_hash,
    sha256,
)


def _subject_overlap(index: pl.DataFrame) -> dict[str, int]:
    members = {
        split: set(
            index.filter(pl.col("split") == split)["subject_id"].unique().to_list()
        )
        for split in SPLITS
    }
    return {
        "train_validation": len(members["train"] & members["validation"]),
        "train_test": len(members["train"] & members["test"]),
        "validation_test": len(members["validation"] & members["test"]),
        "subjects_in_multiple_splits": sum(
            sum(subject in values for values in members.values()) > 1
            for subject in set().union(*members.values())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    task_root = args.task_root.resolve()
    build = load_build_report(cache_root)
    if assert_mimic_release(args.mimic_root.resolve()) != build["release_sha256"]:
        raise AssertionError("KEDGN source report does not match exact MIMIC-III 1.4")
    cohort = load_cohort(cache_root)
    manifest = json.loads((task_root / "manifest.json").read_text())
    task = manifest["task"]
    source = manifest["source"]
    if task.get("modalities") != ["structured"] or task.get("note_codes") != []:
        raise AssertionError("KEDGN selected task must be structured-only")
    if task.get("published_auroc") != 0.851 or task.get("published_auroc_sd") != 0.003:
        raise AssertionError("KEDGN manifest identifies the wrong Table 1 cell")
    if source.get("selection_tier") != 3 or source.get("selected_model_inputs") != [
        "structured"
    ]:
        raise AssertionError("KEDGN selected the wrong experiment/modalities")
    if source.get("fidelity_status") != build["status"]:
        raise AssertionError("KEDGN manifest lost the source-fidelity status")

    index = pl.read_parquet(task_root / manifest["meds"]["task_index"]["file"])
    expected = cohort.select(
        "sample_id",
        pl.col("subject_id").alias("expected_subject_id"),
        pl.col("split").alias("expected_split"),
        pl.col("label").alias("expected_label"),
        pl.col("source_name").alias("expected_source_name"),
    )
    compared = index.join(expected, on="sample_id", how="inner", validate="1:1")
    if compared.height != EXPECTED_TOTAL:
        raise AssertionError("KEDGN MEDS task lost a cohort member")
    if compared.filter(
        (pl.col("subject_id") != pl.col("expected_subject_id"))
        | (pl.col("split") != pl.col("expected_split"))
        | (pl.col("boolean_value") != pl.col("expected_label"))
        | (pl.col("source_name") != pl.col("expected_source_name"))
    ).height:
        raise AssertionError("KEDGN MEDS identity, split, or label differs from source")

    lengths = []
    reader_lengths = []
    observed = 0
    model_observed = 0
    suppressed_zero = 0
    suppressed_negative = 0
    possible = 0
    minimum_time = float("inf")
    maximum_time = float("-inf")
    ordered_names = []
    ordered_labels = []
    ordered_ids = []
    reader_rows = 0
    duplicate_rows_collapsed = 0
    for split in SPLITS:
        for example in iter_source_examples(cache_root, split):
            lengths.append(example.length)
            reader_lengths.append(example.reader_length)
            observed += int(example.measurements.sum())
            finite_values = example.values[example.measurements]
            model_observed += int((finite_values > 0).sum())
            suppressed_zero += int((finite_values == 0).sum())
            suppressed_negative += int((finite_values < 0).sum())
            possible += int(example.measurements.size)
            minimum_time = min(minimum_time, float(example.time.min()))
            maximum_time = max(maximum_time, float(example.time.max()))
            ordered_names.append(example.source_name)
            ordered_labels.append(example.label)
            ordered_ids.append(example.sample_id)
            reader_rows += example.reader_length
            duplicate_rows_collapsed += example.duplicate_rows_collapsed
    missing_ratio = 1.0 - observed / possible
    if len(lengths) != EXPECTED_TOTAL or max(lengths) != EXPECTED_MAX_LENGTH:
        raise AssertionError("KEDGN paper sample count or maximum length changed")
    if (
        max(reader_lengths) != EXPECTED_MAX_READER_LENGTH
        or duplicate_rows_collapsed != EXPECTED_DUPLICATE_ROWS_COLLAPSED
    ):
        raise AssertionError("KEDGN lossless equal-time coalescing evidence changed")
    if missing_ratio != EXPECTED_MISSING_RATIO or round(100.0 * missing_ratio, 1) != 65.5:
        raise AssertionError(f"KEDGN paper missing ratio changed: {missing_ratio}")
    if minimum_time < -1e-6 or maximum_time > 48.0 + 1e-6:
        raise AssertionError("KEDGN source times leave the first-48-hour window")
    if observed != EXPECTED_EVENT_COUNT:
        raise AssertionError("KEDGN exact finite-cell marginal changed")
    expected_export_events = 2 * model_observed + EXPECTED_TOTAL
    if manifest["event_count"] != expected_export_events:
        raise AssertionError("KEDGN MEDS event count differs from exact author-visible fields")

    split_summary = {}
    for split in SPLITS:
        part = index.filter(pl.col("split") == split)
        split_summary[split] = {
            "count": part.height,
            "subjects": part["subject_id"].n_unique(),
            "positive": int(part["boolean_value"].sum()),
        }
        if split_summary[split] != {
            "count": EXPECTED_SPLIT_COUNTS[split],
            "subjects": EXPECTED_SPLIT_SUBJECTS[split],
            "positive": EXPECTED_SPLIT_POSITIVES[split],
        }:
            raise AssertionError(f"KEDGN {split} paper-source marginal changed")
    overlap = _subject_overlap(index)
    if any(overlap.values()):
        raise AssertionError("standard KEDGN split is no longer patient-disjoint")
    codes = tuple(
        pl.read_parquet(task_root / manifest["meds"]["codes"]["file"])["code"].to_list()
    )
    if codes != export_code_vocabulary():
        raise AssertionError("KEDGN MEDS vocabulary differs from author-visible boundary")

    output = {
        "status": "exact",
        "paper_task": "Table 1 KEDGN-ChatGPT MIMIC-III AUROC 0.851 +/- 0.003",
        "samples": EXPECTED_TOTAL,
        "subjects": index["subject_id"].n_unique(),
        "positives": EXPECTED_POSITIVES,
        "prevalence": EXPECTED_POSITIVES / EXPECTED_TOTAL,
        "split_summary": split_summary,
        "cross_split_subject_overlap": overlap,
        "blacklist": {
            "count": len(blacklist()),
            "removals": EXPECTED_BLACKLIST_REMOVALS,
            "ordered_sha256": ordered_text_hash(blacklist()),
        },
        "source_shape": {
            "variables": 16,
            "maximum_length": max(lengths),
            "maximum_public_reader_rows": max(reader_lengths),
            "minimum_length": min(lengths),
            "observed_cells": observed,
            "released_positive_model_mask_cells": model_observed,
            "source_observed_zero_cells_suppressed": suppressed_zero,
            "source_observed_negative_cells_suppressed": suppressed_negative,
            "model_mask_invented_cells": 0,
            "possible_unpadded_cells": possible,
            "public_reader_rows": reader_rows,
            "duplicate_time_rows_losslessly_collapsed": duplicate_rows_collapsed,
            "missing_ratio": missing_ratio,
            "minimum_hour": minimum_time,
            "maximum_hour": maximum_time,
        },
        "ordered_source_name_sha256": ordered_text_hash(ordered_names),
        "ordered_sample_id_sha256": ordered_text_hash(ordered_ids),
        "ordered_label_sha256": ordered_text_hash(ordered_labels),
        "source_build_sha256": sha256(cache_root / "source_build.json"),
        "assertions": {
            "paper_21107_samples_16_variables_max292_missing65_5pct": True,
            "exact_14662_3217_3228_blacklisted_split": True,
            "split_label_and_identifier_membership": True,
            "patient_disjoint_standard_split": True,
            "structured_only_selected_experiment": True,
            "exact_author_visible_value_interval_length_event_count": True,
            "suppressed_nonpositive_raw_values_withheld_from_meds": True,
            "same_timestamp_rows_losslessly_coalesced": True,
            "declared_gcs_none_category_preserved_across_pandas_versions": True,
            "first_48_hour_window": True,
        },
    }
    output_path = task_root / "audit" / "cohort_reconstruction.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "exact", "samples": EXPECTED_TOTAL, "events": observed}))


if __name__ == "__main__":
    main()
