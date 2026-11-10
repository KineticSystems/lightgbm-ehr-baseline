"""Assert REMed's paper-specific 48-hour MIMIC-IV cohort evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from cohort_source import (
    LABEL_COLUMN,
    PUBLISHED_AUROC,
    TASK_ID,
    cross_split_subject_overlap,
    hash_ints,
    load_samples,
    split_summary,
)


PAPER_COUNTS = {"patients": 25_801, "admissions": 30_360, "icu_stays": 32_449}
PAPER_LABELS = {"negative": 25_887, "positive": 6_562}
PAPER_EVENT_LENGTH = {"average": 3992.4, "median": 3023.0, "p90": 5106.2, "max": 266_766}
PAPER_TOKEN_LENGTH = {
    "average": 96_714.8,
    "median": 72_443.0,
    "p90": 133_924.0,
    "max": 6_669_451,
}


def distribution(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.int64)
    return {
        "average": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": int(array.max()),
    }


def assert_paper_distribution(actual, expected, name) -> None:
    for field in ("average", "p90"):
        if not np.isclose(actual[field], expected[field], atol=0.051, rtol=0):
            raise AssertionError(
                f"REMed {name} {field} differs: {actual[field]} != {expected[field]}"
            )
    for field in ("median", "max"):
        if actual[field] != expected[field]:
            raise AssertionError(
                f"REMed {name} {field} differs: {actual[field]} != {expected[field]}"
            )


def distribution_mismatches(actual, expected) -> dict[str, dict[str, float]]:
    """Return every source-to-paper summary difference without rounding it away."""
    return {
        field: {
            "paper": float(expected[field]),
            "reconstructed": float(actual[field]),
            "absolute_difference": float(actual[field] - expected[field]),
            "relative_difference": float(
                (actual[field] - expected[field]) / expected[field]
            ),
        }
        for field in expected
        if actual[field] != expected[field]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.task_dir / "manifest.json").read_text())
    source = manifest["source"]
    if manifest["task"]["id"] != TASK_ID or args.task_dir.name != TASK_ID:
        raise AssertionError("REMed selected task differs")
    if manifest["task"].get("published_auroc") != PUBLISHED_AUROC:
        raise AssertionError("REMed published LOS-7 AUROC differs")
    if source.get("selected_model_inputs") != ["structured"]:
        raise AssertionError("REMed selected input modality differs")
    if source.get("selection_tier") != 1:
        raise AssertionError("REMed selected modality tier differs")
    if source.get("selected_label") != LABEL_COLUMN:
        raise AssertionError("REMed selected source label differs")
    if manifest["task"].get("modalities") != ["structured"]:
        raise AssertionError("REMed common task must be structured-only")
    if manifest["task"].get("note_codes") != []:
        raise AssertionError("REMed must not declare notes")

    samples, observation_hours = load_samples(args.source_root)
    counts = {
        "patients": len({sample.subject_id for sample in samples}),
        "admissions": len({sample.hadm_id for sample in samples}),
        "icu_stays": len(samples),
    }
    if counts != PAPER_COUNTS:
        raise AssertionError(f"REMed Table 4 cohort differs: {counts}")
    labels = {
        "negative": sum(not sample.label for sample in samples),
        "positive": sum(sample.label for sample in samples),
    }
    if labels != PAPER_LABELS:
        raise AssertionError(f"REMed Table 7 LOS-7 labels differ: {labels}")

    event_length = distribution([sample.event_count for sample in samples])
    token_length = distribution([sample.token_count for sample in samples])
    assert_paper_distribution(event_length, PAPER_EVENT_LENGTH, "event length")
    token_length_mismatches = distribution_mismatches(
        token_length, PAPER_TOKEN_LENGTH
    )
    if any(
        abs(record["relative_difference"]) > 0.001
        for record in token_length_mismatches.values()
    ):
        raise AssertionError(
            "REMed reconstructed token-length distribution differs from the "
            f"paper by more than 0.1%: {token_length_mismatches}"
        )

    overlap = cross_split_subject_overlap(samples)
    if any(overlap.values()):
        raise AssertionError(f"REMed split_2020 has cross-split patients: {overlap}")
    summaries = split_summary(samples)
    index_path = args.task_dir / manifest["meds"]["task_index"]["file"]
    index_rows = sorted(
        pq.read_table(index_path).to_pylist(), key=lambda row: row["source_position"]
    )
    if len(index_rows) != len(samples):
        raise AssertionError("MEDS changed the REMed cohort size")
    for expected, actual in zip(samples, index_rows, strict=True):
        expected_fields = {
            "subject_id": expected.subject_id,
            "sample_id": expected.stay_id,
            "hadm_id": expected.hadm_id,
            "split": expected.split,
            "boolean_value": expected.label,
            "source_position": expected.source_position,
            "source_event_count": expected.event_count,
            "source_token_count": expected.token_count,
        }
        for field, value in expected_fields.items():
            if actual[field] != value:
                raise AssertionError(f"MEDS changed REMed cohort field {field}")
    if summaries != manifest["splits"]:
        raise AssertionError("manifest split marginals differ from exact source")
    if manifest["cross_split_subject_overlap"] != {
        **overlap,
        "subjects_in_multiple_splits": 0,
    }:
        raise AssertionError("manifest subject-overlap audit differs")

    report = {
        "status": "exact_source_with_small_paper_token_length_mismatch",
        "selected_paper_cell": source["selected_paper_cell"],
        "published_auroc": PUBLISHED_AUROC,
        "paper_evidence": {
            "table_4": PAPER_COUNTS,
            "table_7_48h_los_7": PAPER_LABELS,
            "table_3_48h_infinite_event_length": PAPER_EVENT_LENGTH,
            "table_3_48h_infinite_token_length": PAPER_TOKEN_LENGTH,
        },
        "reconstructed": {
            "cohort": counts,
            "labels": labels,
            "event_length": event_length,
            "token_length": token_length,
            "splits": summaries,
            "cross_split_subject_overlap": overlap,
            "observation_window_hours": observation_hours,
        },
        "paper_evidence_mismatches": {
            "table_3_48h_infinite_token_length": token_length_mismatches,
        },
        "ordered_sample_id_sha256": hash_ints(
            sample.stay_id for sample in samples
        ),
        "ordered_subject_id_sha256": hash_ints(
            sample.subject_id for sample in samples
        ),
        "ordered_label_sha256": hash_ints(int(sample.label) for sample in samples),
        "selected_cohort_identifier_sha256": hash_ints(
            sorted(sample.stay_id for sample in samples)
        ),
        "checks": [
            "Table 4 patient/admission/ICU-stay counts",
            "Table 7 48-hour LOS-7 label counts",
            "exact Table 3 48-hour infinite-history event distribution",
            "bounded and explicitly reported Table 3 token-distribution mismatch",
            "exact split_2020 membership and per-split labels",
            "patient-disjoint released split",
            "structured-only modality declaration",
        ],
        "qualification": (
            "The exact rebuilt HDF5 cohort matches the paper's cohort, labels, and "
            "event-length distribution. Its token summaries differ from Table 3 "
            "by at most 0.1%, consistent with an unreleased paper-run tokenization "
            "artifact; the exact differences are retained above."
        ),
    }
    output = args.task_dir / "audit" / "cohort_reconstruction.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
