#!/usr/bin/env python3
"""Export GraphCare's exact reconstructable structured PyHealth boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Sequence

import polars as pl

from cohort_source import (
    AuthorSource,
    SourceRecord,
    event_time,
    expected_event_rows,
    load_author_source,
    meds_codes,
    sample_time_bounds,
    source_artifact_hashes,
    source_summary,
)


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402


TASK_ID = "subsequent_visit_mortality"
PUBLISHED_AUROC = 0.703


def sample_frame(source: AuthorSource) -> pl.DataFrame:
    start, prediction = sample_time_bounds(source.max_history)
    records = source.records
    return pl.DataFrame(
        {
            "subject_id": [int(record.sample["patient_id"]) for record in records],
            "sample_id": [int(record.sample["visit_id"]) for record in records],
            "observation_start": [start] * len(records),
            "prediction_time": [prediction] * len(records),
            "boolean_value": [bool(record.sample["label"]) for record in records],
            "split": [record.split for record in records],
            "current_visit_id": [
                int(record.sample["visit_id"]) for record in records
            ],
            "target_visit_id": [int(record.target_visit_id) for record in records],
            "source_index": [record.source_index for record in records],
            "source_split_position": [record.split_position for record in records],
            "history_length": [
                len(record.sample["conditions"]) for record in records
            ],
        },
        schema={
            "subject_id": pl.Int64,
            "sample_id": pl.Int64,
            "observation_start": pl.Datetime("us"),
            "prediction_time": pl.Datetime("us"),
            "boolean_value": pl.Boolean,
            "split": pl.String,
            "current_visit_id": pl.Int64,
            "target_visit_id": pl.Int64,
            "source_index": pl.Int64,
            "source_split_position": pl.Int64,
            "history_length": pl.Int32,
        },
    )


def event_batch(records: Sequence[SourceRecord], max_history: int) -> pl.DataFrame:
    values = {
        "subject_id": [],
        "time": [],
        "code": [],
        "numeric_value": [],
        "sample_id": [],
        "source_row": [],
        "sequence_row": [],
        "source_time": [],
    }
    for record in records:
        for source_row, sequence_row, source_time, code in expected_event_rows(
            record, max_history
        ):
            values["subject_id"].append(int(record.sample["patient_id"]))
            values["time"].append(event_time(sequence_row))
            values["code"].append(code)
            values["numeric_value"].append(None)
            values["sample_id"].append(int(record.sample["visit_id"]))
            values["source_row"].append(source_row)
            values["sequence_row"].append(sequence_row)
            values["source_time"].append(source_time)
    return pl.DataFrame(
        values,
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime("us"),
            "code": pl.String,
            "numeric_value": pl.Float32,
            "sample_id": pl.Int64,
            "source_row": pl.Int64,
            "sequence_row": pl.Int32,
            "source_time": pl.Float64,
        },
    )


def event_batches(
    records: Sequence[SourceRecord], max_history: int, size: int = 2_000
) -> Iterator[pl.DataFrame]:
    for start in range(0, len(records), size):
        yield event_batch(records[start : start + size], max_history)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = load_author_source(PAPER_ROOT, args.mimic_root)
    upstream = json.loads((PAPER_ROOT / "UPSTREAM.json").read_text())
    summary = source_summary(source, PAPER_ROOT)
    artifact_hashes = source_artifact_hashes(PAPER_ROOT, args.mimic_root)
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "graphcare",
            "title": (
                "GraphCare: Enhancing Healthcare Predictions with "
                "Personalized Knowledge Graphs"
            ),
        },
        task={
            "id": TASK_ID,
            "name": "MIMIC-III subsequent-visit mortality",
            "primary_metric": "AUROC",
            "published_auroc": PUBLISHED_AUROC,
            "positive_class": "death at discharge from the subsequent admission",
            "observation_window_hours": float(source.max_history),
            # Every event in a visit carries that visit's single
            # right-aligned slot time, so distinct event time is exactly the
            # visit key and there is no ordering inside a visit.
            "temporal_profile": "multi_visit_untimed",
        },
        codes=meds_codes(PAPER_ROOT),
        note_codes=(),
        events=event_batches(source.records, source.max_history),
        samples=sample_frame(source),
        source={
            "upstream": upstream["repositories"][0],
            "mimic_release": "MIMIC-III 1.4",
            "pyhealth_version": "1.1.2",
            "source_protocol": "single released patient split",
            "selected_experiment": (
                "Table 2 MIMIC-III mortality, GraphCare with BAT"
            ),
            "selected_model_inputs": [
                "conditions",
                "procedures",
                "drugs",
                "external personalized knowledge graph",
            ],
            "selection_tier": 1,
            "retained_projection": ["conditions", "procedures", "drugs"],
            "common_profile": "categorical_occurrence",
            "source_boundary": (
                "MIMIC3Dataset.set_task(mortality_prediction_mimic3_fn) "
                "three aligned cumulative categorical visit histories"
            ),
            "split_boundary": (
                "graphcare_.splitter.split_by_patient([0.8,0.1,0.1], "
                "train_ratio=1.0, seed=528)"
            ),
            "split_assignment_unit": "patient_id",
            "time_projection": {
                "kind": "right-aligned visit ordinal",
                "reason": (
                    "the released model boundary retains visit order and "
                    "whole-cohort maximum padding length but no clinical "
                    "timestamps; real time gaps are intentionally not exposed"
                ),
                "event_time": (
                    "1970-01-01 plus the right-aligned chronological visit slot"
                ),
                "prediction_time": (
                    "1970-01-01 plus the released whole-cohort maximum history"
                ),
            },
            "vocabulary": summary["vocabulary_sizes"],
            "ordered_vocabulary_sha256": summary[
                "ordered_meds_vocabulary_sha256"
            ],
            "author_source_summary": summary,
            "source_artifacts": artifact_hashes,
            "excluded_model_inputs": {
                "knowledge_graph": (
                    "not retained: the complete graph/embedding boundary is "
                    "unreleased and is not a raw structured EHR input"
                ),
                "notes": "not used by the selected method",
                "imaging": "not used by the selected method",
            },
            "comparison_scope": (
                "information-asymmetric structured-only LightGBM comparison; "
                "not a reconstruction of GraphCare's external-KG input"
            ),
            "residual_source_uncertainty": (
                "the paper-time PyHealth crosswalk bytes were not released; "
                "the audited current unversioned payload is pinned by SHA-256"
            ),
        },
        dataset_name="GraphCare PyHealth structured MIMIC-III mortality boundary",
        dataset_version="MIMIC-III 1.4 / PyHealth 1.1.2 / GraphCare fe92aa67",
    )
    print(
        json.dumps(
            {
                "task": str(args.output_dir),
                "samples": len(source.records),
                "events": manifest["event_count"],
                "splits": manifest["splits"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
