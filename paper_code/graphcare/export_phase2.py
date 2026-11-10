#!/usr/bin/env python3
"""Export GraphCare's three remaining published binary outcomes (Table 2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Sequence

import polars as pl

from cohort_phase2 import (
    DATASETS,
    Phase2Record,
    Phase2Source,
    dataset_cache_dir,
    event_time,
    load_phase2_source,
    meds_codes,
    phase2_artifact_hashes,
    phase2_event_rows,
    phase2_source_summary,
    readmission_defect_report,
    sample_time_bounds,
)


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402


TASK_SPECS = {
    "readmission_15d": {
        "dataset": "mimic3",
        "outcome": "readmission",
        "name": "MIMIC-III 15-day readmission",
        "published_auroc": 0.697,
        "published_cell": (
            "Table 2, Task 2 Readmission Prediction, MIMIC-III, "
            "GraphCare w/ BAT AUROC 69.7"
        ),
        "positive_class": (
            "the next admission begins fewer than 15 days after the current "
            "admission begins"
        ),
        "task_function": "readmission_prediction_mimic3_fn",
        "load_dataset_call": 'data_prepare.load_dataset(False, "mimic3", "readmission")',
    },
    "mimic4_subsequent_visit_mortality": {
        "dataset": "mimic4",
        "outcome": "mortality",
        "name": "MIMIC-IV subsequent-visit mortality",
        "published_auroc": 0.731,
        "published_cell": (
            "Table 2, Task 1 Mortality Prediction, MIMIC-IV, "
            "GraphCare w/ BAT AUROC 73.1"
        ),
        "positive_class": "death at discharge from the subsequent admission",
        "task_function": "mortality_prediction_mimic4_fn",
        "load_dataset_call": 'data_prepare.load_dataset(False, "mimic4", "mortality")',
    },
    "mimic4_readmission_15d": {
        "dataset": "mimic4",
        "outcome": "readmission",
        "name": "MIMIC-IV 15-day readmission",
        "published_auroc": 0.685,
        "published_cell": (
            "Table 2, Task 2 Readmission Prediction, MIMIC-IV, "
            "GraphCare w/ BAT AUROC 68.5"
        ),
        "positive_class": (
            "the next admission begins fewer than 15 days after the current "
            "admission begins"
        ),
        "task_function": "readmission_prediction_mimic4_fn",
        "load_dataset_call": (
            'data_prepare.load_dataset(False, "mimic4", "readmission")'
        ),
    },
}


def sample_frame(source: Phase2Source) -> pl.DataFrame:
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
            "current_visit_id": [int(record.sample["visit_id"]) for record in records],
            "target_visit_id": [int(record.target_visit_id) for record in records],
            "source_index": [record.source_index for record in records],
            "source_split_position": [record.split_position for record in records],
            "history_length": [len(record.sample["conditions"]) for record in records],
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


def event_batch(records: Sequence[Phase2Record], max_history: int) -> pl.DataFrame:
    values = {
        "subject_id": [],
        "time": [],
        "code": [],
        "numeric_value": [],
        "sample_id": [],
        "source_row": [],
        "sequence_row": [],
        "source_time": [],
        "visit_id": [],
    }
    for record in records:
        subject_id = int(record.sample["patient_id"])
        sample_id = int(record.sample["visit_id"])
        for (
            source_row,
            sequence_row,
            source_time,
            code,
            visit_id,
        ) in phase2_event_rows(record, max_history):
            values["subject_id"].append(subject_id)
            values["time"].append(event_time(sequence_row))
            values["code"].append(code)
            values["numeric_value"].append(None)
            values["sample_id"].append(sample_id)
            values["source_row"].append(source_row)
            values["sequence_row"].append(sequence_row)
            values["source_time"].append(source_time)
            values["visit_id"].append(visit_id)
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
            "visit_id": pl.Int64,
        },
    )


def event_batches(
    records: Sequence[Phase2Record], max_history: int, size: int = 2_000
) -> Iterator[pl.DataFrame]:
    for start in range(0, len(records), size):
        yield event_batch(records[start : start + size], max_history)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_SPECS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    spec = TASK_SPECS[args.task]
    dataset = spec["dataset"]
    shards = DATASETS[dataset]["shards"]
    cache_dir = dataset_cache_dir(PAPER_ROOT, dataset)
    source = load_phase2_source(
        PAPER_ROOT, dataset, cache_dir, shards, spec["outcome"]
    )
    upstream = json.loads((PAPER_ROOT / "UPSTREAM.json").read_text())
    summary = phase2_source_summary(source, PAPER_ROOT)
    artifact_hashes = phase2_artifact_hashes(PAPER_ROOT, dataset, args.root)

    disclosed = {
        "released_test_evaluation": (
            "no released proposed-model script evaluates the test loader; the "
            "published cell cannot be traced to a released run "
            "(paper_audits/graphcare.md item 1)"
        ),
        "whole_cohort_preprocessing": (
            "the released graph/vocabulary construction observes the whole "
            "sample dataset before the split (paper_audits/graphcare.md item 4)"
        ),
        "drop_last_evaluation": (
            "released validation/test loaders use drop_last=True, so the "
            "authors' scored cohort is batch-size dependent"
        ),
        "unreleased_model_input": (
            "the personalized knowledge graph, embeddings and cluster records "
            "are unreleased; this export retains only the reconstructable "
            "structured EHR branch"
        ),
    }
    if spec["outcome"] == "readmission":
        defects = readmission_defect_report(source)
        disclosed["readmission_label"] = defects["released_label"]
        disclosed["readmission_code_paper_mismatch"] = defects["code_paper_mismatch"]
        disclosed["readmission_prediction_point"] = defects[
            "prediction_point_defect"
        ]
        disclosed["released_visit_order"] = defects["visit_order_defect"]
    if dataset == "mimic4":
        disclosed["mimic4_history_ambiguity"] = (
            "the paper-era MIMIC-IV task functions emitted a single-visit "
            "history; post-publication commit dbedef01 (2024-10-13, 'rollback "
            "task functions') restored the cumulative history that the audited "
            "commit fe92aa67 contains and that this export runs "
            "(paper_audits/graphcare.md, 'MIMIC-IV paper-era history')"
        )

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
            "id": args.task,
            "name": spec["name"],
            "primary_metric": "AUROC",
            "published_auroc": spec["published_auroc"],
            "published_cell": spec["published_cell"],
            "positive_class": spec["positive_class"],
            "label_definition": (
                "released `mortality_prediction_%s_fn`: the label is the "
                "subsequent visit's `discharge_status` (0 when it is neither "
                "0 nor 1)" % dataset
                if spec["outcome"] == "mortality"
                else "released `readmission_prediction_%s_fn` with "
                "`time_window=15`: the label is 1 when "
                "`(next_visit.encounter_time - visit.encounter_time).days < 15`"
                % dataset
            ),
            "observation_window_hours": float(source.max_history),
            # Every event in a visit carries that visit's single right-aligned
            # slot time, so distinct event time is exactly the visit key and
            # there is no ordering inside a visit.
            "temporal_profile": "multi_visit_untimed",
        },
        codes=meds_codes(PAPER_ROOT),
        note_codes=(),
        events=event_batches(source.records, source.max_history),
        samples=sample_frame(source),
        source={
            "upstream": upstream["repositories"][0],
            "mimic_release": DATASETS[dataset]["release"],
            "pyhealth_version": "1.1.2",
            "source_protocol": "single released patient split",
            "selected_experiment": spec["published_cell"],
            "released_entry_point": spec["load_dataset_call"],
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
                f"{DATASETS[dataset]['class']}.set_task({spec['task_function']}) "
                "three aligned cumulative categorical visit histories"
            ),
            "source_execution": (
                "the released parser methods are executed over disjoint "
                "patient shards so the parse fits the 7 GiB per-process "
                "budget; see cohort_phase2.build_parse_cache"
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
            "ordered_vocabulary_sha256": summary["ordered_meds_vocabulary_sha256"],
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
            "disclosed_source_defects": disclosed,
            "residual_source_uncertainty": (
                "the paper-time PyHealth crosswalk bytes were not released; "
                "the audited current unversioned payload is pinned by SHA-256"
            ),
        },
        dataset_name=(
            f"GraphCare PyHealth structured {DATASETS[dataset]['release']} "
            f"{spec['outcome']} boundary"
        ),
        dataset_version=(
            f"{DATASETS[dataset]['release']} / PyHealth 1.1.2 / GraphCare fe92aa67"
        ),
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
