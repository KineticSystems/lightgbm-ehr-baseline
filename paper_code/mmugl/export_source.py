"""Export MMUGL next-visit heart failure directly to the shared MEDS task."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from cohort_source import (  # noqa: E402
    MAX_CODES_PER_VISIT,
    MAX_NOTE_CONCEPTS,
    MAX_VISITS,
    NOTE_CODE,
    PUBLISHED_AUROC,
    TASK_ID,
    build_samples,
    cross_split_overlap,
    event_batches,
    hash_strings,
    load_author_boundary,
    sample_frame,
    split_summary,
    structured_event_count,
)


def hash_ints(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    boundary = load_author_boundary(args.mimic_root, args.mapping_root)
    samples, observation_hours = build_samples(boundary)
    summaries = split_summary(samples)
    note_stats = {}
    source = {
        "upstream_commit": "73556144b14958d41b1fc1d7541c61b51fedf8b3",
        "loader_entrypoint": "kg.data.datasets.CglDatasetDownstream",
        "capture_boundary": (
            "CglDatasetDownstream.patient_data structured projection, with HADM_ID "
            "captured immediately before its released drop; raw notes selected from "
            "the released load_noteevents_table eligibility path"
        ),
        "selected_paper_cell": "Table 2, CGL split, MMUGL heart-failure AuROC",
        "selected_model_inputs": ["structured", "notes"],
        "selection_tier": 2,
        "published_auroc": PUBLISHED_AUROC,
        "split_policy": (
            "released cgl_patient_splits validation/test IDs; training is every "
            "diagnosis-table patient outside validation/test"
        ),
        "transition_policy": (
            "every retained transition for training; final transition among the "
            "earliest 42 visits for validation/test"
        ),
        "split_summary": summaries,
        "cross_split_subject_overlap": cross_split_overlap(samples),
        "ordered_sample_id_sha256": hash_ints([row["sample_id"] for row in samples]),
        "ordered_label_sha256": hash_ints([int(row["boolean_value"]) for row in samples]),
        "released_icd_vocabulary_sha256": hash_strings(boundary["icd"]),
        "released_atc_vocabulary_sha256": hash_strings(boundary["atc"]),
        "released_target_vocabulary_sha256": hash_strings(boundary["target"]),
        "source_artifacts": boundary["inventory"],
        "batch_recreation": {
            "max_codes_per_visit_before_cls": MAX_CODES_PER_VISIT,
            "patient_sequence_length": MAX_VISITS,
            "max_visit_text_length": MAX_NOTE_CONCEPTS,
            "code_shuffle": True,
            "random_state": 1111,
            "retained_modalities": ["structured", "notes"],
            "omitted_fields": [
                "UMLS CUI tokens",
                "note negation token attributes",
                "UMLS graph edges",
                "SapBERT embeddings",
            ],
            "unresolved_full_model_parity": (
                "the paper did not identify the licensed UMLS release or QuickUMLS "
                "index; the retained structured/raw-note projection is exact, but "
                "the CUI/graph branch cannot be claimed byte-identical"
            ),
        },
        "raw_note_projection": note_stats,
        "structured_event_count": structured_event_count(samples),
        "maximum_observation_window_hours": observation_hours,
        "observation_alignment": (
            "each sample retains its exact released history start; the temporal "
            "featurizer uses a label-free prediction-anchored grid whose fixed "
            "absolute boundaries are fitted from structured-row percentiles across "
            "all samples and splits"
        ),
    }
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "mmugl",
            "title": "Multi-modal Graph Learning over UMLS Knowledge Graphs",
        },
        task={
            "id": TASK_ID,
            "name": "Next-visit heart failure from prior visits",
            "primary_metric": "AUROC",
            "published_auroc": PUBLISHED_AUROC,
            "observation_window_hours": observation_hours,
            # Visit-level timestamps; no within-visit ordering is released.
            # Visit bucketing derives its boundaries from each sample's own
            # event times, so no cohort-fitted absolute grid is needed.
            "temporal_profile": "multi_visit_untimed",
        },
        codes=boundary["codes"],
        note_codes=(NOTE_CODE,),
        events=event_batches(boundary, samples, note_stats),
        samples=sample_frame(samples),
        source=source,
        dataset_name="MMUGL CGL-split MIMIC-III next-visit heart failure",
        dataset_version="MIMIC-III 1.4 / released MMUGL CGL producer",
    )
    print(
        json.dumps(
            {
                "task": TASK_ID,
                "samples": len(samples),
                "events": manifest["event_count"],
                "splits": summaries,
                "notes": note_stats,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
