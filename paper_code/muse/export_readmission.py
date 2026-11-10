"""Export MUSE's second released MIMIC-IV task: 15-day readmission to MEDS.

The retained boundary, event construction, vocabulary, modality projection, and
note handling are byte-identical to the sealed mortality export; only the
released label attribute and the released `task:readmission` split differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from export_source import _lookup_frames, event_batches, samples_frame  # noqa: E402
from cohort_source import (  # noqa: E402
    AUTHOR_BATCH_SIZE,
    AUTHOR_MISSING_LAB_WIDTH,
    NOTE_CODE,
    OBSERVATION_HOURS,
    lab_channels,
    sha256_file,
    source_codes,
)
from cohort_readmission import (  # noqa: E402
    AUTHOR_TASK,
    LABEL_COLUMN,
    PUBLISHED_AUPRC_MUSE,
    PUBLISHED_AUPRC_MUSE_PLUS,
    PUBLISHED_AUROC,
    PUBLISHED_AUROC_MUSE_PLUS,
    SELECTED_PAPER_CELL,
    TASK_ID,
    TASK_NAME,
    load_source,
    readmission_cohort,
    write_split_files,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--note-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = load_source(args.mimic_root, args.note_root, args.cache_root)
    labeled, cohort_detail = readmission_cohort(args.cache_root)
    summary = cohort_detail["summary"]
    split_files = write_split_files(args.cache_root, cohort_detail["splits"])
    selected, lookup = _lookup_frames(args.cache_root, labeled)
    channels = lab_channels(args.cache_root)
    author_width = 2 * len(channels["itemids"])
    historical_archive = (
        PAPER_ROOT
        / ".export_assets/external_artifacts/github/"
        "MUSE-3596fd3d5d7d79864e26a80d786ba8c6dc2fbd9b.tar.gz"
    )
    codes = source_codes(args.cache_root)
    source = {
        "upstream_commit": source_manifest["upstream_commit"],
        "historical_producer_commit": "3596fd3d5d7d79864e26a80d786ba8c6dc2fbd9b",
        "historical_producer_archive_sha256": sha256_file(historical_archive),
        "selected_paper_cell": SELECTED_PAPER_CELL,
        "published_cells": {
            "muse": {
                "auroc": PUBLISHED_AUROC,
                "auprc": PUBLISHED_AUPRC_MUSE,
                "location": "Table 1, MIMIC-IV Readmission, MUSE",
                "role": "artifact comparator (labeled-patient supervised branch)",
            },
            "muse_plus": {
                "auroc": PUBLISHED_AUROC_MUSE_PLUS,
                "auprc": PUBLISHED_AUPRC_MUSE_PLUS,
                "location": "Table 1, MIMIC-IV Readmission, MUSE+",
                "role": (
                    "paper headline proposed model; deferred here because its "
                    "semi-supervised branch additionally trains on the "
                    "unlabeled admissions the common supervised task must not "
                    "archive"
                ),
            },
        },
        "comparator_adjudication": (
            "Inherited from the sealed mortality export: the archived "
            "task is the labeled-patient MUSE configuration "
            "(load_no_label=False), so the artifact's published_auroc is the "
            "MUSE cell, exactly as paper_tasks/muse/"
            "mortality_90d_post_discharge carries 0.9004 rather than MUSE+'s "
            "0.9201."
        ),
        "selected_model_inputs": [
            "demographics_and_medical_codes",
            "lab_values",
            "clinical_notes",
        ],
        "selection_tier": 2,
        "retained_modalities": ["structured", "notes"],
        "source_configuration": {
            "dataset": "mimic4",
            "task": AUTHOR_TASK,
            "load_no_label": False,
            "seed": 42,
            "batch_size": AUTHOR_BATCH_SIZE,
            "split": (
                "released admission shuffle 70/10/20, generated after the "
                "mortality shuffle on the same seeded RNG stream"
            ),
            "prediction_cutoff": "hospital discharge",
            "positive_class": (
                "next hospital admission begins at most 15 floored days after "
                "this discharge"
            ),
        },
        "released_label_definition": {
            "producer": (
                "src/preprocess/preprocess_mimic4.py::"
                "process_patients_admissions:43-47"
            ),
            "rule": (
                "readmission = float(readmit_in_days) <= 15 when the notebook's "
                "next-admission gap exists, otherwise the label is missing"
            ),
            "gap_source": (
                "src/preprocess/parse_mimic4_remote.ipynb: "
                "(next admittime - dischtime).dt.days over all admissions of "
                "the subject, floored to whole days"
            ),
            "missing_label_meaning": "the admission is the subject's last visit",
            "prediction_time": "hospital discharge (dischtime)",
        },
        "capture_boundary": {
            "codes": (
                "released last-50 aligned type/code trajectory before vocabulary "
                "indexing; deterministic synthetic event time retains only position"
            ),
            "labs": (
                "lossless sparse-delta parameterization of the released last-50 "
                "forward-filled float32 value/abnormal-flag tensor, with explicit "
                "sequence-row markers; zeros remain authored values"
            ),
            "notes": (
                "raw released Brief Hospital Course text after exact section "
                "extraction and last-row-wins admission attachment, before TinyBERT"
            ),
            "demographics": "released decade bucket, gender, and ethnicity categories",
        },
        "missing_modality_membership": {
            "lab_presence_event": True,
            "note_presence": "exactly the presence of a raw note row, including empty text",
            "codes_flag": "constant true in the released loader",
        },
        "author_lab_boundary": {
            "selected_dictionary_items": channels["selected_dictionary_items"],
            "retained_itemids": channels["itemids"],
            "nonmissing_width": author_width,
            "missing_fallback_width": AUTHOR_MISSING_LAB_WIDTH,
            "model_gru_width": AUTHOR_MISSING_LAB_WIDTH,
            "exact_width_match": author_width == AUTHOR_MISSING_LAB_WIDTH,
        },
        "released_code_vocabulary_sha256": sha256_file(
            args.cache_root / "author_code_vocabulary.json"
        ),
        "released_code_vocabulary_size": source_manifest["codes"][
            "author_vocabulary_size"
        ],
        "meds_event_vocabulary_size": len(codes),
        "readmission_cohort": summary,
        "readmission_split_files": split_files,
        "source_reconstruction": source_manifest,
        "shared_base_task": "paper_tasks/muse/mortality_90d_post_discharge",
        "known_source_defects_preserved": [
            "admission-level random split permits subjects in multiple partitions",
            "whole-cohort lab normalization and code vocabulary precede splitting",
            "MIMIC-IV 2.0 HOSP is joined to MIMIC-IV-Note 2.2 by hadm_id",
            "note charttime/storetime are discarded by the author model path",
            "74 nonpositive-duration admissions are retained by the released one-sided LOS filter",
            "the readmission gap is floored to whole days, so gaps up to 15.999 days are positive",
            "admissions whose next admission begins before this discharge get a negative gap and are still labeled positive",
            "the released labeler never requires the next admission to start after discharge",
            "readmission labels exist only for admissions that have a later admission, so the labeled cohort is selected on a post-discharge fact",
        ],
    }
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "muse",
            "title": (
                "Multimodal Patient Representation Learning with Missing "
                "Modalities and Labels"
            ),
        },
        task={
            "id": TASK_ID,
            "name": TASK_NAME,
            "primary_metric": "AUROC",
            "published_auroc": PUBLISHED_AUROC,
            "published_auroc_muse_plus": PUBLISHED_AUROC_MUSE_PLUS,
            "published_auprc": PUBLISHED_AUPRC_MUSE,
            "published_auprc_muse_plus": PUBLISHED_AUPRC_MUSE_PLUS,
            "published_source": "Table 1, MIMIC-IV Readmission",
            "label_definition": (
                "positive when the subject's next hospital admission begins at "
                "most 15 floored days after this discharge; the label is "
                "missing for the subject's last admission. Prediction time is "
                "hospital discharge."
            ),
            "observation_window_hours": float(OBSERVATION_HOURS),
        },
        codes=codes,
        note_codes=(NOTE_CODE,),
        events=event_batches(selected, lookup, args.cache_root),
        samples=samples_frame(selected, LABEL_COLUMN),
        source=source,
        dataset_name="MUSE MIMIC-IV 15-day readmission structured-plus-notes",
        dataset_version="MIMIC-IV 2.0 / MIMIC-IV-Note 2.2 / MUSE released path",
    )
    print(
        json.dumps(
            {
                "task": str(args.output_dir),
                "samples": len(selected),
                "events": manifest["event_count"],
                "splits": manifest["splits"],
                "exact_lab_width_match": author_width == AUTHOR_MISSING_LAB_WIDTH,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
