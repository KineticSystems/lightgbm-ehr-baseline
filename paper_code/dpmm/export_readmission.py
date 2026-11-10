"""Export DPMM's MIMIC-III partial EHR-note 30-day readmission experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from export_source import event_batches, samples_frame  # noqa: E402
from cohort_source import (  # noqa: E402
    DISCRETIZER_CONFIG,
    HOURS,
    NOTE_CODE,
    TIMESTEP,
    TOKENIZER,
    UPSTREAM_COMMIT,
    ensure_source,
    ordered_text_hash,
    sha256,
    source_codes,
)
from cohort_readmission import (  # noqa: E402
    EXPECTED_COUNTS,
    EXPECTED_PAIRED,
    EXPECTED_PAIRED_POSITIVES,
    EXPECTED_POSITIVES,
    PUBLISHED_AUROC,
    PUBLISHED_CELL,
    PUBLISHED_PAIRED_POSITIVES,
    PUBLISHED_PAIRED_TOTAL,
    PUBLISHED_TOTAL_POSITIVES,
    TASK_ID,
    ensure_readmission_labels,
    readmission_cohort,
    write_readmission_author_view,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--readmission-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = ensure_source(args.mimic_root, args.cache_root)
    readm_manifest = ensure_readmission_labels(
        args.mimic_root, args.readmission_cache
    )
    base_cohort = pd.read_parquet(
        args.cache_root / source_manifest["files"]["cohort"]
    )
    labels = pd.read_parquet(
        args.readmission_cache / readm_manifest["files"]["readmission_labels"]
    )
    cohort = readmission_cohort(base_cohort, labels)
    write_readmission_author_view(args.cache_root, args.readmission_cache, cohort)
    notes = pd.read_parquet(
        args.cache_root / source_manifest["files"]["selected_notes"]
    )
    structured_path = (
        args.cache_root / source_manifest["files"]["structured_events"]
    )

    counts = cohort["split"].value_counts().to_dict()
    positives = cohort.groupby("split")["label"].sum().astype(int).to_dict()
    paired = cohort.loc[cohort["note_present"]].groupby("split").size().to_dict()
    paired_positives = (
        cohort.loc[cohort["note_present"]]
        .groupby("split")["label"]
        .sum()
        .astype(int)
        .to_dict()
    )
    if counts != EXPECTED_COUNTS:
        raise AssertionError(f"DPMM readmission cohort counts differ: {counts}")
    if positives != EXPECTED_POSITIVES:
        raise AssertionError(f"DPMM readmission positives differ: {positives}")
    if paired != EXPECTED_PAIRED:
        raise AssertionError(f"DPMM paired-note counts differ: {paired}")
    if paired_positives != EXPECTED_PAIRED_POSITIVES:
        raise AssertionError(
            f"DPMM paired readmission positives differ: {paired_positives}"
        )

    codes = source_codes()
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "dpmm",
            "title": (
                "Amplifying Prominent Representations in Multimodal Learning "
                "via Variational Dirichlet Process"
            ),
        },
        task={
            "id": TASK_ID,
            "name": (
                "MIMIC-III 30-day hospital readmission from the first 48 ICU hours"
            ),
            "primary_metric": "AUROC",
            "published_auroc": PUBLISHED_AUROC,
            "observation_window_hours": HOURS,
        },
        codes=codes,
        note_codes=(NOTE_CODE,),
        events=event_batches(cohort, structured_path, notes),
        samples=samples_frame(cohort),
        source={
            "upstream_commit": UPSTREAM_COMMIT,
            "selected_paper_cell": PUBLISHED_CELL,
            "published_auroc_interval": [0.734, 0.780],
            "published_aupr": 0.445,
            "published_f1": 0.460,
            "selected_model_inputs": ["structured", "notes"],
            "selection_tier": 2,
            "retained_modalities": ["structured", "notes"],
            "source_configuration": {
                "task": "readmission",
                "labels_set": "readm",
                "data_pairs": "partial_ehr_note",
                "data_ratio": 1.0,
                "timestep": TIMESTEP,
                "store_masks": True,
                "impute_strategy": "previous",
                "start_time": "zero",
                "bert_type": "huawei-noah/TinyBERT_General_4L_312D",
                "batch_size": 16,
                "launcher": "scripts/readmission/mimic3/partial/dp.sh",
            },
            "label_producer": (
                "mimic3extract/mimic3benchmark/scripts/create_readmission_30d.py"
            ),
            "label_definition": readm_manifest["label_definition"],
            "positive_class": (
                "next hospital admission within 30 days of discharge, or "
                "in-hospital death during this admission"
            ),
            "prediction_time": "48 hours after ICU admission (released n_hours=48)",
            "structured_capture_boundary": (
                "released cleaned episode time series before 2-hour discretization, "
                "previous imputation, and validation-leaked normalization"
            ),
            "note_capture_boundary": (
                "released cleaned raw strings for the last five loader-selected "
                "note timestamps, before TinyBERT tokenization"
            ),
            "counts": EXPECTED_COUNTS,
            "positives": EXPECTED_POSITIVES,
            "paired_note_counts": EXPECTED_PAIRED,
            "paired_positives": EXPECTED_PAIRED_POSITIVES,
            "published_total_positives": PUBLISHED_TOTAL_POSITIVES,
            "released_source_total_positives": sum(EXPECTED_POSITIVES.values()),
            "published_paired_positives": PUBLISHED_PAIRED_POSITIVES,
            "published_paired_total_positives": PUBLISHED_PAIRED_TOTAL,
            "released_source_paired_total_positives": sum(
                EXPECTED_PAIRED_POSITIVES.values()
            ),
            "partial_cohort_count": len(cohort),
            "note_rows": len(notes),
            "note_text_sha256": ordered_text_hash(notes["text"]),
            "source_name_sha256": ordered_text_hash(cohort["source_name"]),
            "released_code_vocabulary": list(codes),
            "discretizer_config_sha256": sha256(DISCRETIZER_CONFIG),
            "tokenizer_path": str(TOKENIZER.relative_to(PAPER_ROOT)),
            "tokenizer_revision": "34707a33cd59a94ecde241ac209bf35103691b43",
            "source_reconstruction": source_manifest,
            "readmission_label_reconstruction": readm_manifest,
            "shares_event_stream_with": "mortality_48h_partial_ehr_note",
            "known_source_defects_preserved": [
                "create_readmission_30d.py iterates a sorted DataFrame's index "
                "labels but selects the next stay with .iloc, and detects the "
                "last stay with idx == len-1; measured latent on this cohort "
                "(0 of 5,240 multi-stay subjects reorder, 0 stays change)",
                "the 30-day rule has no lower bound and uses whole-day flooring, "
                "so one negative-gap stay (263279, -450.4 h) is labeled positive",
                "in-hospital death overrides the readmission outcome to positive "
                "(supplement definition; the main-body sentence omits it)",
                "note loader rejects exactly five eligible notes by using <= 5",
                "note cutoff is anchored to the first retained event at least one "
                "full hour after ICU admission",
                "note lower-bound check is disabled and the shifted 48-hour cutoff "
                "can include post-ICU-48-hour text",
                "normalizer moments include the later validation partition",
                "published paired prevalence sums a shuffled drop-last training "
                "loader rather than the full cohort",
                "released training exposes test AUROC every epoch (not used here)",
            ],
            "note_time_serialization": (
                "source_time retains exact hours relative to ICU admission; the MEDS "
                "time coordinate is clamped to [0,48] only because the model uses "
                "note slot order/content and the common schema forbids out-of-window events"
            ),
        },
        dataset_name="DPMM MIMIC-III partial EHR-note 30-day readmission",
        dataset_version="MIMIC-III 1.4 / DPMM released HEAD",
    )
    print(
        json.dumps(
            {
                "task": str(args.output_dir),
                "samples": len(cohort),
                "events": manifest["event_count"],
                "splits": manifest["splits"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
