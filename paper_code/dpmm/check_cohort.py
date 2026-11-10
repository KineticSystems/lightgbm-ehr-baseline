"""Assert DPMM's exact selected partial EHR-note cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from cohort_source import (  # noqa: E402
    EXPECTED_COUNTS,
    EXPECTED_PAIRED,
    EXPECTED_PAIRED_POSITIVES,
    EXPECTED_POSITIVES,
    NOTE_CODE,
    PUBLISHED_TOTAL_POSITIVES,
    PUBLISHED_PAIRED_POSITIVES,
    TASK_ID,
    ensure_source,
    load_source,
    ordered_text_hash,
    sha256,
    source_codes,
)


def ordered_int_hash(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    ensure_source(args.mimic_root, args.cache_root)
    cohort, _, notes, source_manifest = load_source(args.cache_root)
    manifest_path = args.task_dir / "manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["task"]["id"] != TASK_ID:
        raise AssertionError("wrong DPMM task identity")
    if manifest["task"]["modalities"] != ["structured", "notes"]:
        raise AssertionError("selected DPMM task must retain EHR and raw notes")
    if manifest["task"]["note_codes"] != [NOTE_CODE]:
        raise AssertionError("DPMM note-code declaration changed")
    selected = manifest["source"]["source_configuration"]
    expected_config = {
        "task": "in-hospital-mortality",
        "data_pairs": "partial_ehr_note",
        "data_ratio": 1.0,
    }
    for key, expected in expected_config.items():
        if selected.get(key) != expected:
            raise AssertionError(f"wrong selected source configuration: {key}")

    index = pq.read_table(
        args.task_dir / manifest["meds"]["task_index"]["file"]
    ).to_pandas()
    left = index.sort_values(["split", "fusion_position"]).reset_index(drop=True)
    right = cohort.sort_values(["split", "fusion_position"]).reset_index(drop=True)
    comparisons = {
        "subject_id": (left["subject_id"], right["SUBJECT_ID"]),
        "sample_id": (left["sample_id"], right["stay_id"]),
        "boolean_value": (left["boolean_value"], right["label"].astype(bool)),
        "split": (left["split"], right["split"]),
        "source_name": (left["source_name"], right["source_name"]),
        "hadm_id": (left["hadm_id"], right["HADM_ID"]),
        "fusion_position": (left["fusion_position"], right["fusion_position"]),
        "note_present": (left["note_present"], right["note_present"]),
    }
    for name, (actual, expected) in comparisons.items():
        if not np.array_equal(actual.to_numpy(), expected.to_numpy()):
            raise AssertionError(f"MEDS changed exact cohort field: {name}")

    counts = left["split"].value_counts().to_dict()
    positives = (
        left.groupby("split")["boolean_value"].sum().astype(int).to_dict()
    )
    paired = left.loc[left["note_present"]].groupby("split").size().to_dict()
    paired_positives = (
        left.loc[left["note_present"]]
        .groupby("split")["boolean_value"]
        .sum()
        .astype(int)
        .to_dict()
    )
    if counts != EXPECTED_COUNTS:
        raise AssertionError(f"published split counts differ: {counts}")
    if positives != EXPECTED_POSITIVES:
        raise AssertionError(f"released source positive counts differ: {positives}")
    if paired != EXPECTED_PAIRED:
        raise AssertionError(f"published paired-note counts differ: {paired}")
    if paired_positives != EXPECTED_PAIRED_POSITIVES:
        raise AssertionError(
            f"released source paired positive counts differ: {paired_positives}"
        )
    if len(left) != 21139:
        raise AssertionError("published total cohort count differs")
    reconstructed_positive = int(left["boolean_value"].sum())
    if reconstructed_positive != sum(EXPECTED_POSITIVES.values()):
        raise AssertionError("released source total positive count differs")

    subject_sets = {
        split: set(left.loc[left["split"] == split, "subject_id"].astype(int))
        for split in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": len(subject_sets["train"] & subject_sets["validation"]),
        "train_test": len(subject_sets["train"] & subject_sets["test"]),
        "validation_test": len(
            subject_sets["validation"] & subject_sets["test"]
        ),
    }
    if any(overlaps.values()):
        raise AssertionError(f"released patient splits overlap: {overlaps}")

    sampled = pd.read_parquet(args.cache_root / "sampled_note_lists.parquet")
    sampled_counts = sampled.groupby("split").size().to_dict()
    expected_sampled = {"train": 4894, "validation": 1074, "test": 1079}
    if sampled_counts != expected_sampled:
        raise AssertionError(f"released one-third note sampling changed: {sampled_counts}")
    if len(notes) != 5 * sum(EXPECTED_PAIRED.values()):
        raise AssertionError("every paired sample must have exactly five retained notes")

    codes = pq.read_table(
        args.task_dir / manifest["meds"]["codes"]["file"]
    )["code"].to_pylist()
    if codes != list(source_codes()):
        raise AssertionError("MEDS code vocabulary/order differs from the release")
    if any("image" in code.lower() or "cxr" in code.lower() for code in codes):
        raise AssertionError("an imaging code entered the DPMM task")

    report = {
        "status": "exact_source_with_published_drop_last_count",
        "selected_experiment": {
            "paper_cell": manifest["source"]["selected_paper_cell"],
            "task": "MIMIC-III in-hospital mortality",
            "model_inputs": ["structured", "notes"],
            "configuration": expected_config,
            "selection_tier": 2,
        },
        "splits": {
            split: {
                "samples": counts[split],
                "subjects": len(subject_sets[split]),
                "positive": positives[split],
                "prevalence": positives[split] / counts[split],
                "paired_notes": paired[split],
                "paired_positive": paired_positives[split],
                "sampled_for_note_availability": sampled_counts[split],
            }
            for split in ("train", "validation", "test")
        },
        "total": {
            "samples": len(left),
            "subjects": int(left["subject_id"].nunique()),
            "positive": reconstructed_positive,
            "paired_notes": int(left["note_present"].sum()),
            "note_rows": len(notes),
        },
        "paper_statistic_comparison": {
            "paper_and_released_comment_partial_positive": PUBLISHED_TOTAL_POSITIVES,
            "released_source_reconstruction_total_positive": reconstructed_positive,
            "delta": reconstructed_positive - PUBLISHED_TOTAL_POSITIVES,
            "paper_and_released_comment_paired_positive": PUBLISHED_PAIRED_POSITIVES,
            "released_source_reconstruction_paired_positive": sum(
                EXPECTED_PAIRED_POSITIVES.values()
            ),
            "paired_delta": (
                sum(EXPECTED_PAIRED_POSITIVES.values())
                - PUBLISHED_PAIRED_POSITIVES
            ),
            "interpretation": (
                "The released fusion diagnostic computes positives by summing the "
                "training DataLoader, which uses shuffle=True and drop_last=True. "
                "Thus it omits a random nine partial-cohort examples (or four paired "
                "examples) at batch size 16. The exact source labels are retained "
                "rather than altered to reproduce this incomplete-loader count."
            ),
        },
        "cross_split_subject_overlap": overlaps,
        "ordered_sha256": {
            "sample_id": ordered_int_hash(left["sample_id"]),
            "subject_id": ordered_int_hash(left["subject_id"]),
            "label": ordered_int_hash(left["boolean_value"].astype(int)),
            "source_name": ordered_text_hash(left["source_name"]),
            "split": ordered_text_hash(left["split"]),
        },
        "source_reconstruction": source_manifest,
        "retained_projection": {
            "modalities": ["structured", "notes"],
            "note_codes": [NOTE_CODE],
            "codes": codes,
            "no_imaging_or_auxiliary_rows": True,
        },
        "manifest_sha256": manifest_hash,
        "checks": [
            "exact 21,139 partial-EHR cohort",
            "exact 14,681/3,222/3,236 split counts",
            "exact released-source 1,987/436/374 positive counts",
            "paper's 2,795/736 positive counts are drop-last loader observations",
            "exact 3,652/815/806 paired-note counts",
            "released subject resources produce zero overlap",
            "released one-third prefilter counts are exact",
            "exact identifiers, fusion positions, labels, and modality presence",
        ],
    }
    output = args.task_dir / "audit" / "cohort_reconstruction.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if sha256(manifest_path) != manifest_hash:
        raise AssertionError("cohort checker mutated immutable manifest")
    print(
        json.dumps(
            {
                "status": "exact_source_with_published_drop_last_count",
                "report": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
