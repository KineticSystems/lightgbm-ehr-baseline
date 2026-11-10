"""Export FuseMoE's retained structured and raw-note inputs directly as MEDS."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import transformers
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from preprocessing.data_mimiciv import TSNote_Irg  # noqa: E402
from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402


VARIABLE_NAMES = (
    "Absolute Neutrophil Count",
    "Alkaline Phosphate",
    "Anion gap",
    "Bicarbonate",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Diastolic BP",
    "GCS - Eye Opening",
    "GCS - Motor Response",
    "GCS - Verbal Response",
    "Glucose",
    "Heart Rate",
    "Hematocrit",
    "Hemoglobin",
    "Inspired O2 Fraction",
    "Magnesium",
    "Mean BP",
    "O2 Saturation",
    "PH",
    "Platelet Count",
    "Potassium",
    "Respiratory Rate",
    "Sodium",
    "Systolic BP",
    "Temperature",
    "Urea Nitrogen",
    "Vancomycin",
    "WBC",
    "Weight",
)
TASK_STEM = "ihm-48-cxr-notes-ecg-missingInd_stays.pkl"
HOURS = 48.0
NOTE_CODES = ("FUSEMOE//TEXT_SLOT",)
COHORT_ID_SHA256 = "8ed7fd4e61846ff62fc0eee0b73683734205f3a9321af15b211f5f759d74728b"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_source(source_root):
    inventory = {}
    by_stay = {}
    for filename_split in ("train", "val", "test"):
        path = source_root / (filename_split + "_" + TASK_STEM)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            values = pickle.load(handle)
        inventory[path.name] = {
            "sha256": sha256(path),
            "rows": len(values),
        }
        for value in values:
            stay_id = int(value["stay_id"])
            if stay_id in by_stay:
                raise AssertionError("a stay occurs in multiple released artifacts")
            by_stay[stay_id] = value

    # The paper's 8,770-stay initial-analysis population is exactly the
    # notes+CXR intersection in the released 35,129-stay missingness artifacts.
    # ECG availability is deliberately not an inclusion criterion for the
    # selected vital-signs + notes experiment.
    stay_ids = np.asarray(
        sorted(
            stay_id
            for stay_id, value in by_stay.items()
            if int(value["text_missing"]) == 0
            and int(value["cxr_missing"]) == 0
        ),
        dtype=np.int64,
    )
    cohort_hash = hashlib.sha256(np.sort(stay_ids).tobytes()).hexdigest()
    if len(stay_ids) != 8770 or cohort_hash != COHORT_ID_SHA256:
        raise AssertionError("released artifacts do not reconstruct the paper cohort")

    # This is the exact saved producer algorithm after np.intersect1d has
    # sorted the modality intersection.
    random_state = np.random.RandomState(0)
    random_state.shuffle(stay_ids)
    train_end = int(0.70 * len(stay_ids))
    validation_end = int(0.85 * len(stay_ids))
    result = []
    for source_split, selected in (
        ("train", stay_ids[:train_end]),
        ("validation", stay_ids[train_end:validation_end]),
        ("test", stay_ids[validation_end:]),
    ):
        result.append((source_split, [by_stay[int(stay_id)] for stay_id in selected]))
    return result, inventory


def split_overlap(rows, key):
    members = {
        split: {int(row[key]) for row in rows if row["source_split"] == split}
        for split in ("train", "validation", "test")
    }
    return {
        f"{left}_{right}": len(members[left] & members[right])
        for index, left in enumerate(members)
        for right in tuple(members)[index + 1 :]
    }


def loader_args():
    return SimpleNamespace(
        max_length=1024,
        chunk=False,
        pad_to_max_length=False,
        ratio_notes_order=None,
        modeltype="TS_Text",
        model_name="bioLongformer",
        notes_order="Last",
        num_of_notes=5,
        tt_max=48,
        reg_ts=True,
        use_pt_text_embeddings=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", default="yikuan8/Clinical-Longformer"
    )
    args = parser.parse_args()

    split_data, inventory = load_source(args.source_root)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    icustays_path = args.mimic_root / "icu" / "icustays.csv.gz"
    if not icustays_path.is_file():
        raise FileNotFoundError(icustays_path)
    icustays = pd.read_csv(
        icustays_path,
        usecols=["subject_id", "hadm_id", "stay_id", "intime"],
    )
    icustays["intime"] = pd.to_datetime(icustays["intime"], errors="raise")
    icu_lookup = {
        int(row.stay_id): (
            int(row.subject_id),
            int(row.hadm_id),
            int(row.intime.value // 1000),
        )
        for row in icustays.itertuples(index=False)
    }

    retained = []
    filtered_long = []
    for source_split, stays in split_data:
        mode = "train" if source_split == "train" else source_split
        dataset = TSNote_Irg(loader_args(), mode, tokenizer, data=stays)
        for position, stay in enumerate(stays):
            item = dataset[position]
            stay_id = int(stay["stay_id"])
            subject_id, hadm_id, intime_us = icu_lookup[stay_id]
            if int(stay["name"]) != subject_id or int(stay["hadm_id"]) != hadm_id:
                raise AssertionError("released dictionary identifiers disagree with ICUSTAYS")
            if int(item["idx"]) != subject_id:
                raise AssertionError("TSNote_Irg changed the source subject identifier")
            values = item["ts"].numpy()
            mask = item["ts_mask"].numpy()
            times = np.asarray(stay["ts_tt"], dtype=np.float64)
            model_times = item["ts_tt"].numpy().astype(np.float64) * HOURS
            label = int(item["label"].item())
            if values.shape[0] >= 1000:
                filtered_long.append(stay_id)
                continue
            if values.shape != mask.shape or values.shape[1] != len(VARIABLE_NAMES):
                raise AssertionError("unexpected structured loader shape")
            if times.shape != (values.shape[0],):
                raise AssertionError("unexpected structured timestamp shape")
            if model_times.shape != times.shape:
                raise AssertionError("unexpected model timestamp shape")
            if np.count_nonzero(values[mask == 0]):
                raise AssertionError("masked irregular values are unexpectedly nonzero")
            if np.any(mask.sum(axis=1) == 0):
                raise AssertionError("an irregular row has no observed value")
            if label != int(stay["label"]):
                raise AssertionError("TSNote_Irg changed the source label")
            source_texts = [str(value) for value in stay.get("text_data", [])]
            if int(stay["text_missing"]):
                selected_texts = []
            else:
                selected_texts = source_texts[-loader_args().num_of_notes :]
            retained.append(
                {
                    "source_split": source_split,
                    "stay_id": stay_id,
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "intime_us": intime_us,
                    "label": label,
                    "values": values,
                    "mask": mask,
                    "times": times,
                    "model_times": model_times,
                    "source_position": position,
                    "text_values": selected_texts,
                    "text_missing": int(stay["text_missing"]),
                }
            )

    event_subject = []
    event_time_us = []
    event_code = []
    event_value = []
    event_sample = []
    event_source_row = []
    event_sequence_row = []
    event_source_time = []
    notes = {
        "subject": [],
        "time_us": [],
        "sample": [],
        "text": [],
    }

    def append_note(row, text_value):
        notes["subject"].append(row["subject_id"])
        notes["time_us"].append(row["intime_us"])
        notes["sample"].append(row["stay_id"])
        notes["text"].append(str(text_value))

    dropped_nonfinite = 0
    source_row = 0
    for row in retained:
        observed_row, observed_code = np.nonzero(row["mask"] == 1)
        values = row["values"][observed_row, observed_code].astype(np.float32)
        source_times = row["times"][observed_row].astype(np.float64)
        model_times = row["model_times"][observed_row].astype(np.float64)
        keep = (
            np.isfinite(values)
            & np.isfinite(source_times)
            & np.isfinite(model_times)
            & (source_times >= 0.0)
            & (source_times <= HOURS)
            & (model_times >= 0.0)
            & (model_times <= HOURS)
        )
        dropped_nonfinite += int((~keep).sum())
        observed_row = observed_row[keep]
        observed_code = observed_code[keep]
        values = values[keep]
        source_times = source_times[keep]
        model_times = model_times[keep]
        count = len(values)
        event_subject.append(np.full(count, row["subject_id"], dtype=np.int64))
        event_time_us.append(
            row["intime_us"]
            + np.rint(model_times * 3600.0 * 1000000.0).astype(np.int64)
        )
        event_code.append(observed_code.astype(np.int16))
        event_value.append(values)
        event_sample.append(np.full(count, row["stay_id"], dtype=np.int64))
        event_source_row.append(
            source_row + np.arange(count, dtype=np.int64)
        )
        event_sequence_row.append(observed_row.astype(np.int32))
        event_source_time.append(source_times.astype(np.float64))
        source_row += count

        for text_value in row["text_values"]:
            append_note(row, text_value)
    structured_count = source_row
    structured = pl.DataFrame(
        {
            "subject_id": pl.Series(np.concatenate(event_subject), dtype=pl.Int64),
            "time": pl.Series(np.concatenate(event_time_us), dtype=pl.Int64).cast(
                pl.Datetime("us")
            ),
            "code": pl.Series(
                np.asarray(VARIABLE_NAMES)[np.concatenate(event_code)], dtype=pl.String
            ),
            "numeric_value": pl.Series(
                np.concatenate(event_value), dtype=pl.Float32
            ),
            "sample_id": pl.Series(np.concatenate(event_sample), dtype=pl.Int64),
            "source_row": pl.Series(
                np.concatenate(event_source_row), dtype=pl.Int64
            ),
            "sequence_row": pl.Series(
                np.concatenate(event_sequence_row), dtype=pl.Int32
            ),
            "source_time": pl.Series(
                np.concatenate(event_source_time), dtype=pl.Float64
            ),
        }
    )
    note_count = len(notes["sample"])
    note_events = pl.DataFrame(
        {
            "subject_id": pl.Series(notes["subject"], dtype=pl.Int64),
            "time": pl.Series(notes["time_us"], dtype=pl.Int64).cast(
                pl.Datetime("us")
            ),
            "code": pl.Series([NOTE_CODES[0]] * note_count, dtype=pl.String),
            "numeric_value": pl.Series([None] * note_count, dtype=pl.Float32),
            "sample_id": pl.Series(notes["sample"], dtype=pl.Int64),
            "source_row": pl.Series(
                structured_count + np.arange(note_count), dtype=pl.Int64
            ),
            "text_value": pl.Series(notes["text"], dtype=pl.String),
        }
    )
    events = pl.concat([structured, note_events], how="diagonal")
    samples = pl.DataFrame(
        {
            "subject_id": pl.Series(
                [row["subject_id"] for row in retained], dtype=pl.Int64
            ),
            "sample_id": pl.Series(
                [row["stay_id"] for row in retained], dtype=pl.Int64
            ),
            "observation_start": pl.Series(
                [row["intime_us"] for row in retained], dtype=pl.Int64
            ).cast(pl.Datetime("us")),
            "prediction_time": pl.Series(
                [
                    row["intime_us"] + int(HOURS * 3600 * 1_000_000)
                    for row in retained
                ],
                dtype=pl.Int64,
            ).cast(pl.Datetime("us")),
            "boolean_value": pl.Series(
                [bool(row["label"]) for row in retained], dtype=pl.Boolean
            ),
            "split": pl.Series(
                [row["source_split"] for row in retained], dtype=pl.String
            ),
            "source_split": pl.Series(
                [row["source_split"] for row in retained], dtype=pl.String
            ),
            "hadm_id": pl.Series([row["hadm_id"] for row in retained], dtype=pl.Int64),
            "source_position": pl.Series(
                [row["source_position"] for row in retained], dtype=pl.Int64
            ),
            "text_missing": pl.Series(
                [row["text_missing"] for row in retained], dtype=pl.Int64
            ),
        }
    )
    split_counts = {}
    for split in ("train", "validation", "test"):
        chosen = [row for row in retained if row["source_split"] == split]
        split_counts[split] = {
            "rows": len(chosen),
            "subjects": len(set(row["subject_id"] for row in chosen)),
            "positive": sum(row["label"] for row in chosen),
        }
    metadata = {
        "paper": {
            "key": "fusemoe",
            "title": "FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion",
        },
        "task": {
            "id": "mortality_48h",
            "name": "In-hospital mortality from first 48 ICU hours",
            "primary_metric": "AUROC",
            "published_auroc": 0.8103,
            "observation_window_hours": HOURS,
        },
        "dataset_name": "FuseMoE paper vital-signs + notes MIMIC-IV cohort",
        "dataset_version": "MIMIC-IV 2.2 / reconstructed released-artifact intersection",
        "source": {
            "upstream_commit": "52ea655e2103b4daf1033572ca354cabc496bd87",
            "loader_entrypoint": "src/preprocessing/data_mimiciv.py::TSNote_Irg.__getitem__",
            "capture_boundary": "structured and note projection of TS_Text item before released TextTSIrgcollate_fn; imaging omitted",
            "source_artifacts": inventory,
            "selected_paper_cell": "Table 4, 48-IHM, vital signs + clinical notes, Laplace gating",
            "selected_model_inputs": ["structured", "notes"],
            "selection_tier": 2,
            "cohort_selection": {
                "rule": "text_missing == 0 and cxr_missing == 0; ECG availability ignored",
                "paper_rows": 8770,
                "sorted_stay_id_sha256": COHORT_ID_SHA256,
            },
            "split_policy": "paper producer np.intersect1d ordering, NumPy seed-0 shuffle, then 70/15/15 stay slicing",
            "split_counts": split_counts,
            "cross_split_subject_overlap": split_overlap(retained, "subject_id"),
            "cross_split_admission_overlap": split_overlap(retained, "hadm_id"),
            "collate_length_exclusions": filtered_long,
            "nonfinite_observed_cells_excluded": dropped_nonfinite,
            "batch_recreation": {
                "modeltype": "TS_Text",
                "train_batch_size": 1,
                "evaluation_batch_size": 8,
                "num_of_notes": 5,
                "stored_note_rows": "actual author-selected notes only; loader padding omitted",
                "max_length": 1024,
                "notes_order": "Last",
                "tt_max": 48,
                "reg_ts": True,
                "use_pt_text_embeddings": True,
                "retained_modalities": ["structured", "notes"],
                "omitted_modalities": ["CXR", "ECG"],
                "tokenizer": args.tokenizer,
                "tokenizer_class": tokenizer.__class__.__name__,
                "tokenizer_vocabulary_sha256": hashlib.sha256(
                    json.dumps(
                        sorted(tokenizer.get_vocab().items()), separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "transformers_version": transformers.__version__,
            },
            "mimic_icustays_sha256": sha256(icustays_path),
        },
    }
    write_meds_task(
        args.output_dir,
        paper=metadata["paper"],
        task=metadata["task"],
        codes=VARIABLE_NAMES + NOTE_CODES,
        note_codes=NOTE_CODES,
        events=events,
        samples=samples,
        source=metadata["source"],
        dataset_name=metadata["dataset_name"],
        dataset_version=metadata["dataset_version"],
    )
    print(
        json.dumps(
            {
                "samples": len(retained),
                "structured_events": structured_count,
                "note_events": note_count,
                "splits": split_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
