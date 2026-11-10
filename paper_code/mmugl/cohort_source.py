"""Exact released MMUGL boundary used by the MEDS exporter."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(PAPER_ROOT / "kg"))

from kg.data.datasets import CglDatasetDownstream  # noqa: E402
from kg.data.processing import read_set_codes  # noqa: E402
from kg.data.vocabulary import CodeTokenizer  # noqa: E402
from kg.utils.training import get_splits  # noqa: E402


TASK_ID = "next_visit_heart_failure"
NOTE_CODE = "MMUGL//NOTE"
ICD_PREFIX = "MMUGL//ICD9//"
ATC_PREFIX = "MMUGL//ATC4//"
SPLITS = ("train", "validation", "test")
MAX_CODES_PER_VISIT = 63
MAX_VISITS = 42
MAX_NOTE_CONCEPTS = 1279
PUBLISHED_AUROC = 0.8719

EXPECTED_HASHES = {
    "cgl_patient_splits.pkl": "cb33feaa4fd31cbad7b907335fa22c16b2f418ff4762359ddeedfaab58325a53",
    "cgl_mimic3_disease_codes.txt": "d5c2a688e26937d6ceeed6fdc111c35a19a19911e27f2fc16db0b66b3660482b",
    "disease_codes_all.txt": "d96489eefe3f76f1a024344e3210321b207bb278a64409396de9a9aec80fce75",
    "med_codes_all.txt": "9cccc195336a27f35c1c2fd15e465733dc60a45830ea3b1605300ad99470f57d",
    "ndc2atc_level4.csv": "162fefc5cc374ac545501dec0d3738cb7a362e1e08b46b64d1a779eaeda7888e",
    "ndc2rxnorm_mapping.txt": "31b78823f61d23e8f563eeff353fc2d589cfff6808a831178a3736bbac225278",
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _table(root: Path, stem: str) -> Path:
    for name in (stem + ".csv.gz", stem + ".csv"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(root / (stem + ".csv.gz"))


def source_paths(mimic_root: Path, mapping_root: Path) -> Dict[str, Path]:
    paths = {
        "ADMISSIONS.csv.gz": _table(mimic_root, "ADMISSIONS"),
        "DIAGNOSES_ICD.csv.gz": _table(mimic_root, "DIAGNOSES_ICD"),
        "PRESCRIPTIONS.csv.gz": _table(mimic_root, "PRESCRIPTIONS"),
        "NOTEEVENTS.csv.gz": _table(mimic_root, "NOTEEVENTS"),
        "cgl_patient_splits.pkl": PAPER_ROOT / "data" / "cgl_patient_splits.pkl",
        "cgl_mimic3_disease_codes.txt": PAPER_ROOT / "data" / "cgl_mimic3_disease_codes.txt",
        "disease_codes_all.txt": PAPER_ROOT / "data" / "disease_codes_all.txt",
        "med_codes_all.txt": PAPER_ROOT / "data" / "med_codes_all.txt",
        "ndc2atc_level4.csv": mapping_root / "ndc2atc_level4.csv",
        "ndc2rxnorm_mapping.txt": mapping_root / "ndc2rxnorm_mapping.txt",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing MMUGL source {name}: {path}")
    for name, expected in EXPECTED_HASHES.items():
        actual = sha256_file(paths[name])
        if actual != expected:
            raise AssertionError(f"{name} has SHA-256 {actual}, expected {expected}")
    return paths


def artifact_inventory(paths: Dict[str, Path], include_raw_hashes: bool) -> Dict[str, dict]:
    inventory = {}
    for name, path in paths.items():
        value = {"bytes": path.stat().st_size}
        if include_raw_hashes or name in EXPECTED_HASHES:
            value["sha256"] = sha256_file(path)
        inventory[name] = value
    return inventory


def released_vocabularies() -> Tuple[List[str], List[str], List[str]]:
    icd = read_set_codes(str(PAPER_ROOT / "data" / "disease_codes_all.txt"))
    atc = read_set_codes(str(PAPER_ROOT / "data" / "med_codes_all.txt"))
    target = read_set_codes(str(PAPER_ROOT / "data" / "cgl_mimic3_disease_codes.txt"))
    if len(icd) != len(set(icd)) or len(atc) != len(set(atc)):
        raise AssertionError("released MMUGL vocabulary contains duplicate entries")
    return icd, atc, target


def meds_codes(icd: Sequence[str], atc: Sequence[str]) -> Tuple[str, ...]:
    values = [ICD_PREFIX + code for code in icd]
    if "[MASK]" not in icd:
        values.append(ICD_PREFIX + "[MASK]")
    values.extend(ATC_PREFIX + code for code in atc)
    if "[MASK]" not in atc:
        values.append(ATC_PREFIX + "[MASK]")
    values.append(NOTE_CODE)
    if len(values) != len(set(values)):
        raise AssertionError("prefixed MEDS vocabulary is not unique")
    return tuple(values)


def _split_ids(paths: Dict[str, Path]):
    train_ids, validation_ids, test_ids = get_splits(
        str(paths["DIAGNOSES_ICD.csv.gz"]),
        mode="cgl",
        random_state=1111,
        testing=False,
        data_dir=str(PAPER_ROOT / "data"),
    )
    return {
        "train": set(int(value) for value in train_ids),
        "validation": set(int(value) for value in validation_ids),
        "test": set(int(value) for value in test_ids),
    }


def _extract_histories(dataset: CglDatasetDownstream, admissions_path: Path) -> List[dict]:
    admissions = pd.read_csv(
        admissions_path,
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME"],
        converters={"SUBJECT_ID": np.int64, "HADM_ID": np.int64, "ADMITTIME": str},
        index_col=["SUBJECT_ID", "HADM_ID"],
    )
    joined = dataset.data_pd.copy().join(
        admissions, on=["SUBJECT_ID", "HADM_ID"], how="inner"
    )
    retained = []
    author_records = dataset.patient_data
    source_subjects = list(joined["SUBJECT_ID"].unique())
    if [int(record[0]) for record in author_records] != [
        int(subject) for subject in source_subjects
        if (joined["SUBJECT_ID"] == subject).sum() >= 2
    ]:
        raise AssertionError("HADM-preserving adapter changed author patient order")

    author_by_subject = {int(subject): visits for subject, visits in author_records}
    for subject in source_subjects:
        subject = int(subject)
        rows = []
        item = joined[joined["SUBJECT_ID"] == subject]
        for _, row in item.iterrows():
            time = datetime.strptime(row["ADMITTIME"], "%Y-%m-%d %H:%M:%S")
            rows.append(
                {
                    "hadm_id": int(row["HADM_ID"]),
                    "time": time,
                    "icd": tuple(str(value) for value in row["ICD9_CODE"]),
                    "atc": tuple(str(value) for value in row["ATC4"]),
                }
            )
        rows.sort(key=lambda value: value["time"])
        if len(rows) < 2:
            continue
        author_visits = author_by_subject[subject]
        if len(rows) != len(author_visits):
            raise AssertionError("HADM-preserving adapter changed author visit count")
        for row, author in zip(rows, author_visits):
            if (
                list(row["icd"]) != list(author[0])
                or list(row["atc"]) != list(author[1])
                or row["time"] != author[2]
            ):
                raise AssertionError("HADM-preserving adapter changed author visit boundary")
            row["label"] = int(dataset.get_target(row["icd"]))
            row["time_us"] = int(np.datetime64(row["time"], "us").astype(np.int64))
        retained.append({"subject_id": subject, "visits": rows})
    return retained


def load_author_boundary(mimic_root: Path, mapping_root: Path, include_raw_hashes: bool = True):
    """Run the released split and CGL dataset path, retaining HADM at its drop point."""
    paths = source_paths(mimic_root, mapping_root)
    icd, atc, target = released_vocabularies()
    split_ids = _split_ids(paths)
    tokenizer = CodeTokenizer(icd, atc)
    partitions = {}
    for split in SPLITS:
        dataset = CglDatasetDownstream(
            str(paths["DIAGNOSES_ICD.csv.gz"]),
            str(paths["PRESCRIPTIONS.csv.gz"]),
            str(paths["ADMISSIONS.csv.gz"]),
            str(mapping_root),
            diagnosis_codes=icd,
            diagnosis_codes_target=target,
            prescription_codes=atc,
            tokenizer=tokenizer,
            patient_ids=split_ids[split],
            code_count_range=(1, np.inf),
            visit_range=(1, np.inf),
            max_sequence_length=MAX_CODES_PER_VISIT,
            patient_sequence_length=MAX_VISITS,
            target_task="heart_failure",
            umls_graph_data=None,
            notes_concepts_path=None,
            code_shuffle=False,
        )
        partitions[split] = _extract_histories(dataset, paths["ADMISSIONS.csv.gz"])
        del dataset

    return {
        "paths": paths,
        "inventory": artifact_inventory(paths, include_raw_hashes),
        "icd": icd,
        "atc": atc,
        "target": target,
        "codes": meds_codes(icd, atc),
        "split_ids": split_ids,
        "partitions": partitions,
        "tokenizer": tokenizer,
    }


def build_samples(boundary: dict):
    samples = []
    source_position = 0
    for split in SPLITS:
        for patient_position, patient in enumerate(boundary["partitions"][split]):
            visits = patient["visits"][:MAX_VISITS]
            target_indices = (
                range(1, len(visits)) if split == "train" else (len(visits) - 1,)
            )
            for target_index in target_indices:
                target = visits[target_index]
                history = visits[:target_index]
                if not history:
                    raise AssertionError("MMUGL transition has no eligible history")
                sample_id = int(target["hadm_id"])
                samples.append(
                    {
                        "subject_id": int(patient["subject_id"]),
                        "sample_id": sample_id,
                        "split": split,
                        "boolean_value": bool(target["label"]),
                        "prediction_time_us": int(target["time_us"]),
                        "actual_observation_start_us": int(history[0]["time_us"]),
                        "source_patient_position": patient_position,
                        "source_target_visit_index": target_index,
                        "target_hadm_id": int(target["hadm_id"]),
                        "history": history,
                        "source_position": source_position,
                    }
                )
                source_position += 1
    sample_ids = [row["sample_id"] for row in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise AssertionError("target HADM_ID is not a unique transition identifier")
    maximum_span_us = max(
        row["prediction_time_us"] - row["actual_observation_start_us"]
        for row in samples
    )
    if maximum_span_us <= 0:
        raise AssertionError("MMUGL produced a nonpositive history interval")
    observation_hours = math.ceil(maximum_span_us / 3_600_000_000)
    for row in samples:
        # MEDS records the exact boundary at which this patient's released
        # history begins. The shared temporal pilot uses `observation_hours`
        # only as outer support, then fits absolute quantile boundaries from
        # all structured feature rows; it must not fabricate leading history.
        row["observation_start_us"] = row["actual_observation_start_us"]
    return samples, float(observation_hours)


def sample_frame(samples: Sequence[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": pl.Series([row["subject_id"] for row in samples], dtype=pl.Int64),
            "sample_id": pl.Series([row["sample_id"] for row in samples], dtype=pl.Int64),
            "observation_start": pl.Series(
                [row["observation_start_us"] for row in samples], dtype=pl.Int64
            ).cast(pl.Datetime("us")),
            "prediction_time": pl.Series(
                [row["prediction_time_us"] for row in samples], dtype=pl.Int64
            ).cast(pl.Datetime("us")),
            "boolean_value": pl.Series(
                [row["boolean_value"] for row in samples], dtype=pl.Boolean
            ),
            "split": pl.Series([row["split"] for row in samples], dtype=pl.String),
            "source_position": pl.Series(
                [row["source_position"] for row in samples], dtype=pl.Int64
            ),
            "source_patient_position": pl.Series(
                [row["source_patient_position"] for row in samples], dtype=pl.Int64
            ),
            "source_target_visit_index": pl.Series(
                [row["source_target_visit_index"] for row in samples], dtype=pl.Int64
            ),
            "target_hadm_id": pl.Series(
                [row["target_hadm_id"] for row in samples], dtype=pl.Int64
            ),
            "actual_observation_start": pl.Series(
                [row["actual_observation_start_us"] for row in samples], dtype=pl.Int64
            ).cast(pl.Datetime("us")),
        }
    )


def structured_event_count(samples: Sequence[dict]) -> int:
    return sum(
        len(visit["icd"]) + len(visit["atc"])
        for sample in samples
        for visit in sample["history"]
    )


def iter_structured_rows(samples: Sequence[dict]) -> Iterator[dict]:
    source_row = 0
    for sample in samples:
        for visit_index, visit in enumerate(sample["history"]):
            for code in visit["icd"]:
                yield {
                    "subject_id": sample["subject_id"],
                    "time_us": visit["time_us"],
                    "code": ICD_PREFIX + code,
                    "sample_id": sample["sample_id"],
                    "source_row": source_row,
                    "sequence_row": visit_index,
                    "source_time": float(visit_index),
                    "text_value": None,
                }
                source_row += 1
            for code in visit["atc"]:
                yield {
                    "subject_id": sample["subject_id"],
                    "time_us": visit["time_us"],
                    "code": ATC_PREFIX + code,
                    "sample_id": sample["sample_id"],
                    "source_row": source_row,
                    "sequence_row": visit_index,
                    "source_time": float(visit_index),
                    "text_value": None,
                }
                source_row += 1


def _open_csv(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(
        "r", encoding="utf-8", newline=""
    )


def note_consumers(samples: Sequence[dict]) -> Dict[int, List[Tuple[int, int, int]]]:
    consumers = defaultdict(list)
    for sample in samples:
        for visit in sample["history"]:
            consumers[int(visit["hadm_id"])].append(
                (int(sample["sample_id"]), int(sample["subject_id"]), int(visit["time_us"]))
            )
    return dict(consumers)


def iter_note_rows(
    notes_path: Path,
    samples: Sequence[dict],
    first_source_row: int,
    stats: dict | None = None,
) -> Iterator[dict]:
    """Stream the exact non-error, de-duplicated raw notes eligible to each prefix."""
    consumers = note_consumers(samples)
    expected_subject = {
        hadm_id: values[0][1] for hadm_id, values in consumers.items()
    }
    if any(
        subject != expected_subject[hadm_id]
        for hadm_id, values in consumers.items()
        for _, subject, _ in values
    ):
        raise AssertionError("one HADM_ID is owned by multiple MMUGL subjects")
    seen = set()
    raw_position = -1
    selected_source_notes = 0
    text_digest = hashlib.sha256()
    notes_by_hadm = defaultdict(list)
    with _open_csv(notes_path) as handle:
        reader = csv.DictReader(handle)
        for raw_position, row in enumerate(reader):
            hadm = row.get("HADM_ID", "").strip()
            if not hadm or int(hadm) not in consumers:
                continue
            if row.get("ISERROR", "").strip():
                continue
            subject = row.get("SUBJECT_ID", "").strip()
            text = row.get("TEXT", "")
            duplicate_key = (
                subject,
                hadm,
                row.get("CATEGORY", ""),
                row.get("DESCRIPTION", ""),
                row.get("ISERROR", ""),
                text,
            )
            key_hash = hashlib.sha256(
                json.dumps(duplicate_key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).digest()
            if key_hash in seen:
                continue
            seen.add(key_hash)
            hadm_id = int(hadm)
            if int(subject) != expected_subject[hadm_id]:
                raise AssertionError("NOTEEVENTS subject disagrees with source visit")
            selected_source_notes += 1
            text_encoded = text.encode("utf-8")
            text_digest.update(len(text_encoded).to_bytes(8, "little"))
            text_digest.update(text_encoded)
            notes_by_hadm[hadm_id].append((raw_position, text))

    source_row = first_source_row
    selected_note_events = 0
    for sample in samples:
        for visit in sample["history"]:
            for note_position, text in notes_by_hadm.get(int(visit["hadm_id"]), ()):
                yield {
                    "subject_id": int(sample["subject_id"]),
                    "time_us": int(visit["time_us"]),
                    "code": NOTE_CODE,
                    "sample_id": int(sample["sample_id"]),
                    "source_row": source_row,
                    "sequence_row": note_position,
                    "source_time": None,
                    "text_value": text,
                }
                source_row += 1
                selected_note_events += 1
    if stats is not None:
        stats.update(
            {
                "eligible_unique_source_notes": selected_source_notes,
                "duplicated_note_events": selected_note_events,
                "eligible_raw_text_sha256": text_digest.hexdigest(),
                "raw_note_order": (
                    "released chronological visit order per sample, then NOTEEVENTS "
                    "source-row order within each visit after non-error and exact-duplicate filters"
                ),
                "last_raw_position": raw_position,
            }
        )


def rows_to_frame(rows: Sequence[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": pl.Series([row["subject_id"] for row in rows], dtype=pl.Int64),
            "time": pl.Series([row["time_us"] for row in rows], dtype=pl.Int64).cast(
                pl.Datetime("us")
            ),
            "code": pl.Series([row["code"] for row in rows], dtype=pl.String),
            "numeric_value": pl.Series([None] * len(rows), dtype=pl.Float32),
            "sample_id": pl.Series([row["sample_id"] for row in rows], dtype=pl.Int64),
            "source_row": pl.Series([row["source_row"] for row in rows], dtype=pl.Int64),
            "sequence_row": pl.Series(
                [row["sequence_row"] for row in rows], dtype=pl.Int32
            ),
            "source_time": pl.Series([row["source_time"] for row in rows], dtype=pl.Float64),
            "text_value": pl.Series([row["text_value"] for row in rows], dtype=pl.String),
        }
    )


def event_batches(
    boundary: dict,
    samples: Sequence[dict],
    note_stats: dict,
    batch_size: int = 25_000,
) -> Iterable[pl.DataFrame]:
    structured_count = structured_event_count(samples)
    for iterator in (
        iter_structured_rows(samples),
        iter_note_rows(
            boundary["paths"]["NOTEEVENTS.csv.gz"],
            samples,
            structured_count,
            note_stats,
        ),
    ):
        rows = []
        for row in iterator:
            rows.append(row)
            if len(rows) == batch_size:
                yield rows_to_frame(rows)
                rows = []
        if rows:
            yield rows_to_frame(rows)


def split_summary(samples: Sequence[dict]) -> dict:
    result = {}
    for split in SPLITS:
        rows = [row for row in samples if row["split"] == split]
        result[split] = {
            "samples": len(rows),
            "patients": len({row["subject_id"] for row in rows}),
            "positive": sum(row["boolean_value"] for row in rows),
            "prevalence": sum(row["boolean_value"] for row in rows) / len(rows),
        }
    return result


def cross_split_overlap(samples: Sequence[dict]) -> dict:
    members = {
        split: {row["subject_id"] for row in samples if row["split"] == split}
        for split in SPLITS
    }
    return {
        f"{left}_{right}": len(members[left] & members[right])
        for position, left in enumerate(SPLITS)
        for right in SPLITS[position + 1 :]
    }
