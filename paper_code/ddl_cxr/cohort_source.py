"""Exact-source helpers for the DDL-CXR structured projection."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parent
CONFIG_PATH = REPOSITORY / "ehr_utils" / "resources" / "discretizer_config.json"
NORMALIZER_PATH = REPOSITORY / "ldm" / "data" / "normalizers" / "predict_normalizer"
ITEMID_MAP_PATH = (
    REPOSITORY
    / "mimic4extract"
    / "mimic3benchmark"
    / "resources"
    / "itemid_to_variable_map.csv"
)
PHENOTYPE_PATH = (
    REPOSITORY
    / "mimic4extract"
    / "mimic3benchmark"
    / "resources"
    / "icd_9_10_definitions_2.yaml"
)
COMMITTED_COHORT_ROOT = REPOSITORY / "data_pre"
HOURS = 48.0
EPS = 1e-6
SOURCE_SPLITS = ("train", "validate", "test")
COMMON_SPLITS = {"train": "train", "validate": "validation", "test": "test"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def configuration() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def variable_names() -> tuple[str, ...]:
    return tuple(configuration()["id_to_channel"])


def code_vocabulary() -> tuple[str, ...]:
    config = configuration()
    result: list[str] = []
    for variable in config["id_to_channel"]:
        if config["is_categorical_channel"][variable]:
            result.extend(
                f"{variable}//{value}"
                for value in config["possible_values"][variable]
            )
        else:
            result.append(variable)
    return tuple(result)


def committed_subject_ids() -> tuple[int, ...]:
    subjects: set[int] = set()
    for split in SOURCE_SPLITS:
        frame = pd.read_csv(
            COMMITTED_COHORT_ROOT / f"{split}_pred.csv", usecols=["subject_id"]
        )
        subjects.update(frame["subject_id"].astype(int).tolist())
    return tuple(sorted(subjects))


def load_source_rows(work_root: Path) -> list[dict[str, object]]:
    work_root = work_root.resolve()
    subjects_root = work_root / "subjects"
    cohort_root = work_root / "prediction_cohort"
    stays = pd.read_csv(
        subjects_root / "all_stays.csv",
        usecols=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "los",
            "mortality_inhospital",
        ],
    )
    stays["intime"] = pd.to_datetime(stays["intime"], errors="raise")
    stays["outtime"] = pd.to_datetime(stays["outtime"], errors="raise")
    stay_lookup = {int(row.stay_id): row for row in stays.itertuples(index=False)}
    result: list[dict[str, object]] = []
    for source_split in SOURCE_SPLITS:
        frame = pd.read_csv(cohort_root / f"{source_split}_pred.csv")
        for source_position, row in enumerate(frame.itertuples(index=False)):
            stay_id = int(row.stay_id)
            stay = stay_lookup.get(stay_id)
            if stay is None:
                raise AssertionError("prediction cohort references an unknown ICU stay")
            subject_id = int(row.subject_id)
            if subject_id != int(stay.subject_id):
                raise AssertionError("prediction metadata subject disagrees with ICU stay")
            label = int(row.mortality_inhospital)
            if label != int(stay.mortality_inhospital):
                raise AssertionError("prediction metadata label disagrees with extraction")
            timeseries_path = (
                subjects_root
                / str(subject_id)
                / f"episode{stay_id}_timeseries.csv"
            )
            if not timeseries_path.is_file():
                raise FileNotFoundError(timeseries_path)
            result.append(
                {
                    "split": COMMON_SPLITS[source_split],
                    "source_split": source_split,
                    "source_position": source_position,
                    "source_index": int(row.index),
                    "timeseries_path": timeseries_path,
                    "subject_id": subject_id,
                    "hadm_id": int(stay.hadm_id),
                    "stay_id": stay_id,
                    "label": label,
                    "intime": stay.intime.to_pydatetime(),
                    "outtime": stay.outtime.to_pydatetime(),
                    "los_days": float(stay.los),
                    "gender": float(row.Gender),
                    "age": float(row.Age),
                    "study_id": int(row.study_id),
                    "dicom_id": str(row.dicom_id),
                    "cha": float(row.cha),
                }
            )
    stay_ids = [int(row["stay_id"]) for row in result]
    if len(stay_ids) != len(set(stay_ids)):
        raise AssertionError("prediction cohort repeats an ICU stay")
    return result


def iter_raw_rows(sample: dict[str, object]):
    """Yield the exact rows passed to the released prediction discretizer."""
    variables = variable_names()[:-2]
    with Path(sample["timeseries_path"]).open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[0] != "Hours" or tuple(header[1:]) != variables:
            raise AssertionError("unexpected released episode-timeseries header")
        sequence_row = 0
        for values in reader:
            source_time = float(values[0])
            if source_time < 0.0 or source_time > HOURS:
                continue
            # EHR_Pickler resets the filtered row index before appending statics.
            row = list(values)
            row.extend([sample["gender"], sample["age"]])
            yield sequence_row, source_time, row
            sequence_row += 1


def author_preprocessors():
    sys.path.insert(0, str(REPOSITORY))
    from ldm.data.dynamic_data import EHR_Pickler

    previous = Path.cwd()
    try:
        os.chdir(REPOSITORY)
        pickler = EHR_Pickler(
            stage="prediction",
            normalizer_state=str(NORMALIZER_PATH),
            metadata_path="unused",
            mimic_iv_subjects_dir="unused",
        )
    finally:
        os.chdir(previous)
    return pickler.discretizer, pickler.normalizer


def author_tensor_from_rows(rows: list[list[object]]) -> np.ndarray:
    discretizer, normalizer = author_preprocessors()
    values = np.asarray(rows, dtype=object)
    tensor = discretizer.transform(values, end=48)[0]
    return normalizer.transform(tensor)


def load_prediction_pickles(work_root: Path) -> dict[str, dict[str, object]]:
    root = work_root / "prediction_cohort"
    result = {}
    for source_split in SOURCE_SPLITS:
        with (root / f"{source_split}_ehr_pred.pkl").open("rb") as handle:
            result[COMMON_SPLITS[source_split]] = pickle.load(handle)
    return result


def parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return pd.Timestamp(value).to_pydatetime()
