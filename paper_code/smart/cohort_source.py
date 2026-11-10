"""SMART's exact-source MIMIC-III mortality boundary for the export.

The released notebook consumes the standard ``mimic3-benchmarks`` mortality
CSV files, creates a 48 by 17 value tensor and mask, normalizes observed cells
over the full pooled cohort, and only then asks ``data/mimiciii.py`` to redraw
an example-level split.  This module calls those released readers and keeps the
aligned identifiers that SMART drops.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
PRODUCER_COMMIT = "ea0314c7cbd369f62e2237ace6f683740f867e3a"
PRODUCER_ROOT = (
    PAPER_ROOT / ".export_assets" / f"mimic3-benchmarks-{PRODUCER_COMMIT}"
)
EXTERNAL_ROOT = PAPER_ROOT / ".export_assets" / "external_artifacts"
COHORT_EVIDENCE = EXTERNAL_ROOT / "cohort_evidence"
SOURCE_ARCHIVE = (
    EXTERNAL_ROOT
    / "sources"
    / f"mimic3-benchmarks-{PRODUCER_COMMIT}.tar.gz"
)
CHANNEL_INFO_PATH = PAPER_ROOT / "data" / "resources" / "channel_info.json"
CONFIG_PATH = PAPER_ROOT / "data" / "resources" / "discretizer_config.json"
NOTEBOOK_PATH = PAPER_ROOT / "data" / "MIMIC-III" / "preprocess_mimiciii.ipynb"
READER_PATH = PAPER_ROOT / "data" / "MIMIC-III" / "reader.py"
LOADER_PATH = PAPER_ROOT / "data" / "mimiciii.py"
COLLATE_PATH = PAPER_ROOT / "data" / "dataloader.py"

TASK_ID = "mortality_48h"
HOURS = 48.0
SEEDS = (1, 42, 3407)
SPLITS = ("train", "validation", "test")
EXPECTED_TOTAL = 21_139
EXPECTED_SUBJECTS = 18_094
EXPECTED_POSITIVES = 2_797
EXPECTED_OBSERVED_RATE = 0.433020198239663
EXPECTED_SPLIT_COUNTS = {"train": 16_911, "validation": 2_113, "test": 2_115}
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "8215e4df1b1adcb7c885b0b92d5adf697968fa99a8c7413990986d8a5dfa500f"
)
EXPECTED_IHM_INVENTORY_SHA256 = (
    "87ec871dd96a711bc731ed67d049c5ad185edfea17288f16e71888438f81630b"
)
RELEASE_SHA256 = {
    "PATIENTS.csv.gz": "0a13db69044d3b01ab9a397f3192792c2a1d2e98c9cb41c4f94e0403996dc7c5",
    "ADMISSIONS.csv.gz": "68f128f19e95434500982d95e623638f22eadf1c6a07f4bfa6c39c21751d8085",
    "ICUSTAYS.csv.gz": "65f8c8686e791bee80df0115fc18f9f3c1087b85889ba9ce2d9c441204e467ab",
    "D_ICD_DIAGNOSES.csv.gz": "78236ce52b050fc59c0d91b774971a4c3fa43ab13b9929c1cc6b67fdc3e2304d",
    "DIAGNOSES_ICD.csv.gz": "02d3941aa3c7a043af4f138e4ffc9f0a51ceca21c905de1abd2b953211910420",
    "CHARTEVENTS.csv.gz": "5961ac36d8bfbd399088811f9a1d30bd6f482b23a84628593d3122b5a6be8bc4",
    "LABEVENTS.csv.gz": "79d86315dde6e1121f7a88f62c2b91b7b835364ac1e777248a58e4f180224ae6",
    "OUTPUTEVENTS.csv.gz": "70f14321d228b12d243a10974b7e41e41690031d6dbc16354ef78e3d1dfb4445",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ordered_text_hash(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def assert_mimic_release(root: Path) -> dict[str, str]:
    manifest = root / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    text = manifest.read_text()
    for filename, expected in RELEASE_SHA256.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing exact MIMIC-III 1.4 table: {path}")
        if f"{expected} {filename}" not in text:
            raise AssertionError(f"MIMIC-III release manifest changed: {filename}")
    return dict(RELEASE_SHA256)


def variable_names() -> tuple[str, ...]:
    return tuple(json.loads(CONFIG_PATH.read_text())["id_to_channel"])


def configuration() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def channel_info() -> dict:
    return json.loads(CHANNEL_INFO_PATH.read_text())


def code_vocabulary() -> tuple[str, ...]:
    config = configuration()
    result: list[str] = []
    for variable in variable_names():
        if config["is_categorical_channel"][variable]:
            result.extend(
                f"{variable}//{value}"
                for value in config["possible_values"][variable]
            )
        else:
            result.append(variable)
    if len(result) != len(set(result)):
        raise AssertionError("released SMART code vocabulary is not unique")
    return tuple(result)


def replica_task_id(replica: int) -> str:
    if replica not in (1, 2, 3):
        raise ValueError(replica)
    return f"{TASK_ID}_replica_{replica:02d}"


def source_root(cache_root: Path) -> Path:
    return cache_root / "in-hospital-mortality"


def load_build_report(cache_root: Path) -> dict:
    path = cache_root / "source_build.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text())
    if report.get("status") != "exact_source_with_unreleased_pickle_and_filesystem_order":
        raise AssertionError("SMART source has an unapproved fidelity status")
    if report.get("producer_commit") != PRODUCER_COMMIT:
        raise AssertionError("SMART source was not built with the pinned producer")
    return report


def load_cohort(cache_root: Path) -> pl.DataFrame:
    report = load_build_report(cache_root)
    path = cache_root / report["files"]["cohort"]
    cohort = pl.read_parquet(path)
    required = {
        "subject_id",
        "hadm_id",
        "sample_id",
        "intime",
        "outtime",
        "source_name",
        "source_partition",
        "label",
        "source_position",
    }
    if not required.issubset(cohort.columns):
        raise AssertionError(f"SMART cohort is missing {sorted(required-set(cohort.columns))}")
    if cohort.height != EXPECTED_TOTAL:
        raise AssertionError(f"SMART cohort count changed: {cohort.height}")
    if cohort["sample_id"].n_unique() != EXPECTED_TOTAL:
        raise AssertionError("SMART ICU stay IDs are not unique")
    if cohort["source_name"].n_unique() != EXPECTED_TOTAL:
        raise AssertionError("SMART source names are not unique")
    return cohort.sort("source_position")


def split_assignment(cohort: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Reproduce SMART's released example shuffle for one paper seed."""
    if seed not in SEEDS:
        raise ValueError(f"unsupported SMART paper seed: {seed}")
    indices = list(range(cohort.height))
    random.Random(seed).shuffle(indices)
    train_end = int(cohort.height * 0.8)
    validation_end = train_end + int(cohort.height * 0.1)
    split = np.full(cohort.height, "test", dtype="U10")
    split[np.asarray(indices[:train_end], dtype=np.int64)] = "train"
    split[np.asarray(indices[train_end:validation_end], dtype=np.int64)] = "validation"
    result = cohort.with_columns(pl.Series("split", split))
    counts = dict(result.group_by("split").len().iter_rows())
    if counts != EXPECTED_SPLIT_COUNTS:
        raise AssertionError(f"SMART split counts changed for seed {seed}: {counts}")
    return result


def _author_reader():
    reader_parent = str(READER_PATH.parent)
    if reader_parent not in sys.path:
        sys.path.insert(0, reader_parent)
    from reader import InHospitalMortalityReader

    return InHospitalMortalityReader


def load_author_notebook_boundary(cache_root: Path):
    """Call the released reader and reproduce the mortality notebook cells."""
    cohort = load_cohort(cache_root)
    expected_names = cohort["source_name"].to_list()
    task_root = source_root(cache_root)
    Reader = _author_reader()
    config = configuration()
    info = channel_info()
    variables = variable_names()

    data_all: list[np.ndarray] = []
    mask_all: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []
    for mode in ("train", "val", "test"):
        reader = Reader(
            dataset_dir=str(task_root / ("test" if mode == "test" else "train")),
            listfile=str(task_root / f"{mode}_listfile.csv"),
            period_length=HOURS,
        )
        for index in range(reader.get_number_of_examples()):
            example = reader.read_example(index)
            patient = example["X"]
            data = np.zeros((len(variables), int(HOURS)), dtype=np.float32)
            mask = np.zeros_like(data)
            last_time = -1
            for row in patient:
                hour = int(float(row[0]))
                if hour == int(HOURS):
                    hour -= 1
                if hour > HOURS or hour < 0:
                    raise AssertionError("released IHM row lies outside [0,48]")
                for variable_index, value in enumerate(row[1:]):
                    variable = variables[variable_index]
                    if value == "":
                        if mask[variable_index, hour] == 0 and hour - last_time > 0:
                            if last_time >= 0:
                                data[variable_index, last_time + 1 : hour + 1] = data[
                                    variable_index, last_time
                                ]
                            elif config["is_categorical_channel"][variable]:
                                normal = config["normal_values"][variable]
                                data[variable_index, last_time + 1 : hour + 1] = info[
                                    variable
                                ]["values"][normal]
                            else:
                                data[variable_index, last_time + 1 : hour + 1] = float(
                                    config["normal_values"][variable]
                                )
                    else:
                        mask[variable_index, hour] = 1
                        if config["is_categorical_channel"][variable]:
                            data[variable_index, hour] = info[variable]["values"][value]
                        else:
                            data[variable_index, hour] = float(value)
                last_time = hour
            if last_time < int(HOURS) - 1:
                data[:, last_time + 1 : int(HOURS)] = data[:, last_time, None]
            data_all.append(data.T)
            mask_all.append(mask.T)
            labels.append(int(example["y"]))
            names.append(str(example["name"]))

    if names != expected_names:
        raise AssertionError("released reader order differs from frozen source ordering")
    raw = np.asarray(data_all, dtype=np.float32)
    mask = np.asarray(mask_all, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int64)
    masked = np.ma.masked_array(raw.reshape(-1, len(variables)), mask.reshape(-1, len(variables)) == 0)
    mean = np.asarray(np.mean(masked, axis=0), dtype=np.float64)
    std = np.asarray(np.std(masked, axis=0), dtype=np.float64)
    normalized = np.where(
        mask == 1,
        (raw - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1),
        0,
    ).astype(np.float32)
    return {
        "raw": raw,
        "mask": mask,
        "normalized": normalized,
        "labels": labels_array,
        "names": names,
        "mean": mean,
        "std": std,
    }
