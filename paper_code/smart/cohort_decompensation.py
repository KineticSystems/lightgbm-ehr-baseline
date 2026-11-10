"""SMART's exact-source MIMIC-III decompensation boundary (phase 2).

The paper's second binary MIMIC endpoint.  SMART replaces the producer's
decompensation task builder with its own
``data/MIMIC-III/create_decompensation.py``, which emits **one** example per
eligible ICU stay at the released landmark

    t = min(24, int(los), int(lived_time))

with label ``mortality and (lived_time - t < 24)``.  ``data/mimiciii.py::
load_mimic_iii_decompensation`` then redraws an example-level 80/10/10 split
per paper seed, exactly like the mortality task.

This module mirrors ``cohort_source.py`` and reuses its released-resource
helpers; only the task builder, landmark, and variable-length binning differ.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from cohort_source import (  # noqa: F401  (re-exported for the checkers)
    CHANNEL_INFO_PATH,
    COLLATE_PATH,
    CONFIG_PATH,
    LOADER_PATH,
    NOTEBOOK_PATH,
    PAPER_ROOT,
    PRODUCER_COMMIT,
    READER_PATH,
    SEEDS,
    array_hash,
    channel_info,
    code_vocabulary,
    configuration,
    ordered_text_hash,
    sha256,
    variable_names,
)


TASK_ID = "decompensation"
TASK_DIRNAME = "decompensation"
BUILD_REPORT = "decompensation_build.json"
# The pinned producer chain and SMART's own task builder were run here on exact
# MIMIC-III 1.4, but the authors' historical build environment is unreleased and
# reproduces 18 more eligible stays than a clean rerun (see the export notes).
BUILD_STATUS = "exact_source_with_rebuilt_producer_cohort_and_unreleased_order"
SMART_TASK_BUILDER = PAPER_ROOT / "data" / "MIMIC-III" / "create_decompensation.py"
LISTFILE_HEADER = ["stay", "period_length", "y_true"]

# SMART's released notebook cell for `path = 'decompensation'`.
NOTEBOOK_PERIOD_LENGTH = 24

# ---------------------------------------------------------------------------
# Paper/notebook targets.  Saved outputs of the released decompensation cells in
# paper_code/smart/data/MIMIC-III/preprocess_mimiciii.ipynb (29,143 + 6,346 +
# 6,255 = 41,744 examples, positive fraction 0.035310463779225754, observed rate
# 0.4313917233395309) and Appendix A.1 ("We extract 41,744 patients The observed
# rate is 43.2% and the positive rate is 3.5%").
# ---------------------------------------------------------------------------
PAPER_TOTAL = 41_744
PAPER_POSITIVES = 1_474
PAPER_POSITIVE_FRACTION = 0.035310463779225754
PAPER_OBSERVED_RATE = 0.4313917233395309
PAPER_LISTFILE_COUNTS = {"train": 29_143, "validation": 6_346, "test": 6_255}

# float32 moments printed by the released decompensation notebook cell.
NOTEBOOK_MEAN = np.asarray(
    [
        3.6462824e-04, 6.7509583e01, 2.3202808e-01, 4.0216231e00, 6.1931319e00,
        1.4663564e01, 4.8571868e00, 1.3957463e02, 9.2549011e01, 1.6925241e02,
        8.6117294e01, 1.0576494e02, 2.1513866e01, 1.3186606e02, 3.8373699e01,
        8.1611977e01, 7.6491199e00,
    ],
    dtype=np.float64,
)
NOTEBOOK_STD = np.asarray(
    [
        1.9084949e-02, 1.9184895e02, 1.1985364e-01, 1.3376771e00, 1.9729450e00,
        3.5173326e00, 1.7233016e00, 3.4439746e02, 3.7673790e01, 2.2092722e01,
        1.2524643e02, 1.0012802e03, 1.2013160e03, 1.4728951e02, 1.1425072e01,
        1.7897316e01, 2.0367055e00,
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# Frozen facts of THIS reconstruction (pinned producer @ ea0314c7 with SMART's
# documented gzip/Python-3 repairs, run in paper_code/smart/.venv on exact
# MIMIC-III 1.4, then SMART's own create_decompensation.py and the producer's
# mimic3models.split_train_val).  These are asserted exactly; the paper targets
# above are compared and their deltas recorded.
# ---------------------------------------------------------------------------
EXPECTED_TOTAL = 41_726
EXPECTED_POSITIVES = 1_471
EXPECTED_LISTFILE_COUNTS = {"train": 29_131, "validation": 6_344, "test": 6_251}
# data/mimiciii.py::load_mimic_iii_decompensation with training_ratio=0.8.
EXPECTED_SPLIT_COUNTS = {"train": 33_380, "validation": 4_172, "test": 4_174}
# Measured by running the released notebook cell over this reconstruction;
# asserted exactly on every later run so the rebuild stays reproducible.
EXPECTED_OBSERVED_RATE = 0.43041333401883664
EXPECTED_SEQUENCE_LENGTH_SUM = 961_080
EXPECTED_RETAINED_SOURCE_ROWS = 1_741_644
REBUILT_MEAN = np.asarray(
    [
        0.00036469390033744276, 67.54373168945312, 0.2320653647184372,
        4.060528755187988, 6.196147441864014, 14.668959617614746,
        4.859665393829346, 142.33079528808594, 92.59937286376953,
        169.26638793945312, 86.16222381591797, 105.86601257324219,
        21.52414321899414, 131.93333435058594, 38.39031219482422,
        81.62217712402344, 7.814132213592529,
    ],
    dtype=np.float64,
)
REBUILT_STD = np.asarray(
    [
        0.01908666454255581, 191.87051391601562, 0.11994165182113647,
        1.2837061882019043, 1.9760594367980957, 3.522886037826538,
        1.7246007919311523, 513.4312744140625, 37.73469161987305,
        22.074350357055664, 125.26963806152344, 1001.3737182617188,
        1201.4241943359375, 147.32533264160156, 11.452524185180664,
        17.902151107788086, 2.4129319190979004,
    ],
    dtype=np.float64,
)
# Bounds on the gap between this rerun and the authors' unreleased historical
# build, not tolerances that could hide a source change.  The measured gaps are
# 18 examples, 3 positives, 9.78e-4 observed rate, 2.2% on a channel mean and
# 49% on a channel standard deviation (Glucose and pH, whose moments are
# dominated by extreme raw lab values because this producer revision declares
# `--reference_range_file` but never calls `remove_outliers_for_variable`).
MAX_SAMPLE_DELTA = 40
MAX_POSITIVE_DELTA = 10
MAX_OBSERVED_RATE_DELTA = 2e-3
MAX_NOTEBOOK_MEAN_RELATIVE_DELTA = 0.05
MAX_NOTEBOOK_STD_RELATIVE_DELTA = 0.60

PUBLISHED_AUROC = 0.9420
PUBLISHED_AUROC_SD = 0.0012
PUBLISHED_CELL = (
    "Appendix Table 3, MIMIC-III Decompensation, SMART AUROC 94.20 +/- 0.12%"
)


def subject_of(source_name: str) -> int:
    return int(source_name.split("_", 1)[0])


def replica_task_id(replica: int) -> str:
    if replica not in (1, 2, 3):
        raise ValueError(replica)
    return f"{TASK_ID}_replica_{replica:02d}"


def source_root(cache_root: Path) -> Path:
    return cache_root / TASK_DIRNAME


def load_build_report(cache_root: Path) -> dict:
    path = cache_root / BUILD_REPORT
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text())
    if report.get("status") != BUILD_STATUS:
        raise AssertionError("SMART decompensation source has an unapproved status")
    if report.get("producer_commit") != PRODUCER_COMMIT:
        raise AssertionError("SMART decompensation source used another producer")
    return report


def load_cohort(cache_root: Path) -> pl.DataFrame:
    report = load_build_report(cache_root)
    cohort = pl.read_parquet(cache_root / report["files"]["cohort"])
    required = {
        "subject_id",
        "hadm_id",
        "sample_id",
        "intime",
        "outtime",
        "source_name",
        "source_partition",
        "label",
        "period_length",
        "source_position",
    }
    missing = required - set(cohort.columns)
    if missing:
        raise AssertionError(f"SMART decompensation cohort is missing {sorted(missing)}")
    if cohort.height != EXPECTED_TOTAL:
        raise AssertionError(f"SMART decompensation cohort count changed: {cohort.height}")
    if cohort["sample_id"].n_unique() != EXPECTED_TOTAL:
        raise AssertionError("SMART decompensation ICU stay IDs are not unique")
    if cohort["source_name"].n_unique() != EXPECTED_TOTAL:
        raise AssertionError("SMART decompensation source names are not unique")
    if int(cohort["label"].sum()) != EXPECTED_POSITIVES:
        raise AssertionError("SMART decompensation positive marginal changed")
    return cohort.sort("source_position")


def split_assignment(cohort: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Reproduce SMART's released example shuffle for one paper seed."""
    import random

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
        raise AssertionError(
            f"SMART decompensation split counts changed for seed {seed}: {counts}"
        )
    return result


def listfile_facts(task_root: Path) -> dict[str, dict[str, object]]:
    """Read the frozen train/val/test listfiles the notebook consumes."""
    result: dict[str, dict[str, object]] = {}
    for mode, common in (("train", "train"), ("val", "validation"), ("test", "test")):
        path = task_root / f"{mode}_listfile.csv"
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header != LISTFILE_HEADER:
                raise AssertionError(f"decompensation listfile header changed: {header}")
            for position, (name, period_length, label) in enumerate(reader):
                if name in result:
                    raise AssertionError(f"duplicate decompensation sample: {name}")
                result[name] = {
                    "period_length": float(period_length),
                    "label": int(label),
                    "notebook_mode": mode,
                    "common_split": common,
                    "mode_position": position,
                }
    if len(result) != EXPECTED_TOTAL:
        raise AssertionError("frozen decompensation listfiles lost a sample")
    return result


def _author_reader():
    parent = str(READER_PATH.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from reader import DecompensationReader

    return DecompensationReader


def load_author_notebook_boundary(cache_root: Path) -> dict:
    """Reproduce the released decompensation notebook cells exactly.

    ``preprocess_mimiciii.ipynb`` cell ``path = 'decompensation'``: variable
    ``N_bins = min(int(t + 1 - 1e-6), 24)`` bins, per-bin mean of the observed
    values, ``normal_values`` fill for unobserved cells, and (unlike the
    mortality cell) normalization applied to **every** cell rather than only
    observed ones.
    """
    cohort = load_cohort(cache_root)
    expected_names = cohort["source_name"].to_list()
    task_root = source_root(cache_root)
    Reader = _author_reader()
    config = configuration()
    info = channel_info()
    variables = variable_names()
    categorical = config["is_categorical_channel"]
    normal_values = config["normal_values"]

    data_all: list[np.ndarray] = []
    mask_all: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []
    period_lengths: list[float] = []
    retained_rows = 0
    empty_rows = 0
    for mode in ("train", "val", "test"):
        reader = Reader(
            dataset_dir=str(task_root / ("test" if mode == "test" else "train")),
            listfile=str(task_root / f"{mode}_listfile.csv"),
        )
        for index in range(reader.get_number_of_examples()):
            example = reader.read_example(index)
            patient = example["X"]
            t = float(example["t"])
            n_bins = min(int(t + 1 - 1e-6), NOTEBOOK_PERIOD_LENGTH)
            data = np.zeros((len(variables), n_bins), dtype=np.float32)
            mask = np.zeros((len(variables), n_bins), dtype=np.float32)
            last_time = -1
            for row in patient:
                retained_rows += 1
                if all(value == "" for value in row[1:]):
                    empty_rows += 1
                hour = int(float(row[0]))
                if hour == n_bins:
                    hour -= 1
                if hour > n_bins:
                    break
                for variable_index, value in enumerate(row[1:]):
                    variable = variables[variable_index]
                    if value == "":
                        if mask[variable_index, hour] == 0 and hour - last_time > 0:
                            if categorical[variable]:
                                fill = info[variable]["values"][normal_values[variable]]
                            else:
                                fill = float(normal_values[variable])
                            data[variable_index, last_time + 1 : hour + 1] = fill
                    else:
                        mask[variable_index, hour] += 1
                        if categorical[variable]:
                            data[variable_index, hour] += info[variable]["values"][value]
                        else:
                            data[variable_index, hour] += float(value)
                last_time = hour
            with np.errstate(divide="ignore", invalid="ignore"):
                data = np.where(mask > 0, data / mask, data)
            # The released notebook pickles `np.where(mask > 0, 1, 0).tolist()`
            # for decompensation, so SMART's collate_fn builds an **int64** mask
            # here (the mortality cell keeps float32).  Preserve that.
            mask = np.where(mask > 0, 1, 0)
            data_all.append(data.T)
            mask_all.append(np.ascontiguousarray(mask.T, dtype=np.int64))
            labels.append(int(example["y"]))
            names.append(str(example["name"]))
            period_lengths.append(t)

    if names != expected_names:
        raise AssertionError("released reader order differs from the frozen source order")
    concatenated = np.concatenate(data_all, axis=0)
    mean = np.mean(concatenated, axis=0)
    std = np.std(concatenated, axis=0)
    normalized = [
        ((data - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)
        for data in data_all
    ]
    return {
        "raw": data_all,
        "mask": mask_all,
        "normalized": normalized,
        "labels": np.asarray(labels, dtype=np.int64),
        "names": names,
        "period_lengths": np.asarray(period_lengths, dtype=np.float64),
        "mean": np.asarray(mean, dtype=np.float64),
        "std": np.asarray(std, dtype=np.float64),
        "retained_source_rows": retained_rows,
        "retained_source_rows_with_no_observed_cell": empty_rows,
    }


def observed_rate(mask_all: list[np.ndarray]) -> float:
    observed = 0.0
    total = 0
    for mask in mask_all:
        observed += float(mask.sum())
        total += mask.size
    return observed / total
