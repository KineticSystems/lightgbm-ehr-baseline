"""KEDGN's attributable MIMIC-III structured-data source for the export.

The paper README explicitly names the standard ``mimic3-benchmarks`` IHM
producer followed by ``medical_ts_datasets``.  The latter's TensorFlow wrapper
does not add scientific transformations: its source reader returns the exact
irregular rows, concatenates the 16 dynamic variables, constructs
``is_finite`` masks, and reports the sequence length.  This module executes
that released reader without installing TensorFlow by providing only the
``tf.io.gfile.GFile`` file-opening surface it calls.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
TASK_ID = "mortality_48h"
HOURS = 48.0
SPLITS = ("train", "validation", "test")

KEDGN_COMMIT = "c82bc004cd1d1305fc58ca7150369d78682fccad"
BENCHMARK_COMMIT = "ea0314c7cbd369f62e2237ace6f683740f867e3a"
MEDICAL_TS_COMMIT = "051d1cc4d9e657920e0ade31f04ea391571e3fdd"
MEDICAL_TS_ROOT = (
    PAPER_ROOT / ".export_assets" / f"medical_ts_datasets-{MEDICAL_TS_COMMIT}"
)
MEDICAL_SOURCE = MEDICAL_TS_ROOT / "medical_ts_datasets"

EXPECTED_SOURCE_HASHES = {
    "mimic_3_reader.py": "887b3519fd7871e0ea03c266c067ec86d5d477af0b6a63c3c04fcf6ae0a9ba04",
    "mimic_3_mortality.py": "1e4bf0b85d8167f15d21290cce756f4ebc5b334d538990911907294d2770aa50",
    "util.py": "14c4dfe36ab623be57bcfc852c965a55fdb6a398765522ea0678c0d35a1b1e94",
}

RELEASE_SHA256 = {
    "PATIENTS.csv.gz": "0a13db69044d3b01ab9a397f3192792c2a1d2e98c9cb41c4f94e0403996dc7c5",
    "ADMISSIONS.csv.gz": "68f128f19e95434500982d95e623638f22eadf1c6a07f4bfa6c39c21751d8085",
    "ICUSTAYS.csv.gz": "65f8c8686e791bee80df0115fc18f9f3c1087b85889ba9ce2d9c441204e467ab",
    "CHARTEVENTS.csv.gz": "5961ac36d8bfbd399088811f9a1d30bd6f482b23a84628593d3122b5a6be8bc4",
    "LABEVENTS.csv.gz": "79d86315dde6e1121f7a88f62c2b91b7b835364ac1e777248a58e4f180224ae6",
    "OUTPUTEVENTS.csv.gz": "70f14321d228b12d243a10974b7e41e41690031d6dbc16354ef78e3d1dfb4445",
}

EXPECTED_TOTAL = 21_107
EXPECTED_SPLIT_COUNTS = {"train": 14_662, "validation": 3_217, "test": 3_228}
EXPECTED_SPLIT_POSITIVES = {"train": 1_982, "validation": 435, "test": 374}
EXPECTED_SPLIT_SUBJECTS = {"train": 12_551, "validation": 2_762, "test": 2_757}
EXPECTED_POSITIVES = 2_791
EXPECTED_MAX_LENGTH = 292
EXPECTED_MAX_READER_LENGTH = 311
EXPECTED_DUPLICATE_ROWS_COLLAPSED = 137_829
EXPECTED_EVENT_COUNT = 9_089_782
EXPECTED_MISSING_RATIO = 0.6550086656007229
EXPECTED_BLACKLIST_REMOVALS = {"train": 19, "validation": 5, "test": 8}


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


def array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
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


def assert_medical_ts_source() -> dict[str, str]:
    actual = {}
    for filename, expected in EXPECTED_SOURCE_HASHES.items():
        path = MEDICAL_SOURCE / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"pinned medical_ts_datasets source is missing: {path}"
            )
        actual[filename] = sha256(path)
        if actual[filename] != expected:
            raise AssertionError(f"pinned medical_ts_datasets source changed: {filename}")
    return actual


@lru_cache(maxsize=1)
def mortality_reader_class():
    """Load the exact public reader with a file-I/O-only TensorFlow shim."""
    assert_medical_ts_source()
    package_name = "kedgn_export_medical_ts_datasets"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MEDICAL_SOURCE)]
    sys.modules[package_name] = package

    tensorflow = types.ModuleType("tensorflow")
    tensorflow.io = types.SimpleNamespace(
        gfile=types.SimpleNamespace(GFile=lambda path, mode="r": open(path, mode))
    )
    previous_tensorflow = sys.modules.get("tensorflow")
    sys.modules["tensorflow"] = tensorflow

    reader_name = f"{package_name}.mimic_3_reader"
    reader_spec = importlib.util.spec_from_file_location(
        reader_name, MEDICAL_SOURCE / "mimic_3_reader.py"
    )
    if reader_spec is None or reader_spec.loader is None:
        raise ImportError("could not load pinned medical_ts_datasets reader")
    reader_module = importlib.util.module_from_spec(reader_spec)
    sys.modules[reader_name] = reader_module
    reader_spec.loader.exec_module(reader_module)
    original_read_instance = reader_module.MIMICReader._read_data_for_instance

    def read_instance_with_declared_none_category(self, filename):
        # The released reader explicitly maps the literal GCS eye-opening
        # category "None" to 1.  Current pandas added that token to its global
        # default-NA set, swallowing it before the released replacement runs;
        # the dependency was unpinned.  Narrowly restore only this declared
        # category for the duration of the exact released method call.  This
        # also recovers the paper's independently reported 65.5% missingness.
        na_values = reader_module.pd._libs.parsers.STR_NA_VALUES
        had_none = "None" in na_values
        if had_none:
            na_values.remove("None")
        try:
            return original_read_instance(self, filename)
        finally:
            if had_none:
                na_values.add("None")

    reader_module.MIMICReader._read_data_for_instance = (
        read_instance_with_declared_none_category
    )

    util_name = f"{package_name}.util"
    util_module = types.ModuleType(util_name)
    util_module.MedicalTsDatasetBuilder = object
    util_module.MedicalTsDatasetInfo = object
    sys.modules[util_name] = util_module

    tensorflow_datasets = types.ModuleType("tensorflow_datasets")
    tensorflow_datasets.core = types.SimpleNamespace(
        Version=lambda value: value,
        SplitGenerator=lambda **kwargs: kwargs,
    )
    tensorflow_datasets.features = types.SimpleNamespace(
        ClassLabel=lambda **kwargs: kwargs
    )
    tensorflow_datasets.download = types.SimpleNamespace(DownloadManager=object)
    tensorflow_datasets.Split = types.SimpleNamespace(
        TRAIN="train", VALIDATION="validation", TEST="test"
    )
    previous_tfds = sys.modules.get("tensorflow_datasets")
    sys.modules["tensorflow_datasets"] = tensorflow_datasets

    mortality_name = f"{package_name}.mimic_3_mortality"
    mortality_spec = importlib.util.spec_from_file_location(
        mortality_name, MEDICAL_SOURCE / "mimic_3_mortality.py"
    )
    if mortality_spec is None or mortality_spec.loader is None:
        raise ImportError("could not load pinned medical_ts_datasets mortality reader")
    mortality_module = importlib.util.module_from_spec(mortality_spec)
    sys.modules[mortality_name] = mortality_module
    try:
        mortality_spec.loader.exec_module(mortality_module)
    finally:
        if previous_tensorflow is None:
            sys.modules.pop("tensorflow", None)
        else:
            sys.modules["tensorflow"] = previous_tensorflow
        if previous_tfds is None:
            sys.modules.pop("tensorflow_datasets", None)
        else:
            sys.modules["tensorflow_datasets"] = previous_tfds
    return mortality_module.MIMICMortalityReader


def variable_names() -> tuple[str, ...]:
    Reader = mortality_reader_class()
    names = (*Reader.vitals, *Reader.lab_measurements, *Reader.interventions)
    if len(names) != 16 or len(set(names)) != 16:
        raise AssertionError("medical_ts_datasets variable declaration changed")
    return tuple(names)


def code_vocabulary() -> tuple[str, ...]:
    """The ordered names paired with KEDGN's released semantic tensor rows."""
    path = PAPER_ROOT / "data" / "mimic3" / "mimic3_variables_name.json"
    names = tuple(json.loads(path.read_text()).keys())
    if len(names) != 16 or len(set(names)) != 16:
        raise AssertionError("KEDGN MIMIC semantic-variable declaration changed")
    return names


AVERAGE_INTERVAL_SUFFIX = "//KEDGN_AVERAGE_INTERVAL"
SEQUENCE_LENGTH_CODE = "KEDGN_SEQUENCE_LENGTH"


def export_code_vocabulary() -> tuple[str, ...]:
    values = code_vocabulary()
    intervals = tuple(f"{code}{AVERAGE_INTERVAL_SUFFIX}" for code in values)
    return (*values, *intervals, SEQUENCE_LENGTH_CODE)


def kedgn_visible_fields(example: "SourceExample"):
    """Return the exact patient-varying fields visible to KEDGN's model."""
    _, time, values, source_mask, length = example.kedgn_tuple()
    normalized_time = time / 48.0
    model_mask = values > 0
    average = np.zeros((length, 16), dtype=np.float64)
    for channel in range(16):
        positions = np.where(source_mask[:, channel])[0]
        if len(positions) == 1:
            average[positions, channel] = length / 2
        elif len(positions) > 1:
            observed_time = normalized_time[positions]
            gaps = observed_time[1:] - observed_time[:-1]
            right = np.insert(gaps, -1, gaps[-1])
            left = np.insert(gaps, 0, gaps[0])
            average[positions, channel] = (left + right) / 2
    return values, model_mask, normalized_time, average.astype(np.float32), length


def blacklist() -> tuple[str, ...]:
    names = tuple(mortality_reader_class().blacklist)
    if len(names) != 32 or len(set(names)) != 32:
        raise AssertionError("medical_ts_datasets mortality blacklist changed")
    return names


def source_task_root(cache_root: Path) -> Path:
    return cache_root / "in-hospital-mortality"


def split_paths(cache_root: Path, split: str) -> tuple[Path, Path]:
    task_root = source_task_root(cache_root)
    if split == "train":
        return task_root / "train", task_root / "train_listfile.csv"
    if split == "validation":
        return task_root / "train", task_root / "val_listfile.csv"
    if split == "test":
        return task_root / "test", task_root / "test_listfile.csv"
    raise ValueError(f"unsupported split: {split}")


@dataclass(frozen=True)
class SourceExample:
    split: str
    split_position: int
    source_name: str
    subject_id: int
    sample_id: int
    intime: object
    label: int
    demographics: np.ndarray
    time: np.ndarray
    values: np.ndarray
    measurements: np.ndarray
    length: int
    reader_length: int
    duplicate_rows_collapsed: int

    def kedgn_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        # The public reader retains NaN and an explicit mask.  KEDGN's released
        # tensorizer multiplies by a positivity mask, for which NaN*0 remains
        # NaN.  A finite zero fill is therefore the sole mechanically viable
        # serialization into the README-requested .npy tuple.  The original
        # finite-cell mask is retained separately and used for intervals.
        finite_values = np.nan_to_num(
            self.values, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)
        return (
            self.demographics.copy(),
            self.time.copy(),
            finite_values,
            self.measurements.copy(),
            self.length,
        )


def load_build_report(cache_root: Path) -> dict:
    path = cache_root / "source_build.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text())
    if report.get("status") != "exact_structured_source_boundary":
        raise AssertionError("KEDGN source report has an unapproved status")
    if report.get("benchmark_commit") != BENCHMARK_COMMIT:
        raise AssertionError("KEDGN source used a different mimic3-benchmarks revision")
    if report.get("medical_ts_commit") != MEDICAL_TS_COMMIT:
        raise AssertionError("KEDGN source used a different medical_ts_datasets revision")
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
        "split",
        "split_position",
        "source_position",
        "label",
    }
    if not required.issubset(cohort.columns):
        raise AssertionError(f"KEDGN cohort is missing {sorted(required-set(cohort.columns))}")
    if cohort.height != EXPECTED_TOTAL:
        raise AssertionError(f"KEDGN cohort count changed: {cohort.height}")
    return cohort.sort("source_position")


def iter_source_examples(cache_root: Path, split: str):
    """Execute the exact public reader and emit its supervised 5-tuple fields."""
    cohort = load_cohort(cache_root).filter(pl.col("split") == split).sort(
        "split_position"
    )
    data_dir, listfile = split_paths(cache_root, split)
    Reader = mortality_reader_class()
    reader = Reader(str(data_dir), str(listfile))
    if len(reader) != cohort.height:
        raise AssertionError(
            f"medical_ts_datasets {split} count changed: {len(reader)} != {cohort.height}"
        )
    names = variable_names()
    for position in range(len(reader)):
        source_name, instance = reader[position]
        row = cohort.row(position, named=True)
        if source_name != row["source_name"]:
            raise AssertionError("medical_ts_datasets order differs from prepared cohort")
        target = int(instance["targets"]["In_hospital_mortality"])
        if target != int(row["label"]):
            raise AssertionError("medical_ts_datasets label differs from prepared cohort")
        time = np.asarray(instance["time"], dtype=np.float32)
        values = np.concatenate(
            [
                np.asarray(instance["vitals"], dtype=np.float32),
                np.asarray(instance["lab_measurements"], dtype=np.float32),
                np.asarray(instance["interventions"], dtype=np.float32),
            ],
            axis=1,
        )
        raw_length = len(time)
        # mimic3-benchmarks can emit two rows at the same exact Hours value
        # when separate source events contribute disjoint cells.  KEDGN reports
        # a maximum length of 292 rather than the one raw 311-row sequence;
        # coalescing equal adjacent times recovers that maximum.  The full
        # cohort has no conflicting finite channel values within any such
        # time group, so this is a lossless mechanical cache representation.
        unique_time, inverse = np.unique(time, return_inverse=True)
        if len(unique_time) != len(time):
            coalesced = np.full((len(unique_time), values.shape[1]), np.nan, dtype=np.float32)
            for raw_row, target_row in enumerate(inverse):
                observed = np.isfinite(values[raw_row])
                existing = np.isfinite(coalesced[target_row])
                overlap = observed & existing
                if np.any(
                    values[raw_row, overlap] != coalesced[target_row, overlap]
                ):
                    raise AssertionError(
                        "same-time benchmark rows contain conflicting channel values"
                    )
                coalesced[target_row, observed] = values[raw_row, observed]
            time = unique_time.astype(np.float32, copy=False)
            values = coalesced
        measurements = np.isfinite(values)
        if values.shape != (len(time), len(names)):
            raise AssertionError("medical_ts_datasets supervised tuple shape changed")
        if not np.all(np.isfinite(time)) or np.any(np.diff(time) < 0):
            raise AssertionError("medical_ts_datasets time is non-finite or unsorted")
        if len(time) > EXPECTED_MAX_LENGTH:
            raise AssertionError(
                "medical_ts_datasets sequence exceeds paper maximum: "
                f"{source_name} has {len(time)} rows"
            )
        demographics = np.asarray(instance["demographics"], dtype=np.float32)
        yield SourceExample(
            split=split,
            split_position=position,
            source_name=source_name,
            subject_id=int(row["subject_id"]),
            sample_id=int(row["sample_id"]),
            intime=row["intime"],
            label=target,
            demographics=demographics,
            time=time,
            values=values,
            measurements=measurements,
            length=len(time),
            reader_length=raw_length,
            duplicate_rows_collapsed=raw_length - len(time),
        )
