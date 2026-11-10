"""Exact released Stepwise/TLS boundary projected into the MEDS task."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import tables


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import EVENT_SCHEMA  # noqa: E402
from missing_xgboost_lightgbm.landmark_sampling import membership_mask  # noqa: E402


TASK_ID = "decompensation"
AUTHOR_TASK = "decomp_24Hours"
SPLIT_MAP = ("train", "val", "test")
COMMON_SPLIT = {"train": "train", "val": "validation", "test": "test"}
ANCHOR_US = np.datetime64("2000-01-01T00:00:00", "us").astype(np.int64)
HOUR_US = 3_600_000_000
M3B_COMMIT = "ea0314c7cbd369f62e2237ace6f683740f867e3a"
TLS_COMMIT = "0582563440eee1b7998b68165a96cefe92d6077d"
CLINICAL_EMBEDDINGS_COMMIT = "e66cbdde0aff7dde7197d4520ec9df29b3edd58a"
EXPECTED_FEATURE_COUNT = 42
EXPECTED_NUMERIC_COLUMNS = (
    "Height",
    "Hours",
    "Diastolic blood pressure",
    "Fraction inspired oxygen",
    "Glucose",
    "Heart Rate",
    "Mean blood pressure",
    "Oxygen saturation",
    "Respiratory rate",
    "Systolic blood pressure",
    "Temperature",
    "Weight",
    "pH",
)
SOURCE_FILENAME = re.compile(r"^(?P<subject>\d+)_episode(?P<episode>\d+)_timeseries\.csv$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(rows: Iterator[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def decode(values) -> tuple[str, ...]:
    return tuple(value.decode("utf-8") for value in values)


def source_paths(source_root: Path) -> dict[str, Path]:
    source_root = source_root.resolve()
    return {
        "preparation": source_root / "preparation.json",
        "multitask": source_root / "m3b_data" / "multitask",
        "non_scaled": source_root / "tls_hdf" / "non_scaled.h5",
        "standard_scaled": source_root / "tls_hdf" / "standard_scaled.h5",
    }


def assert_source_ready(source_root: Path) -> dict[str, Path]:
    paths = source_paths(source_root)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    preparation = json.loads(paths["preparation"].read_text())
    if preparation.get("status") != "exact_source":
        raise AssertionError("source preparation is not exact")
    return paths


def listfile_path(multitask: Path, split: str) -> Path:
    name = "val" if split == "val" else split
    return multitask / f"{name}_listfile.csv"


def source_filenames(multitask: Path, split: str) -> list[str]:
    frame = pd.read_csv(listfile_path(multitask, split), usecols=["filename"])
    names = frame["filename"].astype(str).tolist()
    if len(names) != len(set(names)):
        raise AssertionError(f"{split} listfile contains duplicate stays")
    for name in names:
        if SOURCE_FILENAME.fullmatch(name) is None:
            raise AssertionError(f"unrecognized source filename: {name}")
    return names


def parse_filename(name: str) -> tuple[int, int]:
    match = SOURCE_FILENAME.fullmatch(name)
    if match is None:
        raise ValueError(name)
    return int(match.group("subject")), int(match.group("episode"))


def paper_encoded_stay(subject_id: int, episode: int) -> int:
    return int(str(episode)[-1]) * 10_000_000 + subject_id


def inspect_hdf(source_root: Path) -> dict[str, object]:
    paths = assert_source_ready(source_root)
    with tables.open_file(paths["standard_scaled"], "r") as handle:
        columns = decode(handle.root.data.columns[:])
        tasks = decode(handle.root.labels.tasks[:])
        if len(columns) != EXPECTED_FEATURE_COUNT:
            raise AssertionError(f"expected 42 author channels, found {len(columns)}")
        if columns[: len(EXPECTED_NUMERIC_COLUMNS)] != EXPECTED_NUMERIC_COLUMNS:
            raise AssertionError("author numeric channel order changed")
        if AUTHOR_TASK not in tasks:
            raise AssertionError(f"missing author task {AUTHOR_TASK}")
        categorical = columns[len(EXPECTED_NUMERIC_COLUMNS) :]
        if not categorical or not all("_cat" in name for name in categorical):
            raise AssertionError("author one-hot channel declaration changed")
        return {
            "columns": columns,
            "numeric_columns": columns[: len(EXPECTED_NUMERIC_COLUMNS)],
            "categorical_columns": categorical,
            "tasks": tasks,
            "task_index": tasks.index(AUTHOR_TASK),
        }


def build_samples(source_root: Path) -> tuple[pl.DataFrame, dict[str, object]]:
    paths = assert_source_ready(source_root)
    metadata = inspect_hdf(source_root)
    task_index = int(metadata["task_index"])
    fields: dict[str, list | np.ndarray] = {
        "subject_id": [],
        "sample_id": [],
        "observation_start": [],
        "prediction_time": [],
        "boolean_value": [],
        "split": [],
        "source_filename": [],
        "source_stay_order": [],
        "author_target_row": [],
        "source_cutoff_hour": [],
        "source_label_row": [],
        "source_order": [],
        "paper_encoded_stay_id": [],
        "episode_number": [],
    }
    source_order = 0
    sample_id = 1
    split_summary: dict[str, dict[str, object]] = {}
    tuple_hasher = hashlib.sha256()
    with tables.open_file(paths["standard_scaled"], "r") as handle:
        root = handle.root
        for split in SPLIT_MAP:
            names = source_filenames(paths["multitask"], split)
            windows = root.patient_windows._f_get_child(split)[:]
            labels = root.labels._f_get_child(split)[:, task_index]
            if len(names) != len(windows):
                raise AssertionError(f"{split} listfile/HDF stay count differs")
            positive = 0
            landmarks = 0
            subjects = set()
            max_target_row = 0
            for stay_order, (name, window) in enumerate(zip(names, windows)):
                start, stop, encoded = (int(value) for value in window)
                subject_id, episode = parse_filename(name)
                if encoded != paper_encoded_stay(subject_id, episode):
                    raise AssertionError("TLS encoded stay ID differs from released formula")
                relative = labels[start:stop]
                target_rows = np.flatnonzero(np.isfinite(relative))
                target_values = relative[target_rows]
                if target_rows.size == 0:
                    continue
                if not np.isin(target_values, (0.0, 1.0)).all():
                    raise AssertionError("eligible author labels are not binary")
                if np.any(target_rows < 1):
                    raise AssertionError("decompensation target lacks shifted source cutoff")
                count = int(target_rows.size)
                subjects.add(subject_id)
                landmarks += count
                positive += int(np.sum(target_values == 1.0))
                max_target_row = max(max_target_row, int(target_rows[-1]))
                ids = np.arange(sample_id, sample_id + count, dtype=np.int64)
                fields["subject_id"].extend([subject_id] * count)
                fields["sample_id"].extend(ids.tolist())
                fields["observation_start"].extend([ANCHOR_US] * count)
                fields["prediction_time"].extend(
                    (ANCHOR_US + target_rows.astype(np.int64) * HOUR_US).tolist()
                )
                fields["boolean_value"].extend((target_values == 1.0).tolist())
                fields["split"].extend([COMMON_SPLIT[split]] * count)
                fields["source_filename"].extend([name] * count)
                fields["source_stay_order"].extend([stay_order] * count)
                fields["author_target_row"].extend(target_rows.astype(int).tolist())
                fields["source_cutoff_hour"].extend((target_rows - 1).astype(int).tolist())
                fields["source_label_row"].extend((start + target_rows).astype(int).tolist())
                fields["source_order"].extend(
                    range(source_order, source_order + count)
                )
                fields["paper_encoded_stay_id"].extend([encoded] * count)
                fields["episode_number"].extend([episode] * count)
                for row, value in zip(target_rows, target_values):
                    tuple_hasher.update(
                        f"{split}|{name}|{int(row)-1}|{int(row)}|{int(value)}\n".encode()
                    )
                source_order += count
                sample_id += count
            split_summary[COMMON_SPLIT[split]] = {
                "stays_in_listfile": len(names),
                "stays_with_targets": int(
                    np.sum(
                        [
                            np.isfinite(labels[int(start) : int(stop)]).any()
                            for start, stop, _ in windows
                        ]
                    )
                ),
                "subjects_with_targets": len(subjects),
                "landmarks": landmarks,
                "positive": positive,
                "negative": landmarks - positive,
                "max_author_target_row": max_target_row,
            }
    samples = pl.DataFrame(
        {
            "subject_id": pl.Series(fields["subject_id"], dtype=pl.Int64),
            "sample_id": pl.Series(fields["sample_id"], dtype=pl.Int64),
            "observation_start": pl.Series(
                fields["observation_start"], dtype=pl.Int64
            ).cast(pl.Datetime("us")),
            "prediction_time": pl.Series(
                fields["prediction_time"], dtype=pl.Int64
            ).cast(pl.Datetime("us")),
            "boolean_value": pl.Series(fields["boolean_value"], dtype=pl.Boolean),
            "split": pl.Series(fields["split"], dtype=pl.String),
            "source_filename": pl.Series(
                fields["source_filename"], dtype=pl.String
            ),
            "source_stay_order": pl.Series(
                fields["source_stay_order"], dtype=pl.Int32
            ),
            "author_target_row": pl.Series(
                fields["author_target_row"], dtype=pl.Int32
            ),
            "source_cutoff_hour": pl.Series(
                fields["source_cutoff_hour"], dtype=pl.Int32
            ),
            "source_label_row": pl.Series(
                fields["source_label_row"], dtype=pl.Int64
            ),
            "source_order": pl.Series(fields["source_order"], dtype=pl.Int64),
            "paper_encoded_stay_id": pl.Series(
                fields["paper_encoded_stay_id"], dtype=pl.Int64
            ),
            "episode_number": pl.Series(fields["episode_number"], dtype=pl.Int32),
        }
    )
    audit = {
        "splits": split_summary,
        "ordered_target_tuple_sha256": tuple_hasher.hexdigest(),
        "landmarks": samples.height,
        "max_author_target_row": int(samples["author_target_row"].max()),
    }
    return samples, audit


def count_events(
    source_root: Path, selected_sample_ids: np.ndarray | None = None
) -> tuple[int, dict[str, int]]:
    paths = assert_source_ready(source_root)
    metadata = inspect_hdf(source_root)
    task_index = int(metadata["task_index"])
    numeric_count = len(metadata["numeric_columns"])
    total = 0
    split_counts = {}
    selected_ids = None
    if selected_sample_ids is not None:
        selected_ids = np.sort(np.asarray(selected_sample_ids, dtype=np.int64))
        if len(selected_ids) == 0:
            raise ValueError("selected_sample_ids must not be empty")
    with tables.open_file(paths["standard_scaled"], "r") as handle:
        root = handle.root
        sample_id = 1
        for split in SPLIT_MAP:
            labels = root.labels._f_get_child(split)[:, task_index]
            data = root.data._f_get_child(split)
            windows = root.patient_windows._f_get_child(split)[:]
            split_total = 0
            for start, stop, _ in windows:
                start = int(start)
                stop = int(stop)
                target_rows = np.flatnonzero(np.isfinite(labels[start:stop]))
                if target_rows.size == 0:
                    continue
                source_sample_ids = np.arange(
                    sample_id, sample_id + len(target_rows), dtype=np.int64
                )
                sample_id += len(target_rows)
                if selected_ids is not None:
                    target_rows = target_rows[
                        membership_mask(source_sample_ids, selected_ids)
                    ]
                    if target_rows.size == 0:
                        continue
                stay = data[start:stop]
                categorical = stay[:, numeric_count:]
                if not np.isin(categorical, (0.0, 1.0)).all():
                    raise AssertionError("author one-hot channels are not binary")
                categorical_cumulative = np.cumsum(
                    np.sum(categorical == 1.0, axis=1), dtype=np.int64
                )
                split_total += int(
                    np.sum(
                        numeric_count * (target_rows + 1)
                        + categorical_cumulative[target_rows]
                    )
                )
            split_counts[COMMON_SPLIT[split]] = split_total
            total += split_total
    return total, split_counts


def _event_frame(buffer: dict[str, list[np.ndarray]]) -> pl.DataFrame:
    combined = {name: np.concatenate(parts) for name, parts in buffer.items()}
    valid = combined.pop("numeric_valid")
    numeric_arrow = pa.array(
        combined.pop("numeric_value"), mask=~valid, type=pa.float32()
    )
    return pl.DataFrame(
        {
            "subject_id": pl.Series(combined["subject_id"], dtype=pl.Int64),
            "time": pl.Series(combined["time"], dtype=pl.Datetime("us")),
            "code": pl.Series(combined["code"], dtype=pl.String),
            "numeric_value": pl.from_arrow(numeric_arrow),
            "sample_id": pl.Series(combined["sample_id"], dtype=pl.Int64),
            "source_row": pl.Series(combined["source_row"], dtype=pl.Int64),
            "sequence_row": pl.Series(combined["sequence_row"], dtype=pl.Int32),
            "source_time": pl.Series(combined["source_time"], dtype=pl.Float64),
        },
        schema={name: dtype for name, dtype in EVENT_SCHEMA.items() if name != "text_value"},
    )


def _prefix_block_coordinates(
    target_rows: np.ndarray,
    categorical_rows: np.ndarray,
    categorical_features: np.ndarray,
    numeric_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorize complete per-landmark prefixes for one bounded stay block."""
    target_rows = np.asarray(target_rows, dtype=np.int64)
    sample_offsets = np.arange(len(target_rows), dtype=np.int64)
    numeric_counts = numeric_count * (target_rows + 1)
    numeric_total = int(numeric_counts.sum())
    numeric_local = np.arange(numeric_total, dtype=np.int64) - np.repeat(
        np.cumsum(numeric_counts) - numeric_counts, numeric_counts
    )
    numeric_rows = (numeric_local // numeric_count).astype(np.int32)
    numeric_features = (numeric_local % numeric_count).astype(np.int32)
    numeric_samples = np.repeat(sample_offsets, numeric_counts)

    categorical_counts = np.searchsorted(
        categorical_rows, target_rows, side="right"
    ).astype(np.int64)
    categorical_total = int(categorical_counts.sum())
    if categorical_total:
        categorical_local = np.arange(
            categorical_total, dtype=np.int64
        ) - np.repeat(
            np.cumsum(categorical_counts) - categorical_counts,
            categorical_counts,
        )
        cat_rows = categorical_rows[categorical_local].astype(np.int32)
        cat_features = categorical_features[categorical_local].astype(np.int32)
        cat_samples = np.repeat(sample_offsets, categorical_counts)
    else:
        cat_rows = np.empty(0, dtype=np.int32)
        cat_features = np.empty(0, dtype=np.int32)
        cat_samples = np.empty(0, dtype=np.int64)
    rows = np.concatenate([numeric_rows, cat_rows])
    features = np.concatenate([numeric_features, cat_features])
    samples = np.concatenate([numeric_samples, cat_samples])
    numeric_valid = np.arange(len(rows), dtype=np.int64) < numeric_total
    return rows, features, samples, numeric_valid


def iter_events(
    source_root: Path,
    *,
    selected_sample_ids: np.ndarray | None = None,
    batch_rows: int = 2_000_000,
) -> Iterator[pl.DataFrame]:
    paths = assert_source_ready(source_root)
    metadata = inspect_hdf(source_root)
    columns = np.asarray(metadata["columns"], dtype=object)
    numeric_count = len(metadata["numeric_columns"])
    feature_count = len(columns)
    task_index = int(metadata["task_index"])
    buffer = {
        name: []
        for name in (
            "subject_id",
            "time",
            "code",
            "numeric_value",
            "numeric_valid",
            "sample_id",
            "source_row",
            "sequence_row",
            "source_time",
        )
    }
    buffered = 0
    sample_id = 1
    emitted_samples = 0
    selected_ids = None
    if selected_sample_ids is not None:
        selected_ids = np.sort(np.asarray(selected_sample_ids, dtype=np.int64))
        if len(selected_ids) == 0:
            raise ValueError("selected_sample_ids must not be empty")
    with tables.open_file(paths["standard_scaled"], "r") as handle:
        root = handle.root
        for split in SPLIT_MAP:
            names = source_filenames(paths["multitask"], split)
            windows = root.patient_windows._f_get_child(split)[:]
            labels = root.labels._f_get_child(split)[:, task_index]
            data = root.data._f_get_child(split)
            for name, window in zip(names, windows):
                start, stop, _ = (int(value) for value in window)
                subject_id, _ = parse_filename(name)
                target_rows = np.flatnonzero(np.isfinite(labels[start:stop]))
                if target_rows.size == 0:
                    continue
                source_sample_ids = np.arange(
                    sample_id, sample_id + len(target_rows), dtype=np.int64
                )
                sample_id += len(target_rows)
                if selected_ids is not None:
                    selected = membership_mask(source_sample_ids, selected_ids)
                    target_rows = target_rows[selected]
                    source_sample_ids = source_sample_ids[selected]
                    if target_rows.size == 0:
                        continue
                stay = np.asarray(data[start:stop], dtype=np.float32)
                if not np.isfinite(stay).all():
                    raise AssertionError("standard-scaled author input contains nonfinite values")
                categorical = stay[:, numeric_count:]
                if not np.isin(categorical, (0.0, 1.0)).all():
                    raise AssertionError("author one-hot channels are not binary")
                cat_row, cat_local = np.nonzero(categorical == 1.0)
                cat_feature = cat_local.astype(np.int32) + numeric_count
                categorical_counts = np.searchsorted(
                    cat_row, target_rows, side="right"
                ).astype(np.int64)
                target_event_counts = (
                    numeric_count * (target_rows.astype(np.int64) + 1)
                    + categorical_counts
                )
                block_start = 0
                while block_start < len(target_rows):
                    remaining = batch_rows - buffered
                    if buffered and target_event_counts[block_start] > remaining:
                        yield _event_frame(buffer)
                        for parts in buffer.values():
                            parts.clear()
                        buffered = 0
                        continue
                    cumulative = np.cumsum(target_event_counts[block_start:])
                    block_size = int(
                        np.searchsorted(cumulative, batch_rows - buffered, side="right")
                    )
                    block_size = max(1, block_size)
                    block_stop = min(len(target_rows), block_start + block_size)
                    rows, features, sample_offsets, valid = _prefix_block_coordinates(
                        target_rows[block_start:block_stop],
                        cat_row,
                        cat_feature,
                        numeric_count,
                    )
                    block_landmarks = block_stop - block_start
                    previous_emitted_samples = emitted_samples
                    count = int(len(rows))
                    values = np.zeros(count, dtype=np.float32)
                    values[valid] = stay[rows[valid], features[valid]]
                    buffer["subject_id"].append(
                        np.full(count, subject_id, dtype=np.int64)
                    )
                    buffer["time"].append(ANCHOR_US + rows.astype(np.int64) * HOUR_US)
                    buffer["code"].append(columns[features])
                    buffer["numeric_value"].append(values)
                    buffer["numeric_valid"].append(valid)
                    buffer["sample_id"].append(
                        source_sample_ids[block_start:block_stop][sample_offsets]
                    )
                    buffer["source_row"].append(
                        rows.astype(np.int64) * feature_count
                        + features.astype(np.int64)
                    )
                    buffer["sequence_row"].append(rows)
                    buffer["source_time"].append(rows.astype(np.float64))
                    buffered += count
                    emitted_samples += block_landmarks
                    block_start = block_stop
                    if buffered >= batch_rows:
                        yield _event_frame(buffer)
                        for parts in buffer.values():
                            parts.clear()
                        buffered = 0
                    if (
                        emitted_samples // 100_000
                        != previous_emitted_samples // 100_000
                    ):
                        print(f"exported event prefixes for {emitted_samples:,} landmarks")
    if buffered:
        yield _event_frame(buffer)


def expected_source(source_root: Path, event_count: int) -> dict[str, object]:
    paths = assert_source_ready(source_root)
    preparation = json.loads(paths["preparation"].read_text())
    standard = preparation["source_artifacts"]["standard_scaled_hdf"]
    non_scaled = preparation["source_artifacts"]["non_scaled_hdf"]
    return {
        "repositories": [
            {
                "url": "https://github.com/ratschlab/clinical-embeddings.git",
                "commit": CLINICAL_EMBEDDINGS_COMMIT,
            },
            {
                "url": "https://github.com/ratschlab/tls.git",
                "commit": TLS_COMMIT,
            },
            {
                "url": "https://github.com/YerevaNN/mimic3-benchmarks.git",
                "commit": M3B_COMMIT,
                "attribution": "TLS-directed de-facto public revision predating the pinned TLS release",
            },
        ],
        "dataset": "MIMIC-III 1.4",
        "preparation_sha256": sha256_file(paths["preparation"]),
        "standard_scaled_hdf_sha256": standard["sha256"],
        "non_scaled_hdf_sha256": non_scaled["sha256"],
        "author_channel_order_sha256": standard["column_hash"],
        "author_task_order_sha256": standard["task_hash"],
        "event_count_precomputed": event_count,
        "retained_boundary": "float32 author-loader values: 13 dense numeric channels and sparse occurrences for 29 exact one-hot channels",
        "cutoff": {
            "faithful_author_target_row": "source_cutoff_hour + 1",
            "known_defect": "the retained causal prefix includes one hour beyond the M3B scientific cutoff",
        },
        "modalities": ["structured"],
        "excluded_modalities": ["notes", "imaging"],
    }
