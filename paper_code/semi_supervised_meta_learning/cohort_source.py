"""Exact SSML/MIMIC3-benchmark decompensation boundary projected to MEDS."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import re
import sys
from itertools import groupby
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
import pyarrow as pa


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import EVENT_SCHEMA  # noqa: E402


TASK_ID = "decompensation"
SPLIT_MAP = ("train", "val", "test")
COMMON_SPLIT = {"train": "train", "val": "validation", "test": "test"}
SOURCE_SPLIT = {"train": "train", "validation": "train", "test": "test"}
ANCHOR_US = np.datetime64("2000-01-01T00:00:00", "us").astype(np.int64)
HOUR_US = 3_600_000_000
EPS = 1e-6
MINIMUM_CUTOFF = 24.0
MAXIMUM_CUTOFF = 2000.0
STUDY_COMMIT = "426c1766e287dec5782855a430ec14c5cacc3a94"
M3B_COMMIT = "ea0314c7cbd369f62e2237ace6f683740f867e3a"
RESOURCE_ROOT = (
    PAPER_ROOT
    / ".export_assets"
    / "external_artifacts"
    / "yerevann_resources"
)
CONFIG_PATH = (
    RESOURCE_ROOT / "mimic3models" / "resources" / "discretizer_config.json"
)
NORMALIZER_PATH = (
    RESOURCE_ROOT
    / "mimic3models"
    / "decompensation"
    / "decomp_ts1.0.input_str-previous.n1e5.start_time-zero.normalizer"
)
SOURCE_FILENAME = re.compile(
    r"^(?P<subject>\d+)_episode(?P<episode>\d+)_timeseries\.csv$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_paths(source_root: Path) -> dict[str, Path]:
    root = source_root.resolve()
    return {
        "preparation": root / "preparation.json",
        "decompensation": root / "decompensation",
        "train_listfile": root / "decompensation" / "train_listfile.csv",
        "val_listfile": root / "decompensation" / "val_listfile.csv",
        "test_listfile": root / "decompensation" / "test_listfile.csv",
    }


def assert_source_ready(source_root: Path) -> dict[str, Path]:
    paths = source_paths(source_root)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    preparation = json.loads(paths["preparation"].read_text())
    if preparation.get("status") != "exact_attributable_benchmark":
        raise AssertionError("source preparation is not the exact attributable benchmark")
    if preparation.get("commits") != {
        "mimic3_benchmarks": M3B_COMMIT,
        "study": STUDY_COMMIT,
    }:
        raise AssertionError("source preparation commits changed")
    return paths


def listfile_path(paths: dict[str, Path], split: str) -> Path:
    return paths[f"{split}_listfile"]


def load_transform_spec() -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text())
    channels = tuple(config["id_to_channel"])
    if len(channels) != 17:
        raise AssertionError("expected the released 17 raw channels")
    value_codes: list[str] = []
    value_positions: dict[str, tuple[int, ...]] = {}
    categorical_positions: dict[str, dict[str, int]] = {}
    numeric_positions: dict[str, int] = {}
    for channel in channels:
        begin = len(value_codes)
        if config["is_categorical_channel"][channel]:
            values = tuple(config["possible_values"][channel])
            value_codes.extend(f"{channel}->{value}" for value in values)
            positions = tuple(range(begin, begin + len(values)))
            categorical_positions[channel] = dict(zip(values, positions))
        else:
            value_codes.append(channel)
            positions = (begin,)
            numeric_positions[channel] = begin
        value_positions[channel] = positions
    value_code_count = len(value_codes)
    model_codes = value_codes + [f"mask->{channel}" for channel in channels]
    if value_code_count != 59 or len(model_codes) != 76:
        raise AssertionError(
            f"released discretizer boundary changed: {value_code_count}+{len(channels)}"
        )
    with NORMALIZER_PATH.open("rb") as handle:
        normalizer = pickle.load(handle, encoding="latin1")
    means = np.asarray(normalizer["means"], dtype=np.float64)
    stds = np.asarray(normalizer["stds"], dtype=np.float64)
    if means.shape != (76,) or stds.shape != (76,) or np.any(stds <= 0):
        raise AssertionError("released normalizer boundary changed")
    defaults = np.zeros(value_code_count, dtype=np.float64)
    for channel in channels:
        if channel in numeric_positions:
            position = numeric_positions[channel]
            defaults[position] = float(config["normal_values"][channel])
        else:
            position = categorical_positions[channel][config["normal_values"][channel]]
            defaults[position] = 1.0
    numeric_indices = np.asarray(tuple(numeric_positions.values()), dtype=np.int64)
    return {
        "config": config,
        "channels": channels,
        # The common task retains the earliest raw occurrence stream.  Masks,
        # hourly bins, and imputed values are derived fields and therefore do
        # not belong in its vocabulary.
        "codes": tuple(value_codes),
        "model_codes": tuple(model_codes),
        "value_code_count": value_code_count,
        "value_positions": value_positions,
        "categorical_positions": categorical_positions,
        "numeric_positions": numeric_positions,
        "mask_positions": {
            channel: value_code_count + index for index, channel in enumerate(channels)
        },
        "means": means,
        "stds": stds,
        "raw_defaults": defaults,
        "numeric_indices": numeric_indices,
    }


def _read_listfile(path: Path, split: str, sample_start: int, order_start: int) -> pl.DataFrame:
    frame = pl.read_csv(
        path,
        schema_overrides={"stay": pl.String, "period_length": pl.Float64, "y_true": pl.Int8},
        row_index_name="source_list_row",
    )
    if frame.filter(~pl.col("y_true").is_in([0, 1])).height:
        raise AssertionError(f"{split} listfile has a nonbinary label")
    eligible = frame.filter(
        (pl.col("period_length") >= MINIMUM_CUTOFF)
        & (pl.col("period_length") < MAXIMUM_CUTOFF)
    )
    eligible = eligible.with_columns(
        pl.col("period_length").round(0).cast(pl.Int32).alias("source_cutoff_hour")
    )
    if eligible.filter(
        (pl.col("period_length") - pl.col("source_cutoff_hour")).abs() > EPS
    ).height:
        raise AssertionError("the released producer emitted a noninteger landmark")
    count = eligible.height
    sample_ids = pl.Series("sample_id", np.arange(sample_start, sample_start + count), dtype=pl.Int64)
    source_orders = pl.Series("source_order", np.arange(order_start, order_start + count), dtype=pl.Int64)
    match = eligible["stay"].str.extract_groups(SOURCE_FILENAME.pattern)
    if match.struct.field("subject").null_count():
        raise AssertionError(f"{split} listfile has an unrecognized stay filename")
    return eligible.with_columns(
        match.struct.field("subject").cast(pl.Int64).alias("subject_id"),
        match.struct.field("episode").cast(pl.Int32).alias("episode_number"),
        sample_ids,
        source_orders,
        pl.lit(COMMON_SPLIT[split]).alias("split"),
        pl.lit(ANCHOR_US).cast(pl.Datetime("us")).alias("observation_start"),
        (
            pl.lit(ANCHOR_US)
            + pl.col("source_cutoff_hour").cast(pl.Int64) * HOUR_US
        ).cast(pl.Datetime("us")).alias("prediction_time"),
        (pl.col("y_true") == 1).alias("boolean_value"),
    ).rename({"stay": "source_filename"}).select(
        "subject_id",
        "sample_id",
        "observation_start",
        "prediction_time",
        "boolean_value",
        "split",
        "source_filename",
        pl.col("source_list_row").cast(pl.Int64),
        "source_cutoff_hour",
        "source_order",
        "episode_number",
    )


def build_samples(source_root: Path) -> tuple[pl.DataFrame, dict[str, object]]:
    paths = assert_source_ready(source_root)
    frames = []
    sample_start = 1
    order_start = 0
    summaries = {}
    tuple_hash = hashlib.sha256()
    preparation = json.loads(paths["preparation"].read_text())
    for split in SPLIT_MAP:
        frame = _read_listfile(
            listfile_path(paths, split), split, sample_start, order_start
        )
        for row in frame.select(
            "source_filename", "source_list_row", "source_cutoff_hour", "boolean_value"
        ).iter_rows():
            tuple_hash.update(
                f"{split}|{row[0]}|{row[1]}|{row[2]}|{int(row[3])}\n".encode()
            )
        common = COMMON_SPLIT[split]
        summaries[common] = {
            "source_rows": preparation["listfiles"][split]["rows"],
            "excluded_below_24": preparation["listfiles"][split]["excluded_below_24"],
            "excluded_at_or_above_2000": preparation["listfiles"][split]["excluded_at_or_above_2000"],
            "landmarks": frame.height,
            "positive": int(frame["boolean_value"].sum()),
            "negative": frame.height - int(frame["boolean_value"].sum()),
            "subjects": frame["subject_id"].n_unique(),
            "stays": frame["source_filename"].n_unique(),
        }
        frames.append(frame)
        sample_start += frame.height
        order_start += frame.height
    samples = pl.concat(frames, how="vertical")
    # The benchmark producer deliberately shuffles the train listfile.  MEDS
    # event batches are sorted by sample_id before being written, so IDs must
    # follow the stay-grouped streaming order; otherwise one writer batch can
    # interleave several stays and later revisit them.  source_order and
    # source_list_row retain the exact released listfile order independently.
    samples = samples.sort(
        "split", "source_filename", "source_cutoff_hour", "source_list_row"
    ).with_columns(
        pl.Series(
            "sample_id",
            np.arange(1, samples.height + 1, dtype=np.int64),
            dtype=pl.Int64,
        )
    )
    return samples, {
        "splits": summaries,
        "landmarks": samples.height,
        "maximum_cutoff_hour": int(samples["source_cutoff_hour"].max()),
        "ordered_eligible_tuple_sha256": tuple_hash.hexdigest(),
    }


def _group_samples(samples: pl.DataFrame) -> Iterator[tuple[str, str, pl.DataFrame]]:
    ordered = samples.sort("split", "source_filename", "source_cutoff_hour", "sample_id")
    split_values = ordered["split"].to_numpy()
    filename_values = ordered["source_filename"].to_numpy()
    if ordered.is_empty():
        return
    changes = np.flatnonzero(
        (split_values[1:] != split_values[:-1])
        | (filename_values[1:] != filename_values[:-1])
    ) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [ordered.height]))
    for start, stop in zip(starts, stops):
        yield str(split_values[start]), str(filename_values[start]), ordered.slice(
            int(start), int(stop - start)
        )


def stay_path(paths: dict[str, Path], common_split: str, filename: str) -> Path:
    return paths["decompensation"] / SOURCE_SPLIT[common_split] / filename


def load_raw_stay(
    path: Path, spec: dict[str, object], maximum_cutoff: int
) -> dict[str, np.ndarray]:
    """Read every attributable pre-discretization structured occurrence.

    Repeated values in one hourly bin are intentionally retained.  The
    released discretizer overwrites them in CSV row order, which the common
    ``source_row`` coordinate preserves mechanically.
    """

    channels = spec["channels"]
    numeric_positions = spec["numeric_positions"]
    categorical_positions = spec["categorical_positions"]
    times: list[float] = []
    source_rows: list[int] = []
    sequence_rows: list[int] = []
    codes: list[str] = []
    values: list[float] = []
    valid: list[bool] = []
    prior_time = -math.inf
    with path.open(newline="") as handle:
        rows = csv.reader(handle)
        header = next(rows)
        if header[0] != "Hours" or tuple(header[1:]) != channels:
            raise AssertionError(f"raw channel order changed in {path}")
        for source_line, row in enumerate(rows):
            time = float(row[0])
            if time + EPS < prior_time:
                raise AssertionError(f"raw times are not monotonic in {path}")
            prior_time = time
            if time > maximum_cutoff + EPS:
                break
            if int(time - EPS) < 0:
                raise AssertionError(f"source observation maps outside cutoff in {path}")
            for channel_index, value in enumerate(row[1:]):
                if not value:
                    continue
                channel = channels[channel_index]
                times.append(time)
                source_rows.append(source_line * len(channels) + channel_index)
                sequence_rows.append(source_line)
                if channel in numeric_positions:
                    codes.append(channel)
                    values.append(float(np.float32(float(value))))
                    valid.append(True)
                else:
                    if value not in categorical_positions[channel]:
                        raise AssertionError(
                            f"unexpected category {value!r} for {channel} in {path}"
                        )
                    codes.append(f"{channel}->{value}")
                    values.append(0.0)
                    valid.append(False)
    if not times:
        raise AssertionError(f"eligible stay has no retained observations: {path}")
    return {
        "times": np.asarray(times, dtype=np.float64),
        "source_rows": np.asarray(source_rows, dtype=np.int64),
        "sequence_rows": np.asarray(sequence_rows, dtype=np.int32),
        "codes": np.asarray(codes, dtype=object),
        "numeric_values": np.asarray(values, dtype=np.float32),
        "numeric_valid": np.asarray(valid, dtype=bool),
    }


def count_events(source_root: Path, samples: pl.DataFrame | None = None) -> tuple[int, dict[str, int]]:
    paths = assert_source_ready(source_root)
    spec = load_transform_spec()
    if samples is None:
        samples, _ = build_samples(source_root)
    totals = {split: 0 for split in ("train", "validation", "test")}
    for split, filename, group in _group_samples(samples):
        cutoffs = group["source_cutoff_hour"].to_numpy()
        stay = load_raw_stay(stay_path(paths, split, filename), spec, int(cutoffs.max()))
        totals[split] += int(
            np.searchsorted(stay["times"], cutoffs + EPS, side="right").sum()
        )
    return sum(totals.values()), totals


def _event_frame(buffer: dict[str, list[np.ndarray]]) -> pl.DataFrame:
    combined = {key: np.concatenate(value) for key, value in buffer.items()}
    valid = combined.pop("numeric_valid")
    numeric = pa.array(combined.pop("numeric_value"), mask=~valid, type=pa.float32())
    return pl.DataFrame(
        {
            "subject_id": pl.Series(combined["subject_id"], dtype=pl.Int64),
            "time": pl.Series(combined["time"], dtype=pl.Datetime("us")),
            "code": pl.Series(combined["code"], dtype=pl.String),
            "numeric_value": pl.from_arrow(numeric),
            "sample_id": pl.Series(combined["sample_id"], dtype=pl.Int64),
            "source_row": pl.Series(combined["source_row"], dtype=pl.Int64),
            "sequence_row": pl.Series(combined["sequence_row"], dtype=pl.Int32),
            "source_time": pl.Series(combined["source_time"], dtype=pl.Float64),
        },
        schema={
            name: dtype
            for name, dtype in EVENT_SCHEMA.items()
            if name not in {"text_value", "visit_id"}
        },
    )


def iter_events(
    source_root: Path,
    samples: pl.DataFrame | None = None,
    batch_rows: int = 2_000_000,
) -> Iterator[pl.DataFrame]:
    paths = assert_source_ready(source_root)
    spec = load_transform_spec()
    if samples is None:
        samples, _ = build_samples(source_root)
    buffer = {
        key: []
        for key in (
            "subject_id", "time", "code", "numeric_value", "numeric_valid",
            "sample_id", "source_row", "sequence_row", "source_time",
        )
    }
    buffered = emitted_samples = 0
    for split, filename, group in _group_samples(samples):
        cutoffs = group["source_cutoff_hour"].to_numpy()
        sample_ids = group["sample_id"].to_numpy()
        subject_id = int(group["subject_id"][0])
        stay = load_raw_stay(stay_path(paths, split, filename), spec, int(cutoffs.max()))
        counts = np.searchsorted(
            stay["times"], cutoffs + EPS, side="right"
        ).astype(np.int64)
        position = 0
        while position < len(cutoffs):
            if buffered and counts[position] > batch_rows - buffered:
                yield _event_frame(buffer)
                for value in buffer.values():
                    value.clear()
                buffered = 0
                continue
            cumulative = np.cumsum(counts[position:])
            block = max(1, int(np.searchsorted(cumulative, batch_rows - buffered, side="right")))
            stop = min(len(cutoffs), position + block)
            selected_counts = counts[position:stop]
            total = int(selected_counts.sum())
            local = np.arange(total, dtype=np.int64) - np.repeat(
                np.cumsum(selected_counts) - selected_counts, selected_counts
            )
            sample_offsets = np.repeat(np.arange(stop - position), selected_counts)
            source_times = stay["times"][local]
            buffer["subject_id"].append(np.full(total, subject_id, dtype=np.int64))
            buffer["time"].append(
                ANCHOR_US + np.rint(source_times * HOUR_US).astype(np.int64)
            )
            buffer["code"].append(stay["codes"][local])
            buffer["numeric_value"].append(stay["numeric_values"][local])
            buffer["numeric_valid"].append(stay["numeric_valid"][local])
            buffer["sample_id"].append(sample_ids[position:stop][sample_offsets])
            buffer["source_row"].append(stay["source_rows"][local])
            buffer["sequence_row"].append(stay["sequence_rows"][local])
            buffer["source_time"].append(source_times)
            buffered += total
            previous = emitted_samples
            emitted_samples += stop - position
            position = stop
            if buffered >= batch_rows:
                yield _event_frame(buffer)
                for value in buffer.values():
                    value.clear()
                buffered = 0
            if emitted_samples // 100_000 != previous // 100_000:
                print(f"exported event prefixes for {emitted_samples:,} landmarks")
    if buffered:
        yield _event_frame(buffer)


def reconstruct_model_tensor(
    events: pl.DataFrame, spec: dict[str, object], cutoff_hour: int
) -> np.ndarray:
    """Apply the released overwrite, imputation, mask, and normalization path."""

    code_to_index = {code: index for index, code in enumerate(spec["codes"])}
    channels = spec["channels"]
    value_positions = spec["value_positions"]
    categorical_positions = spec["categorical_positions"]
    numeric_positions = spec["numeric_positions"]
    defaults = np.asarray(spec["raw_defaults"], dtype=np.float64)
    data = np.zeros((cutoff_hour, 59), dtype=np.float64)
    mask = np.zeros((cutoff_hour, len(channels)), dtype=np.float64)
    by_bin: dict[int, list[tuple[int, str, float | None]]] = {}
    ordered = events.sort("source_row")
    for row in ordered.select("source_time", "source_row", "code", "numeric_value").iter_rows():
        bin_id = int(float(row[0]) - EPS)
        if not 0 <= bin_id < cutoff_hour:
            raise AssertionError("raw MEDS occurrence maps outside the author cutoff")
        by_bin.setdefault(bin_id, []).append((int(row[1]), str(row[2]), row[3]))

    previous: dict[str, tuple[str, float | None]] = {}
    for bin_id in range(cutoff_hour):
        final: dict[str, tuple[str, float | None]] = {}
        for _, code, numeric in by_bin.get(bin_id, []):
            position = code_to_index[code]
            channel = next(
                channel for channel in channels if position in value_positions[channel]
            )
            final[channel] = (code, numeric)
        for channel, value in final.items():
            previous[channel] = value
            mask[bin_id, channels.index(channel)] = 1.0
        for channel in channels:
            if channel in final:
                code, numeric = final[channel]
            elif channel in previous:
                code, numeric = previous[channel]
            elif channel in numeric_positions:
                code = channel
                numeric = defaults[numeric_positions[channel]]
            else:
                category = spec["config"]["normal_values"][channel]
                code = f"{channel}->{category}"
                numeric = None
            position = code_to_index[code]
            positions = value_positions[channel]
            if len(positions) == 1:
                if numeric is None:
                    raise AssertionError("numeric raw occurrence lacks a value")
                data[bin_id, position] = float(numeric)
            else:
                if code not in categorical_positions[channel]:
                    # ``categorical_positions`` is keyed by raw category, while
                    # common codes are channel-qualified.
                    category = code.removeprefix(f"{channel}->")
                    position = categorical_positions[channel][category]
                data[bin_id, list(positions)] = 0.0
                data[bin_id, position] = 1.0

    numeric_indices = spec["numeric_indices"]
    means = spec["means"]
    stds = spec["stds"]
    data[:, numeric_indices] = (
        data[:, numeric_indices] - means[numeric_indices]
    ) / stds[numeric_indices]
    return np.hstack([data, mask]).astype(np.float32)


def expected_source(source_root: Path, event_count: int) -> dict[str, object]:
    paths = assert_source_ready(source_root)
    preparation = json.loads(paths["preparation"].read_text())
    return {
        "repositories": [
            {"url": "https://github.com/lidazhang/ssml-time-series-heterogeneity.git", "commit": STUDY_COMMIT},
            {"url": "https://github.com/YerevaNN/mimic3-benchmarks.git", "commit": M3B_COMMIT},
        ],
        "dataset": "MIMIC-III 1.4",
        "preparation_sha256": sha256_file(paths["preparation"]),
        "listfile_sha256": {
            COMMON_SPLIT[split]: sha256_file(listfile_path(paths, split))
            for split in SPLIT_MAP
        },
        "discretizer_config_sha256": sha256_file(CONFIG_PATH),
        "normalizer_sha256": sha256_file(NORMALIZER_PATH),
        "event_count_precomputed": event_count,
        "retained_boundary": (
            "earliest attributable raw 17-channel occurrence stream with exact "
            "fractional source times, repeated within-bin observations, and CSV "
            "row order; hourly overwrite, 59-channel encoding, 17 masks, previous-"
            "value imputation, and normalization are recreated only in check_batch"
        ),
        "paper_run_evaluation": {
            "exact_membership_recoverable": False,
            "reason": "the released entrypoint freshly and without a seed shuffles 800 source rows for both train and validation, then skips out-of-range cutoffs without replacement",
            "heldout_test_exercised": False,
        },
        "canonical_export": "complete deterministic benchmark train/validation/test membership after the released 24 <= cutoff < 2000 filter",
        "source_order_caveat": preparation["train_list_order_note"],
        "modalities": ["structured"],
        "excluded_modalities": ["notes", "imaging"],
    }
