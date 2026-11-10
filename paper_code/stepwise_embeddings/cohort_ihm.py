"""Exact released Stepwise/TLS boundary for the second published task: `ihm`.

Phase 2 outcome. The paper's second MIMIC-III binary AUROC cell is in-hospital
mortality after 48 hours (Appendix Table 9, FTT embedding 85.8 +/- 0.2). The
released launcher is the same MIMIC example configuration with `TASK='ihm'`,
and the repository README requires `MAXLEN=48` for it, so the author model is
shown exactly author grid rows 0..47 of every eligible stay and predicts at
row 47.

This module mirrors `cohort_source.py` (the sealed decompensation exporter) and
reuses its source-readiness, HDF inspection, listfile, and event-frame helpers;
only the sample enumeration and the fixed 48-row prefix differ.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
import pyarrow as pa
import tables


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import EVENT_SCHEMA  # noqa: E402
from cohort_source import (  # noqa: E402
    ANCHOR_US,
    COMMON_SPLIT,
    HOUR_US,
    SPLIT_MAP,
    assert_source_ready,
    expected_source,
    inspect_hdf,
    parse_filename,
    paper_encoded_stay,
    source_filenames,
)


TASK_ID = "ihm"
AUTHOR_TASK = "ihm"
# `configs`/README: "For `ihm` on MIMIC set `MAXLEN=48`."
AUTHOR_MAXLEN = 48
# mimic3-benchmarks `create_multitask.py`: `ihm_position = 47` whenever the
# in-hospital-mortality target is eligible (`fixed_hours = 48.0`).
AUTHOR_TARGET_ROW = 47


# `cohort_source._event_frame` predates the shared writer's optional
# `visit_id` column and would now raise KeyError; this single-visit task
# declares neither `text_value` nor `visit_id`.
_OPTIONAL_UNUSED = ("text_value", "visit_id")


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
        schema={
            name: dtype
            for name, dtype in EVENT_SCHEMA.items()
            if name not in _OPTIONAL_UNUSED
        },
    )


def ihm_task_index(source_root: Path) -> int:
    metadata = inspect_hdf(source_root)
    tasks = tuple(metadata["tasks"])
    if AUTHOR_TASK not in tasks:
        raise AssertionError(f"missing author task {AUTHOR_TASK}")
    return tasks.index(AUTHOR_TASK)


def build_samples(source_root: Path) -> tuple[pl.DataFrame, dict[str, object]]:
    """One sample per stay carrying the released, non-masked `ihm` target."""
    paths = assert_source_ready(source_root)
    task_index = ihm_task_index(source_root)
    fields: dict[str, list] = {
        name: []
        for name in (
            "subject_id",
            "sample_id",
            "observation_start",
            "prediction_time",
            "boolean_value",
            "split",
            "source_filename",
            "source_stay_order",
            "author_target_row",
            "prefix_rows",
            "source_label_row",
            "source_order",
            "paper_encoded_stay_id",
            "episode_number",
            "author_stay_rows",
        )
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
            eligible = 0
            subjects = set()
            minimum_rows = None
            for stay_order, (name, window) in enumerate(zip(names, windows)):
                start, stop, encoded = (int(value) for value in window)
                subject_id, episode = parse_filename(name)
                if encoded != paper_encoded_stay(subject_id, episode):
                    raise AssertionError(
                        "TLS encoded stay ID differs from released formula"
                    )
                relative = labels[start:stop]
                target_rows = np.flatnonzero(np.isfinite(relative))
                if target_rows.size == 0:
                    continue
                if target_rows.size != 1:
                    raise AssertionError(
                        "released ihm task carries more than one target per stay"
                    )
                target_row = int(target_rows[0])
                if target_row != AUTHOR_TARGET_ROW:
                    raise AssertionError(
                        "released ihm target row differs from M3B ihm_position 47"
                    )
                value = float(relative[target_row])
                if value not in (0.0, 1.0):
                    raise AssertionError("eligible author ihm label is not binary")
                stay_rows = stop - start
                if stay_rows <= AUTHOR_TARGET_ROW:
                    raise AssertionError(
                        "an eligible ihm stay is shorter than the released 48-row window"
                    )
                minimum_rows = (
                    stay_rows if minimum_rows is None else min(minimum_rows, stay_rows)
                )
                eligible += 1
                positive += int(value == 1.0)
                subjects.add(subject_id)
                fields["subject_id"].append(subject_id)
                fields["sample_id"].append(sample_id)
                fields["observation_start"].append(ANCHOR_US)
                fields["prediction_time"].append(
                    ANCHOR_US + AUTHOR_TARGET_ROW * HOUR_US
                )
                fields["boolean_value"].append(value == 1.0)
                fields["split"].append(COMMON_SPLIT[split])
                fields["source_filename"].append(name)
                fields["source_stay_order"].append(stay_order)
                fields["author_target_row"].append(AUTHOR_TARGET_ROW)
                fields["prefix_rows"].append(AUTHOR_MAXLEN)
                fields["source_label_row"].append(start + target_row)
                fields["source_order"].append(source_order)
                fields["paper_encoded_stay_id"].append(encoded)
                fields["episode_number"].append(episode)
                fields["author_stay_rows"].append(stay_rows)
                tuple_hasher.update(
                    f"{split}|{name}|{target_row}|{int(value)}\n".encode()
                )
                source_order += 1
                sample_id += 1
            split_summary[COMMON_SPLIT[split]] = {
                "stays_in_listfile": len(names),
                "eligible_stays": eligible,
                "subjects_with_targets": len(subjects),
                "positive": positive,
                "negative": eligible - positive,
                "minimum_author_stay_rows": minimum_rows,
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
            "source_filename": pl.Series(fields["source_filename"], dtype=pl.String),
            "source_stay_order": pl.Series(
                fields["source_stay_order"], dtype=pl.Int32
            ),
            "author_target_row": pl.Series(
                fields["author_target_row"], dtype=pl.Int32
            ),
            "prefix_rows": pl.Series(fields["prefix_rows"], dtype=pl.Int32),
            "source_label_row": pl.Series(fields["source_label_row"], dtype=pl.Int64),
            "source_order": pl.Series(fields["source_order"], dtype=pl.Int64),
            "paper_encoded_stay_id": pl.Series(
                fields["paper_encoded_stay_id"], dtype=pl.Int64
            ),
            "episode_number": pl.Series(fields["episode_number"], dtype=pl.Int32),
            "author_stay_rows": pl.Series(fields["author_stay_rows"], dtype=pl.Int64),
        }
    )
    audit = {
        "splits": split_summary,
        "ordered_target_tuple_sha256": tuple_hasher.hexdigest(),
        "samples": samples.height,
        "author_maxlen": AUTHOR_MAXLEN,
        "author_target_row": AUTHOR_TARGET_ROW,
    }
    return samples, audit


def _eligible_stays(root, split: str, task_index: int):
    """Yield (stay_order, start, stop) for every stay with an `ihm` target."""
    windows = root.patient_windows._f_get_child(split)[:]
    labels = root.labels._f_get_child(split)[:, task_index]
    for stay_order, window in enumerate(windows):
        start, stop = int(window[0]), int(window[1])
        if np.isfinite(labels[start:stop]).any():
            yield stay_order, start, stop


def count_events(source_root: Path) -> tuple[int, dict[str, int]]:
    paths = assert_source_ready(source_root)
    metadata = inspect_hdf(source_root)
    task_index = ihm_task_index(source_root)
    numeric_count = len(metadata["numeric_columns"])
    total = 0
    split_counts: dict[str, int] = {}
    with tables.open_file(paths["standard_scaled"], "r") as handle:
        root = handle.root
        for split in SPLIT_MAP:
            data = root.data._f_get_child(split)
            split_total = 0
            for _, start, stop in _eligible_stays(root, split, task_index):
                prefix = np.asarray(
                    data[start : start + AUTHOR_MAXLEN], dtype=np.float32
                )
                if prefix.shape[0] != AUTHOR_MAXLEN:
                    raise AssertionError("eligible stay is shorter than MAXLEN=48")
                categorical = prefix[:, numeric_count:]
                if not np.isin(categorical, (0.0, 1.0)).all():
                    raise AssertionError("author one-hot channels are not binary")
                split_total += AUTHOR_MAXLEN * numeric_count + int(
                    np.sum(categorical == 1.0)
                )
            split_counts[COMMON_SPLIT[split]] = split_total
            total += split_total
    return total, split_counts


def iter_events(
    source_root: Path, *, batch_rows: int = 2_000_000
) -> Iterator[pl.DataFrame]:
    """Stream the exact 48-row author prefix of every eligible stay."""
    paths = assert_source_ready(source_root)
    metadata = inspect_hdf(source_root)
    columns = np.asarray(metadata["columns"], dtype=object)
    numeric_count = len(metadata["numeric_columns"])
    feature_count = len(columns)
    task_index = ihm_task_index(source_root)
    numeric_rows = np.repeat(
        np.arange(AUTHOR_MAXLEN, dtype=np.int32), numeric_count
    )
    numeric_features = np.tile(
        np.arange(numeric_count, dtype=np.int32), AUTHOR_MAXLEN
    )
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
    emitted = 0
    with tables.open_file(paths["standard_scaled"], "r") as handle:
        root = handle.root
        for split in SPLIT_MAP:
            names = source_filenames(paths["multitask"], split)
            data = root.data._f_get_child(split)
            for stay_order, start, stop in _eligible_stays(root, split, task_index):
                name = names[stay_order]
                subject_id, _ = parse_filename(name)
                current_sample_id = sample_id
                sample_id += 1
                prefix = np.asarray(
                    data[start : start + AUTHOR_MAXLEN], dtype=np.float32
                )
                if prefix.shape[0] != AUTHOR_MAXLEN:
                    raise AssertionError("eligible stay is shorter than MAXLEN=48")
                if not np.isfinite(prefix).all():
                    raise AssertionError(
                        "standard-scaled author input contains nonfinite values"
                    )
                categorical = prefix[:, numeric_count:]
                if not np.isin(categorical, (0.0, 1.0)).all():
                    raise AssertionError("author one-hot channels are not binary")
                cat_row, cat_local = np.nonzero(categorical == 1.0)
                rows = np.concatenate(
                    [numeric_rows, cat_row.astype(np.int32)]
                )
                features = np.concatenate(
                    [
                        numeric_features,
                        cat_local.astype(np.int32) + numeric_count,
                    ]
                )
                count = int(len(rows))
                valid = np.arange(count, dtype=np.int64) < len(numeric_rows)
                values = np.zeros(count, dtype=np.float32)
                values[valid] = prefix[rows[valid], features[valid]]
                buffer["subject_id"].append(np.full(count, subject_id, dtype=np.int64))
                buffer["time"].append(ANCHOR_US + rows.astype(np.int64) * HOUR_US)
                buffer["code"].append(columns[features])
                buffer["numeric_value"].append(values)
                buffer["numeric_valid"].append(valid)
                buffer["sample_id"].append(
                    np.full(count, current_sample_id, dtype=np.int64)
                )
                buffer["source_row"].append(
                    rows.astype(np.int64) * feature_count + features.astype(np.int64)
                )
                buffer["sequence_row"].append(rows)
                buffer["source_time"].append(rows.astype(np.float64))
                buffered += count
                emitted += 1
                if buffered >= batch_rows:
                    yield _event_frame(buffer)
                    for parts in buffer.values():
                        parts.clear()
                    buffered = 0
                    print(f"exported event prefixes for {emitted:,} stays", flush=True)
    if buffered:
        yield _event_frame(buffer)


def expected_ihm_source(source_root: Path, event_count: int) -> dict[str, object]:
    source = dict(expected_source(source_root, event_count))
    source["retained_boundary"] = (
        "float32 author-loader values for author grid rows 0..47 (MAXLEN=48): "
        "13 dense numeric channels and sparse occurrences for 29 exact one-hot channels"
    )
    source["cutoff"] = {
        "author_target_row": AUTHOR_TARGET_ROW,
        "author_maxlen": AUTHOR_MAXLEN,
        "m3b_definition": (
            "create_multitask.py fixed_hours=48.0: ihm_position 47, mask 1 only when "
            "length of stay >= 48h and the first ICU event is within 48h"
        ),
        "no_label_shift": (
            "TLS writes the ihm target directly at author row 47 and does not "
            "prepend the NaN that shifts its decompensation targets, so this task "
            "carries no one-hour future-input defect"
        ),
        "released_grid_defect": (
            "impute_sample_1h copies the first irregular measurement into grid row 0 "
            "and repeats it until its true timestamp, so a first measurement observed "
            "in (47, 48] hours appears at every retained row; this is the released "
            "author model input and is retained unchanged"
        ),
        "released_preprocessing_defect": (
            "get_one_hot concatenates train, validation, and test covariates before "
            "pd.get_dummies, so held-out category presence determines the released "
            "one-hot dimensionality; preserved, not repaired"
        ),
    }
    return source
