"""Write the MEDS contract from explicitly typed Polars frames."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .common_format import (
    FORMAT_NAME,
    MEDS_VERSION,
    SCHEMA_VERSION,
    SPLITS,
    TASK_ID,
    sha256_file,
    write_json,
)

EVENT_SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.Datetime("us"),
    "code": pl.String,
    "numeric_value": pl.Float32,
    "sample_id": pl.Int64,
    "source_row": pl.Int64,
    "sequence_row": pl.Int32,
    "source_time": pl.Float64,
    "text_value": pl.String,
}
EVENT_REQUIRED = {
    "subject_id",
    "time",
    "code",
    "numeric_value",
    "sample_id",
    "source_row",
}
INDEX_SCHEMA = {
    "subject_id": pl.Int64,
    "sample_id": pl.Int64,
    "observation_start": pl.Datetime("us"),
    "prediction_time": pl.Datetime("us"),
    "boolean_value": pl.Boolean,
    "split": pl.String,
}


def _cast_required(
    frame: pl.DataFrame,
    schema: dict[str, pl.DataType],
    required: set[str],
) -> pl.DataFrame:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"table is missing columns: {sorted(missing)}")
    unknown = set(frame.columns) - set(schema)
    if unknown:
        raise ValueError(f"unsupported event columns: {sorted(unknown)}")
    expressions = []
    for name, dtype in schema.items():
        if name not in frame.columns and name not in required:
            continue
        expression = pl.col(name) if name in frame.columns else pl.lit(None)
        expressions.append(expression.cast(dtype, strict=True).alias(name))
    return frame.select(expressions)


def _cast_index(frame: pl.DataFrame) -> pl.DataFrame:
    missing = set(INDEX_SCHEMA) - set(frame.columns)
    if missing:
        raise ValueError(f"task index is missing columns: {sorted(missing)}")
    expressions = [
        pl.col(name).cast(dtype, strict=True).alias(name)
        for name, dtype in INDEX_SCHEMA.items()
    ]
    expressions.extend(pl.col(name) for name in frame.columns if name not in INDEX_SCHEMA)
    return frame.select(expressions)


# How a task's events carry time, which decides the representations the shared
# runner may offer.
#
#   single_visit_timed   one encounter; timestamps order events inside it. The
#                        window statistics and their fractional sub-windows are
#                        meaningful. This is the default.
#   multi_visit_untimed  a visit sequence with no within-visit ordering: every
#                        event in a visit shares that visit's single timestamp,
#                        so distinct event time is exactly the visit key and the
#                        fractional sub-windows are meaningless.
#                        Needs its own handling; no visit-bucketed
#                        representation is offered for it yet.
TEMPORAL_PROFILES = (
    "single_visit_timed",
    "multi_visit_untimed",
)


def _validate_task(task: dict[str, Any], root: Path) -> float:
    task_id = task.get("id")
    if root.name != task_id or not TASK_ID.fullmatch(str(task_id or "")):
        raise ValueError("task directory and valid task.id must match")
    if not isinstance(task.get("name"), str) or not task["name"].strip():
        raise ValueError("task must declare a nonempty name")
    if task.get("primary_metric") != "AUROC":
        raise ValueError("the MEDS contract supports AUROC tasks only")
    if "input" in task or "modalities" in task or "note_codes" in task:
        raise ValueError("the writer derives modalities and note_codes")
    hours = task.get("observation_window_hours")
    if not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours <= 0:
        raise ValueError("task must declare positive observation_window_hours")
    if "temporal_grid_mode" in task:
        raise ValueError("temporal_grid_mode was removed; declare temporal_profile")
    if task.get("temporal_profile", "single_visit_timed") not in TEMPORAL_PROFILES:
        raise ValueError(
            "task temporal_profile must be one of " + ", ".join(TEMPORAL_PROFILES)
        )
    return float(hours)


def _validate_samples(samples: pl.DataFrame, task_window_hours: float) -> None:
    if samples["sample_id"].n_unique() != samples.height:
        raise ValueError("sample_id must be unique")
    if set(samples["split"].unique().to_list()) != set(SPLITS):
        raise ValueError(f"samples must use exactly the splits {SPLITS}")
    if samples.filter(pl.col("prediction_time") <= pl.col("observation_start")).height:
        raise ValueError("every prediction_time must follow observation_start")
    maximum_width = samples.select(
        (
            (pl.col("prediction_time") - pl.col("observation_start"))
            .dt.total_microseconds()
            / 3_600_000_000.0
        ).max()
    ).item()
    if maximum_width > task_window_hours + 1e-9:
        raise ValueError("a sample observation window exceeds the task window")
    classes = samples.group_by("split").agg(pl.col("boolean_value").n_unique())
    if classes.filter(pl.col("boolean_value") != 2).height:
        raise ValueError("every split must contain both label classes")


def _attach_and_validate_events(
    events: pl.DataFrame,
    samples: pl.DataFrame,
    codes: tuple[str, ...] | pl.Series,
    note_codes: tuple[str, ...],
) -> pl.DataFrame:
    lookup = samples.select(
        "sample_id",
        pl.col("subject_id").alias("expected_subject_id"),
        "observation_start",
        "prediction_time",
        "split",
    )
    events = events.join(
        lookup, on="sample_id", how="left", validate="m:1", coalesce=True
    )
    if events["expected_subject_id"].null_count():
        raise ValueError("an event references an unknown sample_id")
    if events.filter(pl.col("subject_id") != pl.col("expected_subject_id")).height:
        raise ValueError("an event subject_id disagrees with its sample")
    if events.filter(
        (pl.col("time") < pl.col("observation_start"))
        | (pl.col("time") > pl.col("prediction_time"))
    ).height:
        raise ValueError("an event falls outside its sample observation window")
    # Imploding creates a one-row list Series, which Polars 1.8 interprets as
    # a shape-mismatched row-wise comparison for event batches larger than one
    # row.  A plain list works in both the paper environments' older Polars and
    # the shared runner's current Polars without the newer Series deprecation.
    declared = codes.to_list() if isinstance(codes, pl.Series) else codes
    if events.filter(~pl.col("code").is_in(declared)).height:
        raise ValueError("an event uses an undeclared MEDS code")
    if note_codes and "text_value" not in events.columns:
        raise ValueError("a note-using task must retain raw text_value")

    is_note = pl.col("code").is_in(note_codes)
    has_text = (
        pl.col("text_value").is_not_null()
        if "text_value" in events.columns
        else pl.lit(False)
    )
    bad_structured = events.filter(
        (~is_note)
        & (
            (
                pl.col("numeric_value").is_not_null()
                & ~pl.col("numeric_value").is_finite()
            )
            | has_text
        )
    )
    if bad_structured.height:
        raise ValueError(
            "structured events require either a finite numeric value or an "
            "occurrence-only null, and no text"
        )
    bad_note = events.filter(
        is_note
        & (pl.col("numeric_value").is_not_null() | ~has_text)
    )
    if bad_note.height:
        raise ValueError("note events require text and no numeric value")
    # Keep each duplicated prediction example physically contiguous.  This is
    # still a valid MEDS-like event ordering (sample_id identifies the exact
    # eligible prefix) and lets the shared featurizer finish one sample without
    # retaining the task-wide event table.  The remaining keys preserve the
    # deterministic within-sample order used by the reference implementation.
    return events.drop(
        "expected_subject_id", "observation_start", "prediction_time"
    ).sort("sample_id", "time", "source_row", "code")


def _write_code_metadata(
    path: Path,
    codes: tuple[str, ...],
    code_metadata: Path | None,
) -> None:
    if code_metadata is None:
        pl.DataFrame(
            {"code": list(codes)},
            schema={"code": pl.String},
        ).write_parquet(path, compression="zstd")
        return

    source = code_metadata.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"code metadata is missing: {source}")
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    if "code" not in schema.names:
        raise ValueError("code metadata must contain a code column")
    code_type = schema.field("code").type
    if not (pa.types.is_string(code_type) or pa.types.is_large_string(code_type)):
        raise ValueError("code metadata code column must be UTF-8")
    position = 0
    for batch in parquet.iter_batches(columns=["code"], batch_size=250_000):
        actual = batch.column(0).to_pylist()
        expected = codes[position : position + len(actual)]
        if tuple(actual) != expected:
            raise ValueError(
                "code metadata rows must match the declared code order exactly"
            )
        position += len(actual)
    if position != len(codes):
        raise ValueError("code metadata row count differs from declared codes")
    shutil.copyfile(source, path)


def _cross_split_overlap(samples: pl.DataFrame) -> dict[str, int]:
    members = {
        split: set(
            samples.filter(pl.col("split") == split)["subject_id"].unique().to_list()
        )
        for split in SPLITS
    }
    result = {
        f"{left}_{right}": len(members[left] & members[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    all_subjects = set().union(*members.values())
    result["subjects_in_multiple_splits"] = sum(
        sum(subject in values for values in members.values()) > 1
        for subject in all_subjects
    )
    return result


def write_meds_task(
    root: Path,
    *,
    paper: dict[str, str],
    task: dict[str, Any],
    codes: tuple[str, ...],
    code_metadata: Path | None = None,
    note_codes: tuple[str, ...] = (),
    events: Any,
    samples: pl.DataFrame,
    source: dict[str, Any],
    dataset_name: str,
    dataset_version: str,
) -> dict[str, Any]:
    """Write one directly usable MEDS task without importing the MEDS package."""
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    hours = _validate_task(task, root)
    if len(codes) != len(set(codes)) or not codes:
        raise ValueError("codes must be nonempty and unique")
    if len(note_codes) != len(set(note_codes)) or not set(note_codes).issubset(codes):
        raise ValueError("note_codes must be unique members of the MEDS vocabulary")
    if set(codes) == set(note_codes):
        raise ValueError("the task must contain at least one structured code")

    samples = _cast_index(samples)
    _validate_samples(
        samples, hours
    )
    samples = samples.sort("split", "subject_id", "prediction_time", "sample_id")

    event_path = root / "data" / "events.parquet"
    label_path = root / "tasks" / str(task["id"]) / "labels.parquet"
    split_path = root / "metadata" / "subject_splits.parquet"
    codes_path = root / "metadata" / "codes.parquet"
    metadata_path = root / "metadata" / "dataset.json"
    index_path = root / "audit" / "task_index.parquet"
    for path in (event_path, label_path, split_path, codes_path, index_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    event_batches: Iterable[pl.DataFrame]
    if isinstance(events, pl.DataFrame):
        event_batches = (events,)
    else:
        event_batches = events
    parquet_writer = None
    event_count = 0
    declared_codes = pl.Series("__declared_code", codes, dtype=pl.String)
    try:
        for batch in event_batches:
            if not isinstance(batch, pl.DataFrame):
                raise TypeError("every event batch must be a Polars DataFrame")
            if batch.is_empty():
                continue
            batch = _cast_required(batch, EVENT_SCHEMA, EVENT_REQUIRED)
            batch = _attach_and_validate_events(
                batch, samples, declared_codes, note_codes
            )
            table = batch.to_arrow()
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                    event_path, table.schema, compression="zstd"
                )
            elif table.schema != parquet_writer.schema:
                raise ValueError("event batches produced inconsistent schemas")
            parquet_writer.write_table(table, row_group_size=250_000)
            event_count += batch.height
    finally:
        if parquet_writer is not None:
            parquet_writer.close()
    if event_count == 0:
        raise ValueError("the task must contain at least one event")
    samples.select(
        "subject_id", "sample_id", "prediction_time", "boolean_value"
    ).write_parquet(
        label_path, compression="zstd"
    )
    samples.select("subject_id", "split").unique().sort(
        "subject_id", "split"
    ).write_parquet(split_path, compression="zstd")
    _write_code_metadata(codes_path, codes, code_metadata)
    samples.write_parquet(index_path, compression="zstd")

    write_json(
        metadata_path,
        {
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "meds_version": MEDS_VERSION,
        },
    )
    file_paths = {
        "events": event_path,
        "labels": label_path,
        "subject_splits": split_path,
        "codes": codes_path,
        "dataset_metadata": metadata_path,
        "task_index": index_path,
    }
    split_summary = {}
    for split in SPLITS:
        selected = samples.filter(pl.col("split") == split)
        split_summary[split] = {
            "count": selected.height,
            "subjects": selected["subject_id"].n_unique(),
            "positive": int(selected["boolean_value"].sum()),
        }
    manifest = {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "paper": paper,
        "task": {
            **task,
            "modalities": ["structured", "notes"] if note_codes else ["structured"],
            "note_codes": list(note_codes),
            "observation_window_hours": hours,
        },
        "meds": {
            name: {
                "file": str(path.relative_to(root)),
                "sha256": sha256_file(path),
            }
            for name, path in file_paths.items()
        },
        "splits": split_summary,
        "split_assignment_unit": "sample_id",
        "cross_split_subject_overlap": _cross_split_overlap(samples),
        "event_count": event_count,
        "source": source,
    }
    write_json(root / "manifest.json", manifest)
    return manifest
