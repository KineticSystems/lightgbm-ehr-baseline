"""Bound the canonical rolling-prefix Parquet size without starting export."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import tables


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from cohort_source import (  # noqa: E402
    ANCHOR_US,
    COMMON_SPLIT,
    HOUR_US,
    SPLIT_MAP,
    _event_frame,
    _prefix_block_coordinates,
    count_events,
    inspect_hdf,
    parse_filename,
    source_filenames,
    source_paths,
)


LENGTH_BANDS = ("short", "median", "long")
CUTOFF_BANDS = ("early", "middle", "late")
TARGET_EVENTS = 250_000
SAFETY_FACTOR = 2.0
FIXED_ALLOWANCE_BYTES = 2 * 2**30
MINIMUM_FREE_AFTER_BYTES = 20 * 2**30


def cutoff_chunks(target_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically divide a stay's eligible landmarks into thirds."""
    return tuple(np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(target_rows, 3))


def length_band_indexes(lengths: np.ndarray) -> dict[str, np.ndarray]:
    """Return disjoint stay-length tertiles with deterministic tie handling."""
    order = np.argsort(lengths, kind="stable")
    chunks = np.array_split(order, 3)
    result = {
        name: np.asarray(chunk, dtype=np.int64)
        for name, chunk in zip(LENGTH_BANDS, chunks)
    }
    if any(len(indexes) == 0 for indexes in result.values()):
        raise AssertionError("not enough eligible stays for length strata")
    return result


def event_block(
    *,
    stay: np.ndarray,
    target_rows: np.ndarray,
    cat_rows: np.ndarray,
    cat_features: np.ndarray,
    columns: np.ndarray,
    numeric_count: int,
    subject_id: int,
    sample_id: int,
) -> pl.DataFrame:
    rows, features, sample_offsets, valid = _prefix_block_coordinates(
        target_rows, cat_rows, cat_features, numeric_count
    )
    count = len(rows)
    values = np.zeros(count, dtype=np.float32)
    values[valid] = stay[rows[valid], features[valid]]
    return _event_frame(
        {
            "subject_id": [np.full(count, subject_id, dtype=np.int64)],
            "time": [ANCHOR_US + rows.astype(np.int64) * HOUR_US],
            "code": [columns[features]],
            "numeric_value": [values],
            "numeric_valid": [valid],
            "sample_id": [sample_id + sample_offsets],
            "source_row": [
                rows.astype(np.int64) * len(columns) + features.astype(np.int64)
            ],
            "sequence_row": [rows.astype(np.int32)],
            "source_time": [rows.astype(np.float64)],
        }
    )


def write_stratum(
    *,
    handle,
    split: str,
    stay_indexes: np.ndarray,
    cutoff_index: int,
    names: list[str],
    windows: np.ndarray,
    labels: np.ndarray,
    columns: np.ndarray,
    numeric_count: int,
    output_path: Path,
) -> dict[str, object]:
    data = handle.root.data._f_get_child(split)
    frames = []
    events = 0
    samples = 0
    stays = 0
    for stay_index in stay_indexes:
        start, stop, _ = (int(value) for value in windows[int(stay_index)])
        target_rows = np.flatnonzero(np.isfinite(labels[start:stop]))
        selected = cutoff_chunks(target_rows)[cutoff_index]
        if selected.size == 0:
            continue
        stay = np.asarray(data[start:stop], dtype=np.float32)
        categorical = stay[:, numeric_count:]
        cat_row, cat_local = np.nonzero(categorical == 1.0)
        cat_features = cat_local.astype(np.int32) + numeric_count
        counts = (
            numeric_count * (selected + 1)
            + np.searchsorted(cat_row, selected, side="right")
        ).astype(np.int64)
        remaining = TARGET_EVENTS - events
        cumulative = np.cumsum(counts)
        selected_count = int(np.searchsorted(cumulative, remaining, side="right"))
        if selected_count == 0:
            selected_count = 1
        selected = selected[:selected_count]
        subject_id, _ = parse_filename(names[int(stay_index)])
        frame = event_block(
            stay=stay,
            target_rows=selected,
            cat_rows=cat_row,
            cat_features=cat_features,
            columns=columns,
            numeric_count=numeric_count,
            subject_id=subject_id,
            sample_id=samples + 1,
        )
        frames.append(frame)
        events += frame.height
        samples += len(selected)
        stays += 1
        if events >= TARGET_EVENTS:
            break
    if events < TARGET_EVENTS:
        raise AssertionError(
            f"{split} stratum produced only {events:,} events; expected {TARGET_EVENTS:,}"
        )
    combined = (
        pl.concat(frames, rechunk=True)
        .with_columns(pl.lit(COMMON_SPLIT[split]).alias("split"))
        .sort("subject_id", "time", "source_row", "code")
    )
    pq.write_table(
        combined.to_arrow(),
        output_path,
        compression="zstd",
        row_group_size=250_000,
    )
    size = output_path.stat().st_size
    return {
        "events": combined.height,
        "samples": samples,
        "stays": stays,
        "bytes": size,
        "bytes_per_event": size / combined.height,
        "sha256_note": "ephemeral dry-run stratum; not a canonical source artifact",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"preflight output must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = inspect_hdf(args.source_root)
    paths = source_paths(args.source_root)
    columns = np.asarray(metadata["columns"], dtype=object)
    numeric_count = len(metadata["numeric_columns"])
    task_index = int(metadata["task_index"])
    strata: dict[str, dict[str, object]] = {}
    with tables.open_file(paths["standard_scaled"], "r") as handle:
        for split in SPLIT_MAP:
            names = source_filenames(paths["multitask"], split)
            windows = handle.root.patient_windows._f_get_child(split)[:]
            labels = handle.root.labels._f_get_child(split)[:, task_index]
            eligible = np.asarray(
                [
                    index
                    for index, (start, stop, _) in enumerate(windows)
                    if np.isfinite(labels[int(start) : int(stop)]).any()
                ],
                dtype=np.int64,
            )
            lengths = np.asarray(
                [int(windows[index, 1] - windows[index, 0]) for index in eligible],
                dtype=np.int64,
            )
            bands = length_band_indexes(lengths)
            for length_name, relative_indexes in bands.items():
                stay_indexes = eligible[relative_indexes]
                for cutoff_index, cutoff_name in enumerate(CUTOFF_BANDS):
                    key = f"{split}_{length_name}_{cutoff_name}"
                    strata[key] = write_stratum(
                        handle=handle,
                        split=split,
                        stay_indexes=stay_indexes,
                        cutoff_index=cutoff_index,
                        names=names,
                        windows=windows,
                        labels=labels,
                        columns=columns,
                        numeric_count=numeric_count,
                        output_path=args.output_dir / f"{key}.parquet",
                    )

    event_count, event_counts_by_split = count_events(args.source_root)
    worst_key, worst = max(
        strata.items(), key=lambda item: float(item[1]["bytes_per_event"])
    )
    projected = math.ceil(float(worst["bytes_per_event"]) * event_count)
    conservative = math.ceil(projected * SAFETY_FACTOR) + FIXED_ALLOWANCE_BYTES
    free = shutil.disk_usage(args.output_dir).free
    report = {
        "status": "fits" if conservative + MINIMUM_FREE_AFTER_BYTES <= free else "does_not_fit",
        "canonical_event_count": event_count,
        "event_counts_by_split": event_counts_by_split,
        "strata": strata,
        "worst_stratum": worst_key,
        "worst_bytes_per_event": worst["bytes_per_event"],
        "worst_stratum_projection_bytes": projected,
        "safety_factor": SAFETY_FACTOR,
        "fixed_metadata_and_index_allowance_bytes": FIXED_ALLOWANCE_BYTES,
        "conservative_disk_bound_bytes": conservative,
        "minimum_free_after_export_bytes": MINIMUM_FREE_AFTER_BYTES,
        "free_bytes_at_preflight": free,
    }
    (args.output_dir / "preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "fits":
        raise RuntimeError("conservative Stepwise export bound does not fit")


if __name__ == "__main__":
    main()
