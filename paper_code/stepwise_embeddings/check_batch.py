"""Recreate fixed author loader/collate batches from the exported MEDS task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import tables
from torch.utils.data._utils.collate import default_collate

from icu_benchmarks.data.loader import ICUVariableLengthLoaderTables
from cohort_source import (
    ANCHOR_US,
    AUTHOR_TASK,
    COMMON_SPLIT,
    HOUR_US,
    SPLIT_MAP,
    assert_source_ready,
    build_samples,
    count_events,
    inspect_hdf,
    source_filenames,
)
from missing_xgboost_lightgbm.landmark_sampling import select_uniform_landmarks


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def fixed_positions(valid: np.ndarray) -> list[int]:
    if len(valid) < 3:
        return list(range(len(valid)))
    return [0, len(valid) // 2, len(valid) - 1]


def fixed_landmarks(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.height <= 3:
        return frame.sort("author_target_row")
    ordered = frame.sort("author_target_row")
    return ordered[[0, ordered.height // 2, ordered.height - 1]]


def reconstruct_prefix(
    events: pl.DataFrame,
    *,
    columns: tuple[str, ...],
    numeric_count: int,
    author_row: int,
) -> tuple[np.ndarray, dict[str, object]]:
    expected_rows = author_row + 1
    feature_index = {name: index for index, name in enumerate(columns)}
    dense = np.zeros((expected_rows, len(columns)), dtype=np.float32)
    seen_numeric = np.zeros((expected_rows, numeric_count), dtype=bool)
    source_rows = events["source_row"].to_numpy()
    sequence_rows = events["sequence_row"].to_numpy()
    source_times = events["source_time"].to_numpy()
    times = events["time"].cast(pl.Int64).to_numpy()
    codes = events["code"].to_list()
    values = events["numeric_value"].to_numpy()
    valid = events["numeric_value"].is_not_null().to_numpy()
    for position, (row, code) in enumerate(zip(sequence_rows, codes)):
        row = int(row)
        feature = feature_index[code]
        if not 0 <= row <= author_row:
            raise AssertionError("MEDS prefix contains an out-of-prefix sequence row")
        if int(source_rows[position]) != row * len(columns) + feature:
            raise AssertionError("MEDS source_row differs from author flattened coordinate")
        if source_times[position] != float(row):
            raise AssertionError("MEDS source_time differs from author sequence row")
        if int(times[position]) != ANCHOR_US + row * HOUR_US:
            raise AssertionError("MEDS event timestamp differs from author hourly row")
        if feature < numeric_count:
            if not valid[position] or seen_numeric[row, feature]:
                raise AssertionError("numeric author channel is missing or duplicated")
            dense[row, feature] = values[position]
            seen_numeric[row, feature] = True
        else:
            if valid[position]:
                raise AssertionError("one-hot occurrence unexpectedly has a numeric value")
            if dense[row, feature] != 0:
                raise AssertionError("one-hot occurrence is duplicated")
            dense[row, feature] = 1.0
    if not seen_numeric.all():
        raise AssertionError("MEDS prefix omits a dense author numeric channel")
    categorical = dense[:, numeric_count:]
    if not np.isin(categorical, (0.0, 1.0)).all():
        raise AssertionError("reconstructed one-hot channels are not binary")
    return dense, {
        "events": events.height,
        "numeric_events": int(valid.sum()),
        "occurrence_events": int((~valid).sum()),
        "prefix_hash": array_hash(dense),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = assert_source_ready(args.source_root)
    metadata = inspect_hdf(args.source_root)
    columns = tuple(metadata["columns"])
    numeric_count = len(metadata["numeric_columns"])
    manifest = json.loads((args.task_dir / "manifest.json").read_text())
    if manifest["task"]["modalities"] != ["structured"]:
        raise AssertionError("batch check is not bound to the structured-only experiment")
    if manifest["task"]["note_codes"] != []:
        raise AssertionError("structured-only Stepwise task declares note codes")
    codes = pl.read_parquet(args.task_dir / manifest["meds"]["codes"]["file"])
    if codes["code"].to_list() != list(columns):
        raise AssertionError("MEDS code order differs from author HDF channels")
    index = pl.read_parquet(args.task_dir / manifest["meds"]["task_index"]["file"])
    event_path = args.task_dir / manifest["meds"]["events"]["file"]
    full_cohort, _ = build_samples(args.source_root)
    expected_index, sampling_audit = select_uniform_landmarks(full_cohort)
    if not index.sort("source_order").equals(expected_index.sort("source_order")):
        raise AssertionError("task index differs from deterministic landmark subsample")
    if manifest["source"].get("sampling") != sampling_audit:
        raise AssertionError("manifest sampling record differs from deterministic contract")
    selected_sample_ids = np.sort(
        index["sample_id"].to_numpy().astype(np.int64, copy=False)
    )
    expected_event_count, expected_by_split = count_events(
        args.source_root, selected_sample_ids
    )
    parquet_event_count = pq.ParquetFile(event_path).metadata.num_rows
    if manifest["event_count"] != expected_event_count or parquet_event_count != expected_event_count:
        raise AssertionError("MEDS event count differs from exact HDF projection")
    observed_codes = set(
        pl.scan_parquet(event_path)
        .select(pl.col("code").unique())
        .collect(engine="streaming")["code"]
        .to_list()
    )
    if not observed_codes.issubset(set(columns)):
        raise AssertionError("an observed MEDS code is absent from author channels")
    if not set(metadata["numeric_columns"]).issubset(observed_codes):
        raise AssertionError("a dense author numeric channel has no MEDS events")

    chosen = []
    loader_records = []
    for split in SPLIT_MAP:
        loader = ICUVariableLengthLoaderTables(
            paths["standard_scaled"],
            on_RAM=False,
            shuffle=False,
            batch_size=1,
            splits=[split],
            maxlen=-1,
            task=AUTHOR_TASK,
            data_resampling=1,
            label_resampling=1,
            use_feat=False,
            use_presence=False,
        )
        names = source_filenames(paths["multitask"], split)
        common_split = COMMON_SPLIT[split]
        split_index = index.filter(pl.col("split") == common_split).sort(
            "source_order"
        )
        anchor_rows = split_index[fixed_positions(np.arange(split_index.height))]
        raw_stay_indexes = anchor_rows["source_stay_order"].unique(
            maintain_order=True
        ).to_list()
        valid_stays = np.asarray(loader.valid_indexes_samples[split])
        for raw_stay_index in raw_stay_indexes:
            valid_matches = np.flatnonzero(valid_stays == raw_stay_index)
            if len(valid_matches) != 1:
                raise AssertionError("sampled stay is absent or duplicated in author loader")
            valid_position = int(valid_matches[0])
            raw_stay_index = int(raw_stay_index)
            source_name = names[raw_stay_index]
            x, target_mask, y = loader.sample(None, split, valid_position)
            if x.dtype != np.float32 or y.dtype != np.float32 or target_mask.dtype != bool:
                raise AssertionError("author loader boundary dtype changed")
            collated_x, collated_y, collated_mask = default_collate(
                [(x, y, target_mask)]
            )
            np.testing.assert_array_equal(collated_x.numpy()[0], x)
            np.testing.assert_array_equal(collated_y.numpy()[0], y)
            np.testing.assert_array_equal(collated_mask.numpy()[0], target_mask)
            task_stay = index.filter(
                (pl.col("split") == common_split)
                & (pl.col("source_stay_order") == raw_stay_index)
            )
            ordered_task_stay = task_stay.sort("author_target_row")
            target_rows = ordered_task_stay["author_target_row"].to_numpy().astype(
                np.int64
            )
            if not target_mask[target_rows].all():
                raise AssertionError("a sampled landmark is absent from loader target mask")
            if not np.array_equal(
                y[target_rows].astype(bool),
                ordered_task_stay["boolean_value"].to_numpy(),
            ):
                raise AssertionError("sampled labels differ from author loader labels")
            if task_stay["source_filename"].unique().to_list() != [source_name]:
                raise AssertionError("MEDS fixed stay identity differs from source listfile")
            selected = fixed_landmarks(task_stay)
            chosen.extend(selected["sample_id"].to_list())
            loader_records.append(
                {
                    "split": COMMON_SPLIT[split],
                    "source_filename": source_name,
                    "source_stay_order": raw_stay_index,
                    "loader_shape": list(x.shape),
                    "loader_dtype": str(x.dtype),
                    "label_shape": list(y.shape),
                    "target_mask_shape": list(target_mask.shape),
                    "eligible_landmarks": int(target_mask.sum()),
                    "exported_landmarks": task_stay.height,
                    "structural_padding_rows": int(
                        loader.maxlen
                        - (
                            loader.patient_windows[split][raw_stay_index, 1]
                            - loader.patient_windows[split][raw_stay_index, 0]
                        )
                    ),
                    "loader_collate_hash": array_hash(x, y, target_mask),
                }
            )
        loader.data_h5._v_file.close()

    fixed_events = (
        pl.scan_parquet(event_path)
        .filter(pl.col("sample_id").is_in(chosen))
        .collect(engine="streaming")
    )
    fixed_index = index.filter(pl.col("sample_id").is_in(chosen)).sort("sample_id")
    if fixed_index.height != len(chosen):
        raise AssertionError("fixed MEDS sample IDs are missing")
    parity = []
    hdf_handle = tables.open_file(paths["standard_scaled"], "r")
    try:
        root = hdf_handle.root
        for row in fixed_index.iter_rows(named=True):
            sample_id = int(row["sample_id"])
            split = {value: key for key, value in COMMON_SPLIT.items()}[row["split"]]
            target_row = int(row["author_target_row"])
            source_label_row = int(row["source_label_row"])
            expected = np.asarray(
                root.data._f_get_child(split)[
                    source_label_row - target_row : source_label_row + 1
                ],
                dtype=np.float32,
            )
            events = fixed_events.filter(pl.col("sample_id") == sample_id)
            actual, event_summary = reconstruct_prefix(
                events,
                columns=columns,
                numeric_count=numeric_count,
                author_row=target_row,
            )
            np.testing.assert_array_equal(actual, expected)
            label = float(root.labels._f_get_child(split)[source_label_row, int(metadata["task_index"])])
            if bool(label) != bool(row["boolean_value"]):
                raise AssertionError("fixed MEDS label differs from author loader")
            parity.append(
                {
                    "sample_id": sample_id,
                    "split": row["split"],
                    "source_filename": row["source_filename"],
                    "source_cutoff_hour": int(row["source_cutoff_hour"]),
                    "author_target_row": target_row,
                    **event_summary,
                }
            )
    finally:
        hdf_handle.close()

    report = {
        "status": "exact",
        "source_model_input": {
            "task": AUTHOR_TASK,
            "modalities": ["structured"],
            "hdf_channels": len(columns),
            "dense_numeric_channels": numeric_count,
            "one_hot_occurrence_channels": len(columns) - numeric_count,
            "observation_mask": None,
            "target_mask_is_not_an_observation_mask": True,
            "cutoff_defect_preserved": "source target t is evaluated with author rows 0..t+1",
        },
        "lightgbm_boundary": {
            "ordered_codes": list(columns),
            "note_codes": [],
            "observed_code_count": len(observed_codes),
            "declared_code_count": len(columns),
            "unobserved_declared_codes": sorted(set(columns) - observed_codes),
            "imaging_or_note_codes": [],
            "event_count": expected_event_count,
            "event_counts_by_split": expected_by_split,
            "sampling": sampling_audit,
        },
        "fixed_author_loader_and_collate": loader_records,
        "fixed_landmark_prefix_parity": parity,
        "assertions": [
            "exact author HDF channel order and task declaration",
            "exact loader float32 values, labels, target masks, shapes, and padding",
            "exact default-collate tensors",
            "exact MEDS reconstruction of fixed causal prefixes",
            "exact exported-subsample event count and event vocabulary",
            "no notes, imaging, auxiliary, or observation-mask events",
        ],
    }
    output = args.task_dir / "audit" / "full_batch_recreation.json"
    write_json(output, report)
    print(
        json.dumps(
            {
                "fixed_stays": len(loader_records),
                "fixed_landmarks": len(parity),
                "event_count": expected_event_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
