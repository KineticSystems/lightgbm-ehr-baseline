"""Assert that MEDS recreates every deterministic OC-Distill student batch."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch


REPOSITORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY))

from export_source import (  # noqa: E402
    HOURS,
    VARIABLE_NAMES,
    load_cohort,
    sha256,
)
from utils.data import PatientDataset, make_collate  # noqa: E402
from utils.utils import build_vitals_tensors  # noqa: E402


def arrow_lookup(table):
    sample = table["sample_id"].to_numpy()
    order = np.argsort(sample, kind="stable")
    table = table.take(pa.array(order, type=pa.int64()))
    sample = sample[order]
    boundaries = np.flatnonzero(np.r_[True, sample[1:] != sample[:-1], True])
    lookup = {
        int(sample[start]): (int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    }
    return table, lookup


def compare_batch(source, recreated, digests, shapes):
    if set(source) != {"vitals", "y"} or set(recreated) != {"vitals", "y"}:
        raise AssertionError("unexpected deployed-student batch keys")
    for name in ("vitals", "y"):
        left = source[name]
        right = recreated[name]
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            raise AssertionError(f"{name} shape/dtype mismatch")
        if not torch.equal(left, right):
            maximum = float(torch.max(torch.abs(left - right)).item())
            raise AssertionError(f"{name} differs; maximum={maximum}")
        raw = left.detach().cpu().contiguous().numpy()
        digests[name].update(str(raw.dtype).encode())
        digests[name].update(np.asarray(raw.shape, dtype=np.int64).tobytes())
        digests[name].update(raw.tobytes())
        shapes[name].add((tuple(raw.shape), str(raw.dtype)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--auxiliary-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.task_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["task"].get("modalities") != ["structured"]:
        raise AssertionError("OC-Distill task must be structured-only")
    if manifest["task"].get("note_codes") != []:
        raise AssertionError("OC-Distill must not expose notes to the shared model")
    rows, _, retained_frames, _ = load_cohort(
        args.cohort_root,
        args.mimic_root,
        args.auxiliary_root,
        args.cache_root,
        reuse_cache=True,
    )
    event_path = args.task_dir / manifest["meds"]["events"]["file"]
    index_path = args.task_dir / manifest["meds"]["task_index"]["file"]
    events = pq.read_table(
        event_path,
        columns=[
            "sample_id",
            "time",
            "code",
            "numeric_value",
            "source_row",
        ],
    )
    events, event_lookup = arrow_lookup(events)
    gc.collect()
    pa.default_memory_pool().release_unused()
    index = pq.read_table(index_path)
    index_rows = {int(row["sample_id"]): row for row in index.to_pylist()}
    source_order = {
        (row["source_split"], int(row["source_position"])): int(row["sample_id"])
        for row in index.to_pylist()
    }
    variable_index = {name: position for position, name in enumerate(VARIABLE_NAMES)}

    normal_values = np.asarray(
        manifest["source"]["batch_recreation"]["normal_values"], dtype=np.float32
    )

    def recreate(sample_id):
        start, end = event_lookup[sample_id]
        part = events.slice(start, end - start)
        record = index_rows[sample_id]
        observation_start = np.datetime64(record["observation_start"], "us")
        event_time = part["time"].to_numpy()
        elapsed = np.asarray(
            [(time - observation_start) / np.timedelta64(1, "h") for time in event_time]
        )
        bins = np.floor(elapsed - 1e-6).astype(int)
        bins = np.clip(bins, 0, 47)
        code = np.asarray(
            [variable_index[value] for value in part["code"].to_pylist()], dtype=int
        )
        value = part["numeric_value"].to_numpy().astype(np.float32)
        time_us = event_time.astype("datetime64[us]").astype(np.int64)
        observed = np.full((48, len(VARIABLE_NAMES)), np.nan, dtype=np.float32)
        chosen_time = np.full(
            (48, len(VARIABLE_NAMES)), np.iinfo(np.int64).min, dtype=np.int64
        )
        for row, column, candidate_time, candidate_value in zip(
            bins, code, time_us, value, strict=True
        ):
            if candidate_time > chosen_time[row, column] or (
                candidate_time == chosen_time[row, column]
                and (
                    np.isnan(observed[row, column])
                    or candidate_value > observed[row, column]
                )
            ):
                chosen_time[row, column] = candidate_time
                observed[row, column] = candidate_value
        mask = np.isfinite(observed).astype(np.float32)
        values = np.empty_like(observed)
        for column in range(len(VARIABLE_NAMES)):
            previous = normal_values[column]
            for row in range(48):
                if mask[row, column]:
                    previous = observed[row, column]
                values[row, column] = previous
        return np.concatenate([values, mask], axis=1).astype(np.float32)

    position_by_retained = {
        (row["source_split"], row["retained_position"]): row["source_position"]
        for row in rows
    }
    norm_stats_path = REPOSITORY / "vital_norm_stats_48h.pt"
    collate = make_collate()
    digests = {name: hashlib.sha256() for name in ("vitals", "y")}
    shapes = {name: set() for name in ("vitals", "y")}
    ordered_labels = hashlib.sha256()
    batches = 0
    examples = 0
    maximum_pre_normalization_difference = 0.0

    for source_split in ("train", "test"):
        frame = retained_frames[source_split].reset_index(drop=True)
        split_names = []
        for retained_position in range(len(frame)):
            source_position = position_by_retained[(source_split, retained_position)]
            sample_id = source_order[(source_split, source_position)]
            split_names.append(index_rows[sample_id]["split"])
        if source_split == "train":
            partitions = (
                ("train", np.flatnonzero(np.asarray(split_names) == "train")),
                (
                    "validation",
                    np.flatnonzero(np.asarray(split_names) == "validation"),
                ),
            )
        else:
            partitions = (("test", np.arange(len(frame))),)
        for _, selected in partitions:
            for start in range(0, len(selected), 1024):
                stop = min(start + 1024, len(selected))
                batch_positions = selected[start:stop]
                batch_frame = frame.iloc[batch_positions].reset_index(drop=True)
                recreated_x = []
                for local_position, record in enumerate(
                    batch_frame.itertuples(index=False)
                ):
                    retained_position = int(batch_positions[local_position])
                    source_position = position_by_retained[
                        (source_split, retained_position)
                    ]
                    sample_id = source_order[(source_split, source_position)]
                    dense = recreate(sample_id)
                    source_dense = np.asarray(record.ts, dtype=np.float32)
                    difference = float(
                        np.max(np.abs(dense - source_dense), initial=0.0)
                    )
                    maximum_pre_normalization_difference = max(
                        maximum_pre_normalization_difference, difference
                    )
                    if not np.array_equal(dense, source_dense):
                        raise AssertionError(
                            f"MEDS did not exactly reconstruct stay "
                            f"{int(record.ICUSTAY_ID)}; maximum={difference}"
                        )
                    recreated_x.append(dense)
                source_vitals = build_vitals_tensors(batch_frame, norm_stats_path)
                recreated_vitals = build_vitals_tensors(
                    pd.DataFrame({"ts": recreated_x}), norm_stats_path
                )
                labels = torch.tensor(
                    batch_frame[["label"]].values.astype(np.float32)
                )
                local_indices = np.arange(len(batch_frame))
                source_dataset = PatientDataset(
                    local_indices, source_vitals, labels
                )
                recreated_dataset = PatientDataset(
                    local_indices, recreated_vitals, labels
                )
                source_batch = collate(
                    [source_dataset[position] for position in local_indices]
                )
                recreated_batch = collate(
                    [recreated_dataset[position] for position in local_indices]
                )
                compare_batch(source_batch, recreated_batch, digests, shapes)
                ordered_labels.update(
                    source_batch["y"].detach().cpu().contiguous().numpy().tobytes()
                )
                examples += stop - start
                batches += 1

    del (
        events,
        event_lookup,
        source_vitals,
        recreated_vitals,
        source_dataset,
        recreated_dataset,
        recreated_x,
        batch_frame,
        source_batch,
        recreated_batch,
        labels,
        frame,
    )
    gc.collect()
    pa.default_memory_pool().release_unused()
    actual_codes = set()
    included_event_count = 0
    for batch in pq.ParquetFile(event_path).iter_batches(
        batch_size=250_000,
        columns=["code"],
    ):
        structured = pc.is_in(
            batch["code"], value_set=pa.array(VARIABLE_NAMES)
        ).to_numpy(zero_copy_only=False)
        if not np.all(structured):
            raise AssertionError("MEDS contains an undeclared OC-Distill event code")
        actual_codes.update(
            pc.unique(batch.filter(pa.array(structured))["code"]).to_pylist()
        )
        included_event_count += len(batch)
    if actual_codes != set(VARIABLE_NAMES):
        raise AssertionError("MEDS vital code set differs from the released item map")
    report = {
        "status": "exact",
        "comparison": (
            "all note-restricted deployed-student examples in exhaustive fixed batches"
        ),
        "examples": examples,
        "batches": batches,
        "batch_size": 1024,
        "batch_keys": ["vitals", "y"],
        "maximum_pre_normalization_difference": maximum_pre_normalization_difference,
        "ordered_label_sha256": ordered_labels.hexdigest(),
        "field_sha256": {name: digest.hexdigest() for name, digest in digests.items()},
        "field_shapes_and_dtypes": {
            name: [
                {"shape": list(shape), "dtype": dtype}
                for shape, dtype in sorted(values)
            ]
            for name, values in shapes.items()
        },
        "author_dataset": "utils.data::PatientDataset",
        "author_collate": "utils.data::make_collate",
        "normalization_artifact_sha256": sha256(norm_stats_path),
        "structured_feature_audit": {
            "status": "exact",
            "included_event_count": included_event_count,
            "included_codes": sorted(actual_codes),
            "excluded_event_count": 0,
            "excluded_codes": [],
            "rule": "every event is structured because task.note_codes is empty",
        },
    }
    report_path = args.task_dir / "audit" / "full_batch_recreation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
