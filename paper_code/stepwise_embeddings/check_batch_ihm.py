"""Recreate every author loader/collate batch of the `ihm` task from MEDS.

Unlike the sampled rolling decompensation task, the phase-2 `ihm` task is small
enough (21,139 samples, 16.6M events) to reconcile exhaustively: every exported
sample is compared element by element against the released
`ICUVariableLengthLoaderTables` output at the released `MAXLEN=48`.
"""

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
from cohort_ihm import (
    AUTHOR_MAXLEN,
    AUTHOR_TARGET_ROW,
    AUTHOR_TASK,
    build_samples,
    count_events,
    ihm_task_index,
)
from cohort_source import (
    ANCHOR_US,
    COMMON_SPLIT,
    HOUR_US,
    SPLIT_MAP,
    assert_source_ready,
    inspect_hdf,
    source_filenames,
)


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


def reconstruct_prefix(
    sequence_rows: np.ndarray,
    features: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    feature_count: int,
    numeric_count: int,
) -> np.ndarray:
    dense = np.zeros((AUTHOR_MAXLEN, feature_count), dtype=np.float32)
    seen_numeric = np.zeros((AUTHOR_MAXLEN, numeric_count), dtype=bool)
    if sequence_rows.min() < 0 or sequence_rows.max() >= AUTHOR_MAXLEN:
        raise AssertionError("MEDS prefix contains an out-of-window sequence row")
    numeric = features < numeric_count
    if not np.array_equal(numeric, valid):
        raise AssertionError(
            "numeric-valued events and dense numeric channels disagree"
        )
    numeric_rows = sequence_rows[numeric]
    numeric_features = features[numeric]
    if seen_numeric[numeric_rows, numeric_features].any():
        raise AssertionError("a dense author numeric channel is duplicated")
    seen_numeric[numeric_rows, numeric_features] = True
    if not seen_numeric.all():
        raise AssertionError("MEDS prefix omits a dense author numeric channel")
    dense[numeric_rows, numeric_features] = values[numeric]
    occurrence_rows = sequence_rows[~numeric]
    occurrence_features = features[~numeric]
    if dense[occurrence_rows, occurrence_features].any():
        raise AssertionError("a one-hot occurrence is duplicated")
    dense[occurrence_rows, occurrence_features] = 1.0
    return dense


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = assert_source_ready(args.source_root)
    metadata = inspect_hdf(args.source_root)
    columns = tuple(metadata["columns"])
    feature_count = len(columns)
    numeric_count = len(metadata["numeric_columns"])
    task_index = ihm_task_index(args.source_root)
    code_position = {code: position for position, code in enumerate(columns)}

    manifest = json.loads((args.task_dir / "manifest.json").read_text())
    if manifest["task"]["modalities"] != ["structured"]:
        raise AssertionError("batch check is not bound to the structured-only experiment")
    if manifest["task"]["note_codes"] != []:
        raise AssertionError("structured-only Stepwise task declares note codes")
    if manifest["task"]["source_task"] != AUTHOR_TASK:
        raise AssertionError("batch check is not bound to the released ihm task")
    if manifest["task"]["author_maxlen"] != AUTHOR_MAXLEN:
        raise AssertionError("batch check is not bound to the released MAXLEN=48")
    codes = pl.read_parquet(args.task_dir / manifest["meds"]["codes"]["file"])
    if codes["code"].to_list() != list(columns):
        raise AssertionError("MEDS code order differs from author HDF channels")

    index = pl.read_parquet(args.task_dir / manifest["meds"]["task_index"]["file"])
    expected_index, sample_audit = build_samples(args.source_root)
    if not index.sort("source_order").equals(expected_index.sort("source_order")):
        raise AssertionError("task index differs from the exact released cohort")

    event_path = args.task_dir / manifest["meds"]["events"]["file"]
    expected_event_count, expected_by_split = count_events(args.source_root)
    parquet_file = pq.ParquetFile(event_path)
    parquet_event_count = parquet_file.metadata.num_rows
    if (
        manifest["event_count"] != expected_event_count
        or parquet_event_count != expected_event_count
    ):
        raise AssertionError("MEDS event count differs from exact HDF projection")

    by_sample = {
        int(row["sample_id"]): row
        for row in index.select(
            "sample_id",
            "split",
            "source_filename",
            "source_stay_order",
            "source_label_row",
            "boolean_value",
            "subject_id",
        ).iter_rows(named=True)
    }

    loaders = {}
    valid_positions = {}
    for split in SPLIT_MAP:
        loader = ICUVariableLengthLoaderTables(
            paths["standard_scaled"],
            on_RAM=False,
            shuffle=False,
            batch_size=1,
            splits=[split],
            maxlen=AUTHOR_MAXLEN,
            task=AUTHOR_TASK,
            data_resampling=1,
            label_resampling=1,
            use_feat=False,
            use_presence=False,
        )
        if int(loader.maxlen) != AUTHOR_MAXLEN:
            raise AssertionError("released loader did not bind MAXLEN=48")
        loaders[split] = loader
        valid_stays = np.asarray(loader.valid_indexes_samples[split])
        exported_stays = (
            index.filter(pl.col("split") == COMMON_SPLIT[split])
            .sort("source_order")["source_stay_order"]
            .to_numpy()
        )
        if not np.array_equal(valid_stays, exported_stays):
            raise AssertionError(
                f"{split} author loader sample set differs from the export"
            )
        valid_positions[split] = {
            int(stay): position for position, stay in enumerate(valid_stays)
        }

    inverse_split = {value: key for key, value in COMMON_SPLIT.items()}
    listfile_names = {
        split: source_filenames(paths["multitask"], split) for split in SPLIT_MAP
    }

    tensor_hasher = hashlib.sha256()
    meds_hasher = hashlib.sha256()
    observed_codes: set[str] = set()
    checked = 0
    positive_checked = 0
    collated_examples = []
    fixed_records = []
    padding_rows_seen: set[int] = set()
    hdf_handle = tables.open_file(paths["standard_scaled"], "r")
    try:
        root = hdf_handle.root
        pending_sample: int | None = None
        pending: list[np.ndarray] = []

        def flush(sample_id: int, rows: list[np.ndarray]) -> None:
            nonlocal checked, positive_checked
            sequence_rows = np.concatenate([part[0] for part in rows])
            features = np.concatenate([part[1] for part in rows])
            values = np.concatenate([part[2] for part in rows])
            valid = np.concatenate([part[3] for part in rows])
            times = np.concatenate([part[4] for part in rows])
            source_rows = np.concatenate([part[5] for part in rows])
            source_times = np.concatenate([part[6] for part in rows])
            subjects = np.concatenate([part[7] for part in rows])
            meta = by_sample[sample_id]
            if not np.array_equal(
                times, ANCHOR_US + sequence_rows.astype(np.int64) * HOUR_US
            ):
                raise AssertionError("MEDS timestamp differs from the author grid row")
            if not np.array_equal(
                source_rows,
                sequence_rows.astype(np.int64) * feature_count
                + features.astype(np.int64),
            ):
                raise AssertionError("MEDS source_row differs from flattened coordinate")
            if not np.array_equal(source_times, sequence_rows.astype(np.float64)):
                raise AssertionError("MEDS source_time differs from the author row")
            if (subjects != int(meta["subject_id"])).any():
                raise AssertionError("MEDS event subject differs from its sample")
            dense = reconstruct_prefix(
                sequence_rows,
                features,
                values,
                valid,
                feature_count=feature_count,
                numeric_count=numeric_count,
            )
            split = inverse_split[meta["split"]]
            start = int(meta["source_label_row"]) - AUTHOR_TARGET_ROW
            expected = np.asarray(
                root.data._f_get_child(split)[start : start + AUTHOR_MAXLEN],
                dtype=np.float32,
            )
            np.testing.assert_array_equal(dense, expected)
            loader = loaders[split]
            position = valid_positions[split][int(meta["source_stay_order"])]
            x, pad_mask, y = loader.sample(None, split, position)
            if x.dtype != np.float32 or y.dtype != np.float32 or pad_mask.dtype != bool:
                raise AssertionError("author loader boundary dtype changed")
            if x.shape != (AUTHOR_MAXLEN, feature_count):
                raise AssertionError("author loader shape is not (48, 42)")
            if y.shape != (AUTHOR_MAXLEN,) or pad_mask.shape != (AUTHOR_MAXLEN,):
                raise AssertionError("author label/mask shape is not (48,)")
            np.testing.assert_array_equal(x, dense)
            expected_mask = np.zeros(AUTHOR_MAXLEN, dtype=bool)
            expected_mask[AUTHOR_TARGET_ROW] = True
            if not np.array_equal(pad_mask, expected_mask):
                raise AssertionError(
                    "author target mask is not exactly the single row-47 landmark"
                )
            if bool(y[AUTHOR_TARGET_ROW] == 1.0) != bool(meta["boolean_value"]):
                raise AssertionError("MEDS label differs from the author loader label")
            if not np.array_equal(
                y[~expected_mask], np.full(AUTHOR_MAXLEN - 1, -1.0, dtype=np.float32)
            ):
                raise AssertionError("unlabeled author rows are not the released -1 fill")
            stay_rows = int(
                loader.patient_windows[split][int(meta["source_stay_order"]), 1]
                - loader.patient_windows[split][int(meta["source_stay_order"]), 0]
            )
            padding_rows_seen.add(max(0, AUTHOR_MAXLEN - stay_rows))
            tensor_hasher.update(str(sample_id).encode())
            tensor_hasher.update(x.tobytes())
            tensor_hasher.update(y.tobytes())
            tensor_hasher.update(pad_mask.tobytes())
            meds_hasher.update(str(sample_id).encode())
            meds_hasher.update(dense.tobytes())
            observed_codes.update(
                {columns[position] for position in np.unique(features)}
            )
            checked += 1
            positive_checked += int(bool(meta["boolean_value"]))
            if len(collated_examples) < 9 and (
                checked == 1 or checked % 2500 == 0
            ):
                collated_x, collated_y, collated_mask = default_collate(
                    [(x, y, pad_mask)]
                )
                np.testing.assert_array_equal(collated_x.numpy()[0], x)
                np.testing.assert_array_equal(collated_y.numpy()[0], y)
                np.testing.assert_array_equal(collated_mask.numpy()[0], pad_mask)
                collated_examples.append(sample_id)
                fixed_records.append(
                    {
                        "sample_id": sample_id,
                        "split": meta["split"],
                        "source_filename": meta["source_filename"],
                        "listfile_filename": listfile_names[split][
                            int(meta["source_stay_order"])
                        ],
                        "source_stay_order": int(meta["source_stay_order"]),
                        "loader_shape": list(x.shape),
                        "loader_dtype": str(x.dtype),
                        "label_shape": list(y.shape),
                        "target_mask_shape": list(pad_mask.shape),
                        "eligible_landmarks": int(pad_mask.sum()),
                        "author_stay_rows": stay_rows,
                        "structural_padding_rows": max(0, AUTHOR_MAXLEN - stay_rows),
                        "loader_collate_hash": array_hash(x, y, pad_mask),
                    }
                )
            if meta["source_filename"] != listfile_names[split][
                int(meta["source_stay_order"])
            ]:
                raise AssertionError("MEDS stay identity differs from the listfile")

        for batch in parquet_file.iter_batches(
            batch_size=250_000,
            columns=[
                "sample_id",
                "subject_id",
                "time",
                "code",
                "numeric_value",
                "sequence_row",
                "source_row",
                "source_time",
            ],
        ):
            sample_ids = batch.column("sample_id").to_numpy(zero_copy_only=False)
            subject_ids = batch.column("subject_id").to_numpy(zero_copy_only=False)
            times = (
                batch.column("time").cast("int64").to_numpy(zero_copy_only=False)
            )
            code_values = batch.column("code").to_pylist()
            features = np.fromiter(
                (code_position[code] for code in code_values),
                dtype=np.int32,
                count=len(code_values),
            )
            numeric = batch.column("numeric_value")
            valid = ~np.asarray(numeric.is_null())
            values = np.nan_to_num(
                numeric.to_numpy(zero_copy_only=False).astype(np.float32),
                nan=0.0,
            )
            sequence_rows = batch.column("sequence_row").to_numpy(
                zero_copy_only=False
            )
            source_rows = batch.column("source_row").to_numpy(zero_copy_only=False)
            source_times = batch.column("source_time").to_numpy(
                zero_copy_only=False
            )
            boundaries = np.flatnonzero(np.diff(sample_ids)) + 1
            starts = np.concatenate(([0], boundaries))
            stops = np.concatenate((boundaries, [len(sample_ids)]))
            for start, stop in zip(starts, stops):
                sample_id = int(sample_ids[start])
                piece = (
                    sequence_rows[start:stop],
                    features[start:stop],
                    values[start:stop],
                    valid[start:stop],
                    times[start:stop],
                    source_rows[start:stop],
                    source_times[start:stop],
                    subject_ids[start:stop],
                )
                if pending_sample is None:
                    pending_sample = sample_id
                    pending = [piece]
                elif sample_id == pending_sample:
                    pending.append(piece)
                else:
                    flush(pending_sample, pending)
                    if checked % 5000 == 0:
                        print(f"reconciled {checked:,} samples", flush=True)
                    pending_sample = sample_id
                    pending = [piece]
        if pending_sample is not None:
            flush(pending_sample, pending)
    finally:
        hdf_handle.close()
        for loader in loaders.values():
            loader.data_h5._v_file.close()

    if checked != index.height:
        raise AssertionError("not every exported sample was reconciled")
    if not observed_codes.issubset(set(columns)):
        raise AssertionError("an observed MEDS code is absent from author channels")
    if not set(metadata["numeric_columns"]).issubset(observed_codes):
        raise AssertionError("a dense author numeric channel has no MEDS events")

    report = {
        "status": "exact",
        "source_model_input": {
            "task": AUTHOR_TASK,
            "maxlen": AUTHOR_MAXLEN,
            "target_row": AUTHOR_TARGET_ROW,
            "modalities": ["structured"],
            "hdf_channels": feature_count,
            "dense_numeric_channels": numeric_count,
            "one_hot_occurrence_channels": feature_count - numeric_count,
            "observation_mask": None,
            "target_mask_is_not_an_observation_mask": True,
            "structural_padding_rows_observed": sorted(padding_rows_seen),
            "no_label_shift": (
                "TLS writes the ihm target at author row 47 without the NaN "
                "prepend that shifts its decompensation targets"
            ),
        },
        "lightgbm_boundary": {
            "ordered_codes": list(columns),
            "note_codes": [],
            "observed_code_count": len(observed_codes),
            "declared_code_count": feature_count,
            "unobserved_declared_codes": sorted(set(columns) - observed_codes),
            "imaging_or_note_codes": [],
            "event_count": expected_event_count,
            "event_counts_by_split": expected_by_split,
        },
        "full_traversal": {
            "samples_reconciled": checked,
            "positives_reconciled": positive_checked,
            "sample_audit": sample_audit["splits"],
            "ordered_author_loader_tensor_sha256": tensor_hasher.hexdigest(),
            "ordered_meds_reconstruction_sha256": meds_hasher.hexdigest(),
            "coverage": "every exported sample, not a fixed subset",
        },
        "fixed_author_loader_and_collate": fixed_records,
        "assertions": [
            "exact author HDF channel order and task declaration",
            "released MAXLEN=48 bound in the loader and the manifest",
            "exact author loader float32 values, labels, target masks, shapes, "
            "dtypes, and padding for every exported sample",
            "exact default-collate tensors on fixed samples",
            "exact MEDS reconstruction of every 48-row author prefix",
            "exact author loader sample set per split",
            "exact exported event count and event vocabulary",
            "no notes, imaging, auxiliary, or observation-mask events",
        ],
    }
    output = args.task_dir / "audit" / "full_batch_recreation.json"
    write_json(output, report)
    print(
        json.dumps(
            {
                "samples_reconciled": checked,
                "positives": positive_checked,
                "event_count": expected_event_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
