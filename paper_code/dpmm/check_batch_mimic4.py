"""Exhaustively recreate DPMM's retained MIMIC-IV loader and collate boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from cohort_source_mimic4 import (  # noqa: E402
    DISCRETIZER_CONFIG,
    EXPECTED_COUNTS,
    HOURS,
    IHM_TASK_ID,
    READM_TASK_ID,
    author_datasets,
    ensure_source,
    make_discretizer_normalizer,
    sha256,
    source_codes,
)
from dataset_mimic4.fusion import my_collate  # noqa: E402


BATCH_SIZE = 32
ARROW_BATCH_SIZE = 100_000
EXPORT_BATCH_SIZE = 250_000

TASKS = {
    IHM_TASK_ID: ("in-hospital-mortality", "mortality_label", "mortality"),
    READM_TASK_ID: ("readmission", "readmission_label", "readm"),
}


class BatchCursor:
    """Read an Arrow batch iterator in bounded exact-length slices."""

    def __init__(self, batches):
        self._batches = iter(batches)
        self._batch = None
        self._offset = 0

    def take(self, count: int) -> pa.Table:
        pieces = []
        remaining = count
        while remaining:
            if self._batch is None or self._offset == len(self._batch):
                self._batch = next(self._batches)
                self._offset = 0
            length = min(remaining, len(self._batch) - self._offset)
            pieces.append(
                pa.Table.from_batches([self._batch.slice(self._offset, length)])
            )
            self._offset += length
            remaining -= length
        return pa.concat_tables(pieces) if len(pieces) > 1 else pieces[0]


def update_digest(digest: "hashlib._Hash", values) -> None:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.ascontiguousarray(values)
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())


def meds_to_author_array(
    frame: pd.DataFrame, discretizer, normalizer, source_header: list[str]
) -> np.ndarray:
    frame = frame.sort_values(["source_time", "source_row", "code"], kind="stable")
    times = np.sort(frame["source_time"].unique())
    row_by_time = {float(value): index for index, value in enumerate(times)}
    channel_by_name = {name: index + 1 for index, name in enumerate(source_header)}
    values = np.full((len(times), len(source_header) + 1), "", dtype=object)
    for index, value in enumerate(times):
        values[index, 0] = repr(float(value))
    for row in frame.itertuples(index=False):
        if "//" in row.code:
            variable, value = row.code.split("//", 1)
        else:
            variable, value = row.code, repr(float(np.float32(row.numeric_value)))
        values[row_by_time[float(row.source_time)], channel_by_name[variable]] = value
    data, _ = discretizer.transform(values, header=["Hours", *source_header], end=HOURS)
    return normalizer.transform(data)


def max_float32_ulp(left: np.ndarray, right: np.ndarray) -> int:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        return np.iinfo(np.int32).max
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.any():
        return 0
    a = left[finite].view(np.int32).astype(np.int64)
    b = right[finite].view(np.int32).astype(np.int64)
    a = np.where(a < 0, 0x80000000 - a, a)
    b = np.where(b < 0, 0x80000000 - b, b)
    return int(np.max(np.abs(a - b)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = ensure_source(args.mimic_root, args.cache_root)
    cohort = pd.read_parquet(args.cache_root / source_manifest["files"]["cohort"])
    manifest_path = args.task_dir / "manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    task_id = manifest["task"]["id"]
    if task_id not in TASKS:
        raise AssertionError("wrong DPMM MIMIC-IV task")
    author_task, label_column, labels_set = TASKS[task_id]
    if manifest["task"]["modalities"] != ["structured"]:
        raise AssertionError("DPMM MIMIC-IV tasks retain structured inputs only")
    if manifest["task"]["note_codes"] != []:
        raise AssertionError("DPMM MIMIC-IV has no note channel")
    configuration = manifest["source"]["source_configuration"]
    if configuration["task"] != author_task or configuration["labels_set"] != labels_set:
        raise AssertionError("wrong selected source configuration")
    if configuration["data_pairs"] != "partial_ehr_cxr" or configuration[
        "data_ratio"
    ] != 1.0:
        raise AssertionError("wrong selected partial-cohort configuration")

    codes = pq.read_table(args.task_dir / manifest["meds"]["codes"]["file"])[
        "code"
    ].to_pylist()
    if codes != list(source_codes()):
        raise AssertionError("event vocabulary differs from released channels")
    if any(
        "image" in code.lower() or "cxr" in code.lower() or "note" in code.lower()
        for code in codes
    ):
        raise AssertionError("a non-structured code entered the retained projection")

    event_path = args.task_dir / manifest["meds"]["events"]["file"]
    datasets = author_datasets(args.cache_root, author_task)
    discretizer, normalizer, output_header = make_discretizer_normalizer(args.cache_root)
    source_header = json.loads(DISCRETIZER_CONFIG.read_text())["id_to_channel"]

    digest_names = (
        "ehr_float64",
        "ehr_model_float32",
        "label_ehr",
        "sequence_length",
        "pair_presence",
    )
    digests = {name: hashlib.sha256() for name in digest_names}
    shapes = {name: set() for name in digest_names}
    split_report = {}
    maximum_ulp = 0
    maximum_absolute_delta = 0.0
    total_batches = 0
    total_examples = 0
    total_mask_observed = 0

    source_event_path = args.cache_root / source_manifest["files"]["structured_events"]
    source_file = pq.ParquetFile(source_event_path)
    actual_file = pq.ParquetFile(event_path)
    source_event_count = source_file.metadata.num_rows
    if actual_file.metadata.num_rows != source_event_count:
        raise AssertionError("MEDS event count differs from the structured source")

    ordered_lookup = cohort.sort_values("stay_id")
    lookup_ids = ordered_lookup["stay_id"].to_numpy(np.int64)
    lookup_intimes = pd.to_datetime(ordered_lookup["intime"]).to_numpy(
        dtype="datetime64[ns]"
    )
    lookup_subjects = ordered_lookup["subject_id"].to_numpy(np.int64)
    row_by_stay = cohort.set_index("stay_id")
    categorical_indices = [
        index
        for index, channel in enumerate(output_header)
        if "->" in channel and not channel.startswith("mask->")
    ]
    recreated_by_stay: dict[int, np.ndarray] = {}

    def check_loader_group(group: pd.DataFrame) -> None:
        nonlocal maximum_ulp, maximum_absolute_delta, total_mask_observed
        stay_id = int(group["sample_id"].iloc[0])
        source_row = row_by_stay.loc[stay_id]
        author = datasets[str(source_row["split"])]
        author_ehr, author_label = author[str(source_row["source_name"])]
        recreated_ehr = meds_to_author_array(
            group, discretizer, normalizer, source_header
        )
        if author_ehr.shape != (24, 76) or author_ehr.dtype != np.float64:
            raise AssertionError("released EHR loader boundary changed")
        if recreated_ehr.shape != author_ehr.shape:
            raise AssertionError("MEDS transform changed EHR tensor shape")
        maximum_ulp = max(maximum_ulp, max_float32_ulp(recreated_ehr, author_ehr))
        maximum_absolute_delta = max(
            maximum_absolute_delta,
            float(
                np.max(
                    np.abs(
                        recreated_ehr.astype(np.float32)
                        - author_ehr.astype(np.float32)
                    )
                )
            ),
        )
        if not np.array_equal(
            recreated_ehr[:, categorical_indices].astype(np.float32),
            author_ehr[:, categorical_indices].astype(np.float32),
        ):
            raise AssertionError("categorical one-hot channels changed")
        if not np.array_equal(recreated_ehr[:, 59:], author_ehr[:, 59:]):
            raise AssertionError("observation-mask channels changed")
        if int(author_label) != int(source_row[label_column]):
            raise AssertionError("author EHR label differs from MEDS label")
        masks = author_ehr[:, 59:]
        if not np.isin(masks, [0.0, 1.0]).all():
            raise AssertionError("released observation mask is not binary")
        if not np.array_equal(
            group["source_row"].to_numpy(np.int64),
            np.arange(len(group), dtype=np.int64),
        ):
            raise AssertionError("MEDS structured source-row order changed")
        expected_sequence = pd.factorize(group["source_time"], sort=True)[0]
        if not np.array_equal(
            group["sequence_row"].to_numpy(np.int32), expected_sequence.astype(np.int32)
        ):
            raise AssertionError("MEDS structured timestamp grouping changed")
        if group["text_value"].notna().any():
            raise AssertionError("structured events unexpectedly contain text")
        total_mask_observed += int(masks.sum())
        recreated_by_stay[stay_id] = recreated_ehr

    actual_columns = [
        "sample_id",
        "time",
        "code",
        "numeric_value",
        "source_row",
        "sequence_row",
        "source_time",
        "text_value",
    ]
    source_cursor = BatchCursor(source_file.iter_batches(batch_size=ARROW_BATCH_SIZE))
    # The source parquet is written in bounded stay ranges, so its row groups
    # are not all EXPORT_BATCH_SIZE long; the writer mirrors each yielded batch
    # as exactly one MEDS row group, so compare row group by row group.
    row_groups = actual_file.metadata.num_row_groups
    seen_rows: dict[int, int] = {}
    last_time: dict[int, np.datetime64] = {}
    last_sequence: dict[int, int] = {}
    pending = None
    for row_group in range(row_groups):
        actual = actual_file.read_row_group(row_group, columns=actual_columns).to_pandas()
        expected_length = len(actual)
        if expected_length > EXPORT_BATCH_SIZE:
            raise AssertionError("MEDS changed a structured export-batch boundary")
        source = source_cursor.take(expected_length).to_pandas()
        source_ids = source["stay_id"].to_numpy(np.int64)
        positions = np.searchsorted(lookup_ids, source_ids)
        if not np.array_equal(lookup_ids[positions], source_ids):
            raise AssertionError("structured event references an unknown cohort stay")
        source_time = (
            source["charttime"].to_numpy(dtype="datetime64[ns]")
            - lookup_intimes[positions]
        ) / np.timedelta64(1, "h")
        expected_code = source["variable"].astype(str).to_numpy(dtype=object)
        categorical = source["categorical_value"].notna().to_numpy()
        expected_code[categorical] = (
            expected_code[categorical]
            + "//"
            + source.loc[categorical, "categorical_value"].astype(str).to_numpy()
        )
        source_rows = np.empty(len(source), dtype=np.int64)
        sequence_rows = np.empty(len(source), dtype=np.int32)
        for stay_id, indices in source.groupby("stay_id", sort=False).indices.items():
            indices = np.asarray(indices, dtype=np.int64)
            offset = seen_rows.get(int(stay_id), 0)
            source_rows[indices] = offset + np.arange(len(indices), dtype=np.int64)
            seen_rows[int(stay_id)] = offset + len(indices)
            times = source.iloc[indices]["charttime"].to_numpy(dtype="datetime64[ns]")
            changes = np.empty(len(times), dtype=np.int32)
            changes[0] = int(
                int(stay_id) not in last_time or times[0] != last_time[int(stay_id)]
            )
            if len(times) > 1:
                changes[1:] = (times[1:] != times[:-1]).astype(np.int32)
            base = last_sequence.get(int(stay_id), -1)
            sequence_rows[indices] = base + np.cumsum(changes, dtype=np.int32)
            last_time[int(stay_id)] = times[-1]
            last_sequence[int(stay_id)] = int(sequence_rows[indices][-1])
        source_like = pd.DataFrame(
            {
                "sample_id": source_ids,
                "time": source["charttime"].to_numpy(dtype="datetime64[us]"),
                "code": expected_code,
                "numeric_value": source["numeric_value"].to_numpy(np.float32),
                "source_row": source_rows,
                "sequence_row": sequence_rows,
                "source_time": source_time.astype(np.float64),
                "text_value": None,
                "subject_id": lookup_subjects[positions],
            }
        )
        # The shared writer sorts each export batch by
        # (sample_id, time, source_row, code).
        expected = source_like.sort_values(
            ["sample_id", "time", "source_row", "code"], kind="stable"
        ).reset_index(drop=True)
        comparisons = (
            ("sample_id", np.int64, False),
            ("time", "datetime64[us]", False),
            ("code", object, False),
            ("numeric_value", np.float32, True),
            ("source_row", np.int64, False),
            ("sequence_row", np.int32, False),
            ("source_time", np.float64, False),
        )
        for field, dtype, equal_nan in comparisons:
            left = actual[field].to_numpy(dtype=dtype)
            right = expected[field].to_numpy(dtype=dtype)
            if not np.array_equal(left, right, equal_nan=equal_nan):
                raise AssertionError(f"MEDS changed structured field/order: {field}")
        if actual["text_value"].notna().any():
            raise AssertionError("structured events unexpectedly contain text")

        grouped = source_like if pending is None else pd.concat([pending, source_like])
        last_stay = int(grouped["sample_id"].iloc[-1])
        complete = grouped.loc[grouped["sample_id"] != last_stay]
        for _, group in complete.groupby("sample_id", sort=False):
            check_loader_group(group)
        pending = grouped.loc[grouped["sample_id"] == last_stay].copy()
    if pending is not None:
        check_loader_group(pending)
    if set(recreated_by_stay) != set(cohort["stay_id"].astype(int)):
        raise AssertionError("MEDS loader recreation did not cover the full cohort")

    for split in ("train", "validation", "test"):
        ordered = cohort.loc[cohort["split"] == split].sort_values("source_name")
        batch = []
        split_batches = 0
        for split_position, source_row in enumerate(ordered.itertuples(index=False)):
            stay_id = int(source_row.stay_id)
            if stay_id not in recreated_by_stay:
                raise AssertionError(f"MEDS omitted all events for stay {stay_id}")
            # The released `partial_ehr_cxr` collate takes (ehr, cxr, y_ehr,
            # y_cxr). The export never retains imaging, so the imaging branch is
            # passed as absent and its placeholder outputs are asserted empty
            # rather than compared.
            batch.append(
                (
                    recreated_by_stay[stay_id],
                    None,
                    int(getattr(source_row, label_column)),
                    None,
                )
            )
            if len(batch) == BATCH_SIZE or split_position + 1 == len(ordered):
                ehr, img, targets_ehr, targets_cxr, lengths, pairs = my_collate(batch)
                if ehr.dtype != np.float64 or ehr.shape[1:] != (24, 76):
                    raise AssertionError("released fusion collate EHR boundary changed")
                if lengths != [24] * len(batch):
                    raise AssertionError("released sequence lengths changed")
                if any(pairs) or bool(img.abs().sum().item()) or bool(
                    targets_cxr.abs().sum().item()
                ):
                    raise AssertionError("the imaging branch was not left empty")
                values = {
                    "ehr_float64": ehr,
                    "ehr_model_float32": ehr.astype(np.float32),
                    "label_ehr": targets_ehr,
                    "sequence_length": np.asarray(lengths, dtype=np.int64),
                    "pair_presence": np.asarray(pairs, dtype=np.bool_),
                }
                for field, value in values.items():
                    update_digest(digests[field], value)
                    array = (
                        value.detach().cpu().numpy()
                        if isinstance(value, torch.Tensor)
                        else np.asarray(value)
                    )
                    shapes[field].add((tuple(array.shape), str(array.dtype)))
                total_examples += len(batch)
                total_batches += 1
                split_batches += 1
                batch = []
        split_report[split] = {
            "examples": len(ordered),
            "diagnostic_batches": split_batches,
            "released_drop_last": split == "train",
            "released_complete_training_batches": (
                len(ordered) // BATCH_SIZE if split == "train" else split_batches
            ),
            "released_training_remainder": (
                len(ordered) % BATCH_SIZE if split == "train" else 0
            ),
        }

    if total_examples != len(cohort) or split_report["train"]["examples"] != EXPECTED_COUNTS[
        "train"
    ]:
        raise AssertionError("batch recreation was not exhaustive")
    report = {
        "status": "exact_source_with_numeric_ulp_audit",
        "source_entrypoint": (
            "dataset_mimic4.ehr_dataset.EHRdataset.__getitem__ -> "
            "dataset_mimic4.fusion.my_collate"
        ),
        "selected_source_configuration": {
            "task": author_task,
            "labels_set": labels_set,
            "data_pairs": "partial_ehr_cxr",
            "data_ratio": 1.0,
            "timestep": 2.0,
            "batch_size": BATCH_SIZE,
        },
        "retained_projection": {
            "modalities": ["structured"],
            "event_codes": codes,
            "note_codes": [],
            "all_non_note_codes_are_intended_structured_inputs": True,
            "excluded": [
                "all imaging pixels/embeddings/timestamps/masks/flags",
                "identifiers and labels as events",
            ],
            "imaging_branch": (
                "invoked as absent; the collate's zero image placeholder and "
                "empty CXR targets are asserted rather than archived"
            ),
        },
        "structured_boundary": {
            "author_dtype": "float64",
            "author_shape": [24, 76],
            "model_dtype": "float32",
            "value_channels": 59,
            "observation_mask_channels": 17,
            "timestep_hours": 2.0,
            "imputation": "released previous observation / normal default",
            "normalization": (
                "released MIMIC-IV reconstruction from train_listfile.csv only; "
                "the released paths.py points both MIMIC-IV tasks at the single "
                "data_mimic4/readm_ts.normalizer file"
            ),
            "maximum_float32_ulp_delta": maximum_ulp,
            "maximum_absolute_float32_delta": maximum_absolute_delta,
        },
        "diagnostic_batch_order": (
            "fixed exhaustive traversal in released source_name order. The "
            "released partial_ehr_cxr dataset orders CXR-paired stays first, but "
            "that ordering is derived from the imaging modality the export must not "
            "retain; training order is stochastic (shuffle=True) and immaterial, "
            "so only batch membership is frozen and every example is compared "
            "through the released collator."
        ),
        "splits": split_report,
        "examples": total_examples,
        "diagnostic_batches": total_batches,
        "observed_mask_sum": total_mask_observed,
        "batch_shapes_and_dtypes": {
            field: [
                {"shape": list(shape), "dtype": dtype} for shape, dtype in sorted(values)
            ]
            for field, values in shapes.items()
        },
        "ordered_field_sha256": {
            field: digest.hexdigest() for field, digest in digests.items()
        },
        "source_reconstruction": source_manifest,
        "manifest_sha256": manifest_hash,
        "checks": [
            "all 25,071 author EHR examples reconstructed through the released "
            "MIMIC-IV list files",
            "raw numeric events are exact after required MEDS float32 serialization",
            "numeric model-boundary deltas are reported in float32 ULPs without "
            "decimal tolerance",
            "categorical one-hot and observation-mask channels are bit-exact",
            "released fusion collate invoked for every diagnostic batch",
            "labels, dtypes, shapes, sequence lengths and masks exact",
            "exact 59-code event vocabulary with no note or imaging exception",
            "no imaging branch persisted",
        ],
    }
    output = args.task_dir / "audit" / "full_batch_recreation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if sha256(manifest_path) != manifest_hash:
        raise AssertionError("batch checker mutated immutable manifest")
    print(json.dumps({"status": report["status"], "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
