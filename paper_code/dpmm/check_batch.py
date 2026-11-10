"""Exhaustively recreate DPMM's retained loader and collate boundary."""

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
from transformers import AutoTokenizer


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from dataset_mimic3.fusion import my_collate  # noqa: E402
from cohort_source import (  # noqa: E402
    DISCRETIZER_CONFIG,
    HOURS,
    NOTE_CODE,
    TASK_ID,
    TOKENIZER,
    author_datasets,
    ensure_source,
    make_discretizer_normalizer,
    sha256,
    source_codes,
)


BATCH_SIZE = 16
ARROW_BATCH_SIZE = 100_000
EXPORT_BATCH_SIZE = 250_000


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
            pieces.append(pa.Table.from_batches([self._batch.slice(self._offset, length)]))
            self._offset += length
            remaining -= length
        return pa.concat_tables(pieces) if len(pieces) > 1 else pieces[0]

    def remainder(self):
        if self._batch is not None and self._offset < len(self._batch):
            yield self._batch.slice(self._offset)
        yield from self._batches


def update_digest(digest: "hashlib._Hash", values) -> None:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.ascontiguousarray(values)
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())


def meds_to_author_array(
    frame: pd.DataFrame,
    discretizer,
    normalizer,
    source_header: list[str],
) -> np.ndarray:
    frame = frame.loc[frame["code"] != NOTE_CODE].sort_values(
        ["source_time", "source_row", "code"], kind="stable"
    )
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
    data, _ = discretizer.transform(
        values, header=["Hours", *source_header], end=HOURS
    )
    return normalizer.transform(data)


def encode_notes(tokenizer, values: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = []
    masks = []
    for value in values:
        encoded = tokenizer.encode_plus(
            value,
            padding="max_length",
            max_length=512,
            add_special_tokens=True,
            return_attention_mask=True,
            truncation=True,
        )
        tokens.append(encoded["input_ids"])
        masks.append(encoded["attention_mask"])
    return (
        torch.tensor(tokens, dtype=torch.int64),
        torch.tensor(masks, dtype=torch.int64),
    )


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
    # Ordered integer encoding for signed IEEE-754 values.
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
    source_notes = pd.read_parquet(
        args.cache_root / source_manifest["files"]["selected_notes"]
    )
    manifest_path = args.task_dir / "manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["task"]["id"] != TASK_ID:
        raise AssertionError("wrong DPMM task")
    if manifest["task"]["modalities"] != ["structured", "notes"]:
        raise AssertionError("DPMM retained modality declaration changed")
    if manifest["task"]["note_codes"] != [NOTE_CODE]:
        raise AssertionError("DPMM note code changed")

    codes = pq.read_table(
        args.task_dir / manifest["meds"]["codes"]["file"]
    )["code"].to_pylist()
    if codes != list(source_codes()):
        raise AssertionError("event vocabulary differs from released channels")
    if any("image" in code.lower() or "cxr" in code.lower() for code in codes):
        raise AssertionError("imaging entered the retained projection")

    event_path = args.task_dir / manifest["meds"]["events"]["file"]
    datasets = author_datasets(args.cache_root)
    dataset_by_split_name = {
        split: dataset for split, dataset in datasets.items()
    }
    discretizer, normalizer, output_header = make_discretizer_normalizer(
        args.cache_root
    )
    source_header = json.loads(DISCRETIZER_CONFIG.read_text())["id_to_channel"]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)

    digest_names = (
        "ehr_float64",
        "ehr_model_float32",
        "token_ids",
        "attention_mask",
        "label_ehr",
        "label_note",
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
    if actual_file.metadata.num_rows != source_event_count + len(source_notes):
        raise AssertionError("MEDS event count differs from structured plus note source")

    ordered_lookup = cohort.sort_values("stay_id")
    lookup_ids = ordered_lookup["stay_id"].to_numpy(np.int64)
    lookup_intimes = pd.to_datetime(ordered_lookup["INTIME"]).to_numpy(
        dtype="datetime64[ns]"
    )
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
        author = dataset_by_split_name[str(source_row["split"])]
        author_ehr, author_label = author[str(source_row["source_name"])]
        recreated_ehr = meds_to_author_array(
            group, discretizer, normalizer, source_header
        )
        if author_ehr.shape != (24, 76) or author_ehr.dtype != np.float64:
            raise AssertionError("released EHR loader boundary changed")
        if recreated_ehr.shape != author_ehr.shape:
            raise AssertionError("MEDS transform changed EHR tensor shape")
        ulp = max_float32_ulp(recreated_ehr, author_ehr)
        maximum_ulp = max(maximum_ulp, ulp)
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
        if int(author_label) != int(source_row["label"]):
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
            group["sequence_row"].to_numpy(np.int32),
            expected_sequence.astype(np.int32),
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
    source_cursor = BatchCursor(
        source_file.iter_batches(batch_size=ARROW_BATCH_SIZE)
    )
    structured_row_groups = (source_event_count + EXPORT_BATCH_SIZE - 1) // EXPORT_BATCH_SIZE
    if actual_file.metadata.num_row_groups <= structured_row_groups:
        raise AssertionError("MEDS did not retain a separate note-event suffix")
    seen_rows: dict[int, int] = {}
    last_time: dict[int, np.datetime64] = {}
    last_sequence: dict[int, int] = {}
    pending = None
    for row_group in range(structured_row_groups):
        expected_length = min(
            EXPORT_BATCH_SIZE,
            source_event_count - row_group * EXPORT_BATCH_SIZE,
        )
        source = source_cursor.take(expected_length).to_pandas()
        actual = actual_file.read_row_group(
            row_group, columns=actual_columns
        ).to_pandas()
        if len(actual) != expected_length:
            raise AssertionError("MEDS changed a structured export-batch boundary")
        if (actual["code"] == NOTE_CODE).any():
            raise AssertionError("note row entered structured MEDS prefix")
        source_ids = source["stay_id"].to_numpy(np.int64)
        positions = np.searchsorted(lookup_ids, source_ids)
        if not np.array_equal(lookup_ids[positions], source_ids):
            raise AssertionError("structured event references an unknown cohort stay")
        source_time = (
            source["CHARTTIME"].to_numpy(dtype="datetime64[ns]")
            - lookup_intimes[positions]
        ) / np.timedelta64(1, "h")
        expected_code = source["VARIABLE"].astype(str).to_numpy(dtype=object)
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
            times = source.iloc[indices]["CHARTTIME"].to_numpy(dtype="datetime64[ns]")
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
                "time": source["CHARTTIME"].to_numpy(dtype="datetime64[us]"),
                "code": expected_code,
                "numeric_value": source["numeric_value"].to_numpy(np.float32),
                "source_row": source_rows,
                "sequence_row": sequence_rows,
                "source_time": source_time.astype(np.float64),
                "text_value": None,
                "subject_id": lookup_ids[positions] * 0
                + ordered_lookup["SUBJECT_ID"].to_numpy(np.int64)[positions],
            }
        )
        expected = source_like.sort_values(
            ["subject_id", "time", "source_row", "code"], kind="stable"
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

    note_tables = [
        actual_file.read_row_group(index, columns=actual_columns)
        for index in range(structured_row_groups, actual_file.metadata.num_row_groups)
    ]
    actual_notes = (
        pa.concat_tables(note_tables).to_pandas()
        if note_tables
        else pd.DataFrame(columns=actual_columns)
    )
    if len(actual_notes) != len(source_notes):
        raise AssertionError("MEDS changed the retained note count")
    if len(actual_notes) and not (actual_notes["code"] == NOTE_CODE).all():
        raise AssertionError("non-note row entered the MEDS note suffix")
    note_events_by_stay = {
        int(key): value.sort_values("source_row")
        for key, value in actual_notes.groupby("sample_id", sort=False)
    }
    source_note_groups = {
        int(key): value.sort_values("slot")
        for key, value in source_notes.groupby("stay_id")
    }
    for stay_id, source_group in source_note_groups.items():
        actual_group = note_events_by_stay.get(stay_id)
        if actual_group is None or len(actual_group) != 5:
            raise AssertionError("paired DPMM sample does not have five notes")
        if actual_group["text_value"].tolist() != source_group["text"].tolist():
            raise AssertionError("MEDS changed released note slot text/order")
        source_row = row_by_stay.loc[stay_id]
        intime = pd.Timestamp(source_row["INTIME"])
        source_time = (
            pd.to_datetime(source_group["CHARTTIME"]) - intime
        ).dt.total_seconds().to_numpy(np.float64) / 3600.0
        if not np.array_equal(actual_group["source_time"].to_numpy(), source_time):
            raise AssertionError("MEDS changed exact note source times")
        expected_time = pd.to_datetime(source_group["CHARTTIME"]).clip(
            lower=intime, upper=intime + pd.Timedelta(hours=HOURS)
        )
        if not np.array_equal(
            actual_group["time"].to_numpy(dtype="datetime64[us]"),
            expected_time.to_numpy(dtype="datetime64[us]"),
        ):
            raise AssertionError("MEDS changed serialized note ordering times")
        slots = source_group["slot"].to_numpy(np.int32)
        if not np.array_equal(
            actual_group["source_row"].to_numpy(np.int64),
            1_000_000_000 + slots.astype(np.int64),
        ) or not np.array_equal(
            actual_group["sequence_row"].to_numpy(np.int32), slots
        ):
            raise AssertionError("MEDS changed note slot indices")

    for split in ("train", "validation", "test"):
        ordered = cohort.loc[cohort["split"] == split].sort_values("fusion_position")
        batch = []
        split_batches = 0
        split_paired = 0
        for split_position, source_row in enumerate(ordered.itertuples(index=False)):
            stay_id = int(source_row.stay_id)
            if stay_id not in recreated_by_stay:
                raise AssertionError(f"MEDS omitted all events for stay {stay_id}")
            recreated_ehr = recreated_by_stay[stay_id]
            note_events = note_events_by_stay.get(stay_id)
            if bool(source_row.note_present):
                split_paired += 1
                if note_events is None or len(note_events) != 5:
                    raise AssertionError("paired DPMM sample does not have five notes")
                actual_text = note_events["text_value"].tolist()
                token, mask = encode_notes(tokenizer, actual_text)
                note_label = torch.tensor(float(source_row.label), dtype=torch.float32)
            else:
                if note_events is not None:
                    raise AssertionError("unpaired sample unexpectedly has a note row")
                token = mask = note_label = None
            batch.append(
                (recreated_ehr, token, mask, int(source_row.label), note_label)
            )

            if len(batch) == BATCH_SIZE or split_position + 1 == len(ordered):
                collated = my_collate(batch)
                ehr, token_ids, attention, label_ehr, label_note, lengths, pairs = collated
                if ehr.dtype != np.float64 or ehr.shape[1:] != (24, 76):
                    raise AssertionError("released fusion collate EHR boundary changed")
                if token_ids.dtype != torch.int64 or token_ids.shape[1:] != (5, 512):
                    raise AssertionError("released TinyBERT token boundary changed")
                if attention.dtype != torch.int64 or attention.shape != token_ids.shape:
                    raise AssertionError("released attention-mask boundary changed")
                if lengths != [24] * len(batch):
                    raise AssertionError("released sequence lengths changed")
                values = {
                    "ehr_float64": ehr,
                    "ehr_model_float32": ehr.astype(np.float32),
                    "token_ids": token_ids,
                    "attention_mask": attention,
                    "label_ehr": label_ehr,
                    "label_note": label_note,
                    "sequence_length": np.asarray(lengths, dtype=np.int64),
                    "pair_presence": np.asarray(pairs, dtype=np.bool_),
                }
                for field, value in values.items():
                    update_digest(digests[field], value)
                    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                    shapes[field].add((tuple(array.shape), str(array.dtype)))
                total_examples += len(batch)
                total_batches += 1
                split_batches += 1
                batch = []
        split_report[split] = {
            "examples": len(ordered),
            "paired": split_paired,
            "diagnostic_batches": split_batches,
            "released_drop_last": split == "train",
            "released_complete_training_batches": (
                len(ordered) // BATCH_SIZE if split == "train" else split_batches
            ),
            "released_training_remainder": (
                len(ordered) % BATCH_SIZE if split == "train" else 0
            ),
        }

    if total_examples != len(cohort):
        raise AssertionError("batch recreation was not exhaustive")
    report = {
        "status": "exact_source_with_numeric_ulp_audit",
        "source_entrypoint": (
            "dataset_mimic3.ehr_dataset.EHRdataset.__getitem__ -> "
            "dataset_mimic3.fusion.my_collate"
        ),
        "selected_source_configuration": {
            "task": "in-hospital-mortality",
            "data_pairs": "partial_ehr_note",
            "data_ratio": 1.0,
            "timestep": 2.0,
            "batch_size": BATCH_SIZE,
            "bert_type": "huawei-noah/TinyBERT_General_4L_312D",
        },
        "retained_projection": {
            "modalities": ["structured", "notes"],
            "event_codes": codes,
            "note_codes": [NOTE_CODE],
            "all_non_note_codes_are_intended_structured_inputs": True,
            "excluded": [
                "all imaging pixels/embeddings/timestamps/masks/flags",
                "token IDs and attention masks from persistent MEDS",
                "identifiers and labels as events",
            ],
        },
        "structured_boundary": {
            "author_dtype": "float64",
            "author_shape": [24, 76],
            "model_dtype": "float32",
            "value_channels": 59,
            "observation_mask_channels": 17,
            "timestep_hours": 2.0,
            "imputation": "released previous observation / normal default",
            "normalization": "released reconstruction including validation moments",
            "maximum_float32_ulp_delta": maximum_ulp,
            "maximum_absolute_float32_delta": maximum_absolute_delta,
        },
        "note_boundary": {
            "persistent_field": "five released cleaned raw note strings when present",
            "transient_token_shape": [5, 512],
            "token_dtype": "torch.int64",
            "attention_dtype": "torch.int64",
            "missing_branch": "zero token/mask tensors plus pairs=False",
        },
        "splits": split_report,
        "examples": total_examples,
        "diagnostic_batches": total_batches,
        "observed_mask_sum": total_mask_observed,
        "batch_shapes_and_dtypes": {
            field: [
                {"shape": list(shape), "dtype": dtype}
                for shape, dtype in sorted(values)
            ]
            for field, values in shapes.items()
        },
        "ordered_field_sha256": {
            field: digest.hexdigest() for field, digest in digests.items()
        },
        "manifest_sha256": manifest_hash,
        "checks": [
            "all 21,139 author EHR examples reconstructed",
            "raw numeric events are exact after required MEDS float32 serialization",
            "numeric model-boundary deltas are reported in float32 ULPs without decimal tolerance",
            "categorical one-hot and observation-mask channels are bit-exact",
            "all 26,365 paired note slots compared in exact source order",
            "TinyBERT token IDs and masks recreated transiently",
            "released fusion collate invoked for every diagnostic batch",
            "labels, dtypes, shapes, sequence lengths, masks, and presence exact",
            "exact event vocabulary with NOTE as the only exception code",
            "no imaging or auxiliary branch persisted",
        ],
    }
    output = args.task_dir / "audit" / "full_batch_recreation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if sha256(manifest_path) != manifest_hash:
        raise AssertionError("batch checker mutated immutable manifest")
    print(json.dumps({"status": report["status"], "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
