"""Assert that MEDS losslessly recreates every released FuseMoE batch."""

from __future__ import print_function

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from preprocessing.data_mimiciv import (  # noqa: E402
    F_impute,
    TSNote_Irg,
    TextTSIrgcollate_fn,
)
from export_source import (  # noqa: E402
    COHORT_ID_SHA256,
    NOTE_CODES,
    VARIABLE_NAMES,
    load_source,
    loader_args,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def arrow_lookup(table):
    sample = table["sample_id"].to_numpy()
    order = np.argsort(sample, kind="stable")
    table = table.take(pa.array(order, type=pa.int64()))
    sample = sample[order]
    boundaries = np.flatnonzero(np.r_[True, sample[1:] != sample[:-1], True])
    lookup = {
        int(sample[start]): (int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    }
    return table, lookup


RETAINED_BATCH_FIELDS = {
    0: "ts",
    1: "ts_mask",
    2: "ts_tt",
    3: "reg_ts",
    4: "input_ids",
    5: "attention_mask",
    15: "label",
}


def compare_batch(source, recreated, digests, field_shapes):
    if source is None or recreated is None:
        if source is not recreated:
            raise AssertionError("MEDS changed a None batch field")
        return
    if len(source) != 19 or len(recreated) != 19:
        raise AssertionError("unexpected FuseMoE collate arity")
    for field, name in RETAINED_BATCH_FIELDS.items():
        left = source[field]
        right = recreated[field]
        if left is None or right is None:
            if left is not right:
                raise AssertionError(f"retained field {name} changed None status")
            digests[name].update(b"NONE")
            continue
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            raise AssertionError(
                f"retained field {name} shape/dtype mismatch: "
                f"{tuple(left.shape)} {left.dtype} != {tuple(right.shape)} {right.dtype}"
            )
        if not torch.equal(left, right):
            maximum = (
                float(torch.max(torch.abs(left - right)).item())
                if left.is_floating_point()
                else None
            )
            raise AssertionError(f"retained field {name} differs; maximum={maximum}")
        raw = left.detach().cpu().contiguous().numpy()
        digests[name].update(str(raw.dtype).encode())
        digests[name].update(np.asarray(raw.shape, dtype=np.int64).tobytes())
        digests[name].update(raw.tobytes())
        field_shapes[name].add((tuple(raw.shape), str(raw.dtype)))


def summarize_shapes(values):
    grouped = {}
    for shape, dtype in values:
        grouped.setdefault((dtype, len(shape)), []).append(shape)
    return [
        {
            "dtype": dtype,
            "rank": rank,
            "unique_shapes": len(shapes),
            "minimum_shape": [min(shape[axis] for shape in shapes) for axis in range(rank)],
            "maximum_shape": [max(shape[axis] for shape in shapes) for axis in range(rank)],
        }
        for (dtype, rank), shapes in sorted(grouped.items())
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="yikuan8/Clinical-Longformer")
    args = parser.parse_args()

    manifest_path = args.task_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["task"].get("modalities") != ["structured", "notes"]:
        raise AssertionError("FuseMoE task must declare structured data and notes")
    if manifest["task"].get("note_codes") != list(NOTE_CODES):
        raise AssertionError("FuseMoE task declares the wrong note codes")
    source_metadata = manifest.get("source", {})
    if source_metadata.get("selected_model_inputs") != ["structured", "notes"]:
        raise AssertionError("FuseMoE selected the wrong source experiment")
    if source_metadata.get("selection_tier") != 2:
        raise AssertionError("FuseMoE selected the wrong modality-priority tier")
    if source_metadata.get("batch_recreation", {}).get("modeltype") != "TS_Text":
        raise AssertionError("FuseMoE did not use the vital-signs + notes loader")
    event_path = args.task_dir / manifest["meds"]["events"]["file"]
    index_path = args.task_dir / manifest["meds"]["task_index"]["file"]

    events = pq.read_table(event_path)
    note_selected = pc.is_in(
        events["code"], value_set=pa.array(NOTE_CODES)
    ).to_numpy(zero_copy_only=False)
    structured_selected = ~note_selected
    declared = pc.is_in(
        events["code"], value_set=pa.array(VARIABLE_NAMES + NOTE_CODES)
    ).to_numpy(zero_copy_only=False)
    if not np.all(declared):
        raise AssertionError("MEDS contains an undeclared event code")
    actual_structured_codes = set(
        pc.unique(events.filter(pa.array(structured_selected))["code"]).to_pylist()
    )
    actual_note_codes = set(
        pc.unique(events.filter(pa.array(note_selected))["code"]).to_pylist()
    )
    if actual_structured_codes != set(VARIABLE_NAMES):
        raise AssertionError("MEDS structured code set differs from the author loader")
    if actual_note_codes != set(NOTE_CODES):
        raise AssertionError("MEDS note code set is incomplete or unexpected")
    structured = events.filter(pa.array(structured_selected)).select(
        ["sample_id", "code", "numeric_value", "sequence_row", "source_time", "source_row"]
    )
    notes = events.filter(pa.array(note_selected)).select(
        ["sample_id", "code", "text_value", "source_row"]
    )
    structured, structured_lookup = arrow_lookup(structured)
    notes, note_lookup = arrow_lookup(notes)

    index = pq.read_table(index_path)
    if any(
        row["split"] != row["source_split"] for row in index.to_pylist()
    ):
        raise AssertionError("MEDS changed a released FuseMoE split assignment")
    index_rows = {int(row["sample_id"]): row for row in index.to_pylist()}
    cohort_ids = np.sort(index["sample_id"].to_numpy().astype(np.int64))
    cohort_hash = hashlib.sha256(cohort_ids.tobytes()).hexdigest()
    if len(cohort_ids) != 8770 or cohort_hash != COHORT_ID_SHA256:
        raise AssertionError("MEDS does not contain the paper's 8,770-stay cohort")
    source_order = {
        (row["source_split"], int(row["source_position"])): int(row["sample_id"])
        for row in index.to_pylist()
    }
    variable_index = {name: position for position, name in enumerate(VARIABLE_NAMES)}

    def recreate(sample_id, source_item, source_stay):
        start, end = structured_lookup[sample_id]
        part = structured.slice(start, end - start)
        rows = part["sequence_row"].to_numpy()
        length = int(rows.max()) + 1
        values = np.zeros((length, len(VARIABLE_NAMES)), dtype=np.float32)
        mask = np.zeros((length, len(VARIABLE_NAMES)), dtype=np.int64)
        times = np.full(length, np.nan, dtype=np.float64)
        for sequence_row, code, value, source_time in zip(
            rows,
            part["code"].to_pylist(),
            part["numeric_value"].to_numpy(),
            part["source_time"].to_numpy(),
        ):
            column = variable_index[code]
            values[sequence_row, column] = value
            mask[sequence_row, column] = 1
            if np.isnan(times[sequence_row]):
                times[sequence_row] = source_time
            elif times[sequence_row] != source_time:
                raise AssertionError("one irregular row has inconsistent source times")
        if not np.isfinite(times).all():
            raise AssertionError("MEDS lost an irregular row")

        note_start, note_end = note_lookup[sample_id]
        note_part = notes.slice(note_start, note_end - note_start).sort_by(
            [("source_row", "ascending")]
        )
        note_rows = note_part.to_pylist()
        text = [row for row in note_rows if row["code"] == NOTE_CODES[0]]
        raw_text = [row["text_value"] for row in text]
        token_ids = []
        attention_masks = []
        for value in raw_text:
            encoded = tokenizer.encode_plus(
                value,
                padding=False,
                max_length=1024,
                add_special_tokens=True,
                return_attention_mask=True,
                truncation=True,
            )
            attention = encoded["attention_mask"]
            attention[0] += 1
            token_ids.append(torch.tensor(encoded["input_ids"], dtype=torch.long))
            attention_masks.append(torch.tensor(attention, dtype=torch.long))
        while len(token_ids) < 5:
            token_ids.append(torch.tensor([0], dtype=torch.long))
            attention_masks.append(torch.tensor([0], dtype=torch.long))
        if int(source_stay["text_missing"]):
            token_ids = [torch.zeros(100) for _ in raw_text]
            attention_masks = [torch.zeros(100) for _ in raw_text]
        record = index_rows[sample_id]
        reg_ts = F_impute(values, times, mask, 1, 48)
        recreated = {
            "idx": int(record["subject_id"]),
            "ts": torch.tensor(values, dtype=torch.float),
            "ts_mask": torch.tensor(mask, dtype=torch.long),
            "ts_tt": torch.tensor(times / 48.0, dtype=torch.float),
            "reg_ts": torch.tensor(reg_ts, dtype=torch.float),
            "input_ids": token_ids,
            "attention_mask": attention_masks,
            # These source-only fields are required by the author's collator but
            # are intentionally not persisted in the shared raw-note contract.
            "text_embeddings": source_item["text_embeddings"],
            "note_time": source_item["note_time"],
            "text_time_mask": source_item["text_time_mask"],
            "label": torch.tensor(int(record["boolean_value"]), dtype=torch.long),
        }
        return recreated, raw_text

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    digests = {name: hashlib.sha256() for name in RETAINED_BATCH_FIELDS.values()}
    field_shapes = {name: set() for name in RETAINED_BATCH_FIELDS.values()}
    labels = hashlib.sha256()
    batches = 0
    examples = 0
    filtered = 0
    split_data, _ = load_source(args.source_root)
    for (source_split, stays), batch_size in zip(split_data, (1, 8, 8)):
        mode = "train" if source_split == "train" else source_split
        dataset = TSNote_Irg(loader_args(), mode, tokenizer, data=stays)
        source_batch = []
        recreated_batch = []
        for position in range(len(dataset)):
            item = dataset[position]
            sample_id = source_order.get((source_split, position))
            if len(item["ts"]) >= 1000:
                if sample_id is not None:
                    raise AssertionError("a collate-filtered row entered MEDS")
                filtered += 1
                continue
            if sample_id is None:
                raise AssertionError("MEDS omitted a retained source row")
            recreated_item, raw_text_slots = recreate(
                sample_id, item, stays[position]
            )
            if int(stays[position]["text_missing"]) != int(
                index_rows[sample_id]["text_missing"]
            ):
                raise AssertionError("MEDS changed the released text-missing flag")
            source_texts = [str(value) for value in stays[position].get("text_data", [])]
            if int(stays[position]["text_missing"]):
                expected_text_slots = []
            else:
                expected_text_slots = source_texts[-5:]
            if raw_text_slots != expected_text_slots:
                raise AssertionError("MEDS changed an author-selected raw note slot")
            source_batch.append(item)
            recreated_batch.append(recreated_item)
            labels.update(np.asarray([int(item["label"])], dtype=np.int64).tobytes())
            examples += 1
            if len(source_batch) == batch_size:
                compare_batch(
                    TextTSIrgcollate_fn(source_batch),
                    TextTSIrgcollate_fn(recreated_batch),
                    digests,
                    field_shapes,
                )
                source_batch = []
                recreated_batch = []
                batches += 1
        if source_batch:
            compare_batch(
                TextTSIrgcollate_fn(source_batch),
                TextTSIrgcollate_fn(recreated_batch),
                digests,
                field_shapes,
            )
            batches += 1

    report = {
        "status": "exact",
        "comparison": "all persisted structured fields plus raw-note-derived token fields from reconstructed paper-cohort TS_Text batches",
        "examples": examples,
        "batches": batches,
        "collate_filtered_examples": filtered,
        "source_collate_arity": 19,
        "compared_collate_fields": RETAINED_BATCH_FIELDS,
        "retained_modalities": ["structured", "notes"],
        "imaging_retained": False,
        "split_assignment": "paper producer seed-0 70/15/15 stay split",
        "cohort_id_sha256": cohort_hash,
        "cross_split_subject_overlap": manifest[
            "cross_split_subject_overlap"
        ],
        "ordered_label_sha256": labels.hexdigest(),
        "field_sha256": {
            name: digest.hexdigest() for name, digest in digests.items()
        },
        "field_shape_summary": {
            name: summarize_shapes(values) for name, values in field_shapes.items()
        },
        "author_dataset": "src/preprocessing/data_mimiciv.py::TSNote_Irg",
        "author_collate": "src/preprocessing/data_mimiciv.py::TextTSIrgcollate_fn",
        "tokenizer": args.tokenizer,
        "structured_feature_audit": {
            "status": "exact",
            "included_event_count": int(structured_selected.sum()),
            "included_codes": sorted(actual_structured_codes),
            "excluded_event_count": int(note_selected.sum()),
            "excluded_codes": sorted(actual_note_codes),
            "rule": (
                "every event whose code is not in task.note_codes is structured"
            ),
        },
        "raw_note_audit": {
            "status": "exact",
            "documents": examples,
            "note_rows": len(notes),
            "rule": "every actual author-selected MEDS note row concatenated in source order; loader padding regenerated transiently",
        },
    }
    report_path = args.task_dir / "audit" / "full_batch_recreation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
