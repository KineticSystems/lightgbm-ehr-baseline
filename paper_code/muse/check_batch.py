"""Exhaustively recreate MUSE's retained author Dataset/collate boundary."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

import src.dataset.mimic4_dataset as author_dataset_module  # noqa: E402
import src.dataset.tokenizer as author_tokenizer_module  # noqa: E402
from export_source import _lookup_frames, event_batches  # noqa: E402
from src.dataset.data import MIMIC4Data  # noqa: E402
from src.dataset.utils import mimic4_collate_fn  # noqa: E402
from src.dataset.vocab import Vocabulary  # noqa: E402
from cohort_source import (  # noqa: E402
    AUTHOR_BATCH_SIZE,
    AUTHOR_MISSING_LAB_WIDTH,
    CODE_FILE,
    LAB_FILE,
    NOTE_CODE,
    NOTE_FILE,
    ParquetGroupCursor,
    TASK_ID,
    author_vocabulary,
    ensure_source,
    lab_channels,
    reconstruct_labvectors,
    sha256_file,
    source_codes,
)


EVENT_COLUMNS = [
    "subject_id",
    "time",
    "code",
    "numeric_value",
    "sample_id",
    "source_row",
    "sequence_row",
    "source_time",
    "text_value",
]


class EventMultiset:
    """Two independent order-free pandas hashes plus exact row count."""

    def __init__(self) -> None:
        self.count = 0
        self.sum1 = 0
        self.sum2 = 0
        self.xor1 = np.uint64(0)
        self.xor2 = np.uint64(0)

    def update(self, table: Any) -> None:
        if isinstance(table, pl.DataFrame):
            frame = table.select(EVENT_COLUMNS).to_pandas()
        elif isinstance(table, pa.RecordBatch):
            frame = pa.Table.from_batches([table]).select(EVENT_COLUMNS).to_pandas()
        elif isinstance(table, pa.Table):
            frame = table.select(EVENT_COLUMNS).to_pandas()
        else:
            raise TypeError(type(table))
        if frame.empty:
            return
        # Normalize extension/string choices so Arrow string/large_string are
        # equivalent, exactly as in the shared MEDS reader.
        frame["time"] = pd.to_datetime(frame["time"]).astype("int64")
        for name in ("code", "text_value"):
            frame[name] = frame[name].astype(object)
        first = pd.util.hash_pandas_object(
            frame, index=False, hash_key="0123456789abcdef"
        ).to_numpy(np.uint64)
        second = pd.util.hash_pandas_object(
            frame, index=False, hash_key="fedcba9876543210"
        ).to_numpy(np.uint64)
        self.count += len(frame)
        self.sum1 = (self.sum1 + int(first.sum(dtype=np.uint64))) % (1 << 64)
        self.sum2 = (self.sum2 + int(second.sum(dtype=np.uint64))) % (1 << 64)
        self.xor1 = np.bitwise_xor.reduce(first, initial=self.xor1)
        self.xor2 = np.bitwise_xor.reduce(second, initial=self.xor2)

    def result(self) -> dict[str, Any]:
        return {
            "rows": self.count,
            "sum_hash_1": f"{self.sum1:016x}",
            "sum_hash_2": f"{self.sum2:016x}",
            "xor_hash_1": f"{int(self.xor1):016x}",
            "xor_hash_2": f"{int(self.xor2):016x}",
        }


def update_tensor_digest(digest: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())


def _author_code_vocabulary(cache_root: Path) -> Vocabulary:
    vocabulary = Vocabulary()
    for value in author_vocabulary(cache_root):
        vocabulary.add_word(value)
    return vocabulary


def _patch_author_loader(
    objects: dict[str, MIMIC4Data], ids: list[str], vocabulary: Vocabulary
) -> None:
    author_dataset_module.load_pickle = lambda _: objects
    author_dataset_module.read_txt = lambda _: list(ids)
    author_tokenizer_module.load_pickle = lambda _: vocabulary
    # Embeddings are loaded by the Dataset constructor but are not returned or
    # consumed at this deterministic boundary. Avoid inventing a model input.
    author_tokenizer_module.read = lambda _: np.zeros(
        (len(vocabulary), 1), dtype=np.float32
    )


def _make_author_object(
    row: Any,
    code_rows: pd.DataFrame,
    lab_rows: pd.DataFrame,
    note_rows: pd.DataFrame,
    lab_width: int,
) -> MIMIC4Data:
    # The released MIMIC4Data carries both task labels; MIMIC4Dataset selects
    # one with getattr(hosp_adm, task). The readmission attribute is populated
    # only when the caller's cohort frame supplies it.
    readmission = getattr(row, "readmission", None)
    value = MIMIC4Data(
        admission_id=str(int(row.hadm_id)),
        patient_id=str(int(row.subject_id)),
        duration=float(row.duration_minutes),
        mortality=None if pd.isna(row.mortality) else bool(row.mortality),
        readmission=None if readmission is None or pd.isna(readmission) else bool(readmission),
        age=int(row.age),
        gender=str(row.gender),
        ethnicity=str(row.race),
    )
    if code_rows.empty:
        value.trajectory = ([], [])
    else:
        value.trajectory = (
            code_rows["event_type"].astype(str).tolist(),
            code_rows["code_name"].astype(str).tolist(),
        )
    value.labvectors = reconstruct_labvectors(lab_rows, lab_width)
    if note_rows.empty:
        value.discharge = None
    else:
        if len(note_rows) != 1:
            raise AssertionError("released last-write note boundary is not one-to-one")
        value.discharge = str(note_rows["text"].iloc[0])
    return value


def _expected_indices(dataset: Any, value: MIMIC4Data) -> dict[str, Any]:
    tokenizer = dataset.tokenizer
    types, codes = value.trajectory
    return {
        "age": tokenizer.age_vocabs(str(value.age // 10 * 10)),
        "gender": tokenizer.gender_vocabs(value.gender),
        "ethnicity": tokenizer.ethnicity_vocabs(value.ethnicity),
        "types": [tokenizer.type_vocabs("<cls>")]
        + [tokenizer.type_vocabs(item) for item in types]
        + [tokenizer.type_vocabs("<eos>")],
        "codes": [tokenizer.code_vocabs("<cls>")]
        + [tokenizer.code_vocabs(item) for item in codes]
        + [tokenizer.code_vocabs("<eos>")],
    }


def _check_item(
    dataset: Any,
    index: int,
    value: MIMIC4Data,
    *,
    check_singleton_collate: bool = True,
    label_attr: str = "mortality",
) -> dict[str, Any]:
    item = dataset[index]
    expected = _expected_indices(dataset, value)
    if item["id"] != value.admission_id:
        raise AssertionError("author Dataset changed admission ID")
    for name in ("age", "gender", "ethnicity"):
        if item[name].dtype != torch.int64 or int(item[name]) != expected[name]:
            raise AssertionError(f"author Dataset changed {name}")
    for name in ("types", "codes"):
        target = torch.tensor(expected[name], dtype=torch.int64)
        if item[name].dtype != torch.int64 or not torch.equal(item[name], target):
            raise AssertionError(f"author Dataset changed {name}")
    if not bool(item["codes_flag"]):
        raise AssertionError("released codes_flag is not constant true")
    if value.labvectors is None:
        expected_lab = torch.zeros(1, AUTHOR_MISSING_LAB_WIDTH)
        expected_lab_flag = False
    else:
        expected_lab = torch.tensor(value.labvectors, dtype=torch.float32)
        expected_lab_flag = True
    if item["labvectors"].dtype != torch.float32 or not torch.equal(
        item["labvectors"], expected_lab
    ):
        raise AssertionError("author Dataset changed exact labvectors")
    if bool(item["labvectors_flag"]) != expected_lab_flag:
        raise AssertionError("author Dataset changed lab modality membership")
    expected_discharge = value.discharge if value.discharge is not None else ""
    if item["discharge"] != expected_discharge:
        raise AssertionError("author Dataset changed raw discharge text")
    if bool(item["discharge_flag"]) != (value.discharge is not None):
        raise AssertionError("author Dataset changed note modality membership")
    if item["label"].dtype != torch.float32 or float(item["label"]) != float(
        getattr(value, label_attr)
    ):
        raise AssertionError(f"author Dataset changed {label_attr} label")
    if not bool(item["label_flag"]):
        raise AssertionError("labeled MUSE example lost label_flag")
    if check_singleton_collate:
        singleton = mimic4_collate_fn([item])
        if singleton["types"].shape != (1, len(expected["types"])):
            raise AssertionError("singleton code padding changed")
        if singleton["labvectors"].shape != (1, *expected_lab.shape):
            raise AssertionError("singleton lab padding changed")
        if singleton["discharge"] != [expected_discharge]:
            raise AssertionError("singleton raw-note collation changed")
    return item


def _check_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = mimic4_collate_fn(items)
    for name in ("age", "gender", "ethnicity", "label"):
        expected = torch.stack([item[name] for item in items])
        if not torch.equal(batch[name], expected):
            raise AssertionError(f"author collate changed stacked {name}")
    for name in ("types", "codes", "labvectors"):
        maximum = max(item[name].shape[0] for item in items)
        trailing = items[0][name].shape[1:]
        expected = torch.zeros(
            (len(items), maximum, *trailing), dtype=items[0][name].dtype
        )
        for position, item in enumerate(items):
            expected[position, : item[name].shape[0]] = item[name]
        if not torch.equal(batch[name], expected):
            raise AssertionError(f"author collate changed padding for {name}")
    for name in ("codes_flag", "labvectors_flag", "discharge_flag", "label_flag"):
        expected = torch.tensor([item[name] for item in items])
        if not torch.equal(batch[name], expected):
            raise AssertionError(f"author collate changed {name}")
    if batch["discharge"] != [item["discharge"] for item in items]:
        raise AssertionError("author collate changed raw-note order")
    return {
        "examples": len(items),
        "code_shape": list(batch["codes"].shape),
        "lab_shape": list(batch["labvectors"].shape),
    }


def _model_gru_width() -> int:
    path = PAPER_ROOT / "src/core/mimic4_model.py"
    tree = ast.parse(path.read_text())
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if (
                    keyword.arg == "input_size"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, int)
                ):
                    values.append(keyword.value.value)
    if 114 not in values:
        raise AssertionError("released MIMIC4 GRU input width declaration changed")
    return 114


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--note-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    source = ensure_source(args.mimic_root, args.note_root, args.cache_root)
    manifest_path = args.task_dir / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["task"]["id"] != TASK_ID:
        raise AssertionError("wrong MUSE task")
    if manifest["task"]["modalities"] != ["structured", "notes"]:
        raise AssertionError("MUSE retained modality declaration changed")
    if manifest["task"]["note_codes"] != [NOTE_CODE]:
        raise AssertionError("MUSE note-code exception changed")
    if manifest["source"]["source_configuration"]["load_no_label"]:
        raise AssertionError("auxiliary unlabeled MUSE+ rows entered the selected task")

    codes = pq.read_table(
        args.task_dir / manifest["meds"]["codes"]["file"]
    )["code"].to_pylist()
    expected_codes = list(source_codes(args.cache_root))
    if codes != expected_codes:
        raise AssertionError("MEDS vocabulary differs from exact retained inputs")
    if set(manifest["task"]["note_codes"]) != {NOTE_CODE}:
        raise AssertionError("note-code exception set is incomplete")
    structured_prefixes = (
        "MUSE_AGE//",
        "MUSE_GENDER//",
        "MUSE_ETHNICITY//",
        "MUSE_CODE//",
        "MUSE_LAB_PRESENT//",
        "MUSE_LAB_SEQUENCE_ROW",
        "MUSE_LAB_VALUE//",
        "MUSE_LAB_ABNORMAL//",
    )
    for code in codes:
        if code == NOTE_CODE:
            continue
        if not code.startswith(structured_prefixes):
            raise AssertionError(f"undeclared auxiliary code entered MEDS: {code}")

    # Compare every persistent event field as a multiset independently of
    # Parquet row-group boundaries and writer sorting.
    selected, lookup = _lookup_frames(args.cache_root)
    expected_events = EventMultiset()
    for frame in event_batches(selected, lookup, args.cache_root):
        expected_events.update(frame)
    actual_events = EventMultiset()
    event_path = args.task_dir / manifest["meds"]["events"]["file"]
    for batch in pq.ParquetFile(event_path).iter_batches(batch_size=250_000):
        actual_events.update(batch)
    expected_event_result = expected_events.result()
    actual_event_result = actual_events.result()
    if expected_event_result != actual_event_result:
        raise AssertionError("MEDS changed the exact retained event multiset")

    cohort = pd.read_parquet(args.cache_root / "cohort.parquet").sort_values(
        "cohort_position"
    )
    code_cursor = ParquetGroupCursor(
        args.cache_root / CODE_FILE,
        "cohort_position",
        columns=["cohort_position", "event_type", "code_name", "sequence_row"],
    )
    lab_cursor = ParquetGroupCursor(
        args.cache_root / LAB_FILE,
        "cohort_position",
        columns=[
            "cohort_position",
            "sequence_row",
            "source_time",
            "channel_index",
            "numeric_value",
        ],
    )
    note_cursor = ParquetGroupCursor(
        args.cache_root / NOTE_FILE,
        "cohort_position",
        columns=["cohort_position", "text"],
    )
    vocabulary = _author_code_vocabulary(args.cache_root)
    lab_width = 2 * len(lab_channels(args.cache_root)["itemids"])
    if lab_width != AUTHOR_MISSING_LAB_WIDTH:
        raise AssertionError(f"exact source lab width changed: {lab_width}")

    digest_names = (
        "age",
        "gender",
        "ethnicity",
        "types",
        "codes",
        "labvectors",
        "label",
        "modality_flags",
        "raw_note",
    )
    digests = {name: hashlib.sha256() for name in digest_names}
    shapes: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    dtype_sets: dict[str, set[str]] = defaultdict(set)
    fixed_batch: list[dict[str, Any]] = []
    fixed_batch_count = 0
    mixed_presence_batch_count = 0
    examples_by_presence = {True: 0, False: 0}
    total_examples = 0
    total_singleton_batches = 0

    _patch_author_loader({}, [], vocabulary)
    dataset = author_dataset_module.MIMIC4Dataset(
        split="train", task="mortality", load_no_label=False
    )

    def flush_fixed_batch() -> None:
        nonlocal fixed_batch, fixed_batch_count, mixed_presence_batch_count
        if not fixed_batch:
            return
        _check_collate(fixed_batch)
        fixed_batch_count += 1
        mixed_presence_batch_count += int(
            len({bool(item["labvectors_flag"]) for item in fixed_batch}) > 1
        )
        fixed_batch = []

    for row in cohort.itertuples(index=False):
        position = int(row.cohort_position)
        code_rows = code_cursor.take(position)
        lab_rows = lab_cursor.take(position)
        note_rows = note_cursor.take(position)
        if pd.isna(row.mortality):
            continue
        value = _make_author_object(
            row, code_rows, lab_rows, note_rows, lab_width
        )
        dataset.all_hosp_adm_dict = {value.admission_id: value}
        dataset.included_admission_ids = [value.admission_id]
        item = _check_item(dataset, 0, value)
        total_singleton_batches += 1
        total_examples += 1
        present = bool(item["labvectors_flag"])
        examples_by_presence[present] += 1
        fixed_batch.append(item)
        if len(fixed_batch) == AUTHOR_BATCH_SIZE:
            flush_fixed_batch()

        for name in ("age", "gender", "ethnicity", "types", "codes", "labvectors", "label"):
            update_tensor_digest(digests[name], item[name])
            shapes[name].add(tuple(item[name].shape))
            dtype_sets[name].add(str(item[name].dtype))
        flags = np.asarray(
            [
                item["codes_flag"],
                item["labvectors_flag"],
                item["discharge_flag"],
                item["label_flag"],
            ],
            dtype=np.bool_,
        )
        update_tensor_digest(digests["modality_flags"], flags)
        encoded = item["discharge"].encode()
        digests["raw_note"].update(len(encoded).to_bytes(8, "little"))
        digests["raw_note"].update(encoded)

    flush_fixed_batch()
    if total_examples != source["cohort"]["labeled"]:
        raise AssertionError("author Dataset traversal omitted a labeled admission")
    if sum(examples_by_presence.values()) != total_examples:
        raise AssertionError("lab modality membership traversal changed")

    if not all(value > 0 for value in examples_by_presence.values()):
        raise AssertionError("selected cohort does not contain both lab modalities")
    if mixed_presence_batch_count == 0:
        raise AssertionError("fixed exhaustive batches never mixed lab modalities")
    model_width = _model_gru_width()
    if model_width != lab_width:
        raise AssertionError("released GRU width differs from exact source labs")

    report = {
        "status": "exact",
        "reconstruction_status": "exact_author_batch_reconstruction",
        "selected_experiment": {
            "task": "MIMIC-IV 90-day post-discharge mortality",
            "model_inputs": ["structured", "notes"],
            "load_no_label": False,
            "selection_tier": 2,
            "batch_size": AUTHOR_BATCH_SIZE,
        },
        "exact_reconstruction": {
            "examples": total_examples,
            "singleton_author_collate_batches": total_singleton_batches,
            "fixed_author_collate_batches": fixed_batch_count,
            "mixed_lab_presence_batches": mixed_presence_batch_count,
            "examples_by_lab_presence": {
                "present": examples_by_presence[True],
                "missing": examples_by_presence[False],
            },
            "field_sha256": {
                name: digest.hexdigest() for name, digest in digests.items()
            },
            "shapes": {
                name: [list(value) for value in sorted(values)]
                for name, values in shapes.items()
            },
            "dtypes": {
                name: sorted(values) for name, values in dtype_sets.items()
            },
            "event_multiset": actual_event_result,
            "event_vocabulary_size": len(codes),
            "note_codes": [NOTE_CODE],
            "all_non_note_codes_are_intended_structured_inputs": True,
            "no_imaging_or_auxiliary_rows": True,
        },
        "author_lab_width": {
            "lab_item_count": len(lab_channels(args.cache_root)["itemids"]),
            "nonmissing_lab_width": lab_width,
            "missing_fallback_width": AUTHOR_MISSING_LAB_WIDTH,
            "model_gru_width": model_width,
            "exact_width_match": True,
        },
        "source_reconstruction": source,
        "manifest_sha256": manifest_hash,
        "checks": [
            "exhaustive MIMIC4Dataset.__getitem__ reconstruction",
            "exhaustive raw-note, label, flag, dtype, shape, and singleton equality",
            "exhaustive fixed-batch mimic4_collate_fn padding equality",
            "complete MEDS event multiset and vocabulary equality",
            "exact note_codes exception and structured-code proof",
            "mixed lab-presence batches successfully reconstructed",
            "exact 114-channel source/fallback/GRU width equality",
        ],
    }
    output = args.task_dir / "audit" / "full_batch_recreation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if sha256_file(manifest_path) != manifest_hash:
        raise AssertionError("batch checker mutated immutable manifest")
    print(json.dumps({"status": report["status"], "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
