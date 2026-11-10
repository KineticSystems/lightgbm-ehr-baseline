"""Exhaustively verify the reconstructed unlabeled MUSE+ Dataset boundary."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import src.dataset.mimic4_dataset as author_dataset_module
from check_batch import (
    _check_collate,
    _expected_indices,
    _make_author_object,
    update_tensor_digest,
)
from reproduce_muse import CACHE, LazyAdmissionMap, build_vocabulary, install
from cohort_source import (
    AUTHOR_MISSING_LAB_WIDTH,
    CODE_FILE,
    LAB_FILE,
    NOTE_FILE,
    ParquetGroupCursor,
    lab_channels,
    sha256_file,
)


PAPER_ROOT = Path(__file__).resolve().parent
WORK = PAPER_ROOT / ".export_assets" / "repro"
PRIMARY_BATCH_SIZE = 512
EXPECTED_TRAIN = 115_306
EXPECTED_AUXILIARY = 225_038
EXPECTED_MUSE_PLUS_TRAIN = 340_344


def read_ids(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line]


def ordered_string_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def compare_objects(actual: Any, expected: Any) -> None:
    for name in (
        "admission_id",
        "patient_id",
        "duration",
        "mortality",
        "readmission",
        "age",
        "gender",
        "ethnicity",
        "discharge",
    ):
        if getattr(actual, name) != getattr(expected, name):
            raise AssertionError(f"MUSE+ adapter changed auxiliary {name}")
    if actual.trajectory != expected.trajectory:
        raise AssertionError("MUSE+ adapter changed auxiliary code trajectory")
    if actual.labvectors is None or expected.labvectors is None:
        if actual.labvectors is not None or expected.labvectors is not None:
            raise AssertionError("MUSE+ adapter changed auxiliary lab presence")
    elif not np.array_equal(actual.labvectors, expected.labvectors):
        raise AssertionError("MUSE+ adapter changed auxiliary lab values")


def check_auxiliary_item(dataset: Any, index: int, expected: Any) -> dict[str, Any]:
    item = dataset[index]
    indices = _expected_indices(dataset, expected)
    if item["id"] != expected.admission_id:
        raise AssertionError("released Dataset changed auxiliary admission order")
    for name in ("age", "gender", "ethnicity"):
        if item[name].dtype != torch.int64 or int(item[name]) != indices[name]:
            raise AssertionError(f"released Dataset changed auxiliary {name}")
    for name in ("types", "codes"):
        target = torch.tensor(indices[name], dtype=torch.int64)
        if item[name].dtype != torch.int64 or not torch.equal(item[name], target):
            raise AssertionError(f"released Dataset changed auxiliary {name}")
    if not bool(item["codes_flag"]):
        raise AssertionError("released codes_flag is not constant true")

    if expected.labvectors is None:
        expected_lab = torch.zeros(1, AUTHOR_MISSING_LAB_WIDTH)
        expected_lab_flag = False
    else:
        expected_lab = torch.tensor(expected.labvectors, dtype=torch.float32)
        expected_lab_flag = True
    if item["labvectors"].dtype != torch.float32 or not torch.equal(
        item["labvectors"], expected_lab
    ):
        raise AssertionError("released Dataset changed auxiliary labvectors")
    if bool(item["labvectors_flag"]) != expected_lab_flag:
        raise AssertionError("released Dataset changed auxiliary lab membership")

    expected_note = expected.discharge if expected.discharge is not None else ""
    if item["discharge"] != expected_note:
        raise AssertionError("released Dataset changed auxiliary note text")
    if bool(item["discharge_flag"]) != (expected.discharge is not None):
        raise AssertionError("released Dataset changed auxiliary note membership")
    if item["label"].dtype != torch.float32 or float(item["label"]) != 0.0:
        raise AssertionError("released Dataset changed auxiliary placeholder label")
    if bool(item["label_flag"]):
        raise AssertionError("auxiliary MUSE+ example acquired label_flag")
    return item


def main() -> None:
    cache_root = CACHE.resolve()
    source_manifest_path = cache_root / "source_manifest.json"
    source_manifest_hash = sha256_file(source_manifest_path)

    # Source roots are already authenticated in the immutable source manifest;
    # ensure_source's cache-return path verifies every cached artifact hash and
    # does not reopen raw MIMIC files.
    source_manifest = json.loads(source_manifest_path.read_text())
    for record in source_manifest["cache_files"].values():
        path = cache_root / record["file"]
        if sha256_file(path) != record["sha256"]:
            raise AssertionError(f"modified source-cache artifact: {record['file']}")

    cohort = pd.read_parquet(cache_root / "cohort.parquet").sort_values(
        "cohort_position"
    )
    train_ids = read_ids(cache_root / "train_admission_ids.txt")
    auxiliary_ids = read_ids(cache_root / "no_label_admission_ids.txt")
    expected_auxiliary_ids = [
        str(int(value))
        for value in cohort.loc[cohort["mortality"].isna(), "hadm_id"]
    ]
    if auxiliary_ids != expected_auxiliary_ids:
        raise AssertionError("auxiliary ID file differs from cohort-order null labels")
    if len(train_ids) != EXPECTED_TRAIN or len(auxiliary_ids) != EXPECTED_AUXILIARY:
        raise AssertionError("MUSE+ released training cardinality changed")

    vocabulary = build_vocabulary()
    admissions = LazyAdmissionMap(AUTHOR_MISSING_LAB_WIDTH)
    admissions.prepack()
    # Dataset construction stores this matrix in its tokenizer but __getitem__
    # and collate never consume it. The real training path uses the separately
    # hashed author-batch-1,024 ClinicalBERT artifact.
    install(
        vocabulary,
        np.zeros((len(vocabulary), 1), dtype=np.float32),
        admissions,
    )
    dataset = author_dataset_module.MIMIC4Dataset(
        split="train", task="mortality", load_no_label=True
    )
    if dataset.no_label_admission_ids != auxiliary_ids:
        raise AssertionError("released Dataset changed auxiliary ID order")
    if dataset.included_admission_ids != train_ids + auxiliary_ids:
        raise AssertionError("released Dataset changed MUSE+ concatenation order")
    if len(dataset) != EXPECTED_MUSE_PLUS_TRAIN:
        raise AssertionError("released Dataset changed MUSE+ training size")

    code_cursor = ParquetGroupCursor(
        cache_root / CODE_FILE,
        "cohort_position",
        columns=["cohort_position", "event_type", "code_name", "sequence_row"],
    )
    lab_cursor = ParquetGroupCursor(
        cache_root / LAB_FILE,
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
        cache_root / NOTE_FILE,
        "cohort_position",
        columns=["cohort_position", "text"],
    )
    lab_width = 2 * len(lab_channels(cache_root)["itemids"])
    if lab_width != AUTHOR_MISSING_LAB_WIDTH:
        raise AssertionError("released lab width changed")

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
    dtypes: dict[str, set[str]] = defaultdict(set)
    presence = {"labs": 0, "notes": 0}
    fixed_batch: list[dict[str, Any]] = []
    first_auxiliary_items: list[dict[str, Any]] = []
    fixed_batches = 0
    auxiliary_index = 0

    def flush_batch() -> None:
        nonlocal fixed_batch, fixed_batches
        if not fixed_batch:
            return
        _check_collate(fixed_batch)
        fixed_batches += 1
        fixed_batch = []

    for row in cohort.itertuples(index=False):
        if not pd.isna(row.mortality):
            continue
        position = int(row.cohort_position)
        reference = _make_author_object(
            row,
            code_cursor.take(position),
            lab_cursor.take(position),
            note_cursor.take(position),
            lab_width,
        )
        actual_object = admissions[reference.admission_id]
        compare_objects(actual_object, reference)

        dataset_index = len(train_ids) + auxiliary_index
        item = check_auxiliary_item(dataset, dataset_index, reference)
        if len(first_auxiliary_items) < PRIMARY_BATCH_SIZE // 2:
            first_auxiliary_items.append(item)
        fixed_batch.append(item)
        if len(fixed_batch) == PRIMARY_BATCH_SIZE:
            flush_batch()

        presence["labs"] += int(bool(item["labvectors_flag"]))
        presence["notes"] += int(bool(item["discharge_flag"]))
        for name in (
            "age",
            "gender",
            "ethnicity",
            "types",
            "codes",
            "labvectors",
            "label",
        ):
            update_tensor_digest(digests[name], item[name])
            shapes[name].add(tuple(item[name].shape))
            dtypes[name].add(str(item[name].dtype))
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

        auxiliary_index += 1
        if auxiliary_index % 50_000 == 0:
            print(f"verified auxiliary admissions: {auxiliary_index:,}", flush=True)

    flush_batch()
    if auxiliary_index != EXPECTED_AUXILIARY:
        raise AssertionError("exhaustive auxiliary traversal changed cardinality")

    labeled_items = [dataset[index] for index in range(PRIMARY_BATCH_SIZE // 2)]
    mixed = labeled_items + first_auxiliary_items
    _check_collate(mixed)
    mixed_flags = torch.tensor([item["label_flag"] for item in mixed])
    if int(mixed_flags.sum()) != PRIMARY_BATCH_SIZE // 2:
        raise AssertionError("mixed MUSE+ collate changed label flags")

    output = WORK / "audit" / "muse_plus_input_parity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "exact",
        "upstream_commit": source_manifest["upstream_commit"],
        "adapter_sha256": sha256_file(PAPER_ROOT / "reproduce_muse.py"),
        "source_manifest_sha256": source_manifest_hash,
        "counts": {
            "labeled_train": len(train_ids),
            "auxiliary": auxiliary_index,
            "muse_plus_train": len(dataset),
            "fixed_auxiliary_batches": fixed_batches,
            "mixed_batch_examples": len(mixed),
            "mixed_batch_labeled": int(mixed_flags.sum()),
            "auxiliary_labs_present": presence["labs"],
            "auxiliary_notes_present": presence["notes"],
        },
        "ordered_id_sha256": {
            "labeled_train": ordered_string_hash(train_ids),
            "auxiliary": ordered_string_hash(auxiliary_ids),
            "concatenated": ordered_string_hash(train_ids + auxiliary_ids),
        },
        "field_sha256": {
            name: digest.hexdigest() for name, digest in digests.items()
        },
        "shapes": {
            name: [list(value) for value in sorted(values)]
            for name, values in shapes.items()
        },
        "dtypes": {name: sorted(values) for name, values in dtypes.items()},
        "checks": [
            "all source-cache hashes verified",
            "auxiliary IDs equal cohort-order null-label admissions",
            "released train-plus-auxiliary concatenation order exact",
            "all auxiliary reconstructed MIMIC4Data objects exact",
            "all auxiliary released MIMIC4Dataset items exact",
            "all fixed 512-example auxiliary collations exact",
            "one 256-labeled plus 256-auxiliary collation exact",
        ],
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if sha256_file(source_manifest_path) != source_manifest_hash:
        raise AssertionError("MUSE+ checker mutated source manifest")
    print(json.dumps({"status": "exact", "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
