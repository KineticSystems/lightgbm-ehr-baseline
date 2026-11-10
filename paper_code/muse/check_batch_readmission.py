"""Exhaustively recreate MUSE's readmission Dataset/collate boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

import src.dataset.mimic4_dataset as author_dataset_module  # noqa: E402
from check_batch import (  # noqa: E402
    EventMultiset,
    _author_code_vocabulary,
    _check_collate,
    _check_item,
    _make_author_object,
    _model_gru_width,
    _patch_author_loader,
    update_tensor_digest,
)
from export_source import _lookup_frames, event_batches  # noqa: E402
from cohort_source import (  # noqa: E402
    AUTHOR_BATCH_SIZE,
    AUTHOR_MISSING_LAB_WIDTH,
    CODE_FILE,
    COHORT_FILE,
    LAB_FILE,
    NOTE_CODE,
    NOTE_FILE,
    ParquetGroupCursor,
    lab_channels,
    sha256_file,
    source_codes,
)
from cohort_readmission import (  # noqa: E402
    AUTHOR_TASK,
    LABEL_COLUMN,
    TASK_ID,
    load_source,
    readmission_cohort,
    readmission_label,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--note-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    source = load_source(args.mimic_root, args.note_root, args.cache_root)
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
    if manifest["source"]["source_configuration"]["task"] != AUTHOR_TASK:
        raise AssertionError("selected released task changed")

    codes = pq.read_table(args.task_dir / manifest["meds"]["codes"]["file"])[
        "code"
    ].to_pylist()
    if codes != list(source_codes(args.cache_root)):
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

    labeled, detail = readmission_cohort(args.cache_root)
    summary = detail["summary"]
    selected, lookup = _lookup_frames(args.cache_root, labeled)
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

    index = pq.read_table(args.task_dir / manifest["meds"]["task_index"]["file"]).to_pandas()
    index_labels = dict(
        zip(index["sample_id"].astype(int), index["boolean_value"].astype(bool))
    )

    cohort = pd.read_parquet(args.cache_root / COHORT_FILE).sort_values(
        "cohort_position"
    )
    cohort[LABEL_COLUMN] = readmission_label(cohort)
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
        args.cache_root / NOTE_FILE, "cohort_position", columns=["cohort_position", "text"]
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
    positive_examples = 0
    total_examples = 0
    total_singleton_batches = 0

    _patch_author_loader({}, [], vocabulary)
    dataset = author_dataset_module.MIMIC4Dataset(
        split="train", task=AUTHOR_TASK, load_no_label=False
    )
    if dataset.task != AUTHOR_TASK:
        raise AssertionError("author Dataset is not configured for readmission")

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
        if pd.isna(getattr(row, LABEL_COLUMN)):
            continue
        value = _make_author_object(row, code_rows, lab_rows, note_rows, lab_width)
        if value.readmission is None:
            raise AssertionError("released readmission label was lost")
        dataset.all_hosp_adm_dict = {value.admission_id: value}
        dataset.included_admission_ids = [value.admission_id]
        item = _check_item(dataset, 0, value, label_attr=LABEL_COLUMN)
        expected_label = index_labels.get(int(value.admission_id))
        if expected_label is None:
            raise AssertionError("author labeled admission is absent from the MEDS index")
        if bool(float(item["label"])) != bool(expected_label):
            raise AssertionError("MEDS label differs from the author Dataset label")
        positive_examples += int(bool(expected_label))
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
    if total_examples != summary["labeled"]:
        raise AssertionError("author Dataset traversal omitted a labeled admission")
    if total_examples != len(index_labels):
        raise AssertionError("MEDS index and author traversal disagree on membership")
    if positive_examples != summary["positive"]:
        raise AssertionError("author traversal changed the released positive count")
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
        "status": "exact_source_with_recorded_environment_substitution",
        "reconstruction_status": "exact_author_batch_reconstruction",
        "selected_experiment": {
            "task": "MIMIC-IV 15-day readmission after discharge",
            "model_inputs": ["structured", "notes"],
            "load_no_label": False,
            "selection_tier": 2,
            "batch_size": AUTHOR_BATCH_SIZE,
            "author_task_argument": AUTHOR_TASK,
        },
        "exact_reconstruction": {
            "examples": total_examples,
            "positive_examples": positive_examples,
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
            "dtypes": {name: sorted(values) for name, values in dtype_sets.items()},
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
        "recorded_deviations": [
            {
                "kind": "environment_substitution",
                "detail": (
                    "paper_code/muse/.venv (CPython 3.8.20, torch 2.0.1 CPU) is "
                    "unusable - its interpreter under /tmp/medpatch_uv_python was "
                    "deleted. The paper's own "
                    "paper_code/muse/.export_assets/gpu_venv (CPython 3.10.20, "
                    "torch 2.7.0+cu128) was used instead; NumPy, pandas, Polars "
                    "and PyArrow pins are identical to the sealed export."
                ),
            },
            {
                "kind": "adapter_revision_drift",
                "detail": source["adapter_revision_drift"]["note"],
                "sha256_recorded_at_cache_build": source["adapter_revision_drift"][
                    "sha256_recorded_at_cache_build"
                ],
                "sha256_current": source["adapter_revision_drift"]["sha256_current"],
            },
        ],
        "source_reconstruction": source,
        "manifest_sha256": manifest_hash,
        "checks": [
            "exhaustive MIMIC4Dataset.__getitem__ reconstruction with task='readmission'",
            "exhaustive raw-note, label, flag, dtype, shape, and singleton equality",
            "exhaustive fixed-batch mimic4_collate_fn padding equality",
            "every author label equals the MEDS boolean_value for the same admission",
            "complete MEDS event multiset and vocabulary equality",
            "exact note_codes exception and structured-code proof",
            "mixed lab-presence batches successfully reconstructed",
            "exact 114-channel source/fallback/GRU width equality",
        ],
    }
    output = args.task_dir / "audit" / "full_batch_recreation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if sha256_file(manifest_path) != manifest_hash:
        raise AssertionError("batch checker mutated immutable manifest")
    print(json.dumps({"status": report["status"], "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
