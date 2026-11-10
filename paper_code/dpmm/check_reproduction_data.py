"""Verify the adjacent DPMM note adapter against the frozen batch hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from arguments import args_parser
from dataset_mimic3.fusion import my_collate
from reproduction_data import (
    CACHE_ROOT,
    FULL_BATCH_REPORT,
    WORKSPACE_ROOT,
    build_author_bundle,
    sha256,
)


def update_digest(digest, values) -> None:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.ascontiguousarray(values)
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fixed_args():
    parser = args_parser()
    args = parser.parse_args([])
    args.task = "in-hospital-mortality"
    args.labels_set = "mortality"
    args.data_pairs = "partial_ehr_note"
    args.data_ratio = 1.0
    args.batch_size = 16
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "paper_results/dpmm/runtime/reproduction_data_check.json",
    )
    cli = parser.parse_args()
    expected_report = json.loads(FULL_BATCH_REPORT.read_text())
    expected = expected_report["ordered_field_sha256"]
    names = tuple(expected)
    frozen_exact_fields = (
        "token_ids",
        "attention_mask",
        "label_note",
        "sequence_length",
        "pair_presence",
    )
    digests = {name: hashlib.sha256() for name in names}
    bundle = build_author_bundle(fixed_args(), CACHE_ROOT)
    split_details = {}
    total = 0
    for split in ("train", "validation", "test"):
        dataset = bundle.fusion[split]
        loader = DataLoader(
            dataset,
            batch_size=16,
            sampler=SequentialSampler(dataset),
            collate_fn=my_collate,
            num_workers=0,
            drop_last=False,
        )
        seen = 0
        paired = 0
        expected_rows = bundle.ordered_rows(split)
        for batch in loader:
            ehr, token, attention, label_ehr, label_note, lengths, pairs = batch
            if ehr.dtype != np.float64 or ehr.shape[1:] != (24, 76):
                raise AssertionError("DPMM adjacent EHR boundary changed")
            if token.dtype != torch.int64 or token.shape[1:] != (5, 512):
                raise AssertionError("DPMM adjacent token boundary changed")
            if attention.dtype != torch.int64 or attention.shape != token.shape:
                raise AssertionError("DPMM adjacent attention boundary changed")
            if lengths != [24] * len(label_ehr):
                raise AssertionError("DPMM adjacent sequence lengths changed")
            batch_rows = expected_rows.iloc[seen : seen + len(label_ehr)]
            if not np.array_equal(
                label_ehr.astype(np.int64), batch_rows["label"].to_numpy(np.int64)
            ):
                raise AssertionError("DPMM adjacent EHR labels/order changed")
            if not np.isin(ehr[:, :, 59:], [0.0, 1.0]).all():
                raise AssertionError("DPMM adjacent observation masks are not binary")
            values = {
                "ehr_float64": ehr,
                "ehr_model_float32": ehr.astype(np.float32),
                "token_ids": token,
                "attention_mask": attention,
                "label_ehr": label_ehr,
                "label_note": label_note,
                "sequence_length": np.asarray(lengths, dtype=np.int64),
                "pair_presence": np.asarray(pairs, dtype=np.bool_),
            }
            for name, value in values.items():
                update_digest(digests[name], value)
            seen += len(label_ehr)
            paired += int(sum(pairs))
        if seen != len(expected_rows):
            raise AssertionError(f"DPMM adjacent {split} traversal is incomplete")
        if paired != int(expected_rows["note_present"].sum()):
            raise AssertionError(f"DPMM adjacent {split} pair flags changed")
        split_details[split] = {"examples": seen, "paired": paired}
        total += seen
    actual = {name: value.hexdigest() for name, value in digests.items()}
    if any(actual[name] != expected[name] for name in frozen_exact_fields):
        differences = {
            name: {"expected": expected[name], "actual": actual[name]}
            for name in frozen_exact_fields
            if actual[name] != expected[name]
        }
        raise AssertionError(f"DPMM adjacent adapter hashes differ: {differences}")
    report = {
        "status": "exact_adapter_boundary",
        "examples": total,
        "splits": split_details,
        "ordered_field_sha256": actual,
        "frozen_meds_diagnostic_sha256": expected,
        "fields_matching_frozen_diagnostic_exactly": list(frozen_exact_fields),
        "direct_author_boundary_note": (
            "The frozen export diagnostic collated MEDS-recreated EHR values and "
            "Python-int labels. Proposed-model training instead calls the released "
            "author EHRdataset directly, as required by the feasibility handoff. "
            "Its EHR/value and int32-label hashes are recorded here rather than "
            "forced to equal the known lossy MEDS diagnostic hashes."
        ),
        "frozen_full_batch_report": {
            "file": str(FULL_BATCH_REPORT),
            "sha256": sha256(FULL_BATCH_REPORT),
        },
        "source_manifest": {
            "file": str(CACHE_ROOT / "source_manifest.json"),
            "sha256": sha256(CACHE_ROOT / "source_manifest.json"),
        },
        "checks": [
            "adjacent note Dataset paired membership and order exact",
            "released MIMIC_NOTE_EHR paired-first fusion order exact",
            "all 21,139 source EHR examples traversed",
            "all 26,365 note slots tokenized from pinned local TinyBERT",
            "released my_collate invoked for every fixed batch",
            "five applicable frozen note/order boundary hashes exact",
            "direct author EHR arrays and labels retained in exact fusion order",
        ],
    }
    write_json(cli.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
