#!/usr/bin/env python3
"""Exhaustively verify GARLIC's bounded training adapter model boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from main import prepare_batch  # noqa: E402
from reproduction_data import GarlicDataStore, SPLITS  # noqa: E402
from cohort_source import ordered_split_hash, sha256_file  # noqa: E402


VERIFY_BATCH_SIZE = 32
EXPECTED_COUNTS = {"train": 39_500, "validation": 4_938, "test": 4_937}
EXPECTED_POSITIVES = {"train": 4_593, "validation": 574, "test": 575}


def tensor_digest(digest: hashlib._Hash, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous().numpy()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--orders", type=Path, required=True)
    parser.add_argument("--order-report", type=Path, required=True)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    order_report = json.loads(args.order_report.read_text())
    if order_report.get("status") != "exact_membership_and_released_order":
        raise AssertionError("GARLIC split order is not verified")
    export = json.loads(args.export_report.read_text())
    if export.get("status") != "exact":
        raise AssertionError("GARLIC export model boundary is not exact")

    store = GarlicDataStore(args.cache, args.orders)
    try:
        counts = {split: len(store.splits[split]) for split in SPLITS}
        positives = {
            split: sum(store.rows[hadm]["label"] for hadm in store.splits[split].ids)
            for split in SPLITS
        }
        if counts != EXPECTED_COUNTS or positives != EXPECTED_POSITIVES:
            raise AssertionError("GARLIC ordered split counts changed")
        if ordered_split_hash(store.connection) != json.loads(
            (args.export_report.parent.parent / "manifest.json").read_text()
        )["source"]["ordered_identifier_label_split_sha256"]:
            raise AssertionError("GARLIC cache identifier/label/split hash changed")

        digests = {
            name: hashlib.sha256() for name in ("values", "mask", "time", "label")
        }
        source = store.source_order()
        examples = batches = 0
        for start in range(0, len(source), VERIFY_BATCH_SIZE):
            batch = source[start : start + VERIFY_BATCH_SIZE]
            values, mask, times, labels = prepare_batch(batch, "cpu")
            tensors = {
                "values": values,
                "mask": mask,
                "time": times,
                "label": labels,
            }
            if not all(torch.isfinite(value).all() for value in tensors.values()):
                raise AssertionError("GARLIC adapter emitted nonfinite model inputs")
            if not torch.all((mask == 0) | (mask == 1)):
                raise AssertionError("GARLIC adapter mask is not binary")
            for name, value in tensors.items():
                tensor_digest(digests[name], value)
            examples += len(batch)
            batches += 1
        actual_hashes = {name: digest.hexdigest() for name, digest in digests.items()}
        if actual_hashes != export["tensor_sha256"]:
            raise AssertionError(
                f"GARLIC adapter tensors differ from the export: {actual_hashes}"
            )

        edge_batches = {}
        for split in SPLITS:
            sequence = store.splits[split]
            first = sequence[:128]
            last_start = (len(sequence) // 128) * 128
            if last_start == len(sequence):
                last_start -= 128
            last = sequence[last_start:]
            first_tensors = prepare_batch(first, "cpu")
            last_tensors = prepare_batch(last, "cpu")
            edge_batches[split] = {
                "first_batch": len(first),
                "last_batch": len(last),
                "first_shapes": [list(value.shape) for value in first_tensors],
                "last_shapes": [list(value.shape) for value in last_tensors],
            }

        report = {
            "status": "exact",
            "author_entrypoint": "main.py::prepare_batch",
            "adapter": "reproduction_data.py::GarlicDataStore",
            "examples": examples,
            "verification_batches": batches,
            "verification_batch_size": VERIFY_BATCH_SIZE,
            "splits": {
                split: {
                    "count": counts[split],
                    "positive": positives[split],
                    "training_batches_batch128": (counts[split] + 127) // 128,
                }
                for split in SPLITS
            },
            "edge_batches": edge_batches,
            "tensor_sha256": actual_hashes,
            "export_tensor_sha256": export["tensor_sha256"],
            "checks": [
                "released get_data_split membership and within-split order",
                "every source-order admission streamed from the SQLite boundary",
                "all values, masks, timestamps, labels, padding, dtypes, and shapes",
                "exact frozen export tensor hashes at the released prepare_batch boundary",
                "first and final released batch-128 shapes for every split",
            ],
            "artifacts": {
                "orders_sha256": sha256_file(args.orders),
                "order_report_sha256": sha256_file(args.order_report),
                "export_report_sha256": sha256_file(args.export_report),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        store.close()


if __name__ == "__main__":
    main()

