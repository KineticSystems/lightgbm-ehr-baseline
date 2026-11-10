#!/usr/bin/env python3
"""Freeze GARLIC's exact released split order without rebuilding the export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reproduction_data import SPLITS, reconstruct_ordered_splits


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_digest(ids: list[int], by_hadm: dict[int, dict]) -> str:
    digest = hashlib.sha256()
    for hadm in ids:
        row = by_hadm[hadm]
        digest.update(
            np.asarray(
                [hadm, row["subject_id"], row["label"]], dtype=np.int64
            ).tobytes()
        )
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--orders", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.orders, args.report):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    orders, by_hadm, summary = reconstruct_ordered_splits(
        args.cache, args.mimic_root
    )
    payload = {
        "schema_version": 1,
        "source_commit": "aa2cd2c2d434fc80b6705eec66d54060327995de",
        "ordered_hadm_ids": orders,
    }
    args.orders.parent.mkdir(parents=True, exist_ok=True)
    args.orders.write_text(json.dumps(payload, separators=(",", ":")) + "\n")

    split_report = {}
    for split in SPLITS:
        ids = orders[split]
        split_report[split] = {
            "count": len(ids),
            "positive": sum(by_hadm[hadm]["label"] for hadm in ids),
            "ordered_identifier_label_sha256": ordered_digest(ids, by_hadm),
        }
    report = {
        "status": "exact_membership_and_released_order",
        "author_entrypoint": "utils.py::get_data_split",
        "source_rows": len(by_hadm),
        "splits": split_report,
        "cache_summary_status": summary["status"],
        "orders_sha256": sha256_file(args.orders),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

