#!/usr/bin/env python3
"""Build the released PyHealth parse once per dataset, in patient shards.

`data_prepare.load_dataset(False, "mimic3"|"mimic4", ...)` instantiates
PyHealth 1.1.2's `MIMIC3Dataset` / `MIMIC4Dataset` with the released code
mapping.  As one call it materialises the entire patient object graph (>6 GiB
for MIMIC-III, far more for MIMIC-IV), which does not fit the workstation's
7 GiB per-process budget.  This script runs the *same released parser methods*
on disjoint patient shards and stores the released per-visit code lists, which
both tasks of a dataset then reuse.

Equivalence to the unsharded call is argued in `cohort_phase2.py` and proved
empirically for MIMIC-III by `check_cohort_phase2.py`, which reconstructs the
sealed phase-1 MIMIC-III mortality task from this cache and compares it to the
exported artifacts of that task byte for byte.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cohort_phase2 import (
    DATASETS,
    build_parse_cache,
    dataset_cache_dir,
    parse_cache_summary,
    required_raw_files,
)


PAPER_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--shards", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=1_000_000)
    parser.add_argument("--only-shard", type=int, default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    cache_dir = args.cache_dir or dataset_cache_dir(PAPER_ROOT, args.dataset)
    shards = args.shards or DATASETS[args.dataset]["shards"]
    required_raw_files(args.dataset, args.root)
    if not args.summary_only:
        build_parse_cache(
            PAPER_ROOT,
            args.dataset,
            args.root,
            cache_dir,
            shards,
            chunk=args.chunk,
            only_shard=args.only_shard,
        )
    if args.summary_only or args.only_shard is None:
        summary = parse_cache_summary(args.dataset, cache_dir, shards)
        (cache_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
