#!/usr/bin/env python3
"""Derive the sampled decompensation task from the full one.

The paper's decompensation task is scored at every hourly landmark, which is
2.67 million of them and 4.3 billion events. Fitting that directly is out of
reach on ordinary hardware, so the campaign fits a fixed subset drawn under the
shared landmark sampling contract: 250,000 train, 100,000 validation and
250,000 test landmarks, selected without replacement by a SplitMix64
permutation of the sample IDs. Selection is independent of label, subject and
row order, and the seed is fixed, so the subset is the same everywhere.

The subset reuses the base task's already validated event stream rather than
rebuilding it: `derive_task` restricts those events to the selected samples and
carries the base reconciliation forward, so the sampled task satisfies the same
audit contract without a second exhaustive batch check.

    derive_sampled.py --base paper_tasks/semi_supervised_meta_learning/decompensation \
        --output paper_tasks/semi_supervised_meta_learning/decompensation_sampled
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl

PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(
    0, os.environ.get("MXGB_LIGHTGBM_SRC", str(WORKSPACE_ROOT / "lightgbm_code" / "src"))
)

from missing_xgboost_lightgbm.derive_task import derive_task  # noqa: E402
from missing_xgboost_lightgbm.landmark_sampling import (  # noqa: E402
    LANDMARK_SAMPLE_CAPS,
    LANDMARK_SAMPLE_SEED,
    select_uniform_landmarks,
)

TASK_ID = "decompensation_sampled"
TASK_NAME = (
    "Hourly ICU decompensation within the next 24 hours "
    "(600k-landmark subset under the campaign sampling contract)"
)
POSITIVE_CLASS = "in-unit death within 24 hours at the benchmark cutoff"
PUBLISHED_AUROC = 0.906


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True,
                        help="the full decompensation task directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    index_path = args.base / "audit" / "task_index.parquet"
    if not index_path.is_file():
        raise SystemExit(f"base task has no audit index: {index_path}")

    index = pl.read_parquet(index_path)
    selected, audit = select_uniform_landmarks(
        index.select(["sample_id", "subject_id", "boolean_value", "split"])
    )
    labels = selected.select(["sample_id", "boolean_value", "split"])

    manifest = derive_task(
        args.base,
        args.output,
        task={
            "id": TASK_ID,
            "name": TASK_NAME,
            "published_auroc": PUBLISHED_AUROC,
            "positive_class": POSITIVE_CLASS,
        },
        labels=labels,
        derivation={
            "label_source": "inherited unchanged from the base task's index",
            "selection": (
                "uniform landmark sampling without replacement, per split, "
                "independent of label and subject"
            ),
            "caps": dict(LANDMARK_SAMPLE_CAPS),
            "seed": LANDMARK_SAMPLE_SEED,
            "selected_landmarks": int(selected.height),
            "base_landmarks": int(index.height),
            "per_split": audit,
        },
    )
    print(
        f"wrote {args.output} "
        f"({selected.height:,} of {index.height:,} landmarks)"
    )
    return manifest


if __name__ == "__main__":
    main()
