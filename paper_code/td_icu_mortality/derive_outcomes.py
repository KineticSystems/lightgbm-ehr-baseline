"""Phase 2: derive td-icu's 1/3/7/14-day death outcomes as sibling tasks.

The base task index carries every horizon label at the same rolling
landmarks (`{1,3,7,14,28}-day-died`), so each outcome is the base events
with a swapped label (docs/PHASE2_OUTCOMES.md).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import polars as pl

PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(
    0, os.environ.get("MXGB_LIGHTGBM_SRC", str(WORKSPACE_ROOT / "lightgbm_code" / "src"))
)

from missing_xgboost_lightgbm.derive_task import derive_task  # noqa: E402

OUTCOMES = {
    "mortality_1d": ("1-day-died", "1-day mortality at rolling ICU measurement landmarks", 0.874),
    "mortality_3d": ("3-day-died", "3-day mortality at rolling ICU measurement landmarks", 0.841),
    "mortality_7d": ("7-day-died", "7-day mortality at rolling ICU measurement landmarks", 0.832),
    "mortality_14d": ("14-day-died", "14-day mortality at rolling ICU measurement landmarks", 0.838),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    index = pl.read_parquet(args.base / "audit" / "task_index.parquet")
    for task_id, (column, name, published) in OUTCOMES.items():
        if args.only and task_id not in args.only:
            continue
        output = args.output_root / task_id
        if (output / "manifest.json").is_file():
            print(f"skip {task_id}: complete", file=sys.stderr)
            continue
        if output.exists():
            print(f"rebuilding {task_id}: partial output removed", file=sys.stderr)
            shutil.rmtree(output)
        labels = index.select(
            "sample_id",
            "split",
            pl.col(f"source_{column}").cast(pl.Boolean).alias("boolean_value"),
        )
        manifest = derive_task(
            args.base,
            output,
            task={"id": task_id, "name": name, "published_auroc": published,
                  "positive_class": f"death within the horizon (released {column} label)"},
            labels=labels,
            derivation={
                "label_source": f"base task index column source_{column!r} (released labeler output)",
                "published_reference": "TD-ICU Table 2, 24-hour delay TD model",
            },
        )
        print(task_id, {s: manifest["splits"][s]["positive"] for s in manifest["splits"]})


if __name__ == "__main__":
    main()
