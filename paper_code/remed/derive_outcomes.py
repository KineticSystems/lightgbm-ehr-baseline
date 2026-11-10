"""Phase 2: derive REMed's other 48-hour binary outcomes as sibling tasks.

The released cohort CSV carries every Table 10 endpoint as a clean 0/1
column over the same 32,449 stays, so each outcome is the clamped base
task's events with a swapped label (docs/PHASE2_OUTCOMES.md).

    derive_outcomes.py --base paper_tasks/remed/los_7d_at_48h \
        --output-root paper_tasks/remed [--only mortality_14 ...]
"""

from __future__ import annotations

import argparse
import os
import csv
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

csv.field_size_limit(sys.maxsize)

# task id -> (cohort column, name, published AUROC or None, positive class)
OUTCOMES = {
    "mortality_1d_at_48h": ("mortality_1", "1-day mortality at the 48-hour prediction point", None,
                            "death within 1 day of the prediction point (released mortality_1 label)"),
    "mortality_2d_at_48h": ("mortality_2", "2-day mortality at the 48-hour prediction point", None,
                            "death within 2 days (released mortality_2 label)"),
    "mortality_3d_at_48h": ("mortality_3", "3-day mortality at the 48-hour prediction point", None,
                            "death within 3 days (released mortality_3 label)"),
    "mortality_7d_at_48h": ("mortality_7", "7-day mortality at the 48-hour prediction point", None,
                            "death within 7 days (released mortality_7 label)"),
    "mortality_14d_at_48h": ("mortality_14", "14-day mortality at the 48-hour prediction point", 0.866,
                             "death within 14 days (released mortality_14 label)"),
    "los_14d_at_48h": ("los_14", "ICU length of stay greater than fourteen days at the 48-hour prediction point", None,
                       "complete ICU length of stay strictly greater than fourteen days (released los_14 label)"),
    "readmission_at_48h": ("readmission", "Within-admission ICU readmission at the 48-hour prediction point", None,
                           "a later ICU stay within the same hospital admission (released readmission label; "
                           "returns shorter than 48 hours are lost before labeling — released defect)"),
}


def cohort_labels(cohort_csv: Path, column: str) -> dict[int, bool]:
    labels = {}
    with cohort_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            labels[int(row["stay_id"])] = row[column] == "1"
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cohort-csv", type=Path,
                        default=PAPER_ROOT / ".export_work" / "source_48h" / "mimiciv_cohort.csv")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    index = pl.read_parquet(args.base / "audit" / "task_index.parquet")
    for task_id, (column, name, published, positive) in OUTCOMES.items():
        if args.only and task_id not in args.only:
            continue
        output = args.output_root / task_id
        if (output / "manifest.json").is_file():
            print(f"skip {task_id}: complete", file=sys.stderr)
            continue
        if output.exists():
            print(f"rebuilding {task_id}: partial output removed", file=sys.stderr)
            shutil.rmtree(output)
        lookup = cohort_labels(args.cohort_csv, column)
        labels = index.select("sample_id", "split").with_columns(
            pl.col("sample_id").map_elements(lambda s: lookup[int(s)], return_dtype=pl.Boolean)
            .alias("boolean_value")
        )
        manifest = derive_task(
            args.base,
            output,
            task={"id": task_id, "name": name, "published_auroc": published,
                  "positive_class": positive},
            labels=labels,
            derivation={
                "label_source": f"released mimiciv_cohort.csv column {column!r}",
                "label_values": "0/1 over every retained stay; no undefined entries",
                "published_reference": "REMed Appendix Table 10 (48-hour, infinite history)",
            },
        )
        print(task_id, {s: manifest["splits"][s]["positive"] for s in manifest["splits"]})


if __name__ == "__main__":
    main()
