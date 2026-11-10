#!/usr/bin/env python3
"""Aggregate five independently verified MERGE primary seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


PUBLISHED_AUROC = 0.8540
PUBLISHED_AUROC_SD = 0.0024
PUBLISHED_WEIGHTED_F1 = 0.8497
PUBLISHED_WEIGHTED_F1_SD = 0.0035
LIGHTGBM_AUROC = 0.8679307514819312


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_sd": float(array.std(ddof=0)),
        "sample_sd": float(array.std(ddof=1)),
        "sem": float(array.std(ddof=1) / math.sqrt(len(array))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = []
    for seed in range(5):
        path = args.reports_dir / f"seed_{seed}.json"
        report = json.loads(path.read_text())
        if report.get("status") != "exact" or report.get("seed") != seed:
            raise AssertionError(f"seed {seed} report is not exact")
        reports.append(report)

    aurocs = [float(report["test"]["auroc"]) for report in reports]
    auprcs = [float(report["test"]["auprc"]) for report in reports]
    weighted_f1s = [float(report["test"]["weighted_f1"]) for report in reports]
    auroc = stats(aurocs)
    auprc = stats(auprcs)
    weighted_f1 = stats(weighted_f1s)
    auroc_band = [PUBLISHED_AUROC - PUBLISHED_AUROC_SD, PUBLISHED_AUROC + PUBLISHED_AUROC_SD]
    f1_band = [
        PUBLISHED_WEIGHTED_F1 - PUBLISHED_WEIGHTED_F1_SD,
        PUBLISHED_WEIGHTED_F1 + PUBLISHED_WEIGHTED_F1_SD,
    ]
    match = auroc_band[0] <= auroc["mean"] <= auroc_band[1] and f1_band[0] <= weighted_f1["mean"] <= f1_band[1]

    result = {
        "status": "completed_match" if match else "completed_nonmatch",
        "fidelity": "bounded_current_released_head",
        "paper": {
            "auroc_mean": PUBLISHED_AUROC,
            "auroc_sd": PUBLISHED_AUROC_SD,
            "weighted_f1_mean": PUBLISHED_WEIGHTED_F1,
            "weighted_f1_sd": PUBLISHED_WEIGHTED_F1_SD,
        },
        "seeds": [
            {
                "seed": report["seed"],
                "checkpoint_epoch": report["checkpoint_epoch"],
                "validation_auroc": report["checkpoint_best_validation_auroc"],
                **report["test"],
                "artifacts": report["artifacts"],
            }
            for report in reports
        ],
        "aggregate": {
            "auroc": auroc,
            "auprc": auprc,
            "weighted_f1": weighted_f1,
            "auroc_gap_from_paper": auroc["mean"] - PUBLISHED_AUROC,
            "weighted_f1_gap_from_paper": weighted_f1["mean"] - PUBLISHED_WEIGHTED_F1,
            "auroc_gap_from_lightgbm": auroc["mean"] - LIGHTGBM_AUROC,
            "paper_auroc_band": auroc_band,
            "paper_weighted_f1_band": f1_band,
            "descriptive_match": match,
        },
        "lightgbm_auroc": LIGHTGBM_AUROC,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
