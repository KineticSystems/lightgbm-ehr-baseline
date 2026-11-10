#!/usr/bin/env python3
"""Aggregate the five independently verified Step-wise primary seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SEEDS = [1111, 2222, 3333, 4444, 5555]
PAPER = {"auroc": 0.916, "auprc": 0.387}
BANDS = {"auroc": [0.915, 0.917], "auprc": [0.384, 0.390]}
LIGHTGBM_AUROC = 0.9211997005347095


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary-root",
        type=Path,
        default=Path(".export_assets/reproduction/primary"),
    )
    args = parser.parse_args()

    output_path = args.primary_root / "summary.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    reports = []
    for seed in SEEDS:
        report_path = args.primary_root / str(seed) / "verification" / "report.json"
        report = json.loads(report_path.read_text())
        if report.get("status") != "passed" or report.get("seed") != seed:
            raise AssertionError(f"seed {seed} is not independently verified")
        if report.get("source_test_passes") != 1:
            raise AssertionError(f"seed {seed} source test access differs")
        if report.get("independent_audit_test_passes") != 1:
            raise AssertionError(f"seed {seed} audit test access differs")
        reports.append(report)

    metrics = {}
    for metric in ("auroc", "auprc"):
        values = np.asarray(
            [report["independent_metrics"]["test"][metric] for report in reports],
            dtype=np.float64,
        )
        mean = float(values.mean())
        population_sd = float(values.std(ddof=0))
        sample_sd = float(values.std(ddof=1))
        metrics[metric] = {
            "values": values.tolist(),
            "mean": mean,
            "population_sd": population_sd,
            "sample_sd": sample_sd,
            "sem": sample_sd / math.sqrt(len(values)),
            "paper": PAPER[metric],
            "paper_band": BANDS[metric],
            "signed_gap": mean - PAPER[metric],
            "inside_paper_band": BANDS[metric][0] <= mean <= BANDS[metric][1],
        }

    report = {
        "status": "completed_match"
        if all(metrics[name]["inside_paper_band"] for name in metrics)
        else "completed_nonmatch",
        "fidelity": "bounded_reconstructed_configuration_retraining",
        "seeds": SEEDS,
        "metrics": metrics,
        "lightgbm_auroc": LIGHTGBM_AUROC,
        "reproduction_minus_lightgbm_auroc": metrics["auroc"]["mean"]
        - LIGHTGBM_AUROC,
        "source_test_passes": len(SEEDS),
        "independent_audit_test_passes": len(SEEDS),
        "total_post_selection_test_passes": 2 * len(SEEDS),
        "preserved_source_defect": (
            "faithful target row consumes through one hour after the nominal "
            "MIMIC-III Benchmark cutoff"
        ),
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
