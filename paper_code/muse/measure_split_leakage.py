"""Quantify exploitable patient overlap in the released MUSE splits.

Diagnostic only. Reports (a) how much of the test set shares a patient with
MUSE's and MUSE+'s training pools, and (b) how far a predictor with no
features at all gets using only patient identity and training labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

CACHE = Path(__file__).resolve().parent / ".export_assets" / "source_cache"


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="stable")
    y, s = y[order], s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    df = pl.read_parquet(CACHE / "cohort.parquet")
    tr = df.filter(pl.col("split") == "train")
    te = df.filter(pl.col("split") == "test")
    nl = df.filter(pl.col("split").is_null())
    tr_pat, te_pat, nl_pat = (set(x["subject_id"]) for x in (tr, te, nl))

    result = {
        "admissions": {"train": tr.height, "test": te.height, "unlabeled": nl.height},
        "patients": {"train": len(tr_pat), "test": len(te_pat), "unlabeled": len(nl_pat)},
        "test_patients_in_train": len(te_pat & tr_pat),
        "test_patients_in_unlabeled": len(te_pat & nl_pat),
        "test_patients_in_muse_training": len(te_pat & tr_pat),
        "test_patients_in_muse_plus_training": len(te_pat & (tr_pat | nl_pat)),
        "test_admissions_with_patient_in_train": te.filter(
            pl.col("subject_id").is_in(list(te_pat & tr_pat))).height,
    }
    for key in ("muse", "muse_plus"):
        seen = result[f"test_patients_in_{key}_training"]
        result[f"{key}_test_patient_coverage"] = seen / len(te_pat)

    # zero-feature predictor: the patient's own training-label rate
    rate = tr.group_by("subject_id").agg(pl.col("mortality").mean().alias("pat_rate"))
    joined = te.join(rate, on="subject_id", how="left")
    y = joined["mortality"].to_numpy().astype(int)
    prior = float(tr["mortality"].mean())
    result["test_prevalence"] = float(y.mean())
    result["leak_only_auroc"] = auroc(y, joined["pat_rate"].fill_null(prior).to_numpy())
    seen_rows = joined.filter(pl.col("pat_rate").is_not_null())
    result["leak_only_auroc_on_leaked_admissions"] = auroc(
        seen_rows["mortality"].to_numpy().astype(int),
        seen_rows["pat_rate"].to_numpy())
    result["published"] = {"muse": 0.9004, "muse_plus": 0.9201, "best_baseline_m3care": 0.8896}

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("split_leakage.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
