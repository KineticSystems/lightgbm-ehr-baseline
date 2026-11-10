"""Evaluate deliberately leaky same-patient temporal rules for MUSE.

This is a diagnostic oracle, not a candidate clinical model.  It joins test
admissions to the released admission-level training pools by ``subject_id``.
The primary variant applies only conclusions entailed by admission timing and
binary labels, then falls back to the existing structured LightGBM score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, roc_auc_score


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.common_format import (  # noqa: E402
    sha256_file,
    write_json,
)


TASK_ID = "mortality_90d_post_discharge"
WINDOW_DAYS = 90
PROTOCOL_VERSION = "muse-same-patient-temporal-cheat-v1"
REQUIRED_COHORT_COLUMNS = {
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "mortality",
    "split",
}


def _load_inputs(
    cohort_path: Path,
    structured_result_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    cohort = pl.read_parquet(cohort_path)
    missing = REQUIRED_COHORT_COLUMNS - set(cohort.columns)
    if missing:
        raise ValueError(f"cohort is missing columns: {sorted(missing)}")
    if cohort["hadm_id"].n_unique() != cohort.height:
        raise ValueError("cohort hadm_id is not unique")
    if cohort.select(
        pl.any_horizontal(
            pl.col("subject_id").is_null(),
            pl.col("hadm_id").is_null(),
            pl.col("admittime").is_null(),
            pl.col("dischtime").is_null(),
        ).any()
    ).item():
        raise ValueError("cohort identity or admission timestamps are missing")

    prediction_path = structured_result_dir / "predictions.npz"
    with np.load(prediction_path) as archive:
        required = {
            "test_probability",
            "test_labels",
            "test_group_id",
            "test_sample_id",
        }
        missing_arrays = required - set(archive.files)
        if missing_arrays:
            raise ValueError(
                f"structured predictions are missing: {sorted(missing_arrays)}"
            )
        sample_id = np.asarray(archive["test_sample_id"], dtype=np.int64)
        labels = np.asarray(archive["test_labels"], dtype=np.int8)
        group_id = np.asarray(archive["test_group_id"], dtype=np.int64)
        baseline = np.asarray(archive["test_probability"], dtype=np.float64)
    lengths = {len(sample_id), len(labels), len(group_id), len(baseline)}
    if len(lengths) != 1 or len(np.unique(sample_id)) != len(sample_id):
        raise ValueError("structured test arrays have inconsistent or duplicate rows")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("structured test labels are not binary")
    if not np.isfinite(baseline).all() or np.any((baseline < 0) | (baseline > 1)):
        raise ValueError("structured test probabilities are invalid")

    prediction_index = pl.DataFrame(
        {
            "prediction_position": np.arange(len(sample_id), dtype=np.int64),
            "hadm_id": sample_id,
            "archive_label": labels,
        }
    )
    test = prediction_index.join(
        cohort.filter(
            pl.col("split") == "test", pl.col("mortality").is_not_null()
        ).select(sorted(REQUIRED_COHORT_COLUMNS)),
        on="hadm_id",
        how="left",
        validate="1:1",
    ).sort("prediction_position")
    if test["subject_id"].null_count():
        raise ValueError("a structured test admission is absent from the MUSE cohort")
    reconstructed_labels = test["mortality"].cast(pl.Int8).to_numpy()
    if not np.array_equal(reconstructed_labels, labels):
        raise ValueError("structured labels disagree with the reconstructed cohort")
    reconstructed_groups = test["subject_id"].to_numpy()
    if not np.array_equal(reconstructed_groups, group_id):
        raise ValueError("structured patient IDs disagree with the reconstructed cohort")
    return cohort, test, labels, group_id, baseline


def _attach_training_evidence(
    cohort: pl.DataFrame,
    test: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    labeled = cohort.filter(
        pl.col("split") == "train", pl.col("mortality").is_not_null()
    )
    unlabeled = cohort.filter(pl.col("mortality").is_null())
    if set(labeled["hadm_id"].to_list()) & set(test["hadm_id"].to_list()):
        raise ValueError("an exact admission occurs in labeled train and test")
    if set(unlabeled["hadm_id"].to_list()) & set(test["hadm_id"].to_list()):
        raise ValueError("an exact admission occurs in unlabeled train and test")

    labeled_summary = labeled.group_by("subject_id").agg(
        pl.col("admittime").max().alias("latest_labeled_admittime"),
        pl.col("mortality").any().alias("has_positive_labeled_admission"),
    )
    positive_summary = (
        labeled.filter(pl.col("mortality") == True)  # noqa: E712
        .group_by("subject_id")
        .agg(
            pl.col("dischtime")
            .min()
            .alias("earliest_positive_labeled_dischtime")
        )
    )
    unlabeled_summary = unlabeled.group_by("subject_id").agg(
        pl.col("admittime").max().alias("latest_unlabeled_admittime")
    )
    enriched = (
        test.join(labeled_summary, on="subject_id", how="left", validate="m:1")
        .join(positive_summary, on="subject_id", how="left", validate="m:1")
        .join(unlabeled_summary, on="subject_id", how="left", validate="m:1")
        .with_columns(
            pl.col("has_positive_labeled_admission").fill_null(False)
        )
        .sort("prediction_position")
    )
    return enriched, {
        "labeled_admissions": labeled.height,
        "labeled_subjects": labeled["subject_id"].n_unique(),
        "labeled_positive_admissions": labeled.filter(
            pl.col("mortality") == True  # noqa: E712
        ).height,
        "unlabeled_admissions": unlabeled.height,
        "unlabeled_subjects": unlabeled["subject_id"].n_unique(),
    }


def _mask_counts(mask: np.ndarray, labels: np.ndarray) -> dict[str, int | float]:
    count = int(mask.sum())
    positive = int(labels[mask].sum())
    return {
        "count": count,
        "fraction_of_test": count / len(labels),
        "positive_test": positive,
        "negative_test": count - positive,
    }


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "accuracy_at_0_5": float(accuracy_score(labels, probability >= 0.5)),
    }


def _evaluate_rule_set(
    *,
    labels: np.ndarray,
    baseline: np.ndarray,
    survival_rule: np.ndarray,
    death_rule: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if survival_rule.dtype != np.bool_ or death_rule.dtype != np.bool_:
        raise TypeError("rule masks must be boolean")
    conflict = survival_rule & death_rule
    # Survival beyond the target window is direct timestamp evidence, so it
    # takes precedence for the intentionally over-broad naive death rule.
    active_death = death_rule & ~survival_rule
    overridden = survival_rule | active_death
    probability = baseline.copy()
    probability[survival_rule] = 0.0
    probability[active_death] = 1.0
    correctness = (probability[overridden] >= 0.5) == labels[overridden]
    result: dict[str, Any] = {
        "test": _metrics(labels, probability),
        "survival_rule": {
            **_mask_counts(survival_rule, labels),
            "correct_fraction": float(np.mean(labels[survival_rule] == 0))
            if survival_rule.any()
            else None,
        },
        "death_rule_before_precedence": {
            **_mask_counts(death_rule, labels),
            "correct_fraction": float(np.mean(labels[death_rule] == 1))
            if death_rule.any()
            else None,
        },
        "conflicts_resolved_as_survival": _mask_counts(conflict, labels),
        "active_death_rule": {
            **_mask_counts(active_death, labels),
            "correct_fraction": float(np.mean(labels[active_death] == 1))
            if active_death.any()
            else None,
        },
        "all_overrides": {
            **_mask_counts(overridden, labels),
            "correct_fraction": float(np.mean(correctness))
            if overridden.any()
            else None,
            "baseline_accuracy_at_0_5_on_overridden_rows": float(
                np.mean((baseline[overridden] >= 0.5) == labels[overridden])
            )
            if overridden.any()
            else None,
        },
        "fallback": _mask_counts(~overridden, labels),
    }
    fallback = ~overridden
    if fallback.any() and len(np.unique(labels[fallback])) == 2:
        result["fallback"]["baseline_auroc"] = float(
            roc_auc_score(labels[fallback], baseline[fallback])
        )
    return probability, result


def _evaluate_variant(
    *,
    test: pl.DataFrame,
    labels: np.ndarray,
    baseline: np.ndarray,
    include_unlabeled: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if include_unlabeled:
        latest_admission = pl.max_horizontal(
            "latest_labeled_admittime", "latest_unlabeled_admittime"
        )
        pool = "labeled train plus every missing-label auxiliary admission"
    else:
        latest_admission = pl.col("latest_labeled_admittime")
        pool = "labeled train only"
    cutoff = pl.col("dischtime") + pl.duration(days=WINDOW_DAYS)
    flags = test.select(
        (latest_admission > cutoff)
        .fill_null(False)
        .alias("survival_beyond_window"),
        (
            pl.col("earliest_positive_labeled_dischtime")
            <= pl.col("dischtime")
        )
        .fill_null(False)
        .alias("entailed_death_within_window"),
        pl.col("has_positive_labeled_admission")
        .fill_null(False)
        .alias("naive_any_positive_label"),
    )
    survival = flags["survival_beyond_window"].to_numpy()
    entailed_death = flags["entailed_death_within_window"].to_numpy()
    naive_death = flags["naive_any_positive_label"].to_numpy()
    entailed_probability, entailed_report = _evaluate_rule_set(
        labels=labels,
        baseline=baseline,
        survival_rule=survival,
        death_rule=entailed_death,
    )
    naive_probability, naive_report = _evaluate_rule_set(
        labels=labels,
        baseline=baseline,
        survival_rule=survival,
        death_rule=naive_death,
    )
    return (
        {
            "survival_rule": survival,
            "entailed_death_rule": entailed_death,
            "naive_death_rule": naive_death,
            "entailed_probability": entailed_probability,
            "naive_probability": naive_probability,
        },
        {
            "accessible_pool": pool,
            "entailed_rules": entailed_report,
            "naive_user_proposed_death_rule": {
                **naive_report,
                "logical_status": (
                    "not entailed: a positive label for a later admission can "
                    "refer to a death after the current test window"
                ),
            },
        },
    )


def run(
    cohort_path: Path,
    structured_result_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort, test, labels, group_id, baseline = _load_inputs(
        cohort_path, structured_result_dir
    )
    test, pool_counts = _attach_training_evidence(cohort, test)
    arrays: dict[str, np.ndarray] = {}
    reports: dict[str, Any] = {}
    for name, include_unlabeled in (
        ("labeled_only", False),
        ("labeled_plus_unlabeled", True),
    ):
        variant_arrays, reports[name] = _evaluate_variant(
            test=test,
            labels=labels,
            baseline=baseline,
            include_unlabeled=include_unlabeled,
        )
        arrays.update(
            {f"{name}_{key}": value for key, value in variant_arrays.items()}
        )

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "paper": "muse",
        "task_id": TASK_ID,
        "diagnostic_only": True,
        "interpretation": (
            "Deliberately joins admissions by subject_id and uses absolute "
            "calendar time. MUSE does not receive either as an input feature, "
            "so this quantifies exploitable information in the split rather "
            "than reproducing the released neural model."
        ),
        "prediction_time": "test admission discharge",
        "target_window_days": WINDOW_DAYS,
        "precedence": "survival beyond the target window, then death, then LightGBM",
        "rules": {
            "survival": (
                "predict 0 when any accessible same-patient training admission "
                "begins more than 90 days after the test discharge"
            ),
            "entailed_death": (
                "predict 1 when a labeled same-patient training admission has "
                "a positive 90-day mortality label and discharged no later "
                "than the test admission"
            ),
            "fallback": "retain the structured LightGBM probability",
            "naive_death_audit": (
                "also evaluate the proposed any-positive-label rule, but do not "
                "claim it is logically entailed"
            ),
        },
        "inputs": {
            "cohort": str(cohort_path),
            "cohort_sha256": sha256_file(cohort_path),
            "structured_result_dir": str(structured_result_dir),
            "structured_predictions_sha256": sha256_file(
                structured_result_dir / "predictions.npz"
            ),
            "structured_manifest_sha256": sha256_file(
                structured_result_dir / "manifest.json"
            ),
        },
        "pool_counts": pool_counts,
        "test_counts": {
            "admissions": len(labels),
            "subjects": int(len(np.unique(group_id))),
            "positive": int(labels.sum()),
            "negative": int(len(labels) - labels.sum()),
        },
        "structured_lightgbm": {"test": _metrics(labels, baseline)},
        "variants": reports,
    }
    np.savez_compressed(
        output_dir / "predictions.npz",
        test_sample_id=test["hadm_id"].to_numpy(),
        test_group_id=group_id,
        test_label=labels,
        baseline_probability=baseline,
        **arrays,
    )
    write_json(output_dir / "manifest.json", result)
    write_json(
        output_dir / "result.json",
        {
            "paper": "muse",
            "task_id": TASK_ID,
            "sensitivity": "same_patient_temporal_cheat",
            "structured_lightgbm_auroc": result["structured_lightgbm"]["test"][
                "auroc"
            ],
            "labeled_only_entailed_auroc": reports["labeled_only"][
                "entailed_rules"
            ]["test"]["auroc"],
            "labeled_plus_unlabeled_entailed_auroc": reports[
                "labeled_plus_unlabeled"
            ]["entailed_rules"]["test"]["auroc"],
            "labeled_plus_unlabeled_naive_auroc": reports[
                "labeled_plus_unlabeled"
            ]["naive_user_proposed_death_rule"]["test"]["auroc"],
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--structured-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cohort, args.structured_result_dir, args.output_dir)
    print(json.dumps(result["variants"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
