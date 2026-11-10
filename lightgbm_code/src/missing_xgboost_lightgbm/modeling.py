"""Joint Optuna search, LightGBM fitting, and clustered evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
from scipy import sparse
from sklearn.metrics import roc_auc_score


# Disposable-histogram budget for one trial. It is a per-trial figure, never a
# total divided across concurrent trials: when the pool cannot hold a node's
# histograms LightGBM derives them by subtracting a sibling from its parent
# instead of computing them directly, and the two routes round differently, so
# a divided total would let the number of concurrent trials change the fitted
# model. A fixed figure keeps the fit invariant to concurrency at any value.
#
# Eviction is not a penalty to avoid: the subtraction path it forces is cheaper
# than computing histograms directly, so raising the budget costs wall clock
# and buys nothing.
HISTOGRAM_POOL_PER_WORKER_MB = 256.0


@dataclass
class FitResult:
    best_iteration: int
    auroc: float
    probability: np.ndarray
    model: lgb.LGBMClassifier


def point_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    return {"auroc": float(roc_auc_score(y_true, probability))}


def recoverable_lightgbm_trial_failure(error: BaseException) -> bool:
    """Recognize LightGBM's sampled-split empty-child pathology exactly.

    Both executors fail only the affected trial on this message; anything
    else stays fatal.
    """
    if not isinstance(error, lgb.basic.LightGBMError):
        return False
    message = str(error)
    return any(
        f"Check failed: (best_split_info.{side}_count) > (0)" in message
        for side in ("left", "right")
    )


LOGIT_EPSILON = 1e-7


def ensemble_probability(predictions: list[np.ndarray]) -> np.ndarray:
    """Uniform logit-scale average of member probabilities (Caruana-style
    sorted-initialization ensemble of the validation top-k)."""
    if not predictions:
        raise ValueError("ensemble requires at least one member prediction")
    logits = [
        np.log(clipped / (1.0 - clipped))
        for clipped in (
            np.clip(
                np.asarray(prediction, dtype=np.float64),
                LOGIT_EPSILON,
                1.0 - LOGIT_EPSILON,
            )
            for prediction in predictions
        )
    ]
    return 1.0 / (1.0 + np.exp(-np.mean(logits, axis=0)))


def evaluation_callable(y_true: np.ndarray, probability: np.ndarray):
    return "auroc", float(roc_auc_score(y_true, probability)), True


def suggest_parameters(
    trial: optuna.trial.Trial,
    representations: tuple[str, ...],
    negative_to_positive_ratio: float,
) -> tuple[str, dict[str, Any]]:
    """Jointly sample representation and standard GBDT parameters."""
    representation = trial.suggest_categorical("representation", representations)
    parameters = {
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.20, log=True
        ),
        # Tree capacity is controlled by num_leaves alone, with max_depth a
        # fixed cap. The floors matter: selection takes the validation argmax,
        # which will happily pick a degenerate configuration -- boosted stumps,
        # or leaves holding a single sample -- that scores highest on
        # validation and worst on test.
        "num_leaves": trial.suggest_int("num_leaves", 8, 200, log=True),
        "max_depth": 12,
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 10, 200, log=True
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-3, 10.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.60, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.60, 1.0
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-8, 100.0, log=True
        ),
    }
    upper_weight = max(1.0, float(negative_to_positive_ratio))
    parameters["scale_pos_weight"] = (
        1.0
        if upper_weight == 1.0
        else trial.suggest_float(
            "scale_pos_weight", 1.0, upper_weight, log=True
        )
    )
    return representation, parameters


def fixed_parameters(seed: int, threads: int) -> dict[str, Any]:
    return {
        "boosting_type": "gbdt",
        "objective": "binary",
        "device_type": "cpu",
        "metric": "None",
        "n_estimators": 5000,
        "max_bin": 255,
        "histogram_pool_size": HISTOGRAM_POOL_PER_WORKER_MB,
        "min_split_gain": 0.0,
        "use_missing": True,
        # Absent measurements are unshown rather than stored as NaN, so an
        # unshown entry must read as missing. Statistics that can be a
        # legitimate 0.0 have that value vacated by the featurizer's
        # zero-vacating offset; statistics that fill 0.0 when absent are
        # exempt because missing and zero denote the same state there.
        "zero_as_missing": True,
        "feature_pre_filter": False,
        "is_unbalance": False,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "n_jobs": threads,
        "random_state": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
    }


def native_fixed_parameters(seed: int, threads: int) -> dict[str, Any]:
    """Fixed parameters for the low-level, reusable native Dataset path."""
    return {
        "boosting_type": "gbdt",
        "objective": "binary",
        "device_type": "cpu",
        "metric": "auc",
        "max_bin": 255,
        "histogram_pool_size": HISTOGRAM_POOL_PER_WORKER_MB,
        "min_split_gain": 0.0,
        "use_missing": True,
        # See fixed_parameters for why an unshown entry must read as missing.
        "zero_as_missing": True,
        "is_enable_sparse": True,
        "enable_bundle": True,
        "feature_pre_filter": False,
        "is_unbalance": False,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "num_threads": threads,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
    }


def native_trial_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Translate the stable Optuna search names to LightGBM's native names."""
    return {
        "learning_rate": parameters["learning_rate"],
        "num_leaves": parameters["num_leaves"],
        "max_depth": parameters["max_depth"],
        "min_data_in_leaf": parameters["min_child_samples"],
        "min_sum_hessian_in_leaf": parameters["min_child_weight"],
        "bagging_fraction": parameters["subsample"],
        "bagging_freq": parameters["subsample_freq"],
        "feature_fraction": parameters["colsample_bytree"],
        "lambda_l1": parameters["reg_alpha"],
        "lambda_l2": parameters["reg_lambda"],
        "scale_pos_weight": parameters["scale_pos_weight"],
    }


def fit_with_early_stopping(
    parameters: dict[str, Any],
    x_fit: np.ndarray | sparse.csr_matrix,
    y_fit: np.ndarray,
    x_validation: np.ndarray | sparse.csr_matrix,
    y_validation: np.ndarray,
    seed: int,
    threads: int,
) -> FitResult:
    model = lgb.LGBMClassifier(**fixed_parameters(seed, threads), **parameters)
    model.fit(
        x_fit,
        y_fit,
        eval_X=x_validation,
        eval_y=y_validation,
        eval_metric=evaluation_callable,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                first_metric_only=True,
                verbose=False,
                min_delta=0.0,
            )
        ],
    )
    best_iteration = int(model.best_iteration_)
    probability = model.predict_proba(
        x_validation, num_iteration=best_iteration
    )[:, 1]
    return FitResult(
        best_iteration,
        point_metrics(y_validation, probability)["auroc"],
        probability.astype(np.float32),
        model,
    )


def patient_clustered_bootstrap(
    labels: np.ndarray,
    probability: np.ndarray,
    group_ids: np.ndarray,
    replicates: int = 2000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    groups = np.unique(group_ids)
    rows = [np.flatnonzero(group_ids == group) for group in groups]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        draw = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([rows[index] for index in draw])
        if np.unique(labels[indices]).size == 2:
            values.append(point_metrics(labels[indices], probability[indices])["auroc"])
    if not values:
        raise ValueError("all bootstrap replicates contained one class")
    values = np.asarray(values)
    return {
        "auroc": {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "valid_replicates": int(len(values)),
        }
    }
