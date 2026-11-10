import numpy as np
import optuna

from missing_xgboost_lightgbm.modeling import (
    fixed_parameters,
    fit_with_early_stopping,
    native_fixed_parameters,
    native_trial_parameters,
    patient_clustered_bootstrap,
    suggest_parameters,
)


def _binary_data(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(200, 6)).astype(np.float32)
    x[rng.random(x.shape) < 0.1] = np.nan
    y = (np.nan_to_num(x[:, 0]) + 0.5 * np.nan_to_num(x[:, 1]) > 0).astype(int)
    return x, y


def test_joint_parameters_include_representation_and_native_missing_support():
    trial = optuna.trial.FixedTrial(
        {
            "representation": "temporal",
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "min_child_weight": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": 2.0,
        }
    )
    representation, parameters = suggest_parameters(
        trial, ("summary", "temporal"), negative_to_positive_ratio=3.0
    )

    assert representation == "temporal"
    assert parameters["scale_pos_weight"] == 2.0


def test_fixed_parameters_bound_lightgbm_histogram_cache():
    parameters = fixed_parameters(seed=0, threads=1)

    assert parameters["histogram_pool_size"] == 256.0
    assert parameters["force_col_wise"] is True


def test_native_parameters_enable_sparse_missing_aware_lightgbm():
    fixed = native_fixed_parameters(seed=7, threads=3)
    assert fixed["metric"] == "auc"
    assert fixed["is_enable_sparse"] is True
    assert fixed["enable_bundle"] is True
    assert fixed["use_missing"] is True
    assert fixed["zero_as_missing"] is True
    assert fixed["num_threads"] == 3
    assert fixed["seed"] == 7

    translated = native_trial_parameters(
        {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "min_child_weight": 0.1,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.2,
            "reg_lambda": 1.5,
            "scale_pos_weight": 2.0,
        }
    )
    assert translated["min_data_in_leaf"] == 20
    assert translated["bagging_fraction"] == 0.8
    assert translated["feature_fraction"] == 0.75


def test_early_stopping_uses_auroc():
    x, y = _binary_data()
    parameters = {
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 10,
        "min_child_weight": 0.01,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
    }
    selected = fit_with_early_stopping(
        parameters, x[:140], y[:140], x[140:], y[140:], seed=0, threads=1
    )
    assert selected.best_iteration > 0
    assert selected.auroc > 0.8
    assert set(selected.model.evals_result_["valid_0"]) == {"auroc"}


def test_patient_bootstrap_resamples_labels_and_probabilities_together():
    labels = np.array([0, 0, 1, 1, 0, 1])
    probability = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    group_ids = np.array([1, 1, 2, 3, 4, 4])

    interval = patient_clustered_bootstrap(
        labels, probability, group_ids, replicates=25, seed=0
    )

    assert interval["auroc"]["valid_replicates"] > 0
    assert 0 <= interval["auroc"]["lower"] <= interval["auroc"]["upper"] <= 1
