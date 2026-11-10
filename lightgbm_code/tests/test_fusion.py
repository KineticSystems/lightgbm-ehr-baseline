import numpy as np

from missing_xgboost_lightgbm.fusion import (
    LOGISTIC_C_GRID,
    fit_logistic_stacker,
    fit_tfidf_logistic,
    late_fusion_features,
    predict_probability,
)


def test_tfidf_vocabulary_is_fit_on_training_documents_only():
    documents = ["normal clear lungs"] * 6 + ["severe pulmonary edema"] * 6
    labels = np.asarray([0] * 6 + [1] * 6)
    validation_documents = ["normal leakagetoken"] * 3 + ["edema leakagetoken"] * 3
    validation_labels = np.asarray([0] * 3 + [1] * 3)
    estimator, selection = fit_tfidf_logistic(
        documents,
        labels,
        validation_documents,
        validation_labels,
        seed=1,
    )
    vocabulary = estimator.named_steps["tfidf"].vocabulary_
    classifier = estimator.named_steps["classifier"]
    assert "normal" in vocabulary
    assert "edema" in vocabulary
    assert "leakagetoken" not in vocabulary
    probability = predict_probability(
        estimator, ["normal leakagetoken", "edema leakagetoken"]
    )
    assert probability.shape == (2,)
    assert classifier.C in LOGISTIC_C_GRID
    assert selection["selected_C"] == classifier.C
    assert len(selection["candidates"]) == len(LOGISTIC_C_GRID)
    assert selection["tie_break"] == "smallest C"

def test_logistic_stacker_uses_two_finite_log_odds_features():
    structured = np.linspace(0.01, 0.99, 20)
    text = np.asarray([0.1, 0.2] * 5 + [0.8, 0.9] * 5)
    labels = np.asarray([0] * 10 + [1] * 10)
    features = late_fusion_features(structured, text)
    assert features.shape == (20, 2)
    assert np.isfinite(features).all()
    estimator = fit_logistic_stacker(features, labels)
    assert estimator.predict_proba(features).shape == (20, 2)
    assert np.isinf(estimator.C)
    assert estimator.coef_.shape == (1, 2)
    assert estimator.intercept_.shape == (1,)
