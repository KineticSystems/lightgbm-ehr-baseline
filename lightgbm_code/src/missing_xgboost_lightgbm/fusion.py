"""Shared sparse-note learner and prediction-level fusion primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


TFIDF_CONTRACT: dict[str, Any] = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.98,
    "max_features": 50_000,
    "sublinear_tf": True,
    "norm": "l2",
    "dtype": np.float64,
}
LOGISTIC_C_GRID = tuple(float(10**power) for power in range(-4, 5))


def _logistic(c: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        solver="liblinear",
        class_weight="balanced",
        max_iter=2_000,
        random_state=seed,
    )


def fit_tfidf_logistic(
    training_documents: list[str],
    training_labels: np.ndarray,
    validation_documents: list[str],
    validation_labels: np.ndarray,
    *,
    seed: int,
) -> tuple[Pipeline, dict[str, Any]]:
    """Select one training-fit note model solely by validation AUROC."""
    training_labels = np.asarray(training_labels, dtype=np.int8)
    validation_labels = np.asarray(validation_labels, dtype=np.int8)
    if len(training_documents) != len(training_labels) or len(
        validation_documents
    ) != len(validation_labels):
        raise ValueError("document and label counts differ")
    if np.unique(training_labels).size != 2 or np.unique(validation_labels).size != 2:
        raise ValueError("text training and validation labels must contain both classes")

    vectorizer = TfidfVectorizer(**TFIDF_CONTRACT)
    training_matrix = vectorizer.fit_transform(training_documents)
    validation_matrix = vectorizer.transform(validation_documents)
    candidates = []
    best_classifier = None
    best_auroc = -np.inf
    for c in LOGISTIC_C_GRID:
        classifier = _logistic(c, seed).fit(training_matrix, training_labels)
        probability = classifier.predict_proba(validation_matrix)[:, 1]
        auroc = float(roc_auc_score(validation_labels, probability))
        candidates.append({"C": c, "validation_auroc": auroc})
        # The grid is ascending, so exact ties choose the more regularized model.
        if auroc > best_auroc:
            best_auroc = auroc
            best_classifier = classifier
    if best_classifier is None:  # pragma: no cover - the fixed grid is nonempty
        raise RuntimeError("the note-model grid produced no candidate")
    estimator = Pipeline(
        steps=[("tfidf", vectorizer), ("classifier", best_classifier)]
    )
    return estimator, {
        "selected_C": float(best_classifier.C),
        "selected_validation_auroc": best_auroc,
        "candidates": candidates,
        "tie_break": "smallest C",
    }


def predict_probability(estimator: Any, documents: list[str]) -> np.ndarray:
    probability = np.asarray(estimator.predict_proba(documents)[:, 1], dtype=np.float64)
    if probability.shape != (len(documents),) or not np.isfinite(probability).all():
        raise ValueError("text model returned invalid probabilities")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("text probabilities lie outside [0, 1]")
    return probability


def late_fusion_features(
    structured_probability: np.ndarray, text_probability: np.ndarray
) -> np.ndarray:
    """Convert two base probabilities into stable log-odds features."""
    structured_probability = np.asarray(structured_probability, dtype=np.float64)
    text_probability = np.asarray(text_probability, dtype=np.float64)
    if structured_probability.ndim != 1 or text_probability.ndim != 1:
        raise ValueError("base probabilities must be one-dimensional")
    if structured_probability.shape != text_probability.shape:
        raise ValueError("base probability shapes differ")
    if not (
        np.isfinite(structured_probability).all()
        and np.isfinite(text_probability).all()
    ):
        raise ValueError("base probabilities must be finite")
    if np.any(
        (structured_probability < 0.0)
        | (structured_probability > 1.0)
        | (text_probability < 0.0)
        | (text_probability > 1.0)
    ):
        raise ValueError("base probabilities lie outside [0, 1]")
    epsilon = 1e-6
    clipped = [
        np.clip(values, epsilon, 1.0 - epsilon)
        for values in (structured_probability, text_probability)
    ]
    return np.column_stack(
        [np.log(values / (1.0 - values)) for values in clipped]
    )


def fit_logistic_stacker(
    features: np.ndarray, labels: np.ndarray
) -> LogisticRegression:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("stacker input must contain exactly two columns")
    if len(features) != len(labels) or np.unique(labels).size != 2:
        raise ValueError("invalid stacker training rows or labels")
    return LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=2_000,
    ).fit(features, labels)
