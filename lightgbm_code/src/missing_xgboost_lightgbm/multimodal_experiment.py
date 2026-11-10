"""Stack structured LightGBM and raw-note TF-IDF logistic predictions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common_format import open_task, sha256_file, sha256_strings, write_json
from .experiment import (
    BOOTSTRAP_REPLICATES,
    MODEL_SEED,
    _audit_inventory,
    _validate_partitions,
)
from .fusion import (
    LOGISTIC_C_GRID,
    TFIDF_CONTRACT,
    fit_logistic_stacker,
    fit_tfidf_logistic,
    late_fusion_features,
    predict_probability,
)
from .modeling import patient_clustered_bootstrap, point_metrics


PROTOCOL_VERSION = "tfidf-logistic-stacker-v5"


def _log(message: str, start: float) -> None:
    print(f"[{time.monotonic() - start:8.1f}s] {message}", flush=True)


def _load_structured_manifest(
    source,
    structured_result_dir: Path,
) -> dict[str, Any]:
    manifest_path = structured_result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("task_id") != source.task_id:
        raise ValueError("structured result task does not match the MEDS task")
    if manifest.get("paper", {}).get("key") != source.manifest["paper"]["key"]:
        raise ValueError("structured result paper does not match the MEDS task")
    if manifest.get("source_manifest_sha256") != sha256_file(
        source.root / "manifest.json"
    ):
        raise ValueError("structured result was produced from a different MEDS export")
    predictions = structured_result_dir / "predictions.npz"
    if not predictions.is_file():
        raise FileNotFoundError(f"structured predictions are missing: {predictions}")
    return manifest


def _load_structured_probability(
    structured_result_dir: Path,
    split: str,
    partition: dict[str, np.ndarray],
) -> np.ndarray:
    with np.load(structured_result_dir / "predictions.npz") as archive:
        if not np.array_equal(
            archive[f"{split}_sample_id"], partition["sample_id"]
        ):
            raise ValueError(f"structured {split} sample order differs")
        if not np.array_equal(archive[f"{split}_labels"], partition["y"]):
            raise ValueError(f"structured {split} labels differ")
        probability = np.asarray(
            archive[f"{split}_probability"], dtype=np.float64
        )
    if probability.shape != partition["y"].shape or not np.isfinite(
        probability
    ).all():
        raise ValueError(f"structured {split} probabilities are invalid")
    return probability


def _evaluated(
    labels: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    return {
        "test": point_metrics(labels, probability),
        "patient_clustered_95pct_ci": patient_clustered_bootstrap(
            labels,
            probability,
            groups,
            replicates=bootstrap_replicates,
            seed=0,
        ),
    }


def run(
    task_dir: Path,
    structured_result_dir: Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Fit the fixed note model and validation-fit stacker."""
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    start = time.monotonic()
    source = open_task(task_dir)
    export_audits = _audit_inventory(source)
    if source.modalities != ("structured", "notes"):
        raise ValueError("stacking requires a structured-plus-notes MEDS task")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    structured_manifest = _load_structured_manifest(
        source, structured_result_dir
    )

    partitions = {
        split: source.load(split) for split in ("train", "validation")
    }
    _validate_partitions(source.task_id, partitions)
    documents = {
        split: source.load_documents(split)
        for split in ("train", "validation")
    }
    _log("selecting training-fit TF-IDF logistic model on validation AUROC", start)
    text_model, text_selection = fit_tfidf_logistic(
        documents["train"],
        partitions["train"]["y"],
        documents["validation"],
        partitions["validation"]["y"],
        seed=MODEL_SEED,
    )
    text_probability = {
        "validation": predict_probability(
            text_model, documents["validation"]
        ),
    }
    structured_probability = {
        "validation": _load_structured_probability(
            structured_result_dir,
            "validation",
            partitions["validation"],
        )
    }
    validation_fusion = late_fusion_features(
        structured_probability["validation"], text_probability["validation"]
    )
    stacker = fit_logistic_stacker(
        validation_fusion,
        partitions["validation"]["y"],
    )

    _log("stacker fit complete; loading sealed test", start)
    partitions["test"] = source.load("test")
    _validate_partitions(source.task_id, partitions)
    documents["test"] = source.load_documents("test")
    text_probability["test"] = predict_probability(
        text_model, documents["test"]
    )
    structured_probability["test"] = _load_structured_probability(
        structured_result_dir,
        "test",
        partitions["test"],
    )
    test_fusion = late_fusion_features(
        structured_probability["test"], text_probability["test"]
    )
    stacker_probability = stacker.predict_proba(test_fusion)[:, 1]

    vocabulary = text_model.named_steps["tfidf"].vocabulary_
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "paper": source.manifest["paper"],
        "task_id": source.task_id,
        "primary_metric": "AUROC",
        "export_audits": export_audits,
        "text_model": {
            "learner": "word TF-IDF plus balanced L2 logistic regression",
            "fit_split": "train",
            "vocabulary_fit_split": "train",
            "vocabulary_size": len(vocabulary),
            "vocabulary_sha256": sha256_strings(sorted(vocabulary)),
            "contract": {
                "lowercase": True,
                "ngram_range": [1, 2],
                "min_df": TFIDF_CONTRACT["min_df"],
                "max_df": TFIDF_CONTRACT["max_df"],
                "max_features": TFIDF_CONTRACT["max_features"],
                "sublinear_tf": True,
                "norm": "l2",
                "C_grid": list(LOGISTIC_C_GRID),
                "selected_C": text_selection["selected_C"],
                "class_weight": "balanced",
                "solver": "liblinear",
                "selection_metric": "AUROC",
                "selection_split": "validation",
                "cross_validation": False,
                "candidate_fit_split": "train",
                "tie_break": text_selection["tie_break"],
            },
            "selection": text_selection,
            "document_sha256": {
                split: sha256_strings(values)
                for split, values in documents.items()
            },
            "empty_documents": {
                split: sum(not value.strip() for value in values)
                for split, values in documents.items()
            },
            "validation_auroc": text_selection["selected_validation_auroc"],
            "test": point_metrics(
                partitions["test"]["y"], text_probability["test"]
            ),
        },
        "structured_model": {
            "result_directory": str(structured_result_dir),
            "protocol_version": structured_manifest["protocol_version"],
            "selected_representation": structured_manifest["selection"][
                "selected_representation"
            ],
            "validation_prediction_source": "predictions.npz",
            "validation_auroc": point_metrics(
                partitions["validation"]["y"],
                structured_probability["validation"],
            )["auroc"],
            "test": point_metrics(
                partitions["test"]["y"], structured_probability["test"]
            ),
        },
        "stacker": {
            "features": ["structured_log_odds", "text_log_odds"],
            "fit_split": "validation",
            "fit_rows": len(partitions["validation"]["y"]),
            "learner": "unregularized logistic regression",
            "penalty": "none (C=infinity)",
            "solver": "lbfgs",
            "fitted_parameter_count": 3,
            "hyperparameter_selection": None,
            "coefficients": stacker.coef_[0].tolist(),
            "intercept": float(stacker.intercept_[0]),
            **_evaluated(
                partitions["test"]["y"],
                stacker_probability,
                partitions["test"]["group_id"],
                bootstrap_replicates,
            ),
        },
    }
    np.savez_compressed(
        output_dir / "predictions.npz",
        test_label=partitions["test"]["y"],
        test_group_id=partitions["test"]["group_id"],
        test_sample_id=partitions["test"]["sample_id"],
        text_probability=text_probability["test"],
        structured_probability=structured_probability["test"],
        stacked_probability=stacker_probability,
        validation_text_probability=text_probability["validation"],
        validation_structured_probability=structured_probability["validation"],
    )
    write_json(output_dir / "manifest.json", result)
    write_json(
        output_dir / "result.json",
        {
            "paper": source.manifest["paper"]["key"],
            "task_id": source.task_id,
            "stacked_auroc": result["stacker"]["test"]["auroc"],
        },
    )
    _log("stacking experiment complete", start)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--structured-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.task_dir, args.structured_result_dir, args.output_dir)


if __name__ == "__main__":
    main()
