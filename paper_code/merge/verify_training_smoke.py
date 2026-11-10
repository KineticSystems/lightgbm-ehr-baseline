#!/usr/bin/env python3
"""Independently validate MERGE's exact one-epoch training smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT / "mimiciv"))

from utils.checkpoint_utils import create_model_from_checkpoint  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_close(name: str, actual: float, expected: float) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
        raise AssertionError(f"{name} differs: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--validation-pickle", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--rus-report", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--evaluation-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_report = json.loads(args.input_report.read_text())
    rus_report = json.loads(args.rus_report.read_text())
    if input_report.get("status") != "exact" or rus_report.get("status") != "exact":
        raise AssertionError("smoke inputs are not independently verified")
    if sha256_file(args.validation_pickle) != input_report["artifacts"]["pickles"]["validation"]:
        raise AssertionError("validation pickle differs from the verified input")

    torch.serialization.add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != 0:
        raise AssertionError("smoke checkpoint is not epoch zero")
    train_args = checkpoint["args"]
    expected_args = {
        "task": "ihm",
        "seq_len": 48,
        "truncate_from_end": True,
        "batch_size": 512,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "epochs": 1,
        "seed": 0,
        "lambda_u": 1.0,
        "lambda_r": 1.0,
        "lambda_s": 1.0,
        "lambda_load": 0.02,
        "d_model": 32,
        "nhead": 4,
        "d_ff": 64,
        "num_encoder_layers": 6,
        "num_moe_layers": 2,
        "modality_encoder_layers": 2,
        "moe_num_experts": 8,
        "moe_num_synergy_experts": 2,
        "moe_k": 2,
    }
    for name, expected in expected_args.items():
        actual = getattr(train_args, name)
        if isinstance(expected, float):
            require_close(f"checkpoint arg {name}", float(actual), expected)
        elif actual != expected:
            raise AssertionError(f"checkpoint arg {name} differs: {actual} != {expected}")

    modality_names = list(checkpoint["modality_names"])
    if modality_names != ["cxr", "labs_vitals", "notes"]:
        raise AssertionError(f"modality order changed: {modality_names}")
    modality_configs = checkpoint["modality_configs"]
    if [int(config["input_dim"]) for config in modality_configs] != [1024, 30, 768]:
        raise AssertionError("modality input dimensions changed")
    state_dict = checkpoint["model_state_dict"]
    for name, tensor in state_dict.items():
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"checkpoint tensor {name} is nonfinite")
    model = create_model_from_checkpoint(
        state_dict, train_args, modality_configs, num_classes=2, device=torch.device("cpu")
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model

    with args.validation_pickle.open("rb") as handle:
        stays = pickle.load(handle)
    expected_labels = np.asarray(
        [int(stay["label"]) for stay in stays if len(stay["ts_tt"]) > 12],
        dtype=np.int64,
    )
    del stays
    if len(expected_labels) != input_report["splits"]["validation"]["postfilter_count"]:
        raise AssertionError("validation label count changed")

    archive = np.load(args.predictions)
    predictions = archive["predictions"]
    probabilities = archive["probabilities"]
    labels = archive["labels"]
    if not np.array_equal(labels, expected_labels):
        raise AssertionError("serialized validation label order changed")
    if probabilities.shape != (len(labels), 2) or not np.isfinite(probabilities).all():
        raise AssertionError("serialized validation probabilities are invalid")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise AssertionError("serialized class probabilities do not sum to one")
    if not np.array_equal(predictions, np.argmax(probabilities, axis=1)):
        raise AssertionError("serialized predictions are not probability argmax")

    recomputed = {
        "auroc": float(roc_auc_score(labels, probabilities[:, 1])),
        "auprc": float(average_precision_score(labels, probabilities[:, 1])),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "accuracy": float(accuracy_score(labels, predictions)),
    }
    released_metrics = json.loads(args.metrics.read_text())
    require_close("released AUROC", float(released_metrics["auc_score"]), recomputed["auroc"])
    require_close("released weighted F1", float(released_metrics["f1_score"]), recomputed["weighted_f1"])
    require_close("released accuracy", float(released_metrics["accuracy"]), recomputed["accuracy"])
    require_close("checkpoint best validation AUROC", float(checkpoint["best_val_auc"]), recomputed["auroc"])

    report = {
        "status": "passed",
        "scope": "one exact released epoch followed by validation-only checkpoint evaluation",
        "test_loader_accesses": 0,
        "seed": 0,
        "checkpoint_epoch": 0,
        "checkpoint_best_validation_auroc": float(checkpoint["best_val_auc"]),
        "parameter_count": parameter_count,
        "state_keys": len(state_dict),
        "validation": {
            "count": len(labels),
            "positives": int(labels.sum()),
            **recomputed,
        },
        "artifacts": {
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "predictions_sha256": sha256_file(args.predictions),
            "metrics_sha256": sha256_file(args.metrics),
            "train_log_sha256": sha256_file(args.train_log),
            "evaluation_log_sha256": sha256_file(args.evaluation_log),
            "rus_sha256": rus_report["artifacts"]["rus_sha256"],
            "input_report_sha256": sha256_file(args.input_report),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
