#!/usr/bin/env python3
"""Independently verify one MERGE primary seed from its saved artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--test-pickle", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--rus-report", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.seed not in range(5):
        raise AssertionError("unexpected primary seed")
    input_report = json.loads(args.input_report.read_text())
    rus_report = json.loads(args.rus_report.read_text())
    if input_report.get("status") != "exact" or rus_report.get("status") != "exact":
        raise AssertionError("primary inputs are not independently verified")
    if sha256_file(args.test_pickle) != input_report["artifacts"]["pickles"]["test"]:
        raise AssertionError("test pickle differs from the verified input")

    torch.serialization.add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(checkpoint["epoch"])
    if checkpoint_epoch not in range(20):
        raise AssertionError("checkpoint epoch is outside the frozen 20-epoch run")
    train_args = checkpoint["args"]
    expected_args = {
        "task": "ihm",
        "seq_len": 48,
        "truncate_from_end": True,
        "d_model": 32,
        "nhead": 4,
        "d_ff": 64,
        "num_encoder_layers": 6,
        "num_moe_layers": 2,
        "dropout": 0.1,
        "modality_encoder_layers": 2,
        "use_cnn_encoders": False,
        "moe_num_experts": 8,
        "moe_num_synergy_experts": 2,
        "moe_k": 2,
        "moe_expert_hidden_dim": 128,
        "moe_capacity_factor": 1.25,
        "moe_drop_tokens": False,
        "moe_router_gru_hidden_dim": 32,
        "moe_router_token_processed_dim": 32,
        "moe_router_attn_key_dim": 16,
        "moe_router_attn_value_dim": 16,
        "epochs": 20,
        "batch_size": 512,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "clip_grad_norm": 1.0,
        "use_lr_scheduler": False,
        "use_gradient_checkpointing": False,
        "threshold_u": 0.01,
        "threshold_r": 0.01,
        "threshold_s": 0.01,
        "lambda_u": 1.0,
        "lambda_r": 1.0,
        "lambda_s": 1.0,
        "lambda_load": 0.02,
        "epsilon_loss": 1e-8,
        "seed": args.seed,
        "num_workers": 0,
        "use_wandb": False,
        "plot_expert_activations": False,
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
        state_dict,
        train_args,
        modality_configs,
        num_classes=2,
        device=torch.device("cpu"),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model

    with args.test_pickle.open("rb") as handle:
        stays = pickle.load(handle)
    expected_labels = np.asarray(
        [int(stay["label"]) for stay in stays if len(stay["ts_tt"]) > 12],
        dtype=np.int64,
    )
    del stays
    expected_split = input_report["splits"]["test"]
    if len(expected_labels) != expected_split["postfilter_count"]:
        raise AssertionError("test label count changed")
    if int(expected_labels.sum()) != expected_split["positive"]:
        raise AssertionError("test positive count changed")

    archive = np.load(args.predictions)
    predictions = archive["predictions"]
    probabilities = archive["probabilities"]
    labels = archive["labels"]
    if not np.array_equal(labels, expected_labels):
        raise AssertionError("serialized test label order changed")
    if probabilities.shape != (len(labels), 2) or not np.isfinite(probabilities).all():
        raise AssertionError("serialized test probabilities are invalid")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise AssertionError("serialized class probabilities do not sum to one")
    if not np.array_equal(predictions, np.argmax(probabilities, axis=1)):
        raise AssertionError("serialized predictions are not probability argmax")

    recomputed = {
        "auroc": float(roc_auc_score(labels, probabilities[:, 1])),
        "auprc": float(average_precision_score(labels, probabilities[:, 1])),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_precision": float(precision_score(labels, predictions, average="weighted")),
        "weighted_recall": float(recall_score(labels, predictions, average="weighted")),
    }
    released_metrics = json.loads(args.metrics.read_text())
    require_close("released AUROC", float(released_metrics["auc_score"]), recomputed["auroc"])
    require_close("released weighted F1", float(released_metrics["f1_score"]), recomputed["weighted_f1"])
    require_close("released accuracy", float(released_metrics["accuracy"]), recomputed["accuracy"])
    require_close(
        "released weighted precision",
        float(released_metrics["precision"]),
        recomputed["weighted_precision"],
    )
    require_close(
        "released weighted recall",
        float(released_metrics["recall"]),
        recomputed["weighted_recall"],
    )

    train_log = args.train_log.read_text(errors="replace")
    test_log = args.test_log.read_text(errors="replace")
    best_val_auc = float(checkpoint["best_val_auc"])
    best_line = (
        f"Best Validation AU-ROC: {best_val_auc:.4f} at epoch {checkpoint_epoch + 1}"
    )
    if best_line not in train_log:
        raise AssertionError("training log does not confirm the saved selected checkpoint")
    if "Evaluating on Test set" in train_log:
        raise AssertionError("test evaluation occurred during training")
    if test_log.count("Evaluating on Test set") != 1:
        raise AssertionError("released evaluator did not make exactly one test pass")

    report = {
        "status": "exact",
        "seed": args.seed,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_best_validation_auroc": best_val_auc,
        "parameter_count": parameter_count,
        "state_keys": len(state_dict),
        "test_loader_passes": 1,
        "test": {
            "count": len(labels),
            "positives": int(labels.sum()),
            **recomputed,
        },
        "artifacts": {
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "predictions_sha256": sha256_file(args.predictions),
            "metrics_sha256": sha256_file(args.metrics),
            "train_log_sha256": sha256_file(args.train_log),
            "test_log_sha256": sha256_file(args.test_log),
            "rus_sha256": rus_report["artifacts"]["rus_sha256"],
            "input_report_sha256": sha256_file(args.input_report),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
