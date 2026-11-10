#!/usr/bin/env python3
"""Independently verify and aggregate the five FuseMoE Table 4 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score


SEEDS = (32, 42, 52, 62, 72)
PAPER_AUROC = 0.8103
PAPER_AUROC_PM = 0.0025
PAPER_F1 = 0.4653
PAPER_F1_PM = 0.0057
LIGHTGBM_AUROC = 0.8194417189952905
EXPECTED_ROWS = 1316
EXPECTED_POSITIVE = 224
EXPECTED_INPUT_HASHES = {
    "manifest.json": "1a84aef9c55bde7bfb446b2b6eb50dd93e318f6a37549bf7c3e3bad4036f1c6d",
    "verification.json": "647118aa289b2c7eb49efe1dca8e96d5ee7d25c58c347c408bf91445bdf4f0e4",
    "test_ihm-48-table4_stays.pkl": "922846d629abb1fb52753fa4e0db9bad38910e4617ebb588421849bb1c9b04f7",
}
EXPECTED_CONFIG = {
    "task": "ihm-48-table4",
    "mode": "train",
    "modeltype": "TS_Text",
    "eval_score": ["auc", "auprc", "f1"],
    "num_labels": 2,
    "train_batch_size": 2,
    "eval_batch_size": 8,
    "num_train_epochs": 8,
    "txt_learning_rate": 2e-5,
    "ts_learning_rate": 4e-4,
    "gradient_accumulation_steps": 16,
    "num_heads": 8,
    "layers": 3,
    "cross_layers": 3,
    "embed_dim": 128,
    "reg_ts": True,
    "tt_max": 48,
    "embed_time": 64,
    "model_name": "bioLongformer",
    "num_of_notes": 5,
    "notes_order": "Last",
    "fine_tune": False,
    "TS_mixup": True,
    "mixup_level": "batch",
    "cross_method": "moe",
    "hidden_size": 512,
    "gating_function": ["laplace"],
    "num_of_experts": [16],
    "top_k": [4],
    "num_modalities": 2,
    "use_pt_text_embeddings": True,
    "router_type": "joint",
    "use_balance_loss": True,
    "balance_loss_coef": 0.01,
    "mixed_precision": "fp16",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{label}: {actual} != {expected}")


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    sample_sd = float(np.std(array, ddof=1))
    return {
        "values": [float(value) for value in array],
        "mean": float(np.mean(array)),
        "sample_sd": sample_sd,
        "sem": float(sample_sd / math.sqrt(len(array))),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def verify_config(config: dict, seed: int) -> None:
    if config.get("seed") != seed:
        raise AssertionError(f"seed_{seed}: config seed changed")
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise AssertionError(
                f"seed_{seed}: configuration {key} changed: {config.get(key)!r}"
            )
    target = config.get("paper_target", {})
    if target != {
        "auroc": PAPER_AUROC,
        "f1": PAPER_F1,
        "paper_seeds": list(SEEDS),
    }:
        raise AssertionError(f"seed_{seed}: paper target changed")


def verify_history(history: list[dict], result: dict, seed: int) -> int:
    if len(history) != 8:
        raise AssertionError(f"seed_{seed}: expected 8 validation events")
    best_f1 = -math.inf
    selected_epoch = None
    for epoch, event in enumerate(history):
        if event.get("zero_based_epoch") != epoch:
            raise AssertionError(f"seed_{seed}: validation epoch order changed")
        validation = event.get("validation", {})
        for metric in ("auc", "auprc", "f1"):
            if metric not in validation or not math.isfinite(float(validation[metric])):
                raise AssertionError(f"seed_{seed}: invalid validation {metric}")
        expected_selected = float(validation["f1"]) > best_f1
        if event.get("selected") is not expected_selected:
            raise AssertionError(f"seed_{seed}: checkpoint selection trace changed")
        if expected_selected:
            best_f1 = float(validation["f1"])
            selected_epoch = epoch
    if result.get("selected_zero_based_epoch") != selected_epoch:
        raise AssertionError(f"seed_{seed}: selected epoch changed")
    selected_validation = history[selected_epoch]["validation"]
    for metric in ("auc", "auprc", "f1"):
        assert_close(
            result["checkpoint_validation"][metric],
            selected_validation[metric],
            f"seed_{seed}: checkpoint validation {metric}",
        )
    return int(selected_epoch)


def verify_seed(
    primary_dir: Path,
    seed: int,
    expected_arrays: dict[str, np.ndarray],
) -> dict:
    run_dir = primary_dir / f"seed_{seed}"
    required = (
        "config.json",
        "console.log",
        "predictions.npz",
        "result.json",
        "validation_history.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"seed_{seed}: missing {missing}")

    result = load_json(run_dir / "result.json")
    if result.get("status") != "complete" or result.get("seed") != seed:
        raise AssertionError(f"seed_{seed}: incomplete or mismatched result")
    if result.get("target") != (
        "Table 4 MIMIC-IV 48-IHM vital signs + notes Laplace FuseMoE"
    ):
        raise AssertionError(f"seed_{seed}: target changed")
    verify_config(load_json(run_dir / "config.json"), seed)
    selected_epoch = verify_history(
        load_json(run_dir / "validation_history.json"), result, seed
    )

    with np.load(run_dir / "predictions.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(expected_arrays) | {"y_score"}:
            raise AssertionError(f"seed_{seed}: prediction fields changed")
        for name, expected in expected_arrays.items():
            np.testing.assert_array_equal(saved[name], expected)
        y_score = np.asarray(saved["y_score"], dtype=np.float64)
    if y_score.shape != (EXPECTED_ROWS,):
        raise AssertionError(f"seed_{seed}: score shape changed: {y_score.shape}")
    if not np.isfinite(y_score).all() or np.any((y_score < 0) | (y_score > 1)):
        raise AssertionError(f"seed_{seed}: invalid probability scores")

    y_true = expected_arrays["y_true"]
    y_pred = (y_score > 0.5).astype(np.int64)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    recomputed = {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(auc(recall, precision)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    for metric, value in recomputed.items():
        assert_close(result["test"][metric], value, f"seed_{seed}: test {metric}")
    if (result["test"]["rows"], result["test"]["positive"]) != (
        EXPECTED_ROWS,
        EXPECTED_POSITIVE,
    ):
        raise AssertionError(f"seed_{seed}: test denominator changed")
    if result["test"]["filtered_examples"] != 0 or result["test"]["nan_logits"] != 0:
        raise AssertionError(f"seed_{seed}: filtered or nonfinite test examples")

    artifacts = result["artifacts"]
    prediction_path = run_dir / artifacts["predictions"]["path"]
    history_path = run_dir / artifacts["validation_history"]["path"]
    checkpoint_path = run_dir / artifacts["checkpoint"]["path"]
    for label, path, expected_hash in (
        ("predictions", prediction_path, artifacts["predictions"]["sha256"]),
        ("validation history", history_path, artifacts["validation_history"]["sha256"]),
        ("checkpoint", checkpoint_path, artifacts["checkpoint"]["sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise AssertionError(f"seed_{seed}: {label} hash changed")

    return {
        "seed": seed,
        "selected_zero_based_epoch": selected_epoch,
        "checkpoint_validation": result["checkpoint_validation"],
        "test": recomputed,
        "runtime_seconds": float(result["runtime_seconds"]),
        "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "predictions_sha256": artifacts["predictions"]["sha256"],
        "config_sha256": sha256_file(run_dir / "config.json"),
        "console_log_sha256": sha256_file(run_dir / "console.log"),
        "validation_history_sha256": artifacts["validation_history"]["sha256"],
    }


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for name, expected_hash in EXPECTED_INPUT_HASHES.items():
        path = args.input_dir / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise AssertionError(f"input artifact changed: {name}")

    with (args.input_dir / "test_ihm-48-table4_stays.pkl").open("rb") as handle:
        test_rows = pickle.load(handle)
    if len(test_rows) != EXPECTED_ROWS:
        raise AssertionError("test input denominator changed")
    expected_arrays = {
        "stay_id": np.asarray([int(row["stay_id"]) for row in test_rows], dtype=np.int64),
        "subject_id": np.asarray([int(row["name"]) for row in test_rows], dtype=np.int64),
        "hadm_id": np.asarray([int(row["hadm_id"]) for row in test_rows], dtype=np.int64),
        "y_true": np.asarray([int(row["label"]) for row in test_rows], dtype=np.int64),
    }
    if int(expected_arrays["y_true"].sum()) != EXPECTED_POSITIVE:
        raise AssertionError("test positive count changed")

    seed_results = [
        verify_seed(args.primary_dir, seed, expected_arrays) for seed in SEEDS
    ]
    aggregate = {
        metric: summarize([entry["test"][metric] for entry in seed_results])
        for metric in ("auroc", "auprc", "f1")
    }
    aggregate["auroc"].update(
        {
            "published": PAPER_AUROC,
            "published_plus_minus": PAPER_AUROC_PM,
            "signed_gap_from_published": aggregate["auroc"]["mean"] - PAPER_AUROC,
            "inside_descriptive_paper_band": (
                PAPER_AUROC - PAPER_AUROC_PM
                <= aggregate["auroc"]["mean"]
                <= PAPER_AUROC + PAPER_AUROC_PM
            ),
            "matched_lightgbm": LIGHTGBM_AUROC,
            "signed_gap_from_lightgbm": (
                aggregate["auroc"]["mean"] - LIGHTGBM_AUROC
            ),
        }
    )
    aggregate["f1"].update(
        {
            "published": PAPER_F1,
            "published_plus_minus": PAPER_F1_PM,
            "signed_gap_from_published": aggregate["f1"]["mean"] - PAPER_F1,
            "inside_descriptive_paper_band": (
                PAPER_F1 - PAPER_F1_PM
                <= aggregate["f1"]["mean"]
                <= PAPER_F1 + PAPER_F1_PM
            ),
        }
    )
    report = {
        "status": "exact",
        "paper": "fusemoe",
        "target": "Table 4 MIMIC-IV 48-IHM vital signs + notes Laplace FuseMoE",
        "seeds": list(SEEDS),
        "test_rows": EXPECTED_ROWS,
        "test_positive": EXPECTED_POSITIVE,
        "input_sha256": EXPECTED_INPUT_HASHES,
        "checks": [
            "all five prespecified seeds complete",
            "frozen Table 4 configuration exact",
            "strict validation-F1 checkpoint history exact",
            "ordered test identifiers and labels exact",
            "test metrics independently recomputed from saved probabilities",
            "all checkpoint, prediction, history, config, and console artifacts hashed",
        ],
        "seed_results": seed_results,
        "aggregate": aggregate,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
