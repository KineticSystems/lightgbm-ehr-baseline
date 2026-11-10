#!/usr/bin/env python3
"""Independently evaluate one completed Step-wise source-command run."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import random
import re
import time
from pathlib import Path

import gin
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from icu_benchmarks.data.loader import ICUVariableLengthDataset
from icu_benchmarks.models import train as _train  # noqa: F401; Gin registration
from icu_benchmarks.models.wrappers import DLWrapper


ALLOWED_SEEDS = {1111, 2222, 3333, 4444, 5555}
EXPECTED = {
    "test": {"landmarks": 523_200, "positive": 9_683},
    "val": {"landmarks": 530_638, "positive": 11_752},
}
PAPER = {"auroc": 0.916, "auprc": 0.387}
LIGHTGBM_AUROC = 0.9211997005347095


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def close_dataset(dataset: ICUVariableLengthDataset) -> None:
    h5_file = dataset.h5_loader.data_h5._v_file
    if h5_file.isopen:
        h5_file.close()


def evaluate_split(model: DLWrapper, dataset: ICUVariableLengthDataset):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    probabilities = []
    labels = []
    model.encoder.eval()
    with torch.no_grad():
        for element in loader:
            _, logits, targets, _, _ = model.step_fn(element, None)
            probability = torch.softmax(logits, dim=1)[:, 1]
            probabilities.append(probability.detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())
    probabilities_np = np.concatenate(probabilities).astype(np.float32, copy=False)
    labels_np = np.concatenate(labels).astype(np.float32, copy=False)
    if not np.isfinite(probabilities_np).all():
        raise FloatingPointError("nonfinite probabilities in independent evaluation")
    return labels_np, probabilities_np


def read_metric_pickle(path: Path) -> dict[str, float]:
    # These pickles are outputs created locally by the just-completed pinned
    # source process. No downloaded or untrusted pickle is opened here.
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    return {str(key): float(value) for key, value in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-console-log", type=Path, required=True)
    parser.add_argument(
        "--config",
        default=(
            "configs/reproduce/mimic/embeddings/ftt-no-group/"
            "transformer_ftt_decomp.gin"
        ),
    )
    args = parser.parse_args()

    if args.seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed must be one of {sorted(ALLOWED_SEEDS)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Step-wise checkpoint evaluation")
    if torch.cuda.get_device_capability() != (12, 0):
        raise AssertionError(torch.cuda.get_device_capability())
    if args.run_dir.name != str(args.seed):
        raise AssertionError("run directory is not associated with requested seed")

    verification_dir = args.run_dir / "verification"
    if verification_dir.exists():
        raise FileExistsError(f"refusing to overwrite {verification_dir}")

    required = [
        args.run_dir / "val_metrics.pkl",
        args.run_dir / "test_metrics.pkl",
        args.run_dir / "post_val_metrics.pkl",
        args.run_dir / "train_config.gin",
        args.source_console_log,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    checkpoints = [Path(path) for path in glob.glob(str(args.run_dir / "model_e*.torch"))]
    if len(checkpoints) != 1:
        raise AssertionError("expected exactly one retained best checkpoint")
    checkpoint_path = checkpoints[0]
    selected_epoch = int(re.fullmatch(r"model_e(\d+)\.torch", checkpoint_path.name).group(1))

    source_console = args.source_console_log.read_text(errors="replace")
    source_test_markers = source_console.count("TESTING")
    if source_test_markers != 1:
        raise AssertionError(f"source test marker count is {source_test_markers}, not 1")

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    gin.parse_config_file(args.config)
    model = DLWrapper()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    incompatible = model.encoder.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(incompatible)

    data_path = gin.query_parameter("train_common.data_path")
    train_dataset = ICUVariableLengthDataset(data_path, split="train")
    test_dataset = ICUVariableLengthDataset(data_path, split="test")
    val_dataset = ICUVariableLengthDataset(data_path, split="val")
    test_dataset.set_scaler(train_dataset.scaler)
    val_dataset.set_scaler(train_dataset.scaler)
    model.set_scaler(train_dataset.scaler)
    if train_dataset.scale_feature:
        test_dataset.set_feat_scaler(train_dataset.feat_scaler)
        val_dataset.set_feat_scaler(train_dataset.feat_scaler)
    model.n_worker = 1
    model.set_metrics(allow_distributed=False, with_kappa=True)
    model.encoder = model.accelerator.prepare(model.encoder)

    started = time.monotonic()
    test_labels, test_probabilities = evaluate_split(model, test_dataset)
    val_labels, val_probabilities = evaluate_split(model, val_dataset)

    split_arrays = {
        "test": (test_labels, test_probabilities),
        "val": (val_labels, val_probabilities),
    }
    recomputed = {}
    for split, (labels, probabilities) in split_arrays.items():
        expected = EXPECTED[split]
        if len(labels) != expected["landmarks"]:
            raise AssertionError(f"{split} landmark count differs")
        if int(labels.sum()) != expected["positive"]:
            raise AssertionError(f"{split} positive count differs")
        recomputed[split] = {
            "landmarks": len(labels),
            "positive": int(labels.sum()),
            "auroc": float(roc_auc_score(labels, probabilities)),
            "auprc": float(average_precision_score(labels, probabilities)),
            "labels_sha256": sha256_array(labels),
            "probabilities_sha256": sha256_array(probabilities),
        }

    source_test = read_metric_pickle(args.run_dir / "test_metrics.pkl")
    source_post_val = read_metric_pickle(args.run_dir / "post_val_metrics.pkl")
    source_selected_val = read_metric_pickle(args.run_dir / "val_metrics.pkl")
    comparisons = [
        (source_test["AUC"], recomputed["test"]["auroc"], "test AUROC"),
        (source_test["PR"], recomputed["test"]["auprc"], "test AUPRC"),
        (source_post_val["AUC"], recomputed["val"]["auroc"], "val AUROC"),
        (source_post_val["PR"], recomputed["val"]["auprc"], "val AUPRC"),
    ]
    for source_value, recomputed_value, name in comparisons:
        if not np.isclose(source_value, recomputed_value, rtol=0.0, atol=1e-10):
            raise AssertionError(f"{name} differs: {source_value} vs {recomputed_value}")

    verification_dir.mkdir()
    predictions_path = verification_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        test_labels=test_labels,
        test_probabilities=test_probabilities,
        val_labels=val_labels,
        val_probabilities=val_probabilities,
    )
    report = {
        "status": "passed",
        "seed": args.seed,
        "selected_epoch_zero_based": selected_epoch,
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
            "state_keys": len(state),
            "strict_reload": True,
        },
        "source_metrics": {
            "selected_validation": source_selected_val,
            "post_selection_validation": source_post_val,
            "test": source_test,
        },
        "independent_metrics": recomputed,
        "predictions": {
            "path": str(predictions_path),
            "bytes": predictions_path.stat().st_size,
            "sha256": sha256_file(predictions_path),
        },
        "paper": PAPER,
        "lightgbm_auroc": LIGHTGBM_AUROC,
        "source_test_passes": 1,
        "independent_audit_test_passes": 1,
        "total_post_selection_test_passes": 2,
        "elapsed_seconds": time.monotonic() - started,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(),
        "test_metric_tolerance": 1e-10,
    }
    report_path = verification_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))

    close_dataset(train_dataset)
    close_dataset(test_dataset)
    close_dataset(val_dataset)
    gin.clear_config()


if __name__ == "__main__":
    main()
