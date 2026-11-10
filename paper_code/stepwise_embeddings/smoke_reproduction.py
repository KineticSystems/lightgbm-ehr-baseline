#!/usr/bin/env python3
"""One exact-loader Step-wise FTT update without touching the test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import gin
import numpy as np
import torch
from torch.utils.data import DataLoader

from icu_benchmarks.data.loader import ICUVariableLengthDataset
from icu_benchmarks.models import train as _train  # noqa: F401; Gin registration
from icu_benchmarks.models.wrappers import DLWrapper


ALLOWED_SEEDS = {1111, 2222, 3333, 4444, 5555}
EXPECTED_CONFIG_STATUS = "exact_reconstructed_config"
EXPECTED_BATCH_SIZE = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_parameter(name: str):
    value = gin.query_parameter(name)
    if isinstance(value, gin.config.ConfigurableReference) and value.evaluate:
        value = value.scoped_configurable_fn()
    return value


def close_dataset(dataset: ICUVariableLengthDataset) -> None:
    h5_file = dataset.h5_loader.data_h5._v_file
    if h5_file.isopen:
        h5_file.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument(
        "--config",
        default=(
            "configs/reproduce/mimic/embeddings/ftt-no-group/"
            "transformer_ftt_decomp.gin"
        ),
    )
    parser.add_argument(
        "--config-report",
        default=".export_assets/reproduction/audit/config_preflight.json",
    )
    parser.add_argument(
        "--output-dir",
        default=".export_assets/reproduction/smoke/seed_1111",
    )
    args = parser.parse_args()

    if args.seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed must be one of {sorted(ALLOWED_SEEDS)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real Step-wise smoke")
    if torch.cuda.get_device_capability() != (12, 0):
        raise AssertionError(torch.cuda.get_device_capability())

    config_report_path = Path(args.config_report)
    config_report = json.loads(config_report_path.read_text())
    if config_report.get("status") != EXPECTED_CONFIG_STATUS:
        raise AssertionError("reconstructed configuration gate is not exact")
    if config_report.get("test_accesses") != 0:
        raise AssertionError("configuration preflight unexpectedly accessed test")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    config_path = Path(args.config)
    gin.parse_config_file(str(config_path))
    if query_parameter("DLWrapper.train.batch_size") != EXPECTED_BATCH_SIZE:
        raise AssertionError("smoke no longer exercises released batch size 64")

    started = time.monotonic()
    model = DLWrapper()
    train_dataset = ICUVariableLengthDataset(
        query_parameter("train_common.data_path"), split="train"
    )
    val_dataset = ICUVariableLengthDataset(
        query_parameter("train_common.data_path"), split="val"
    )
    val_dataset.set_scaler(train_dataset.scaler)
    model.set_scaler(train_dataset.scaler)
    if train_dataset.scale_feature:
        val_dataset.set_feat_scaler(train_dataset.feat_scaler)
    model.set_logdir(str(output_dir))
    model.set_metrics()

    if train_dataset.h5_loader.on_RAM:
        raise AssertionError("frozen configuration unexpectedly loads HDF into RAM")
    model.n_worker = 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=EXPECTED_BATCH_SIZE,
        shuffle=True,
        num_workers=model.n_worker,
        pin_memory=model.pin_memory,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EXPECTED_BATCH_SIZE,
        shuffle=False,
        num_workers=model.n_worker,
        pin_memory=model.pin_memory,
        prefetch_factor=2,
    )
    (
        model.encoder,
        model.optimizer,
        train_loader,
        val_loader,
    ) = model.accelerator.prepare(
        model.encoder,
        model.optimizer,
        train_loader,
        val_loader,
    )

    torch.cuda.reset_peak_memory_stats()
    model.encoder.train()
    train_batch = next(iter(train_loader))
    train_loss, train_logits, train_labels, _, _ = model.step_fn(train_batch, None)
    model.accelerator.backward(train_loss)
    finite_gradients = 0
    for parameter in model.encoder.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("nonfinite gradient in exact smoke batch")
            finite_gradients += 1
    if finite_gradients == 0:
        raise AssertionError("no gradients were produced")
    model.optimizer.step()
    model.optimizer.zero_grad()
    model.accelerator.wait_for_everyone()

    checkpoint_path = output_dir / "model_after_one_step.torch"
    unwrapped = model.accelerator.unwrap_model(model.encoder)
    model.accelerator.save(unwrapped.state_dict(), checkpoint_path)
    model.accelerator.wait_for_everyone()

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    fresh_model = DLWrapper()
    incompatible = fresh_model.encoder.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(incompatible)

    model.encoder.eval()
    with torch.no_grad():
        val_batch = next(iter(val_loader))
        val_loss, val_logits, val_labels, _, _ = model.step_fn(val_batch, None)
        val_probabilities = torch.softmax(val_logits, dim=1)[:, 1]
    val_probabilities_np = val_probabilities.detach().cpu().numpy()
    val_labels_np = val_labels.detach().cpu().numpy()
    if not np.isfinite(val_probabilities_np).all():
        raise FloatingPointError("nonfinite validation probability")
    if len(val_probabilities_np) != len(val_labels_np) or not len(val_labels_np):
        raise AssertionError("invalid serialized validation output")
    predictions_path = output_dir / "validation_batch_predictions.npz"
    np.savez_compressed(
        predictions_path,
        labels=val_labels_np,
        probabilities=val_probabilities_np,
    )

    report = {
        "status": "passed",
        "seed": args.seed,
        "config": str(config_path),
        "config_report_sha256": sha256_file(config_report_path),
        "batch_size": EXPECTED_BATCH_SIZE,
        "train_batch_tensor_shape": list(train_batch[0].shape),
        "train_landmarks_scored": int(train_labels.numel()),
        "train_loss": float(train_loss.detach().cpu()),
        "finite_gradient_tensors": finite_gradients,
        "validation_batch_tensor_shape": list(val_batch[0].shape),
        "validation_landmarks_scored": int(val_labels.numel()),
        "validation_loss": float(val_loss.detach().cpu()),
        "checkpoint": {
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
            "state_keys": len(state),
            "strict_reload": True,
        },
        "predictions": {
            "bytes": predictions_path.stat().st_size,
            "sha256": sha256_file(predictions_path),
        },
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.monotonic() - started,
        "cuda_device": torch.cuda.get_device_name(),
        "cuda_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "test_accesses": 0,
    }
    report_path = output_dir / "smoke.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))

    close_dataset(train_dataset)
    close_dataset(val_dataset)
    gin.clear_config()


if __name__ == "__main__":
    main()
