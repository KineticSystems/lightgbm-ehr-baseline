#!/usr/bin/env python3
"""Time one exact initial GARLIC epoch without accessing the test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time as walltime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from main import prepare_batch  # noqa: E402
from model import Model  # noqa: E402
from reproduction_data import GarlicDataStore  # noqa: E402


SEED = 0
BATCH_SIZE = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_model(device: str) -> Model:
    return Model(103, 16, 128, 5, 200, device).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--orders", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for output in (args.checkpoint, args.output):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    input_report = json.loads(args.input_report.read_text())
    if input_report.get("status") != "exact":
        raise AssertionError("GARLIC input adapter has not passed exact verification")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("GARLIC epoch benchmark requires real CUDA")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    store = GarlicDataStore(args.cache, args.orders)
    train_data = store.splits["train"]
    val_data = store.splits["validation"]
    model = make_model(args.device)
    rec_params = []
    pred_params = []
    for name, parameter in model.named_parameters():
        if any(key in name for key in ("t_attention", "classifier", "aggregator", "predictor")):
            pred_params.append(parameter)
        else:
            rec_params.append(parameter)
    optimizer = optim.Adam(
        [
            {"params": rec_params, "lr": 1e-3, "weight_decay": 1e-5},
            {"params": pred_params, "lr": 1e-5, "weight_decay": 1e-4},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-5,
        eps=1e-8,
    )

    model.train()
    total_loss = 0.0
    all_labels = torch.tensor([], device=device)
    all_outputs = torch.tensor([], device=device)
    torch.cuda.synchronize(device)
    train_started = walltime.perf_counter()
    train_batches = 0
    for start in range(0, len(train_data), BATCH_SIZE):
        batch = train_data[start : start + BATCH_SIZE]
        arr, mask, times, labels = prepare_batch(batch, device)
        optimizer.zero_grad()
        logits, reconstruction_loss, _ = model(arr, mask, times)
        classification_loss = nn.BCEWithLogitsLoss()(logits, labels)
        graph_loss = torch.sum(torch.abs(model.message_passing.adj)) * 1e-2
        loss = reconstruction_loss + classification_loss + graph_loss
        if not torch.isfinite(loss):
            raise AssertionError(f"nonfinite GARLIC training loss in batch {train_batches}")
        probabilities = torch.sigmoid(logits)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_labels = torch.cat([all_labels, labels], 0)
        all_outputs = torch.cat([all_outputs, probabilities], 0)
        train_batches += 1
        if train_batches % 25 == 0 or start + len(batch) == len(train_data):
            print(f"train batches {train_batches}/309; admissions {start + len(batch)}/{len(train_data)}", flush=True)
    torch.cuda.synchronize(device)
    train_seconds = walltime.perf_counter() - train_started
    train_loss = total_loss / len(train_data)
    train_labels = all_labels.detach().cpu().numpy()
    train_scores = all_outputs.detach().cpu().numpy()
    train_auc = roc_auc_score(train_labels, train_scores)
    train_auprc = average_precision_score(train_labels, train_scores)
    scheduler.step(train_auc)

    model.eval()
    validation_loss = 0.0
    validation_labels = torch.tensor([], device=device)
    validation_outputs = torch.tensor([], device=device)
    torch.cuda.synchronize(device)
    validation_started = walltime.perf_counter()
    validation_batches = 0
    with torch.no_grad():
        for start in range(0, len(val_data), BATCH_SIZE):
            batch = val_data[start : start + BATCH_SIZE]
            arr, mask, times, labels = prepare_batch(batch, device)
            logits, _, _ = model(arr, mask, times)
            loss = nn.BCEWithLogitsLoss()(logits, labels)
            if not torch.isfinite(loss):
                raise AssertionError("nonfinite GARLIC validation loss")
            validation_loss += loss.item()
            validation_labels = torch.cat([validation_labels, labels], 0)
            validation_outputs = torch.cat(
                [validation_outputs, torch.sigmoid(logits)], 0
            )
            validation_batches += 1
    torch.cuda.synchronize(device)
    validation_seconds = walltime.perf_counter() - validation_started
    validation_loss /= len(val_data)
    val_labels = validation_labels.detach().cpu().numpy()
    val_scores = validation_outputs.detach().cpu().numpy()
    validation_auc = roc_auc_score(val_labels, val_scores)
    validation_auprc = average_precision_score(val_labels, val_scores)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    report = {
        "status": "passed",
        "test_loader_accesses": 0,
        "seed": SEED,
        "epoch": 0,
        "mode": "rec",
        "batch_size": BATCH_SIZE,
        "train": {
            "admissions": len(train_data),
            "batches": train_batches,
            "loss": float(train_loss),
            "auroc": float(train_auc),
            "auprc": float(train_auprc),
            "seconds": train_seconds,
            "seconds_per_batch": train_seconds / train_batches,
        },
        "validation": {
            "admissions": len(val_data),
            "batches": validation_batches,
            "classification_loss": float(validation_loss),
            "auroc": float(validation_auc),
            "auprc": float(validation_auprc),
            "seconds": validation_seconds,
        },
        "projected_five_seed_100_epoch_gpu_hours_without_early_stopping": (
            (train_seconds + validation_seconds) * 500 / 3600
        ),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "checkpoint": {
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256_file(args.checkpoint),
        },
        "input_report_sha256": sha256_file(args.input_report),
        "orders_sha256": sha256_file(args.orders),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    store.close()


if __name__ == "__main__":
    main()

