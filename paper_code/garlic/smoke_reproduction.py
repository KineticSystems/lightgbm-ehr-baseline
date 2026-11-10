#!/usr/bin/env python3
"""One exact GARLIC batch through CUDA, optimizer, and checkpoint round trip."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def make_model(device: str) -> Model:
    return Model(
        input_dim=103,
        feature_dim=16,
        hidden_size=128,
        window_size=5,
        length=200,
        device=device,
    ).to(device)


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
        raise RuntimeError("GARLIC smoke requires real CUDA")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    store = GarlicDataStore(args.cache, args.orders)
    try:
        batch = store.splits["train"][:BATCH_SIZE]
        arr, mask, time, label = prepare_batch(batch, device)
    finally:
        store.close()
    expected_shapes = {
        "arr": (128, 200, 103),
        "mask": (128, 200, 103),
        "time": (200, 128),
        "label": (128,),
    }
    tensors = {"arr": arr, "mask": mask, "time": time, "label": label}
    for name, value in tensors.items():
        if tuple(value.shape) != expected_shapes[name] or value.dtype != torch.float32:
            raise AssertionError(f"GARLIC smoke {name} boundary changed")
        if not torch.isfinite(value).all():
            raise AssertionError(f"GARLIC smoke {name} contains nonfinite values")

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
    optimizer.zero_grad()
    logits, reconstruction_loss, _ = model(arr, mask, time)
    classification_loss = nn.BCEWithLogitsLoss()(logits, label)
    graph_loss = torch.sum(torch.abs(model.message_passing.adj)) * 1e-2
    total_loss = reconstruction_loss + classification_loss + graph_loss
    if not torch.isfinite(total_loss):
        raise AssertionError("GARLIC smoke loss is nonfinite")
    total_loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise AssertionError("GARLIC smoke gradients are missing or nonfinite")
    optimizer.step()

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    reloaded = make_model(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    load_result = reloaded.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise AssertionError("GARLIC checkpoint strict reload failed")
    reloaded.eval()
    with torch.no_grad():
        reloaded_logits, reloaded_reconstruction, _ = reloaded(arr, mask, time)
        probabilities = torch.sigmoid(reloaded_logits)
    if not torch.isfinite(probabilities).all() or not torch.isfinite(reloaded_reconstruction):
        raise AssertionError("GARLIC reloaded predictions are nonfinite")

    report = {
        "status": "passed",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "configuration": {
            "window_size": 5,
            "hidden_size": 128,
            "feature_dim": 16,
            "lr_cls": 1e-5,
            "lr_rec": 1e-3,
            "wd_cls": 1e-4,
            "wd_rec": 1e-5,
            "lambda_cls": 1.0,
            "lambda_graph": 1e-2,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "losses": {
            "reconstruction": float(reconstruction_loss.detach().cpu()),
            "classification": float(classification_loss.detach().cpu()),
            "graph": float(graph_loss.detach().cpu()),
            "total": float(total_loss.detach().cpu()),
        },
        "input_sha256": {name: tensor_sha256(value) for name, value in tensors.items()},
        "reloaded_probability_sha256": tensor_sha256(probabilities),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "checkpoint": {
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256_file(args.checkpoint),
            "state_keys": len(state),
        },
        "input_report_sha256": sha256_file(args.input_report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

