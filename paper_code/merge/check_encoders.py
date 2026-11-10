#!/usr/bin/env python3
"""Verify frozen MERGE encoders and their released feature boundaries."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import distributions, version
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torchxrayvision as xrv


CXR_ALIAS = "densenet121-res224-chex"
CXR_FILENAME = "chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array.astype("<f4", copy=False))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def verify_manifest(root: Path) -> dict[str, object]:
    manifest_path = root / "encoder_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    for encoder in ("biobert", "cxr"):
        encoder_root = root / (
            "biobert-v1.1" if encoder == "biobert" else "torchxrayvision-chex"
        )
        for record in manifest[encoder]["files"]:
            path = encoder_root / record["path"]
            if path.stat().st_size != record["bytes"]:
                raise RuntimeError(f"byte-count mismatch: {path}")
            if sha256(path) != record["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch: {path}")
            checked += 1
    return {"manifest_sha256": sha256(manifest_path), "files_checked": checked}


def package_lock() -> list[str]:
    records = {
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in distributions()
        if dist.metadata.get("Name")
    }
    return sorted(records, key=str.casefold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root = args.encoder_root.resolve()
    verification = verify_manifest(root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    biobert_root = root / "biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(biobert_root, local_files_only=True)
    biobert = AutoModel.from_pretrained(biobert_root, local_files_only=True)
    biobert = biobert.to(device).eval()
    token_batch = tokenizer(
        ["No acute cardiopulmonary process."],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    token_batch = {key: value.to(device) for key, value in token_batch.items()}
    with torch.no_grad():
        note_embedding = biobert(**token_batch).pooler_output
    if tuple(note_embedding.shape) != (1, 768) or not torch.isfinite(note_embedding).all():
        raise RuntimeError(f"invalid BioBERT output: {tuple(note_embedding.shape)}")

    cxr_root = root / "torchxrayvision-chex"
    if not (cxr_root / CXR_FILENAME).is_file():
        raise FileNotFoundError(cxr_root / CXR_FILENAME)
    cxr_model = xrv.models.DenseNet(
        weights=CXR_ALIAS,
        cache_dir=str(cxr_root),
    ).to(device).eval()
    image = torch.zeros((1, 1, 224, 224), dtype=torch.float32, device=device)
    with torch.no_grad():
        feature_map = cxr_model.features(image)
        feature_map = F.relu(feature_map, inplace=False)
        cxr_embedding = F.adaptive_avg_pool2d(feature_map, (1, 1)).reshape(1, -1)
        cxr_predictions = cxr_model(image)
    if tuple(cxr_embedding.shape) != (1, 1024) or not torch.isfinite(cxr_embedding).all():
        raise RuntimeError(f"invalid CXR embedding: {tuple(cxr_embedding.shape)}")
    if tuple(cxr_predictions.shape) != (1, 18) or not torch.isfinite(cxr_predictions).all():
        raise RuntimeError(f"invalid CXR predictions: {tuple(cxr_predictions.shape)}")

    lock = package_lock()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / "environment-lock.txt"
    lock_path.write_text("\n".join(lock) + "\n", encoding="utf-8")

    report = {
        "status": "exact_frozen_artifacts_and_feature_shapes",
        "device": {
            "requested": args.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "capability": list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None,
        },
        "environment": {
            "torch": torch.__version__,
            "torchvision": version("torchvision"),
            "torchxrayvision": version("torchxrayvision"),
            "transformers": version("transformers"),
            "lock_path": lock_path.name,
            "lock_sha256": sha256(lock_path),
        },
        "artifacts": verification,
        "feature_checks": {
            "biobert_shape": list(note_embedding.shape),
            "biobert_float32_sha256": array_sha256(note_embedding.cpu().numpy()),
            "cxr_shape": list(cxr_embedding.shape),
            "cxr_float32_sha256": array_sha256(cxr_embedding.cpu().numpy()),
            "cxr_prediction_shape": list(cxr_predictions.shape),
        },
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
