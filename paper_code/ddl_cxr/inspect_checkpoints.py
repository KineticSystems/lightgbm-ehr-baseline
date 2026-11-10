#!/usr/bin/env python3
"""Safely inspect released Lightning checkpoints without executing pickle globals."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


PAPER_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = (
    PAPER_ROOT
    / ".export_assets/external_artifacts"
    / "huggingface_jasmineChen_ddl-cxr_1a7b6d3776745194f1426f8b2c9f7e8b1ebd5033"
)
OUTPUT = PAPER_ROOT / ".export_assets/reproduction/checkpoints.json"
CHECKPOINTS = {
    "ldm": {
        "path": ARTIFACT_ROOT / "ddl-cxr-ldm.ckpt",
        "sha256": "bdd11363779e6136a409c2e94d90913e59243ec438e9e6b3bf2e97885e823b22",
    },
    "vae": {
        "path": ARTIFACT_ROOT / "ddl-cxr-vae.ckpt",
        "sha256": "6bd4f13cab9810486a61e713d67a809a4b2fe92439746e96bff6dce230d31de0",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_one(name: str, spec: dict[str, object]) -> dict[str, object]:
    path = Path(spec["path"])
    expected_sha = str(spec["sha256"])
    assert path.is_file(), path
    observed_sha = sha256(path)
    assert observed_sha == expected_sha, (observed_sha, expected_sha)

    # weights_only rejects arbitrary pickle globals; mmap avoids materializing
    # the multi-GB optimizer/state payload while inventorying tensor metadata.
    with torch.serialization.safe_globals([ModelCheckpoint]):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert isinstance(payload, dict), type(payload)
    state = payload.get("state_dict")
    assert isinstance(state, dict) and state, f"{name}: missing state_dict"
    assert all(isinstance(value, torch.Tensor) for value in state.values())

    dtypes = Counter(str(value.dtype) for value in state.values())
    return {
        "bytes": path.stat().st_size,
        "checkpoint_top_level_keys": sorted(payload),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "sha256": observed_sha,
        "state_dict": {
            "dtype_counts": dict(sorted(dtypes.items())),
            "keys": len(state),
            "parameters_and_buffers_numel": sum(value.numel() for value in state.values()),
        },
        "weights_only_load": "pass",
    }


def main() -> None:
    report = {
        "checkpoints": {
            name: inspect_one(name, spec) for name, spec in CHECKPOINTS.items()
        },
        "status": "pass",
        "torch": torch.__version__,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    OUTPUT.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
