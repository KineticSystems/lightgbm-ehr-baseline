#!/usr/bin/env python3
"""Validate and record the isolated Blackwell REMed environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import distributions, version
from pathlib import Path


EXPECTED = {
    "accelerate": "0.20.3",
    "einops": "0.8.2",
    "h5pickle": "0.4.2",
    "h5py": "3.16.0",
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "performer-pytorch": "1.1.4",
    "polars": "1.43.2",
    "recurrent-memory-transformer-pytorch": "0.2.2",
    "scikit-learn": "1.7.2",
    "tokenizers": "0.13.3",
    "torch": "2.7.0+cu128",
    "torchaudio": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "tqdm": "4.70.0",
    "transformers": "4.30.1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    args = parser.parse_args()

    import torch

    from src.dataset import EHRDataset, EHRforReprGen, ReprDataset
    from src.trainer import TRAINER_REGISTRY

    if platform.python_version() != "3.10.20":
        raise AssertionError(platform.python_version())
    observed = {name: version(name) for name in EXPECTED}
    if observed != EXPECTED:
        raise AssertionError((observed, EXPECTED))
    if torch.version.cuda != "12.8":
        raise AssertionError(torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible to the REMed GPU environment")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise AssertionError(torch.cuda.get_device_capability(0))

    matrix = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
    product = matrix @ matrix
    torch.cuda.synchronize()
    if not torch.isfinite(product).all().item():
        raise AssertionError("non-finite BF16 matrix multiplication")

    assert EHRDataset is not None
    assert EHRforReprGen is not None
    assert ReprDataset is not None
    assert "short" in TRAINER_REGISTRY and "remed" in TRAINER_REGISTRY

    installed = sorted(
        {
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in distributions()
            if distribution.metadata.get("Name")
        },
        key=str.casefold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.freeze.write_text("\n".join(installed) + "\n")
    report = {
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_device": torch.cuda.get_device_name(0),
        "expected_selected_packages": EXPECTED,
        "freeze": {
            "bytes": args.freeze.stat().st_size,
            "path": str(args.freeze),
            "sha256": sha256(args.freeze),
        },
        "installed_distribution_count": len(installed),
        "python": platform.python_version(),
        "real_cuda_bf16_matrix_multiply": True,
        "remed_registry": sorted(TRAINER_REGISTRY),
        "requirements_sha256": sha256(args.requirements),
        "status": "passed",
        "torch_cuda": torch.version.cuda,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
