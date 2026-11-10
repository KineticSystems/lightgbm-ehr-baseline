#!/usr/bin/env python3
"""Fail-closed Blackwell/runtime import check for the DDL-CXR reproduction."""

from __future__ import annotations

import importlib.metadata
import argparse
import json
import platform
from pathlib import Path

import torch
import torchvision


EXPECTED = {
    "einops": "0.8.0",
    "h5py": "3.10.0",
    "kornia": "0.7.2",
    "matplotlib": "3.7.5",
    "numpy": "1.21.6",
    "omegaconf": "2.3.0",
    "pandas": "2.0.3",
    "Pillow": "9.5.0",
    "pytorch-lightning": "1.9.5",
    "scikit-learn": "1.3.0",
    "taming-transformers-rom1504": "0.0.6",
    "torchmetrics": "0.11.4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert platform.python_version() == "3.10.20", platform.python_version()
    assert torch.__version__ == "2.7.0+cu128", torch.__version__
    assert torchvision.__version__ == "0.22.0+cu128", torchvision.__version__
    assert torch.version.cuda == "12.8", torch.version.cuda
    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert torch.cuda.get_device_capability(0) == (12, 0)

    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    assert versions == EXPECTED, versions

    # Import the exact released module surfaces needed by generation and fit.
    from ldm.data.dynamic_data import GenerateZ1, PredictDataset  # noqa: F401
    from ldm.models.diffusion.ddpm import DDL_CXR_LDM  # noqa: F401
    from ldm.models.predict_model import FusionTokens3inputAttnFuse  # noqa: F401

    lhs = torch.randn(1024, 1024, device="cuda")
    rhs = torch.randn(1024, 1024, device="cuda")
    product = lhs @ rhs
    torch.cuda.synchronize()
    assert torch.isfinite(product).all()

    report = {
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "device_capability": list(torch.cuda.get_device_capability(0)),
                "imports": "pass",
                "matmul": "pass",
                "packages": versions,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
            }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
