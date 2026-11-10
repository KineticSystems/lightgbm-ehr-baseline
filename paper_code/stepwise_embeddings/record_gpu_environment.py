#!/usr/bin/env python3
"""Freeze and verify the completed Step-wise Blackwell environment."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import distributions, version
from pathlib import Path

import torch


EXPECTED = {
    "accelerate": "0.18.0",
    "gin-config": "0.5.0",
    "h5py": "3.9.0",
    "joblib": "1.5.3",
    "lightgbm": "3.3.5",
    "matplotlib": "3.7.3",
    "numpy": "1.24.4",
    "pandas": "1.5.3",
    "pytorch-ignite": "0.4.12",
    "scikit-learn": "1.2.2",
    "scipy": "1.10.1",
    "seaborn": "0.12.2",
    "tables": "3.8.0",
    "tensorboard": "2.12.3",
    "torch": "2.7.0+cu128",
    "torchaudio": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "tqdm": "4.65.0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    output_dir = Path(".export_assets/reproduction/audit")
    report_path = output_dir / "gpu_environment.json"
    freeze_path = output_dir / "gpu_environment_freeze.txt"
    if report_path.exists() or freeze_path.exists():
        raise FileExistsError("refusing to overwrite the GPU environment record")

    observed = {name: version(name) for name in EXPECTED}
    if observed != EXPECTED:
        raise AssertionError((observed, EXPECTED))
    if platform.python_version() != "3.9.25":
        raise AssertionError(platform.python_version())
    if torch.version.cuda != "12.8":
        raise AssertionError(torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (12, 0):
        raise AssertionError(torch.cuda.get_device_capability())

    matrix = torch.arange(4096, device="cuda", dtype=torch.float32).reshape(64, 64)
    product = matrix @ matrix.T
    torch.cuda.synchronize()
    if not torch.isfinite(product).all().item():
        raise FloatingPointError("nonfinite real-GPU matrix product")

    package_rows = sorted(
        {
            f"{dist.metadata['Name'].lower().replace('_', '-')}=={dist.version}"
            for dist in distributions()
            if dist.metadata.get("Name")
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text("\n".join(package_rows) + "\n")

    requirements = Path("requirements-gpu.txt")
    setup = Path("setup_gpu_env.sh")
    report = {
        "status": "passed",
        "python": platform.python_version(),
        "expected_selected_packages": EXPECTED,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(),
        "cuda_capability": list(torch.cuda.get_device_capability()),
        "real_cuda_matrix_multiply": True,
        "installed_distribution_count": len(package_rows),
        "freeze": {
            "path": str(freeze_path),
            "bytes": freeze_path.stat().st_size,
            "sha256": sha256_file(freeze_path),
        },
        "requirements_sha256": sha256_file(requirements),
        "setup_sha256": sha256_file(setup),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
