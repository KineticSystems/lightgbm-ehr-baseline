#!/usr/bin/env python3
"""Check the minimized FuseMoE Blackwell runtime and local encoder assets."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import accelerate
import numpy as np
import pandas as pd
import sklearn
import tensorboardX
import tokenizers
import torch
import transformers


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT / "src"))

from core.model import MULTCrossModel  # noqa: E402, F401
from core.module import TransformerCrossEncoderLayer  # noqa: E402, F401
from preprocessing.data_mimiciv import TSNote_Irg, TextTSIrgcollate_fn  # noqa: E402, F401


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if platform.python_version() != "3.10.20":
        raise AssertionError(platform.python_version())
    if torch.__version__ != "2.7.0+cu128":
        raise AssertionError(torch.__version__)
    if np.__version__ != "1.21.6":
        raise AssertionError(np.__version__)
    if sklearn.__version__ != "1.0.2":
        raise AssertionError(sklearn.__version__)
    if transformers.__version__ != "4.16.2":
        raise AssertionError(transformers.__version__)
    if accelerate.__version__ != "0.6.0":
        raise AssertionError(accelerate.__version__)
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is unavailable")
    capability = torch.cuda.get_device_capability(0)
    if capability < (12, 0):
        raise AssertionError(f"unexpected GPU capability {capability}")

    model_root = PAPER_ROOT / ".export_assets/reproduction/clinical_longformer_model"
    weight_path = model_root / "pytorch_model.bin"
    weight_hash = sha256(weight_path)
    if weight_hash != "87e239f555459c3668993fbf5b53bea93f503be86f2f4285f431fdb2c6c33c75":
        raise AssertionError("Clinical-Longformer weight hash changed")
    report = {
        "status": "pass",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(capability),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "accelerate": accelerate.__version__,
        "tensorboardX": tensorboardX.__version__,
        "clinical_longformer_weight_sha256": weight_hash,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

