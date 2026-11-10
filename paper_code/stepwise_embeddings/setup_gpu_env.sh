#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
UV_CACHE_ROOT="$REPRO_ROOT/uv_cache"
PIP_TMP_ROOT="$REPRO_ROOT/pip_tmp"

mkdir -p "$REPRO_ROOT" "$UV_CACHE_ROOT" "$PIP_TMP_ROOT"

if [[ ! -x "$PAPER_ROOT/.venv/bin/python" ]]; then
  echo "Missing frozen paper environment: $PAPER_ROOT/.venv" >&2
  exit 1
fi

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  env UV_CACHE_DIR="$UV_CACHE_ROOT" \
    uv venv --python "$PAPER_ROOT/.venv/bin/python" "$GPU_ENV"
fi

"$GPU_ENV/bin/python" -c \
  'import platform; assert platform.python_version() == "3.9.25", platform.python_version()'

"$GPU_ENV/bin/python" -m ensurepip --upgrade

# Torch 2.7.0 is the oldest official cu128 wheel family supporting Blackwell.
# All non-Torch direct runtime versions remain those in the verified CPU env.
env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install \
  -r "$PAPER_ROOT/requirements-gpu.txt"

env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install --no-deps -e "$PAPER_ROOT"

"$GPU_ENV/bin/python" -m pip check

MPLCONFIGDIR="$REPRO_ROOT/matplotlib" \
  "$GPU_ENV/bin/python" - <<'PY'
import json
import platform
from importlib.metadata import version

import accelerate
import gin
import ignite
import lightgbm
import numpy
import pandas
import sklearn
import tables
import torch

from icu_benchmarks.data.loader import ICUVariableLengthDataset
from icu_benchmarks.models.wrappers import DLWrapper

versions = {
    "accelerate": accelerate.__version__,
    "gin": version("gin-config"),
    "ignite": ignite.__version__,
    "lightgbm": lightgbm.__version__,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "python": platform.python_version(),
    "scikit_learn": sklearn.__version__,
    "tables": tables.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}
expected = {
    "accelerate": "0.18.0",
    "gin": "0.5.0",
    "ignite": "0.4.12",
    "lightgbm": "3.3.5",
    "numpy": "1.24.4",
    "pandas": "1.5.3",
    "python": "3.9.25",
    "scikit_learn": "1.2.2",
    "tables": "3.8.0",
    "torch": "2.7.0+cu128",
    "torch_cuda": "12.8",
}
assert versions == expected, (versions, expected)
assert ICUVariableLengthDataset is not None
assert DLWrapper is not None

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not visible to the Step-wise GPU environment")
if torch.cuda.get_device_capability() != (12, 0):
    raise AssertionError(torch.cuda.get_device_capability())
a = torch.arange(4096, device="cuda", dtype=torch.float32).reshape(64, 64)
b = a @ a.T
torch.cuda.synchronize()
assert torch.isfinite(b).all().item()
versions["cuda_device"] = torch.cuda.get_device_name()
versions["cuda_capability"] = list(torch.cuda.get_device_capability())
print(json.dumps(versions, sort_keys=True))
PY

echo "Prepared $GPU_ENV"
