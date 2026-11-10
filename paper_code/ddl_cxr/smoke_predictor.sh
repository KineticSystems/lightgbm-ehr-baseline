#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"
: "${MIMIC_CXR_JPG_2_0_0_ROOT:?MIMIC_CXR_JPG_2_0_0_ROOT must identify the exact 2.0.0 payload}"

REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
WEIGHT="$PAPER_ROOT/.export_assets/external_artifacts/torchvision_resnet34_imagenet1k_v1/resnet34-b627a593.pth"
CACHE_WEIGHT="$REPRO_ROOT/torch_home/hub/checkpoints/resnet34-b627a593.pth"
EXPECTED_SHA=b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f
[[ "$(sha256sum "$WEIGHT" | cut -d' ' -f1)" == "$EXPECTED_SHA" ]]
mkdir -p "$(dirname "$CACHE_WEIGHT")"
if [[ -L "$CACHE_WEIGHT" ]]; then
  [[ "$(readlink -f "$CACHE_WEIGHT")" == "$(readlink -f "$WEIGHT")" ]]
elif [[ -e "$CACHE_WEIGHT" ]]; then
  echo "Refusing non-link cached ResNet weight: $CACHE_WEIGHT" >&2
  exit 1
else
  ln -s "$WEIGHT" "$CACHE_WEIGHT"
fi

cd "$PAPER_ROOT"
env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
  PYTHONUNBUFFERED=1 TORCH_HOME="$REPRO_ROOT/torch_home" \
  "$REPRO_ROOT/gpu_venv/bin/python" "$PAPER_ROOT/smoke_predictor.py" \
  --input-root "$REPRO_ROOT/predictor_input" \
  --cxr-root "$MIMIC_CXR_JPG_2_0_0_ROOT/files" \
  --resnet-weight "$WEIGHT" \
  --output-root "$REPRO_ROOT/smoke/predictor_one_step"
