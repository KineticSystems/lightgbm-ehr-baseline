#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"
: "${MIMIC_CXR_JPG_2_0_0_ROOT:?MIMIC_CXR_JPG_2_0_0_ROOT must identify the exact 2.0.0 payload}"

REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
env PYTHONHASHSEED=0 \
  "$REPRO_ROOT/gpu_venv/bin/python" \
  "$PAPER_ROOT/assemble_verify_predictor_inputs.py" \
  --author-root "$PAPER_ROOT/.export_assets/author_source/prediction_cohort" \
  --generation-root "$REPRO_ROOT/generation" \
  --output-root "$REPRO_ROOT/predictor_input" \
  --cxr-root "$MIMIC_CXR_JPG_2_0_0_ROOT/files"
