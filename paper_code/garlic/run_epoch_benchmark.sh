#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU=${1:-0}

mkdir -p "$REPRO_ROOT/audit" "$REPRO_ROOT/benchmark"
env CUDA_VISIBLE_DEVICES="$GPU" \
  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/benchmark_epoch.py" \
  --cache "$PAPER_ROOT/.export_assets/author_boundary.sqlite" \
  --orders "$REPRO_ROOT/ordered_splits.json" \
  --input-report "$REPRO_ROOT/audit/reproduction_inputs.json" \
  --checkpoint "$REPRO_ROOT/benchmark/epoch0_checkpoint.pt" \
  --output "$REPRO_ROOT/audit/epoch_benchmark.json" \
  --device cuda:0 2>&1 | tee "$REPRO_ROOT/audit/epoch_benchmark.log"

