#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"

for seed in 1111 2222 3333 4444 5555; do
  "$PAPER_ROOT/run_reproduction_seed.sh" "$seed"
done

cd "$PAPER_ROOT"
"$REPRO_ROOT/gpu_venv/bin/python" summarize_reproduction.py
