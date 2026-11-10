#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for seed in 32 42 52 62 72; do
  "$PAPER_ROOT/run_table4_seed.sh" "$seed"
done
