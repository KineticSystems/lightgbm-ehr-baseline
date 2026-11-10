#!/usr/bin/env bash
# Build the released PyHealth parse for one dataset, in patient shards.
# Usage: build_parse_cache.sh <mimic3|mimic4> [extra build_parse_cache.py args]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_phase2_common.sh"

DATASET="${1:-}"
shift || true
case "$DATASET" in
  mimic3) ROOT="${MIMIC_III_1_4_ROOT:?Set the exact MIMIC-III 1.4 source root}" ;;
  mimic4) ROOT="${MIMIC_IV_2_0_ROOT:?Set the exact MIMIC-IV 2.0 source root}/hosp" ;;
  *) echo "usage: $(basename "$0") <mimic3|mimic4> [args]" >&2; exit 2 ;;
esac

run_capped "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/build_parse_cache.py" \
  --dataset "$DATASET" --root "$ROOT" "$@"
