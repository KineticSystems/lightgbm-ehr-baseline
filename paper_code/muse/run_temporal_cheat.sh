#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
COHORT="$PAPER_ROOT/.export_assets/source_cache/cohort.parquet"
STRUCTURED_RESULT="$WORKSPACE_ROOT/paper_results/muse/mortality_90d_post_discharge"
OUTPUT="$WORKSPACE_ROOT/paper_results/muse/sensitivities/temporal_cheat"

[[ -f "$COHORT" ]] || { echo "Missing reconstructed MUSE cohort: $COHORT" >&2; exit 1; }
[[ -f "$STRUCTURED_RESULT/predictions.npz" ]] || {
  echo "Missing structured predictions: $STRUCTURED_RESULT/predictions.npz" >&2
  exit 1
}
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing sensitivity: $OUTPUT" >&2
  exit 1
fi

POLARS_MAX_THREADS=1 MXGB_THREADS=1 \
  "$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" \
  "$PAPER_ROOT/evaluate_temporal_cheat.py" \
  --cohort "$COHORT" \
  --structured-result-dir "$STRUCTURED_RESULT" \
  --output-dir "$OUTPUT"

echo "Wrote $OUTPUT"
