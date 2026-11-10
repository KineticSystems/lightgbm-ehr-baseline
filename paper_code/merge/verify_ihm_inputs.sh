#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
VERIFY_PYTHON="$PAPER_ROOT/.venv/bin/python"

if [[ ! -x "$VERIFY_PYTHON" ]]; then
  echo "Missing MERGE paper environment" >&2
  exit 1
fi

mkdir -p "$REPRO_ROOT/matplotlib"
mkdir -p "$REPRO_ROOT/audit"
export MPLCONFIGDIR="$REPRO_ROOT/matplotlib"
# A concurrent guarded job can legitimately hold enough of the fixed 36 GiB
# aggregate budget to delay this 18 GiB verifier. Keep the identical verifier
# queued for up to one day instead of treating the guard's 15-minute default
# acquisition timeout as a scientific failure.
"$VERIFY_PYTHON" "$PAPER_ROOT/verify_ihm_inputs.py" \
  --mimic-root "$MIMIC_IV_3_1_ROOT" \
  --cache-root "$PAPER_ROOT/.export_assets/source_cache" \
  --repro-root "$REPRO_ROOT" \
  --export-task "$WORKSPACE_ROOT/paper_tasks/merge/mortality_48h" \
  2>&1 | tee "$REPRO_ROOT/audit/verify_ihm_inputs.log"
