#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SEED" >&2
  exit 2
fi

SEED=$1
case "$SEED" in
  1111|2222|3333|4444|5555) ;;
  *) echo "seed must be one of 1111, 2222, 3333, 4444, 5555" >&2; exit 2 ;;
esac

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
PRIMARY_ROOT="$REPRO_ROOT/primary"
RUN_DIR="$PRIMARY_ROOT/$SEED"
CONSOLE_LOG="$PRIMARY_ROOT/seed_${SEED}.console.log"
CONFIG="configs/reproduce/mimic/embeddings/ftt-no-group/transformer_ftt_decomp.gin"

if [[ ! -x "$GPU_ENV/bin/accelerate" ]]; then
  echo "GPU environment has not passed setup_gpu_env.sh" >&2
  exit 1
fi
if [[ ! -f "$REPRO_ROOT/smoke/seed_1111/smoke.json" ]]; then
  echo "Exact-loader real-GPU smoke has not completed" >&2
  exit 1
fi
if [[ ! -f "$REPRO_ROOT/epoch_benchmark/report.json" ]]; then
  echo "Complete one-epoch zero-test benchmark has not completed" >&2
  exit 1
fi
if ! env SMOKE_REPORT="$REPRO_ROOT/smoke/seed_1111/smoke.json" \
  "$GPU_ENV/bin/python" -c \
  'import json, os, pathlib; p=pathlib.Path(os.environ["SMOKE_REPORT"]); r=json.loads(p.read_text()); assert r["status"] == "passed" and r["test_accesses"] == 0' \
  2>/dev/null; then
  echo "Exact-loader smoke report is not a passing zero-test-access gate" >&2
  exit 1
fi
if ! env BENCHMARK_REPORT="$REPRO_ROOT/epoch_benchmark/report.json" \
  "$GPU_ENV/bin/python" -c \
  'import json, os, pathlib; p=pathlib.Path(os.environ["BENCHMARK_REPORT"]); r=json.loads(p.read_text()); assert r["status"] == "passed" and r["epochs"] == 1 and r["batch_size"] == 64 and r["test_accesses"] == 0' \
  2>/dev/null; then
  echo "One-epoch benchmark report is not an exact zero-test-access gate" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" || -e "$CONSOLE_LOG" ]]; then
  echo "Refusing to overwrite existing seed-$SEED artifacts" >&2
  exit 1
fi
mkdir -p "$PRIMARY_ROOT" "$REPRO_ROOT/matplotlib"

cd "$PAPER_ROOT"
env MPLCONFIGDIR="$REPRO_ROOT/matplotlib" PYTHONUNBUFFERED=1 \
  "$GPU_ENV/bin/accelerate" launch \
  --config_file configs/accelerate/accel_config_gpu1.yml \
  ./icu_benchmarks/run.py train \
  -c "$CONFIG" \
  -l "$PRIMARY_ROOT" \
  -sd "$SEED" 2>&1 | tee "$CONSOLE_LOG"

checkpoints=("$RUN_DIR"/model_e*.torch)
if [[ ${#checkpoints[@]} -ne 1 || ! -s "${checkpoints[0]}" ]]; then
  echo "Expected exactly one validation-loss-selected checkpoint" >&2
  exit 1
fi
test -s "$RUN_DIR/val_metrics.pkl"
test -s "$RUN_DIR/test_metrics.pkl"
test -s "$RUN_DIR/post_val_metrics.pkl"
test -s "$RUN_DIR/train_config.gin"

VERIFICATION_LOG="$RUN_DIR/verification.console.log"
env MPLCONFIGDIR="$REPRO_ROOT/matplotlib" PYTHONUNBUFFERED=1 \
  "$GPU_ENV/bin/python" evaluate_reproduction_seed.py \
  --seed "$SEED" \
  --run-dir "$RUN_DIR" \
  --source-console-log "$CONSOLE_LOG" 2>&1 | tee "$VERIFICATION_LOG"

test -s "$RUN_DIR/verification/report.json"
