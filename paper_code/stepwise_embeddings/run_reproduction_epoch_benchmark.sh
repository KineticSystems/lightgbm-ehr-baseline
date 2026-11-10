#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
BENCHMARK_ROOT="$REPRO_ROOT/epoch_benchmark"
RUN_DIR="$BENCHMARK_ROOT/1111"
CONSOLE_LOG="$REPRO_ROOT/epoch_benchmark.console.log"
CONFIG="configs/reproduce/mimic/embeddings/ftt-no-group/transformer_ftt_decomp.gin"
OVERRIDE="configs/reproduce/mimic/embeddings/ftt-no-group/one_epoch_no_test.gin"

if [[ ! -x "$GPU_ENV/bin/accelerate" ]]; then
  echo "GPU environment has not passed setup_gpu_env.sh" >&2
  exit 1
fi
if [[ ! -f "$REPRO_ROOT/smoke/seed_1111/smoke.json" ]]; then
  echo "Exact-loader real-GPU smoke has not completed" >&2
  exit 1
fi
if [[ -e "$BENCHMARK_ROOT" || -e "$CONSOLE_LOG" ]]; then
  echo "Refusing to overwrite existing epoch-benchmark artifacts" >&2
  exit 1
fi
mkdir -p "$REPRO_ROOT/matplotlib"

cd "$PAPER_ROOT"
env MPLCONFIGDIR="$REPRO_ROOT/matplotlib" PYTHONUNBUFFERED=1 \
  "$GPU_ENV/bin/accelerate" launch \
  --config_file configs/accelerate/accel_config_gpu1.yml \
  ./icu_benchmarks/run.py train \
  -c "$CONFIG" "$OVERRIDE" \
  -l "$BENCHMARK_ROOT" \
  -sd 1111 2>&1 | tee "$CONSOLE_LOG"

checkpoints=("$RUN_DIR"/model_e*.torch)
if [[ ${#checkpoints[@]} -ne 1 || ! -s "${checkpoints[0]}" ]]; then
  echo "Expected one epoch-0 benchmark checkpoint" >&2
  exit 1
fi
test -s "$RUN_DIR/val_metrics.pkl"
test -s "$RUN_DIR/train_config.gin"
if [[ -e "$RUN_DIR/test_metrics.pkl" || -e "$RUN_DIR/post_val_metrics.pkl" ]]; then
  echo "Benchmark unexpectedly accessed a test or post-validation path" >&2
  exit 1
fi

"$GPU_ENV/bin/python" verify_epoch_benchmark.py \
  --benchmark-root "$BENCHMARK_ROOT" \
  --console-log "$CONSOLE_LOG"
test -s "$BENCHMARK_ROOT/report.json"
