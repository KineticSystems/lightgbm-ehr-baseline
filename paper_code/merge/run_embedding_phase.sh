#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_PYTHON="$REPRO_ROOT/gpu_venv/bin/python"
SOURCE_ROOT="$PAPER_ROOT/.export_assets/source_cache"
ENCODER_ROOT="$PAPER_ROOT/.export_assets/external_artifacts/merge_encoders"
DATA_ROOT="$REPRO_ROOT/data"
BOUNDED_ROOT="$REPRO_ROOT/bounded_inputs"
BENCHMARK_ROOT="$REPRO_ROOT/benchmarks"
AUDIT_ROOT="$REPRO_ROOT/audit"
GPU=${2:-0}
PHASE=${1:-}

if [[ ! -x "$GPU_PYTHON" ]]; then
  echo "Missing MERGE GPU environment; run setup_gpu_env.sh first" >&2
  exit 1
fi
mkdir -p "$DATA_ROOT" "$AUDIT_ROOT" "$REPRO_ROOT/matplotlib"

common_env=(
  CUDA_VISIBLE_DEVICES="$GPU"
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  TOKENIZERS_PARALLELISM=false
  MPLCONFIGDIR="$REPRO_ROOT/matplotlib"
)

case "$PHASE" in
  prepare)
    "$GPU_PYTHON" "$PAPER_ROOT/prepare_embedding_inputs.py" \
      --structured "$SOURCE_ROOT/ts_labs_vitals.parquet" \
      --notes "$SOURCE_ROOT/rad_notes_text.parquet" \
      --cxr-metadata "$DATA_ROOT/cxr_metadata_with_time_delta.parquet" \
      --cxr-root "$MIMIC_CXR_JPG_2_0_0_ROOT" \
      --output-root "$REPRO_ROOT"
    ;;
  benchmark-notes)
    env "${common_env[@]}" \
      "$GPU_PYTHON" \
      "$PAPER_ROOT/materialize_embeddings.py" \
      --modality notes \
      --input "$BENCHMARK_ROOT/notes/rad_notes_text.parquet" \
      --output "$BENCHMARK_ROOT/notes/rad_notes_text_embeddings.parquet" \
      --report "$AUDIT_ROOT/benchmark_notes.json" \
      --encoder-root "$ENCODER_ROOT" --device cuda:0 --batch-size 16
    ;;
  benchmark-cxr)
    env "${common_env[@]}" \
      "$GPU_PYTHON" \
      "$PAPER_ROOT/materialize_embeddings.py" \
      --modality cxr \
      --input "$BENCHMARK_ROOT/cxr/cxr_metadata_with_time_delta.parquet" \
      --output "$BENCHMARK_ROOT/cxr/mimic_cxr_embeddings.parquet" \
      --report "$AUDIT_ROOT/benchmark_cxr.json" \
      --encoder-root "$ENCODER_ROOT" --cxr-root "$MIMIC_CXR_JPG_2_0_0_ROOT" \
      --device cuda:0 --batch-size 64 --num-workers 4
    ;;
  notes)
    env "${common_env[@]}" \
      "$GPU_PYTHON" \
      "$PAPER_ROOT/materialize_embeddings.py" \
      --modality notes --input "$SOURCE_ROOT/rad_notes_text.parquet" \
      --output "$DATA_ROOT/rad_notes_text_embeddings.parquet" \
      --report "$AUDIT_ROOT/notes_embeddings.json" \
      --encoder-root "$ENCODER_ROOT" --device cuda:0 --batch-size 16
    ;;
  cxr)
    env "${common_env[@]}" \
      "$GPU_PYTHON" \
      "$PAPER_ROOT/materialize_embeddings.py" \
      --modality cxr --input "$BOUNDED_ROOT/cxr/cxr_metadata_with_time_delta.parquet" \
      --output "$DATA_ROOT/mimic_cxr_embeddings.parquet" \
      --report "$AUDIT_ROOT/cxr_embeddings.json" \
      --encoder-root "$ENCODER_ROOT" --cxr-root "$MIMIC_CXR_JPG_2_0_0_ROOT" \
      --device cuda:0 --batch-size 64 --num-workers 4
    ;;
  *)
    echo "Usage: $0 {prepare|benchmark-notes|benchmark-cxr|notes|cxr} [gpu]" >&2
    exit 2
    ;;
esac
