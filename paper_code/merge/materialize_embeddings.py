#!/usr/bin/env python3
"""Call the released MERGE embedding functions on a frozen bounded input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import torchxrayvision as xrv


PAPER_ROOT = Path(__file__).resolve().parent
PREPROCESS_ROOT = PAPER_ROOT / "mimiciv" / "data_preprocess"
sys.path.insert(0, str(PREPROCESS_ROOT))

from preprocess_cxr_embeddings import process_img_embeddings_batched  # noqa: E402
from preprocess_notes_embeddings import process_notes_embeddings_batched  # noqa: E402


CXR_ALIAS = "densenet121-res224-chex"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    frame.to_parquet(partial)
    os.replace(partial, output)


def validate_vectors(values: pd.Series, width: int, name: str) -> int:
    count = 0
    for value in values:
        vectors = value if name == "biobert_embeddings" else [value]
        if vectors is None or len(vectors) == 0:
            raise RuntimeError(f"empty {name} value")
        for vector in vectors:
            array = np.asarray(vector)
            if array.shape != (width,) or not np.isfinite(array).all():
                raise RuntimeError(f"invalid {name} vector: {array.shape}")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=("notes", "cxr"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--encoder-root", type=Path, required=True)
    parser.add_argument("--cxr-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.reset_peak_memory_stats() if args.device.startswith("cuda") else None
    frame = pd.read_parquet(args.input)
    started = time.perf_counter()

    if args.modality == "notes":
        batch_size = args.batch_size or 16
        model_root = args.encoder_root / "biobert-v1.1"
        tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
        model = AutoModel.from_pretrained(model_root, local_files_only=True)
        result = process_notes_embeddings_batched(
            frame,
            tokenizer,
            model,
            args.device,
            chunk_batch_size=batch_size,
        )
        vector_count = validate_vectors(
            result["biobert_embeddings"], 768, "biobert_embeddings"
        )
        validation = {
            "rows": len(result),
            "chunk_embeddings": vector_count,
            "vector_width": 768,
        }
    else:
        if args.cxr_root is None:
            raise ValueError("--cxr-root is required for CXR embedding")
        batch_size = args.batch_size or 64
        model = xrv.models.DenseNet(
            weights=CXR_ALIAS,
            cache_dir=str(args.encoder_root / "torchxrayvision-chex"),
        ).to(args.device)
        model.eval()
        result = process_img_embeddings_batched(
            frame,
            model,
            args.device,
            str(args.cxr_root),
            batch_size=batch_size,
            num_workers=args.num_workers,
        )
        vector_count = validate_vectors(result["densefeatures"], 1024, "densefeatures")
        prediction_count = validate_vectors(result["predictions"], 18, "predictions")
        validation = {
            "rows": len(result),
            "dense_feature_vectors": vector_count,
            "prediction_vectors": prediction_count,
            "vector_width": 1024,
        }

    elapsed = time.perf_counter() - started
    atomic_parquet(result, args.output)
    report = {
        "status": "complete",
        "modality": args.modality,
        "released_function": (
            "preprocess_notes_embeddings.py::process_notes_embeddings_batched"
            if args.modality == "notes"
            else "preprocess_cxr_embeddings.py::process_img_embeddings_batched"
        ),
        "batch_size": batch_size,
        "num_workers": args.num_workers if args.modality == "cxr" else None,
        "input_sha256": sha256(args.input),
        "output_sha256": sha256(args.output),
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated()
        if args.device.startswith("cuda")
        else 0,
        "validation": validation,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
