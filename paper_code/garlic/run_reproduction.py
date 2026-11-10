#!/usr/bin/env python3
"""Run the released five-run GARLIC primary through the exact data adapter."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import sklearn.metrics
import torch


PAPER_ROOT = Path(__file__).resolve().parent
SOURCE_MAIN = (PAPER_ROOT / "main.py").resolve()
sys.path.insert(0, str(PAPER_ROOT))

import utils  # noqa: E402
from reproduction_data import GarlicDataStore  # noqa: E402


SEEDS = tuple(range(5))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def source_run_context() -> tuple[int, int]:
    frame = inspect.currentframe().f_back
    while frame is not None:
        if Path(frame.f_code.co_filename).resolve() == SOURCE_MAIN:
            if "run" in frame.f_locals and "seed" in frame.f_locals:
                return int(frame.f_locals["run"]), int(frame.f_locals["seed"])
        frame = frame.f_back
    raise AssertionError("could not identify the active released GARLIC run")


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "values": values,
        "mean": float(array.mean()),
        "population_sd": float(array.std(ddof=0)),
        "sample_sd": float(array.std(ddof=1)),
        "sem": float(array.std(ddof=1) / math.sqrt(len(array))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--orders", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()

    input_report = json.loads(args.input_report.read_text())
    if input_report.get("status") != "exact":
        raise AssertionError("GARLIC exact input adapter has not passed")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("GARLIC primary reproduction requires real CUDA")

    args.output_root.mkdir(parents=True)
    store = GarlicDataStore(args.cache, args.orders)
    train_data = store.splits["train"]
    validation_data = store.splits["validation"]
    test_data = store.splits["test"]
    if (len(train_data), len(validation_data), len(test_data)) != (39500, 4938, 4937):
        raise AssertionError("GARLIC split sizes changed")

    # Only storage/split injection and passive artifact capture are patched.
    # The released main.py loop, five-run process, losses, phase switching,
    # checkpoint selection, and one final test pass per run execute unchanged.
    original_load_data = utils.load_data
    original_get_data_split = utils.get_data_split
    original_roc_auc = sklearn.metrics.roc_auc_score
    original_average_precision = sklearn.metrics.average_precision_score
    original_torch_save = torch.save
    save_events = {seed: [] for seed in SEEDS}
    test_captures = {}

    def adapted_load_data(data_name):
        if data_name != "MIMICIII":
            raise AssertionError(f"unexpected GARLIC dataset {data_name}")
        return object()

    def adapted_get_data_split(_data):
        return train_data, validation_data, test_data

    def captured_roc_auc(labels, scores, *positional, **keywords):
        result = original_roc_auc(labels, scores, *positional, **keywords)
        labels_array = np.asarray(labels)
        scores_array = np.asarray(scores)
        if len(labels_array) == len(test_data):
            run, seed = source_run_context()
            if run != seed or seed not in SEEDS or seed in test_captures:
                raise AssertionError(f"unexpected or repeated GARLIC test context {run}/{seed}")
            test_captures[seed] = {
                "labels": labels_array.copy(),
                "scores": scores_array.copy(),
                "auroc": float(result),
            }
        return result

    def captured_average_precision(labels, scores, *positional, **keywords):
        result = original_average_precision(labels, scores, *positional, **keywords)
        labels_array = np.asarray(labels)
        scores_array = np.asarray(scores)
        if len(labels_array) == len(test_data):
            run, seed = source_run_context()
            if run != seed or seed not in test_captures:
                raise AssertionError(f"test AUPRC has no matching AUROC for {run}/{seed}")
            capture = test_captures[seed]
            if not np.array_equal(capture["labels"], labels_array):
                raise AssertionError("test metric calls received different labels")
            if not np.array_equal(capture["scores"], scores_array):
                raise AssertionError("test metric calls received different scores")
            capture["auprc"] = float(result)
        return result

    def captured_torch_save(obj, path, *positional, **keywords):
        result = original_torch_save(obj, path, *positional, **keywords)
        frame = inspect.currentframe().f_back
        run, seed = source_run_context()
        if run != seed or seed not in SEEDS:
            raise AssertionError(f"unexpected GARLIC checkpoint context {run}/{seed}")
        event = {
            "epoch": int(frame.f_locals["epoch"]),
            "mode": str(frame.f_locals["mode"]),
            "validation_classification_loss": float(frame.f_locals["val_pred_loss"]),
            "best_validation_loss": float(frame.f_locals["best_val_loss"]),
            "path": str(Path(path).resolve().relative_to(args.output_root)),
            "sha256": sha256_file(Path(path)),
        }
        save_events[seed].append(event)
        return result

    utils.load_data = adapted_load_data
    utils.get_data_split = adapted_get_data_split
    sklearn.metrics.roc_auc_score = captured_roc_auc
    sklearn.metrics.average_precision_score = captured_average_precision
    torch.save = captured_torch_save

    resolved_argv = [
        str(SOURCE_MAIN),
        "--epochs", "100",
        "--data", "MIMICIII",
        "--seed", "0",
        "--run", "5",
        "--early_stopping", "10",
        "--lr_rec", "0.001",
        "--lr_cls", "0.00001",
        "--wd_rec", "0.00001",
        "--wd_cls", "0.0001",
        "--batch_size", "128",
        "--device", args.device,
        "--feature_dim", "16",
        "--hidden_size", "128",
        "--window_size", "5",
        "--lambda_cls", "1",
        "--lambda_graph", "0.01",
        "--patience", "3",
        "--warm_up", "40",
        "--model", "GARLIC",
    ]
    original_argv = sys.argv
    original_cwd = Path.cwd()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        os.chdir(args.output_root)
        sys.argv = resolved_argv
        runpy.run_path(str(SOURCE_MAIN), run_name="__main__")
        torch.cuda.synchronize(device)
    finally:
        runtime_seconds = time.perf_counter() - started
        sys.argv = original_argv
        os.chdir(original_cwd)
        utils.load_data = original_load_data
        utils.get_data_split = original_get_data_split
        sklearn.metrics.roc_auc_score = original_roc_auc
        sklearn.metrics.average_precision_score = original_average_precision
        torch.save = original_torch_save
        store.close()

    logs = list(args.output_root.glob("logs/MIMICIII/GARLIC_5_*.log"))
    checkpoints = list(args.output_root.glob("models/MIMICIII/GARLIC_*_best_model.pth"))
    if len(logs) != 1 or len(checkpoints) != 5:
        raise AssertionError(
            f"expected one log and five checkpoints, found {len(logs)}/{len(checkpoints)}"
        )
    if set(test_captures) != set(SEEDS):
        raise AssertionError(f"released loop test captures are incomplete: {sorted(test_captures)}")

    ids = np.asarray(test_data.ids, dtype=np.int64)
    subjects = np.asarray([store.rows[int(hadm)]["subject_id"] for hadm in ids], dtype=np.int64)
    expected_labels = np.asarray([store.rows[int(hadm)]["label"] for hadm in ids], dtype=np.int64)
    rows = []
    for seed in SEEDS:
        events = save_events[seed]
        capture = test_captures[seed]
        if not events or "auprc" not in capture:
            raise AssertionError(f"GARLIC seed {seed} lacks selection or test evidence")
        for earlier, later in zip(events, events[1:]):
            if not later["validation_classification_loss"] < earlier[
                "validation_classification_loss"
            ]:
                raise AssertionError(f"seed {seed} checkpoint saves are not strict improvements")
        checkpoint_path = args.output_root / events[-1]["path"]
        if sha256_file(checkpoint_path) != events[-1]["sha256"]:
            raise AssertionError(f"seed {seed} selected checkpoint changed after saving")
        labels = np.asarray(capture["labels"], dtype=np.int64)
        scores = np.asarray(capture["scores"], dtype=np.float32)
        if not np.array_equal(labels, expected_labels):
            raise AssertionError(f"seed {seed} test label order differs from the adapter")
        predictions_path = args.output_root / f"test_predictions_seed_{seed}.npz"
        np.savez_compressed(
            predictions_path,
            hadm_id=ids,
            subject_id=subjects,
            label=labels,
            probability=scores,
        )
        rows.append(
            {
                "run": seed,
                "seed": seed,
                "selection": {
                    "criterion": "strict minimum validation classification loss",
                    "save_events": events,
                    "selected_epoch": events[-1]["epoch"],
                    "selected_validation_classification_loss": events[-1][
                        "validation_classification_loss"
                    ],
                },
                "test": {
                    "accesses": 1,
                    "count": len(labels),
                    "positives": int(labels.sum()),
                    "auroc": capture["auroc"],
                    "auprc": capture["auprc"],
                    "ordered_hadm_id_sha256": array_sha256(ids),
                    "ordered_subject_id_sha256": array_sha256(subjects),
                    "ordered_label_sha256": array_sha256(labels),
                    "probability_sha256": array_sha256(scores),
                },
                "artifacts": {
                    "checkpoint_relative_path": events[-1]["path"],
                    "checkpoint_bytes": checkpoint_path.stat().st_size,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "predictions_relative_path": str(predictions_path.relative_to(args.output_root)),
                    "predictions_bytes": predictions_path.stat().st_size,
                    "predictions_sha256": sha256_file(predictions_path),
                },
            }
        )

    auroc = summarize([float(row["test"]["auroc"]) for row in rows])
    auprc = summarize([float(row["test"]["auprc"]) for row in rows])
    report = {
        "status": "completed",
        "fidelity": "unchanged released main.py five-run process with exact storage/split adapter",
        "resolved_argv": resolved_argv[1:],
        "split_sizes": {"train": 39500, "validation": 4938, "test": 4937},
        "test_loader_accesses": 5,
        "rows": rows,
        "aggregate": {"auroc": auroc, "auprc": auprc},
        "runtime_seconds": runtime_seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "artifacts": {
            "author_log_relative_path": str(logs[0].relative_to(args.output_root)),
            "author_log_bytes": logs[0].stat().st_size,
            "author_log_sha256": sha256_file(logs[0]),
            "input_report_sha256": sha256_file(args.input_report),
            "orders_sha256": sha256_file(args.orders),
        },
    }
    report_path = args.output_root / "result.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
