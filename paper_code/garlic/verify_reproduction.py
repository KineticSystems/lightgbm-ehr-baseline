#!/usr/bin/env python3
"""Independently verify and classify the five-run GARLIC primary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from model import Model  # noqa: E402
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


def require_close(name: str, actual: float, expected: float) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
        raise AssertionError(f"{name} differs: {actual} != {expected}")


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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--lightgbm-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    result_path = args.run_root / "result.json"
    result = json.loads(result_path.read_text())
    input_report = json.loads(args.input_report.read_text())
    if result.get("status") != "completed" or input_report.get("status") != "exact":
        raise AssertionError("GARLIC primary or input verification is incomplete")
    expected_argv = [
        "--epochs", "100", "--data", "MIMICIII", "--seed", "0",
        "--run", "5", "--early_stopping", "10", "--lr_rec", "0.001",
        "--lr_cls", "0.00001", "--wd_rec", "0.00001", "--wd_cls", "0.0001",
        "--batch_size", "128", "--device", "cuda:0", "--feature_dim", "16",
        "--hidden_size", "128", "--window_size", "5", "--lambda_cls", "1",
        "--lambda_graph", "0.01", "--patience", "3", "--warm_up", "40",
        "--model", "GARLIC",
    ]
    if result["resolved_argv"] != expected_argv:
        raise AssertionError("GARLIC resolved source arguments changed")
    if result["split_sizes"] != {"train": 39500, "validation": 4938, "test": 4937}:
        raise AssertionError("GARLIC split sizes changed")
    if int(result["test_loader_accesses"]) != 5:
        raise AssertionError("GARLIC primary did not make one test pass per source run")

    store = GarlicDataStore(args.cache, args.orders)
    test_ids = np.asarray(store.splits["test"].ids, dtype=np.int64)
    expected_subjects = np.asarray(
        [store.rows[int(hadm)]["subject_id"] for hadm in test_ids], dtype=np.int64
    )
    expected_labels = np.asarray(
        [store.rows[int(hadm)]["label"] for hadm in test_ids], dtype=np.int64
    )
    store.close()

    rows = result["rows"]
    if [int(row["seed"]) for row in rows] != list(SEEDS):
        raise AssertionError("GARLIC result seed order changed")
    verified_rows = []
    checkpoint_hashes = set()
    prediction_hashes = set()
    parameter_count = None
    state_keys = None
    for row in rows:
        seed = int(row["seed"])
        if int(row["run"]) != seed or int(row["test"]["accesses"]) != 1:
            raise AssertionError(f"GARLIC run/seed/test count changed for seed {seed}")
        predictions_path = args.run_root / row["artifacts"]["predictions_relative_path"]
        with np.load(predictions_path) as archive:
            ids = archive["hadm_id"]
            subjects = archive["subject_id"]
            labels = archive["label"]
            scores = archive["probability"]
        if not np.array_equal(ids, test_ids):
            raise AssertionError(f"seed {seed} test admission order changed")
        if not np.array_equal(subjects, expected_subjects):
            raise AssertionError(f"seed {seed} test subject order changed")
        if not np.array_equal(labels, expected_labels):
            raise AssertionError(f"seed {seed} test label order changed")
        if scores.shape != (4937,) or scores.dtype != np.float32:
            raise AssertionError(f"seed {seed} probability shape/dtype changed")
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise AssertionError(f"seed {seed} probabilities are invalid")
        recomputed_auroc = float(roc_auc_score(labels, scores))
        recomputed_auprc = float(average_precision_score(labels, scores))
        require_close(f"seed {seed} AUROC", float(row["test"]["auroc"]), recomputed_auroc)
        require_close(f"seed {seed} AUPRC", float(row["test"]["auprc"]), recomputed_auprc)
        for name, values in {
            "ordered_hadm_id": ids,
            "ordered_subject_id": subjects,
            "ordered_label": labels,
            "probability": scores,
        }.items():
            if array_sha256(values) != row["test"][f"{name}_sha256"]:
                raise AssertionError(f"seed {seed} {name} array hash changed")

        selection = row["selection"]
        events = selection["save_events"]
        if not events or int(selection["selected_epoch"]) != int(events[-1]["epoch"]):
            raise AssertionError(f"seed {seed} selected checkpoint event changed")
        for earlier, later in zip(events, events[1:]):
            if not float(later["validation_classification_loss"]) < float(
                earlier["validation_classification_loss"]
            ):
                raise AssertionError(f"seed {seed} checkpoint history is not strict")
        require_close(
            f"seed {seed} selected validation loss",
            float(selection["selected_validation_classification_loss"]),
            float(events[-1]["validation_classification_loss"]),
        )

        checkpoint_path = args.run_root / row["artifacts"]["checkpoint_relative_path"]
        for path, bytes_key, hash_key in (
            (checkpoint_path, "checkpoint_bytes", "checkpoint_sha256"),
            (predictions_path, "predictions_bytes", "predictions_sha256"),
        ):
            if path.stat().st_size != int(row["artifacts"][bytes_key]):
                raise AssertionError(f"seed {seed} artifact size changed: {path}")
            if sha256_file(path) != row["artifacts"][hash_key]:
                raise AssertionError(f"seed {seed} artifact hash changed: {path}")
        if sha256_file(checkpoint_path) != events[-1]["sha256"]:
            raise AssertionError(f"seed {seed} checkpoint differs from its save event")

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not state_dict or any(not torch.isfinite(tensor).all() for tensor in state_dict.values()):
            raise AssertionError(f"seed {seed} checkpoint contains invalid tensors")
        model = Model(103, 16, 128, 5, 200, "cpu")
        model.load_state_dict(state_dict, strict=True)
        current_parameters = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count is None:
            parameter_count = current_parameters
            state_keys = len(state_dict)
        elif current_parameters != parameter_count or len(state_dict) != state_keys:
            raise AssertionError("GARLIC checkpoint architecture changed between seeds")
        checkpoint_hashes.add(sha256_file(checkpoint_path))
        prediction_hashes.add(sha256_file(predictions_path))
        verified_rows.append(
            {
                "seed": seed,
                "selected_epoch": int(selection["selected_epoch"]),
                "test_auroc": recomputed_auroc,
                "test_auprc": recomputed_auprc,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "predictions_sha256": sha256_file(predictions_path),
            }
        )

    if len(checkpoint_hashes) != 5 or len(prediction_hashes) != 5:
        raise AssertionError("GARLIC seeds did not produce five distinct artifacts")
    auroc = summarize([row["test_auroc"] for row in verified_rows])
    auprc = summarize([row["test_auprc"] for row in verified_rows])
    for metric, recomputed in (("auroc", auroc), ("auprc", auprc)):
        released = result["aggregate"][metric]
        for key in ("mean", "population_sd", "sample_sd", "sem"):
            require_close(f"aggregate {metric} {key}", float(released[key]), recomputed[key])
        if released["values"] != recomputed["values"]:
            raise AssertionError(f"aggregate {metric} seed values changed")

    auroc_match = 0.8964 <= auroc["mean"] <= 0.9054
    auprc_match = 0.6317 <= auprc["mean"] <= 0.6653
    lightgbm = json.loads(args.lightgbm_result.read_text())
    lightgbm_auroc = float(lightgbm["auroc"])
    author_log_path = args.run_root / result["artifacts"]["author_log_relative_path"]
    if author_log_path.stat().st_size != int(result["artifacts"]["author_log_bytes"]):
        raise AssertionError("GARLIC author log size changed")
    if sha256_file(author_log_path) != result["artifacts"]["author_log_sha256"]:
        raise AssertionError("GARLIC author log hash changed")

    report = {
        "status": "completed_match" if auroc_match and auprc_match else "completed_nonmatch",
        "fidelity": result["fidelity"],
        "seeds": list(SEEDS),
        "test_loader_accesses": 5,
        "rows": verified_rows,
        "auroc": {
            **auroc,
            "published_mean": 0.9009,
            "published_population_sd": 0.0045,
            "signed_mean_gap": auroc["mean"] - 0.9009,
        },
        "auprc": {
            **auprc,
            "published_mean": 0.6485,
            "published_population_sd": 0.0168,
            "signed_mean_gap": auprc["mean"] - 0.6485,
        },
        "match_rule": {
            "auroc_band": [0.8964, 0.9054],
            "auprc_band": [0.6317, 0.6653],
            "auroc_match": auroc_match,
            "auprc_match": auprc_match,
            "both_required": True,
        },
        "lightgbm": {
            "auroc": lightgbm_auroc,
            "garlic_mean_minus_lightgbm": auroc["mean"] - lightgbm_auroc,
            "result_sha256": sha256_file(args.lightgbm_result),
        },
        "checkpoint_state_keys": state_keys,
        "parameter_count": parameter_count,
        "runtime_seconds": float(result["runtime_seconds"]),
        "peak_cuda_allocated_bytes": int(result["peak_cuda_allocated_bytes"]),
        "peak_cuda_reserved_bytes": int(result["peak_cuda_reserved_bytes"]),
        "checks": [
            "unchanged released five-run argument vector",
            "one ordered source test pass per seed",
            "all prediction arrays and AUROC/AUPRC independently verified",
            "strict validation-loss checkpoint histories",
            "five selected checkpoints finite and strictly reloaded",
            "mean, population/sample SD, SEM, match rule, and LightGBM gap recomputed",
        ],
        "artifacts": {
            "result_sha256": sha256_file(result_path),
            "run_log_sha256": sha256_file(args.run_log),
            "author_log_sha256": sha256_file(author_log_path),
            "input_report_sha256": sha256_file(args.input_report),
            "orders_sha256": sha256_file(args.orders),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
