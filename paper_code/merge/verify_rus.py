#!/usr/bin/env python3
"""Independently validate the complete released MERGE RUS artifact."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT / "mimiciv"))

from train_mimiciv_multimodal import load_mimiciv_rus_data  # noqa: E402


SEQUENCE_LENGTH = 48
EXPECTED_LAGS = list(range(0, SEQUENCE_LENGTH, 6))
GENERATION_MODALITIES = ["labs_vitals", "cxr", "notes"]
TRAINING_MODALITIES = sorted(GENERATION_MODALITIES)
MODALITY_DIMS = {"labs_vitals": 30, "cxr": 1024, "notes": 768}
EXPECTED_PAIRS = [
    ("labs_vitals", "cxr"),
    ("labs_vitals", "notes"),
    ("cxr", "notes"),
]
VALUE_KEYS = ("R_value", "U1_value", "U2_value", "S_value", "MI_value")
NORM_KEYS = ("R_norm", "U1_norm", "U2_norm", "S_norm")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def require_close(name: str, actual: float, expected: float) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
        raise AssertionError(f"{name} differs: {actual} != {expected}")


def verify_pair(result: dict, expected_pair: tuple[str, str]) -> dict:
    pair = tuple(result.get("feature_pair", ()))
    if pair != expected_pair:
        raise AssertionError(f"RUS pair order changed: {pair} != {expected_pair}")
    mod1, mod2 = pair
    if int(result["n_features_mod1"]) != MODALITY_DIMS[mod1]:
        raise AssertionError(f"{pair} first feature width changed")
    if int(result["n_features_mod2"]) != MODALITY_DIMS[mod2]:
        raise AssertionError(f"{pair} second feature width changed")

    lag_results = list(result.get("lag_results", []))
    lags = [int(row["lag"]) for row in lag_results]
    if lags != EXPECTED_LAGS:
        raise AssertionError(f"{pair} lag grid changed: {lags}")

    for index, row in enumerate(lag_results):
        values = np.asarray([row[key] for key in VALUE_KEYS + NORM_KEYS], dtype=float)
        if not np.isfinite(values).all():
            raise AssertionError(f"{pair} lag {lags[index]} contains nonfinite values")
        if any(float(row[key]) < 0.0 for key in VALUE_KEYS):
            raise AssertionError(f"{pair} lag {lags[index]} contains negative PID values")
        component_sum = sum(float(row[key]) for key in VALUE_KEYS[:4])
        require_close(f"{pair} lag {lags[index]} total", float(row["MI_value"]), component_sum)
        if float(row["MI_value"]) <= 1e-9:
            raise AssertionError(f"{pair} lag {lags[index]} has degenerate total information")
        for value_key, norm_key in zip(VALUE_KEYS[:4], NORM_KEYS):
            require_close(
                f"{pair} lag {lags[index]} {norm_key}",
                float(row[norm_key]),
                float(row[value_key]) / float(row["MI_value"]),
            )

    expected_averages = {}
    actual_averages = result.get("avg_metrics", {})
    for key in VALUE_KEYS + NORM_KEYS:
        # Match the released generator's dtype-preserving NumPy reduction.
        # Several estimator outputs are np.float32; coercing them to Python
        # floats before np.mean incorrectly changes the expected last bits.
        expected = np.mean([row[key] for row in lag_results])
        require_close(f"{pair} average {key}", float(actual_averages[key]), expected)
        expected_averages[key] = float(expected)
    return {
        "feature_pair": list(pair),
        "feature_dimensions": [MODALITY_DIMS[mod1], MODALITY_DIMS[mod2]],
        "lags": lags,
        "average_metrics": expected_averages,
    }


def independently_interpolate(results: list[dict]) -> dict[str, np.ndarray]:
    modality_to_index = {name: index for index, name in enumerate(TRAINING_MODALITIES)}
    unique = np.zeros((3, SEQUENCE_LENGTH), dtype=np.float32)
    redundancy = np.zeros((3, 3, SEQUENCE_LENGTH), dtype=np.float32)
    synergy = np.zeros((3, 3, SEQUENCE_LENGTH), dtype=np.float32)
    full_times = np.arange(SEQUENCE_LENGTH, dtype=np.float64)

    for result in results:
        mod1, mod2 = tuple(result["feature_pair"])
        first = modality_to_index[mod1]
        second = modality_to_index[mod2]
        lag_results = result["lag_results"]
        lag_times = np.asarray([int(row["lag"]) for row in lag_results], dtype=float)

        def interpolate(key: str) -> np.ndarray:
            values = np.asarray([float(row[key]) for row in lag_results], dtype=float)
            return np.interp(full_times, lag_times, values)

        r_values = interpolate("R_value").astype(np.float32)
        s_values = interpolate("S_value").astype(np.float32)
        redundancy[first, second] = r_values
        redundancy[second, first] = r_values
        synergy[first, second] = s_values
        synergy[second, first] = s_values
        unique[first] = np.maximum(unique[first].astype(float), interpolate("U1_value")).astype(np.float32)
        unique[second] = np.maximum(unique[second].astype(float), interpolate("U2_value")).astype(np.float32)
    return {"U": unique, "R": redundancy, "S": synergy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rus", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--train-pickle", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_report = json.loads(args.input_report.read_text())
    if input_report.get("status") != "exact":
        raise AssertionError("IHM input verification is not exact")
    train_sha256 = sha256_file(args.train_pickle)
    if train_sha256 != input_report["artifacts"]["pickles"]["train"]:
        raise AssertionError("RUS train pickle differs from verified input")
    if not args.run_log.is_file():
        raise AssertionError("RUS console log is missing")

    loaded = np.load(args.rus, allow_pickle=True)
    results = loaded.tolist()
    if not isinstance(results, list) or len(results) != len(EXPECTED_PAIRS):
        raise AssertionError("RUS artifact does not contain all three modality pairs")
    pair_reports = [
        verify_pair(result, expected_pair)
        for result, expected_pair in zip(results, EXPECTED_PAIRS)
    ]

    expected_tensors = independently_interpolate(results)
    with contextlib.redirect_stdout(io.StringIO()):
        released_tensors = load_mimiciv_rus_data(
            str(args.rus), TRAINING_MODALITIES, SEQUENCE_LENGTH
        )
    tensor_report = {}
    for name, expected in expected_tensors.items():
        actual = released_tensors[name].detach().cpu().numpy()
        if actual.dtype != np.float32 or not np.array_equal(actual, expected):
            maximum = float(np.max(np.abs(actual.astype(float) - expected.astype(float))))
            raise AssertionError(f"released {name} tensor differs; maximum={maximum}")
        if not np.isfinite(actual).all():
            raise AssertionError(f"released {name} tensor contains nonfinite values")
        tensor_report[name] = {
            "shape": list(actual.shape),
            "sha256": array_sha256(actual),
            "minimum": float(actual.min()),
            "maximum": float(actual.max()),
            "mean": float(actual.mean()),
        }

    report = {
        "status": "exact",
        "author_entrypoints": {
            "generator": "mimiciv/mimiciv_rus_multimodal.py",
            "estimator": "pid/temporal_pid_multi_sequence.py",
            "loader": "mimiciv/train_mimiciv_multimodal.py::load_mimiciv_rus_data",
        },
        "configuration": {
            "train_stays": input_report["splits"]["train"]["postfilter_count"],
            "sequence_length": SEQUENCE_LENGTH,
            "num_lags": len(EXPECTED_LAGS),
            "lags": EXPECTED_LAGS,
            "sequence_pooling": "mean",
            "batch_size": 256,
            "n_batches": 10,
            "discriminator_epochs": 100,
            "alignment_epochs": 10,
            "seed": 42,
        },
        "checks": [
            "verified train-pickle identity",
            "all three released modality pairs and feature widths",
            "all eight released lags for every pair",
            "finite nonnegative PID components and exact component totals",
            "normalized and average metrics independently recomputed",
            "released training interpolation independently reconstructed",
        ],
        "pairs": pair_reports,
        "training_tensors": tensor_report,
        "artifacts": {
            "rus_sha256": sha256_file(args.rus),
            "train_pickle_sha256": train_sha256,
            "input_report_sha256": sha256_file(args.input_report),
            "run_log_sha256": sha256_file(args.run_log),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
