#!/usr/bin/env python3
"""Verify the complete one-epoch Step-wise benchmark without test access."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import pickle
import re
from pathlib import Path

import torch


EXPECTED_STATE_KEYS = 84
EXPECTED_STATE_TENSOR_ELEMENTS = 518_530


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path(".export_assets/reproduction/epoch_benchmark"),
    )
    parser.add_argument(
        "--console-log",
        type=Path,
        default=Path(".export_assets/reproduction/epoch_benchmark.console.log"),
    )
    args = parser.parse_args()

    run_dir = args.benchmark_root / "1111"
    report_path = args.benchmark_root / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    required = [run_dir / "val_metrics.pkl", run_dir / "train_config.gin", args.console_log]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    for forbidden in (run_dir / "test_metrics.pkl", run_dir / "post_val_metrics.pkl"):
        if forbidden.exists():
            raise AssertionError(f"zero-test benchmark created {forbidden.name}")

    checkpoints = [Path(path) for path in glob.glob(str(run_dir / "model_e*.torch"))]
    if len(checkpoints) != 1 or checkpoints[0].name != "model_e0.torch":
        raise AssertionError("benchmark did not retain exactly the epoch-0 checkpoint")
    checkpoint_path = checkpoints[0]
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if len(state) != EXPECTED_STATE_KEYS:
        raise AssertionError(f"state has {len(state)} keys, expected {EXPECTED_STATE_KEYS}")
    if not all(torch.is_tensor(value) for value in state.values()):
        raise AssertionError("checkpoint is not a plain tensor state dictionary")
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise FloatingPointError("benchmark checkpoint contains a nonfinite tensor")
    parameter_count = sum(value.numel() for value in state.values())
    if parameter_count != EXPECTED_STATE_TENSOR_ELEMENTS:
        raise AssertionError(
            f"state tensor count {parameter_count} differs from "
            f"{EXPECTED_STATE_TENSOR_ELEMENTS}"
        )

    with (run_dir / "val_metrics.pkl").open("rb") as handle:
        raw_metrics = pickle.load(handle)
    metrics = {str(key): float(value) for key, value in raw_metrics.items()}
    if set(metrics) != {"AUC", "PR", "loss"}:
        raise AssertionError(f"unexpected validation metrics: {sorted(metrics)}")
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("benchmark validation metric is nonfinite")

    operative_config = (run_dir / "train_config.gin").read_text()
    expected_bindings = [
        "DLWrapper.train.batch_size = 64",
        "DLWrapper.train.epochs = 1",
        "train_common.do_test = False",
    ]
    for binding in expected_bindings:
        if binding not in operative_config:
            raise AssertionError(f"operative benchmark config lacks {binding!r}")

    console = args.console_log.read_text(errors="replace")
    if "TESTING" in console:
        raise AssertionError("benchmark console contains a test marker")
    if console.count("Train Epoch:1") != 1 or console.count("Val Epoch:1") != 1:
        raise AssertionError("benchmark console does not contain exactly one full epoch")
    matches = re.findall(r"training took:\s*([^\n]+)", console)
    if len(matches) != 1:
        raise AssertionError("could not identify one source-reported benchmark duration")

    report = {
        "status": "passed",
        "seed": 1111,
        "epochs": 1,
        "batch_size": 64,
        "test_accesses": 0,
        "source_reported_duration": matches[0].strip(),
        "validation_metrics": metrics,
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
            "state_keys": len(state),
            "state_tensor_elements": parameter_count,
            "all_finite": True,
        },
        "operative_config_sha256": sha256_file(run_dir / "train_config.gin"),
        "console_sha256": sha256_file(args.console_log),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
