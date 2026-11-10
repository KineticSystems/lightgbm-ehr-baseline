#!/usr/bin/env python3
"""Freeze one completed source REMed stage without opening clinical data."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, include_hash: bool = True) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    result = {"bytes": path.stat().st_size, "path": str(path)}
    if include_hash:
        result["sha256"] = sha256(path)
    return result


def source_test_metrics(path: Path) -> list[dict[str, float]]:
    found = []
    for line in path.read_text(errors="replace").splitlines():
        position = line.find("{'epoch':")
        if position < 0:
            continue
        try:
            candidate = ast.literal_eval(line[position:])
        except (SyntaxError, ValueError):
            continue
        if isinstance(candidate, dict) and "test" in candidate:
            found.append(
                {key: float(value) for key, value in candidate["test"].items()}
            )
    if not found:
        raise AssertionError(f"no source test metric dictionary in {path}")
    return found


def command(stage: str, seed: int) -> list[str]:
    shared = [
        "accelerate",
        "launch",
        "--config_file",
        "config/single.json",
        "--num_processes",
        "1",
        "--gpu_ids",
        "0",
        "main.py",
        "--src_data",
        "mimiciv",
        "--input_path",
        ".export_assets/reproduction/input",
        "--save_dir",
        ".export_assets/reproduction/runs",
    ]
    if stage == "encoder":
        return shared + [
            "--train_type",
            "short",
            "--time",
            "-99999",
            "--pred_time",
            "48",
            "--lr",
            "5e-5",
            "--random_sample",
            "--encode_events",
            "--resume_name",
            f"encoder_{seed}",
        ]
    return shared + [
        "--train_type",
        "remed",
        "--time",
        "-99999",
        "--pred_time",
        "48",
        "--lr",
        "1e-5",
        "--scorer",
        "--scorer_use_time",
        "--pretrained",
        f"encoder_{seed}",
        "--no_pretrained_checkpoint",
        "--resume_name",
        f"remed_{seed}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("encoder", "remed"), required=True)
    parser.add_argument("--seed", type=int, choices=(2020, 2021, 2022), required=True)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--console-log", type=Path, required=True)
    parser.add_argument("--environment-report", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.attempts < 1:
        raise AssertionError(args.attempts)
    environment = json.loads(args.environment_report.read_text())
    input_report = json.loads(args.input_report.read_text())
    if environment.get("status") != "passed":
        raise AssertionError(environment.get("status"))
    if input_report.get("status") != "exact_paper_local_input_shim":
        raise AssertionError(input_report.get("status"))

    run_dir = args.save_root / f"{args.stage}_{args.seed}"
    checkpoint = run_dir / "checkpoint_best.pt"
    resume_state = run_dir / "checkpoint"
    if resume_state.exists():
        raise AssertionError(f"source resume state remains after terminal stage: {resume_state}")
    metrics = source_test_metrics(args.console_log)
    artifacts = {
        "checkpoint_best": artifact(checkpoint),
        "console_log": artifact(args.console_log),
    }
    if args.stage == "encoder":
        encoded_hdf = run_dir / "mimiciv_encoded.h5"
        artifacts["encoded_hdf"] = artifact(encoded_hdf, include_hash=False)
        artifacts["encoded_hdf"]["sha256"] = "pending_full_cache_verification"
    else:
        encoder_report = args.save_root / f"encoder_{args.seed}" / "encoded_cache_report.json"
        exact_batch_report = (
            args.save_root / f"encoder_{args.seed}" / "exact_maximum_batch_report.json"
        )
        artifacts["encoded_cache_report"] = artifact(encoder_report)
        artifacts["exact_maximum_batch_report"] = artifact(exact_batch_report)
        if json.loads(encoder_report.read_text()).get("status") != "passed":
            raise AssertionError("encoded cache report is not passing")
        if json.loads(exact_batch_report.read_text()).get("status") != "passed":
            raise AssertionError("exact maximum batch report is not passing")

    report = {
        "artifacts": artifacts,
        "attempts": args.attempts,
        "command": command(args.stage, args.seed),
        "environment_report_sha256": sha256(args.environment_report),
        "input_report_sha256": sha256(args.input_report),
        "literal_readme_scheduler_flag": False,
        "seed": args.seed,
        "source_test_accesses": len(metrics),
        "source_test_metrics": metrics,
        "stage": args.stage,
        "status": "terminal_source_stage",
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
