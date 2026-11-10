#!/usr/bin/env python3
"""Compare the staged REMed smoke configuration with released CLI semantics."""

from __future__ import annotations

import json
from pathlib import Path

from evaluate_reproduction_seed import resolved_args as evaluation_args
from main import get_parser
from record_reproduction_stage import command as stage_command
from smoke_reproduction import (
    EXPECTED_MAX_EVENTS,
    EXPECTED_MAX_SEQUENCE,
    PRED_TARGETS,
    model_args,
)


SHARED_FIELDS = (
    "alibi_const",
    "batch_size",
    "dropout",
    "encoder_pooling",
    "max_retrieve_len",
    "max_word_len",
    "n_agg_layers",
    "n_enc_layers",
    "n_flatten_layers",
    "n_heads",
    "pos_enc",
    "pred_dim",
    "pred_pooling",
    "pred_time",
    "query_gen",
    "rejection_cutoff",
    "scorer",
    "scorer_use_time",
    "src_data",
    "train_cls_header",
    "train_type",
)


def released_args(train_type: str):
    command = [
        "--input_path",
        "unused",
        "--save_dir",
        "unused",
        "--src_data",
        "mimiciv",
        "--train_type",
        train_type,
        "--time",
        "-99999",
        "--pred_time",
        "48",
    ]
    if train_type == "short":
        command.extend(("--lr", "5e-5", "--random_sample", "--encode_events"))
    else:
        command.extend(
            (
                "--lr",
                "1e-5",
                "--scorer",
                "--scorer_use_time",
                "--pretrained",
                "encoder_2020",
                "--no_pretrained_checkpoint",
            )
        )
    return get_parser().parse_args(command)


def compare(train_type: str, maximum_sequence: int) -> dict:
    released = released_args(train_type)
    staged = model_args(train_type, maximum_sequence)
    if tuple(released.pred_targets) != PRED_TARGETS:
        raise AssertionError((released.pred_targets, PRED_TARGETS))
    mismatches = {}
    for field in SHARED_FIELDS:
        left, right = getattr(released, field), getattr(staged, field)
        if left != right:
            mismatches[field] = {"released": left, "staged": right}
    if mismatches:
        raise AssertionError(mismatches)
    if staged.local_batch_size != released.batch_size or released.batch_size != 8:
        raise AssertionError((staged.local_batch_size, released.batch_size))
    if train_type == "short":
        if released.max_seq_len != staged.max_seq_len or staged.max_seq_len != 512:
            raise AssertionError((released.max_seq_len, staged.max_seq_len))
        if not released.random_sample or released.lr != 5e-5:
            raise AssertionError((released.random_sample, released.lr))
    else:
        if staged.max_seq_len != EXPECTED_MAX_SEQUENCE:
            raise AssertionError(staged.max_seq_len)
        if released.lr != 1e-5 or not released.no_pretrained_checkpoint:
            raise AssertionError((released.lr, released.no_pretrained_checkpoint))
    if released.scheduler:
        raise AssertionError("literal README primary unexpectedly enables warmup")
    tasks = [
        {"name": task.name, "num_classes": task.num_classes, "property": task.property}
        for task in staged.tasks
    ]
    if len(tasks) != 27 or tasks[1]["name"] != "los_7":
        raise AssertionError(tasks)
    return {
        "batch_size": released.batch_size,
        "learning_rate": released.lr,
        "max_sequence": staged.max_seq_len,
        "scheduler": "LinearLR(1 -> 1 over 1 step; README omits --scheduler)",
        "tasks": tasks,
        "train_type": train_type,
    }


def verify_source_stage(stage: str, seed: int) -> dict:
    command = stage_command(stage, seed)
    if command[:7] != [
        "accelerate",
        "launch",
        "--config_file",
        "config/single.json",
        "--num_processes",
        "1",
        "--gpu_ids",
    ] or command[7:9] != ["0", "main.py"]:
        raise AssertionError(command[:9])
    if "--scheduler" in command:
        raise AssertionError(command)
    parsed = get_parser().parse_args(command[9:])
    if parsed.resume_name is not None:
        parsed.seed = int(parsed.resume_name.split("_")[-1])
    expected = {
        "encoder": {
            "encode_events": True,
            "lr": 5e-5,
            "no_pretrained_checkpoint": False,
            "pretrained": None,
            "random_sample": True,
            "resume_name": f"encoder_{seed}",
            "scorer": False,
            "scorer_use_time": False,
            "train_type": "short",
        },
        "remed": {
            "encode_events": False,
            "lr": 1e-5,
            "no_pretrained_checkpoint": True,
            "pretrained": f"encoder_{seed}",
            "random_sample": False,
            "resume_name": f"remed_{seed}",
            "scorer": True,
            "scorer_use_time": True,
            "train_type": "remed",
        },
    }[stage]
    observed = {key: getattr(parsed, key) for key in expected}
    if observed != expected:
        raise AssertionError((observed, expected))
    if (
        parsed.src_data != "mimiciv"
        or parsed.time != -99999
        or parsed.pred_time != 48
        or parsed.batch_size != 8
        or parsed.scheduler
        or parsed.seed != seed
        or parsed.test_only
        or parsed.encode_only
        or tuple(parsed.pred_targets) != PRED_TARGETS
    ):
        raise AssertionError(vars(parsed))
    return {
        "batch_size": parsed.batch_size,
        "encode_events": parsed.encode_events,
        "learning_rate": parsed.lr,
        "resume_name": parsed.resume_name,
        "scheduler": parsed.scheduler,
        "seed": parsed.seed,
        "source_test_after_selection": True,
        "train_type": parsed.train_type,
    }


def main() -> None:
    evaluation = evaluation_args(2020, Path("input"), Path("runs"))
    if (
        evaluation.train_type != "remed"
        or evaluation.src_data != "mimiciv"
        or evaluation.time != -99999
        or evaluation.pred_time != 48
        or evaluation.batch_size != 8
        or evaluation.lr != 1e-5
        or not evaluation.scorer
        or not evaluation.scorer_use_time
        or evaluation.scheduler
        or not evaluation.no_pretrained_checkpoint
        or not evaluation.test_only
        or evaluation.pretrained != "encoder_2020"
        or evaluation.resume_name != "remed_2020"
        or evaluation.seed != 2020
        or tuple(evaluation.pred_targets) != PRED_TARGETS
    ):
        raise AssertionError(vars(evaluation))
    report = {
        "evaluation": {
            "batch_size": evaluation.batch_size,
            "input_path": evaluation.input_path,
            "pretrained": evaluation.pretrained,
            "resume_name": evaluation.resume_name,
            "save_dir": evaluation.save_dir,
            "scheduler": evaluation.scheduler,
            "seed": evaluation.seed,
            "test_only": evaluation.test_only,
        },
        "maximum_source_events": EXPECTED_MAX_EVENTS,
        "maximum_rounded_sequence": EXPECTED_MAX_SEQUENCE,
        "remed": compare("remed", EXPECTED_MAX_SEQUENCE),
        "short": compare("short", 512),
        "source_stages": {
            stage: verify_source_stage(stage, 2020) for stage in ("encoder", "remed")
        },
        "status": "exact_released_smoke_configuration",
        "test_accesses": 0,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
