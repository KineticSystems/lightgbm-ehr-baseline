#!/usr/bin/env python3
"""Run both released REMed updates on the longest exact training-loader batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from h5pickle import File
from torch.utils.data import DataLoader, Subset

from main import get_parser
from record_reproduction_stage import command as stage_command
from src.models import REMed
from src.trainer.remed import REMedTrainer
from src.utils.trainer_utils import PredLoss, PredMetric


EXPECTED_GLOBAL_MAX = 266_766
EXPECTED_MAX_SEQUENCE = 266_880


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_args(seed: int, input_root: Path, save_root: Path):
    command = stage_command("remed", seed)
    args = get_parser().parse_args(command[9:])
    args.input_path = str(input_root)
    args.save_dir = str(save_root)
    args.seed = seed
    return args


def memory() -> dict[str, int]:
    torch.cuda.synchronize()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def loader_for_key(dataset, key: str, batch_size: int):
    positions = np.flatnonzero(dataset.keys == key)
    if len(positions) != 1:
        raise AssertionError((key, positions.tolist()))
    position = int(positions[0])
    start = position - position % batch_size
    stop = min(start + batch_size, len(dataset))
    indices = list(range(start, stop))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        persistent_workers=True,
    )
    return loader, position, indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(2020, 2021, 2022), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--environment-report", type=Path, required=True)
    parser.add_argument("--small-smoke-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args_cli = parser.parse_args()

    if args_cli.output.exists():
        raise FileExistsError(args_cli.output)
    environment = json.loads(args_cli.environment_report.read_text())
    cache = json.loads(args_cli.cache_report.read_text())
    small_smoke = json.loads(args_cli.small_smoke_report.read_text())
    if environment.get("status") != "passed":
        raise AssertionError(environment.get("status"))
    if cache.get("status") != "passed":
        raise AssertionError(cache.get("status"))
    if small_smoke.get("status") != "passed":
        raise AssertionError(small_smoke.get("status"))
    if cache["global_longest"]["events"] != EXPECTED_GLOBAL_MAX:
        raise AssertionError(cache["global_longest"])

    args = resolved_args(args_cli.seed, args_cli.input_root, args_cli.save_root)
    trainer = REMedTrainer(args)
    if args.max_seq_len != EXPECTED_MAX_SEQUENCE:
        raise AssertionError(args.max_seq_len)
    accelerator = Accelerator(mixed_precision="bf16", split_batches=True)
    if accelerator.num_processes != 1:
        raise AssertionError(accelerator.num_processes)
    args.local_batch_size = args.batch_size // accelerator.num_processes
    if args.local_batch_size != 8:
        raise AssertionError(args.local_batch_size)

    report = {
        "cache_report_sha256": sha256(args_cli.cache_report),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_device": torch.cuda.get_device_name(0),
        "environment_report_sha256": sha256(args_cli.environment_report),
        "global_longest_events": EXPECTED_GLOBAL_MAX,
        "seed": args_cli.seed,
        "small_smoke_report_sha256": sha256(args_cli.small_smoke_report),
        "status": "running",
        "test_metric_accesses": 0,
    }
    trainer.data = File(trainer.data_path, "r")
    try:
        dataset = trainer.dataset(args, trainer.train_subset, trainer.data, trainer.df)
        longest_key = str(cache["per_split_longest"]["train"]["stay_id"])
        loader, position, indices = loader_for_key(dataset, longest_key, args.batch_size)
        model = REMed(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1, total_iters=1
        )
        loader, optimizer, scheduler, model = accelerator.prepare(
            loader, optimizer, scheduler, model
        )
        criterion = PredLoss(args)
        metric = PredMetric(args)
        sample = next(iter(loader))
        if sample["repr"].shape[0] != len(indices):
            raise AssertionError((sample["repr"].shape, indices))
        longest_training_events = cache["per_split_longest"]["train"]["events"]
        expected_padding = min(
            2 ** math.ceil(math.log(longest_training_events, 2)),
            EXPECTED_MAX_SEQUENCE,
        )
        if sample["repr"].shape[1] != expected_padding:
            raise AssertionError((sample["repr"].shape, expected_padding))
        training_sample_shape = list(sample["repr"].shape)

        model.train()
        steps = {}
        for mode in ("scorer", "predictor"):
            model.set_mode(mode)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            output, representations = model(**sample)
            loss, logging_outputs = criterion(output, representations)
            if not torch.isfinite(loss).item():
                raise AssertionError(loss)
            accelerator.backward(loss)
            finite_gradient_tensors = 0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all().item():
                        raise AssertionError(f"non-finite {mode} gradient")
                    finite_gradient_tensors += 1
            if finite_gradient_tensors == 0:
                raise AssertionError(f"no {mode} gradients")
            optimizer.step()
            scheduler.step()
            metric(logging_outputs, accelerator=None)
            los_probability = output["pred"]["los_7"].detach().float()
            if not torch.isfinite(los_probability).all().item():
                raise AssertionError("non-finite LOS-7 probability")
            steps[mode] = {
                "finite_gradient_tensors": finite_gradient_tensors,
                "loss": float(loss.detach().float().cpu()),
                "los_7_probability_first": float(los_probability.reshape(-1)[0].cpu()),
                "memory": memory(),
            }
            del output, representations, loss, logging_outputs, los_probability

        global_key = str(cache["global_longest"]["stay_id"])
        matching_splits = [
            split
            for split, details in cache["per_split_longest"].items()
            if str(details["stay_id"]) == global_key
        ]
        if len(matching_splits) != 1:
            raise AssertionError((global_key, matching_splits))
        global_split = matching_splits[0]
        if global_split == trainer.train_subset and global_key == longest_key:
            global_sample = sample
            global_position = position
            global_indices = indices
        else:
            del sample
            torch.cuda.empty_cache()
            global_dataset = trainer.dataset(args, global_split, trainer.data, trainer.df)
            global_loader, global_position, global_indices = loader_for_key(
                global_dataset, global_key, args.batch_size
            )
            global_loader = accelerator.prepare(global_loader)
            global_sample = next(iter(global_loader))
        if global_sample["repr"].shape[1] != EXPECTED_MAX_SEQUENCE:
            raise AssertionError(global_sample["repr"].shape)

        model.eval()
        model.set_mode("predictor")
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            global_output, global_representations = model(**global_sample)
            global_loss, global_logging = criterion(
                global_output, global_representations
            )
        if not torch.isfinite(global_loss).item():
            raise AssertionError(global_loss)
        global_batch = {
            "batch_indices": global_indices,
            "batch_size": len(global_indices),
            "loss": float(global_loss.detach().float().cpu()),
            "memory": memory(),
            "sample_shape": list(global_sample["repr"].shape),
            "source_position": global_position,
            "split": global_split,
            "stay_id": global_key,
        }

        report.update(
            {
                "batch_indices": indices,
                "batch_size": len(indices),
                "longest_training_events": longest_training_events,
                "longest_training_stay_id": longest_key,
                "sample_shape": training_sample_shape,
                "source_position": position,
                "status": "passed",
                "steps": steps,
                "global_longest_predictor_batch": global_batch,
                "test_loader_batches_accessed": int(global_split == trainer.test_subset),
            }
        )
        args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
    except torch.cuda.OutOfMemoryError as error:
        report.update({"error": str(error), "status": "cuda_oom"})
        args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        trainer.data.close()


if __name__ == "__main__":
    main()
