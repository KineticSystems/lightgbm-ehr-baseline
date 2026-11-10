#!/usr/bin/env python3
"""Shape-faithful CUDA smokes for the released REMed two-stage model path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from src.models import REMed, UniHPF
from src.utils.trainer_utils import PredLoss, get_task


PRED_TARGETS = (
    "readmission",
    "los_7",
    "los_14",
    "mortality_1",
    "mortality_2",
    "mortality_3",
    "mortality_7",
    "mortality_14",
    "diagnosis",
    "creatinine_1",
    "creatinine_2",
    "creatinine_3",
    "platelets_1",
    "platelets_2",
    "platelets_3",
    "wbc_1",
    "wbc_2",
    "wbc_3",
    "hb_1",
    "hb_2",
    "hb_3",
    "bicarbonate_1",
    "bicarbonate_2",
    "bicarbonate_3",
    "sodium_1",
    "sodium_2",
    "sodium_3",
)
EXPECTED_MAX_EVENTS = 266_766
EXPECTED_MAX_SEQUENCE = 266_880


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_report(path: Path, status: str) -> dict:
    report = json.loads(path.read_text())
    if report.get("status") != status:
        raise AssertionError((path, report.get("status"), status))
    return report


def model_args(train_type: str, maximum_sequence: int) -> SimpleNamespace:
    args = SimpleNamespace(
        alibi_const=3,
        batch_size=8,
        dropout=0.2,
        encoder_pooling="mean",
        local_batch_size=8,
        max_retrieve_len=128,
        max_seq_len=maximum_sequence,
        max_word_len=128,
        n_agg_layers=2,
        n_enc_layers=2,
        n_flatten_layers=4,
        n_heads=8,
        pos_enc="sinusoidal_time",
        pred_dim=512,
        pred_pooling="mean",
        pred_time=48,
        query_gen=False,
        rejection_cutoff=0.0,
        scorer=train_type == "remed",
        scorer_use_time=train_type == "remed",
        src_data="mimiciv",
        tasks=[get_task(target, "mimiciv") for target in PRED_TARGETS],
        train_cls_header=False,
        train_type=train_type,
    )
    return args


def labels(args: SimpleNamespace, batch_size: int, device: torch.device) -> dict:
    result = {}
    for index, task in enumerate(args.tasks):
        if task.property == "binary":
            value = (torch.arange(batch_size, device=device) + index) % 2
            result[task.name] = value.float().unsqueeze(1)
        elif task.property == "multilabel":
            value = torch.zeros((batch_size, task.num_classes), device=device)
            value[:, index % task.num_classes] = 1
            result[task.name] = value
        elif task.property == "multiclass":
            value = torch.zeros((batch_size, task.num_classes), device=device)
            value[torch.arange(batch_size, device=device), index % task.num_classes] = 1
            result[task.name] = value
        else:
            raise AssertionError(task)
    return result


def peak_memory() -> dict[str, int]:
    torch.cuda.synchronize()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def probabilities(output: dict) -> dict[str, float]:
    result = {}
    for task, tensor in output["pred"].items():
        value = tensor[0].detach().float().cpu().reshape(-1)
        if not torch.isfinite(value).all().item():
            raise AssertionError((task, value))
        result[task] = float(value[0])
    return result


def optimizer_step(
    model,
    criterion,
    sample,
    optimizer,
    scheduler,
    mode: str | None = None,
) -> tuple[dict, dict]:
    if mode is not None:
        model.set_mode(mode)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, representations = model(**sample)
        loss, _ = criterion(output, representations)
    if not torch.isfinite(loss).item():
        raise AssertionError(loss)
    loss.backward()
    finite_gradients = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all().item():
                raise AssertionError("non-finite gradient")
            finite_gradients += 1
    if finite_gradients == 0:
        raise AssertionError("no gradients produced")
    optimizer.step()
    scheduler.step()
    return output, {
        "finite_gradient_tensors": finite_gradients,
        "loss": float(loss.detach().float().cpu()),
        "memory": peak_memory(),
    }


def strict_checkpoint_roundtrip(model, args, path: Path) -> dict:
    torch.save(model.state_dict(), path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    clone = model.__class__(args).cpu()
    incompatible = clone.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(incompatible)
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def smoke_event_encoder(output_dir: Path, device: torch.device) -> dict:
    args = model_args("short", 512)
    torch.manual_seed(2020)
    model = UniHPF(args).to(device).train()
    criterion = PredLoss(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=1, total_iters=1
    )
    batch_size, sequence, word_length = 8, 512, 128
    sample = {
        "dpe_ids": torch.randint(0, 100, (batch_size, sequence, word_length), device=device),
        "input_ids": torch.randint(1, 28_996, (batch_size, sequence, word_length), device=device),
        "label": labels(args, batch_size, device),
        "times": torch.arange(sequence, device=device, dtype=torch.int32).repeat(batch_size, 1),
        "type_ids": torch.randint(0, 8, (batch_size, sequence, word_length), device=device),
    }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output, step = optimizer_step(
        model, criterion, sample, optimizer=optimizer, scheduler=scheduler
    )
    prediction = probabilities(output)
    checkpoint = strict_checkpoint_roundtrip(model, args, output_dir / "event_encoder.pt")
    return {
        "checkpoint": checkpoint,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "prediction": prediction,
        "sample_shape": [batch_size, sequence, word_length],
        "step": step,
    }


def smoke_remed(output_dir: Path, device: torch.device, sequence: int) -> dict:
    args = model_args("remed", sequence)
    torch.manual_seed(2020)
    model = REMed(args).to(device).train()
    criterion = PredLoss(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=1, total_iters=1
    )
    batch_size = 8
    sample = {
        "label": labels(args, batch_size, device),
        "repr": torch.randn((batch_size, sequence, 512), device=device),
        "times": torch.arange(sequence, device=device, dtype=torch.int32).repeat(batch_size, 1),
    }
    steps = {}
    predictions = {}
    for mode in ("scorer", "predictor"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        output, step = optimizer_step(
            model,
            criterion,
            sample,
            optimizer=optimizer,
            scheduler=scheduler,
            mode=mode,
        )
        steps[mode] = step
        predictions[mode] = probabilities(output)
    checkpoint = strict_checkpoint_roundtrip(model, args, output_dir / "remed.pt")
    return {
        "checkpoint": checkpoint,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "predictions": predictions,
        "sample_shape": [batch_size, sequence, 512],
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("small", "maximum"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment-report", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--small-report", type=Path)
    parser.add_argument("--cohort-report", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    require_report(args.environment_report, "passed")
    require_report(args.input_report, "exact_paper_local_input_shim")
    cohort = require_report(
        args.cohort_report, "exact_source_with_small_paper_token_length_mismatch"
    )
    max_events = cohort["reconstructed"]["event_length"]["max"]
    max_sequence = math.ceil(max_events / 128) * 128
    if max_events != EXPECTED_MAX_EVENTS or max_sequence != EXPECTED_MAX_SEQUENCE:
        raise AssertionError((max_events, max_sequence))
    if args.profile == "maximum":
        if args.small_report is None:
            raise ValueError("--small-report is required for maximum profile")
        require_report(args.small_report, "passed")

    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "report.json"
    device = torch.device("cuda:0")
    report = {
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_device": torch.cuda.get_device_name(device),
        "maximum_source_events": max_events,
        "profile": args.profile,
        "status": "running",
        "test_accesses": 0,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    try:
        if args.profile == "small":
            report["event_encoder"] = smoke_event_encoder(args.output_dir, device)
            report["remed"] = smoke_remed(args.output_dir, device, 1024)
        else:
            report["remed"] = smoke_remed(args.output_dir, device, max_sequence)
        predictions_path = args.output_dir / "predictions.json"
        predictions_path.write_text(
            json.dumps(
                {
                    key: value.get("prediction", value.get("predictions"))
                    for key, value in report.items()
                    if isinstance(value, dict)
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        report["predictions"] = {
            "bytes": predictions_path.stat().st_size,
            "sha256": sha256(predictions_path),
        }
        report["status"] = "passed"
    except torch.cuda.OutOfMemoryError as error:
        report["error"] = str(error)
        report["memory_at_failure"] = peak_memory()
        report["status"] = "oom"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["status"] = "failed"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        torch.cuda.empty_cache()

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
