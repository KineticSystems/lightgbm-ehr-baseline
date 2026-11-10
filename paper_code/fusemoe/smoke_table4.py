#!/usr/bin/env python3
"""Exercise the full primary FuseMoE stack before any headline run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import set_seed

from reproduce_table4 import (
    EXPECTED_SPLITS,
    atomic_json,
    build_author_args,
    create_stack,
    sha256_file,
    ts_text_loss,
)


def count_loader(name, loader, dataset):
    cursor = 0
    positive = 0
    batches = 0
    for batch in loader:
        if batch is None:
            raise AssertionError(f"{name} collator returned None")
        labels = batch[15].detach().cpu().numpy().astype(np.int64)
        rows = dataset.data[cursor : cursor + len(labels)]
        expected = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        np.testing.assert_array_equal(labels, expected)
        cursor += len(labels)
        positive += int(labels.sum())
        batches += 1
    if (cursor, positive) != EXPECTED_SPLITS[name]:
        raise AssertionError(f"{name} denominator changed: {(cursor, positive)}")
    return {"rows": cursor, "positive": positive, "batches": batches, "none_batches": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args_cli = parser.parse_args()
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)

    author_args = build_author_args(42, args_cli.input_dir, args_cli.output_dir)
    set_seed(author_args.seed)
    torch.cuda.reset_peak_memory_stats()
    stack = create_stack(author_args, args_cli.model_root)
    validation_count = count_loader(
        "validation", stack["validation_loader"], stack["validation_dataset"]
    )
    test_count = count_loader("test", stack["test_loader"], stack["test_dataset"])

    stack["model"].train()
    batch = next(iter(stack["train_loader"]))
    if batch is None or len(batch[15]) != author_args.train_batch_size:
        raise AssertionError("unexpected training smoke batch")
    task_loss, balance_loss, total_loss = ts_text_loss(
        stack["model"], batch, author_args
    )
    if not all(torch.isfinite(value).all() for value in (task_loss, balance_loss, total_loss)):
        raise AssertionError("smoke loss is nonfinite")
    stack["accelerator"].backward(total_loss / author_args.gradient_accumulation_steps)
    gradients = [
        parameter.grad
        for parameter in stack["model"].parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("smoke gradients are absent or nonfinite")
    stack["optimizer"].step()
    stack["model"].zero_grad()

    with tempfile.NamedTemporaryFile(
        prefix="fusemoe-smoke-", suffix=".pth.tar", dir=args_cli.output_dir, delete=False
    ) as handle:
        checkpoint_path = Path(handle.name)
    try:
        torch.save({"network": stack["model"].state_dict()}, checkpoint_path)
        checkpoint_hash = sha256_file(checkpoint_path)
        checkpoint_bytes = checkpoint_path.stat().st_size
        first_parameter = next(stack["model"].parameters())
        expected_parameter = first_parameter.detach().clone()
        with torch.no_grad():
            first_parameter.add_(1)
        checkpoint = torch.load(checkpoint_path, map_location=stack["device"], weights_only=True)
        stack["model"].load_state_dict(checkpoint["network"])
        if not torch.equal(first_parameter.detach(), expected_parameter):
            raise AssertionError("checkpoint round trip changed model state")
    finally:
        checkpoint_path.unlink(missing_ok=True)

    stack["model"].eval()
    with torch.no_grad():
        scores = stack["model"](
            x_ts=batch[0],
            x_ts_mask=batch[1],
            ts_tt_list=batch[2],
            input_ids_sequences=batch[4],
            attn_mask_sequences=batch[5],
            text_emb=batch[6],
            note_time_list=batch[7],
            note_time_mask_list=batch[8],
            reg_ts=batch[3],
        )
    if scores.shape != (author_args.train_batch_size,) or not torch.isfinite(scores).all():
        raise AssertionError("post-round-trip inference failed")

    report = {
        "status": "pass",
        "seed": author_args.seed,
        "configuration": {
            "modeltype": author_args.modeltype,
            "num_modalities": author_args.num_modalities,
            "num_experts": author_args.num_of_experts,
            "top_k": author_args.top_k,
            "gating_function": author_args.gating_function,
            "router_type": author_args.router_type,
            "train_batch_size": author_args.train_batch_size,
            "eval_batch_size": author_args.eval_batch_size,
            "mixed_precision": author_args.mixed_precision,
        },
        "loss": {
            "task": float(task_loss.detach().float().cpu()),
            "balance": float(balance_loss.detach().float().cpu()),
            "total": float(total_loss.detach().float().cpu()),
        },
        "gradients": len(gradients),
        "validation_loader": validation_count,
        "test_loader": test_count,
        "checkpoint_round_trip": {
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_hash,
            "retained": False,
        },
        "post_round_trip_scores": scores.detach().float().cpu().tolist(),
        "parameter_count": int(
            sum(parameter.numel() for parameter in stack["model"].parameters())
        ),
        "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    atomic_json(args_cli.output_dir / "smoke.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

