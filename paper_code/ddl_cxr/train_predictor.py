#!/usr/bin/env python3
"""Train one DDL-CXR seed around the exact released model and loader methods."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.seed import pl_worker_init_function
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score

from ldm.util import instantiate_from_config


PAPER_ROOT = Path(__file__).resolve().parent
EXPECTED_INPUT_REPORT_SHA = "65148ffa8030803d543e3393d993f73aff984c42c6e465b6812d98c7e82e2a5c"
EXPECTED_RESNET_SHA = "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f"
EXPECTED_ROWS = {"train": 5142, "validation": 861, "test": 1483}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, value: dict[str, object]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def configure(input_root: Path, cxr_root: Path):
    config_path = PAPER_ROOT / "configs/Prediction/Pred_fusion.yaml"
    config = OmegaConf.load(config_path)
    config.model.params.task = "mortality"
    for split in ("train", "validation", "test"):
        params = config.data.params[split].params
        params.task = "mortality"
        params.metadata_path = str(input_root)
        params.mimic_cxr_jpg_dir = str(cxr_root)
    assert config.data.params.batch_size == 64
    assert config.data.params.num_workers == 16
    assert config.model.base_learning_rate == 0.000002
    assert config.model.params.max_epoch == 100
    assert config.lightning.trainer.max_epochs == 100
    assert config.lightning.trainer.deterministic is True
    assert config.lightning.callbacks.EarlyStopping.params.patience == 10
    assert config.model.params.get("monitor", "val/pr_auc") == "val/pr_auc"
    assert config.model.params.mode == "max"
    assert config.model.params.ehr_modal
    assert config.model.params.z1_modal
    assert config.model.params.x0_modal
    assert config.model.params.z1_embed_type == "patchembedding"
    assert config.model.params.fusion_way == "attention"
    return config_path, config


def evaluate(model, loader, *, include_ids: bool):
    losses: list[float] = []
    targets: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []
    sample_ids: list[torch.Tensor] = []
    rows = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            ehr, target, latent, sample_id, image = model.get_input(batch)
            logits = model(ehr=ehr, z=latent, x0=image)
            loss = model.loss(logits, target)
            probability = torch.sigmoid(logits)
            assert torch.isfinite(loss)
            assert torch.isfinite(logits).all()
            losses.append(float(loss.cpu()))
            targets.append(target.cpu())
            probabilities.append(probability.cpu())
            logits_list.append(logits.cpu())
            if include_ids:
                sample_ids.append(sample_id.cpu())
            rows += int(target.shape[0])
    target = torch.cat(targets).numpy()
    probability = torch.cat(probabilities).numpy()
    logits = torch.cat(logits_list).numpy()
    result = {
        "auprc": float(average_precision_score(target, probability)),
        "auroc": float(roc_auc_score(target, probability)),
        "batch_mean_loss": float(np.mean(losses)),
        "batches": len(losses),
        "logits": logits,
        "probabilities": probability,
        "rows": rows,
        "sample_ids": torch.cat(sample_ids).numpy() if include_ids else None,
        "targets": target,
    }
    return result


def checkpoint_payload(model, optimizer, scheduler, epoch, global_step, validation):
    return {
        "epoch": epoch,
        "global_step": global_step,
        "lr_schedulers": [scheduler.state_dict()],
        "optimizer_states": [optimizer.state_dict()],
        "state_dict": model.state_dict(),
        "validation_auprc": validation["auprc"],
        "validation_auroc": validation["auroc"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cxr-root", type=Path, required=True)
    parser.add_argument("--resnet-weight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    input_root = args.input_root.resolve(strict=True)
    cxr_root = args.cxr_root.resolve(strict=True)
    resnet_weight = args.resnet_weight.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing prior seed artifact directory: {output_root}")
    output_root.mkdir(parents=True)
    assert sha256(input_root / "verification.json") == EXPECTED_INPUT_REPORT_SHA
    assert sha256(resnet_weight) == EXPECTED_RESNET_SHA

    seed = args.seed
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    config_path, config = configure(input_root, cxr_root)
    OmegaConf.save(config, output_root / "resolved_config.yaml")

    started = time.perf_counter()
    model = instantiate_from_config(config.model)
    model.learning_rate = 1 * 1 * 64 * float(config.model.base_learning_rate)
    assert model.learning_rate == 0.000128
    optimizer_list, scheduler_list = model.configure_optimizers()
    assert len(optimizer_list) == len(scheduler_list) == 1
    optimizer = optimizer_list[0]
    scheduler = scheduler_list[0]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    assert {key: len(value) for key, value in data.datasets.items()} == EXPECTED_ROWS
    train_loader = data.train_dataloader()
    validation_loader = data.val_dataloader()
    test_loader = data.test_dataloader()
    for loader in (train_loader, validation_loader, test_loader):
        assert loader.worker_init_fn is None
        loader.worker_init_fn = partial(pl_worker_init_function, rank=0)
    assert len(train_loader) == 81
    assert len(validation_loader) == 14
    assert len(test_loader) == 24

    device = torch.device("cuda")
    model.to(device)
    torch.cuda.reset_peak_memory_stats()
    checkpoint_path = output_root / "best.ckpt"
    trace_path = output_root / "epochs.jsonl"
    best_auprc = -math.inf
    best_epoch = -1
    global_step = 0
    wait_count = 0
    stopped_early = False

    for epoch in range(100):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_rows = 0
        train_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            ehr, target, latent, _sample_id, image = model.get_input(batch)
            logits = model(ehr=ehr, z=latent, x0=image)
            loss = model.loss(logits, target)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            batch_rows = int(target.shape[0])
            train_loss_sum += float(loss.detach().cpu()) * batch_rows
            train_rows += batch_rows
            train_batches += 1
            global_step += 1
        assert train_rows == EXPECTED_ROWS["train"]
        assert train_batches == 81

        validation = evaluate(model, validation_loader, include_ids=False)
        assert validation["rows"] == EXPECTED_ROWS["validation"]
        improved = validation["auprc"] > best_auprc
        if improved:
            best_auprc = validation["auprc"]
            best_epoch = epoch
            wait_count = 0
            temporary = output_root / "best.ckpt.tmp"
            torch.save(
                checkpoint_payload(
                    model, optimizer, scheduler, epoch, global_step, validation
                ),
                temporary,
            )
            os.replace(temporary, checkpoint_path)
        else:
            wait_count += 1

        scheduler.step()
        record = {
            "best_epoch": best_epoch,
            "best_validation_auprc": best_auprc,
            "epoch": epoch,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "global_step": global_step,
            "improved": improved,
            "learning_rate_next_epoch": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss_sum / train_rows,
            "train_rows": train_rows,
            "validation_auprc": validation["auprc"],
            "validation_auroc": validation["auroc"],
            "validation_batch_mean_loss": validation["batch_mean_loss"],
            "validation_rows": validation["rows"],
            "wait_count": wait_count,
        }
        append_jsonl(trace_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)
        del validation
        if wait_count >= 10:
            stopped_early = True
            break

    assert checkpoint_path.is_file()
    fit_seconds = time.perf_counter() - started
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=True)
    assert not missing and not unexpected
    assert int(payload["epoch"]) == best_epoch
    assert float(payload["validation_auprc"]) == best_auprc
    del payload, optimizer, scheduler
    gc.collect()

    test = evaluate(model, test_loader, include_ids=True)
    assert test["rows"] == EXPECTED_ROWS["test"]
    sample_ids = np.asarray(test["sample_ids"], dtype=np.int64)
    targets = np.asarray(test["targets"], dtype=np.int64)
    probabilities = np.asarray(test["probabilities"], dtype=np.float32)
    logits = np.asarray(test["logits"], dtype=np.float32)
    assert np.array_equal(sample_ids, np.arange(EXPECTED_ROWS["test"]))
    assert int(targets.sum()) == 219
    assert np.isfinite(probabilities).all() and np.isfinite(logits).all()
    predictions_path = output_root / "test_predictions.csv"
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "target": targets,
            "logit": logits,
            "probability": probabilities,
        }
    ).to_csv(predictions_path, index=False)
    replay = pd.read_csv(predictions_path)
    assert replay["sample_id"].tolist() == sample_ids.tolist()
    assert replay["target"].tolist() == targets.tolist()
    assert np.array_equal(replay["logit"].to_numpy(dtype=np.float32), logits)
    assert np.array_equal(
        replay["probability"].to_numpy(dtype=np.float32), probabilities
    )
    assert float(roc_auc_score(targets, probabilities)) == test["auroc"]
    assert float(average_precision_score(targets, probabilities)) == test["auprc"]
    torch.cuda.synchronize()

    result = {
        "best_epoch_zero_based": best_epoch,
        "best_validation_auprc": best_auprc,
        "checkpoint": {
            "bytes": checkpoint_path.stat().st_size,
            "missing_keys": len(missing),
            "sha256": sha256(checkpoint_path),
            "unexpected_keys": len(unexpected),
            "weights_only_reload": "pass",
        },
        "configuration": {
            "batch_size": 64,
            "effective_learning_rate": 0.000128,
            "max_epochs": 100,
            "num_workers": 16,
            "optimizer": "released AdamW defaults",
            "scheduler": "released CosineAnnealingLR(T_max=100, eta_min=1e-7)",
            "selection": "strict maximum validation AUPRC",
            "stopping_patience": 10,
            "yaml_sha256": sha256(config_path),
        },
        "environment": {
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "epochs_completed": epoch + 1,
        "fit_seconds": fit_seconds,
        "global_steps": global_step,
        "input_verification_sha256": EXPECTED_INPUT_REPORT_SHA,
        "max_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "source_class": type(model).__name__,
        },
        "prediction_sha256": sha256(predictions_path),
        "resnet_weight_sha256": EXPECTED_RESNET_SHA,
        "runtime_seconds": time.perf_counter() - started,
        "seed": seed,
        "status": "complete",
        "stopped_early": stopped_early,
        "test": {
            "auprc": test["auprc"],
            "auroc": test["auroc"],
            "batch_mean_loss": test["batch_mean_loss"],
            "batches": test["batches"],
            "positives": int(targets.sum()),
            "rows": test["rows"],
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (output_root / "result.json").write_text(text)
    print(text, end="", flush=True)


if __name__ == "__main__":
    main()
