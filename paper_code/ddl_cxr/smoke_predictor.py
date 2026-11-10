#!/usr/bin/env python3
"""Smoke the exact released DDL-CXR predictor model, data, optimizer, and IO."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score

from ldm.util import instantiate_from_config


PAPER_ROOT = Path(__file__).resolve().parent
EXPECTED_RESNET_SHA = "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_batch(loader):
    iterator = iter(loader)
    try:
        return next(iterator)
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        del iterator


def configure(input_root: Path, cxr_root: Path):
    config_path = PAPER_ROOT / "configs/Prediction/Pred_fusion.yaml"
    config = OmegaConf.load(config_path)
    config.model.params.task = "mortality"
    for split in ("train", "validation", "test"):
        data_config = config.data.params[split].params
        data_config.task = "mortality"
        data_config.metadata_path = str(input_root)
        data_config.mimic_cxr_jpg_dir = str(cxr_root)
    assert config.data.params.batch_size == 64
    assert config.data.params.num_workers == 16
    assert config.model.base_learning_rate == 0.000002
    assert config.model.params.ehr_modal
    assert config.model.params.z1_modal
    assert config.model.params.x0_modal
    assert config.model.params.z1_embed_type == "patchembedding"
    assert config.model.params.fusion_way == "attention"
    return config_path, config


def main() -> None:
    parser = argparse.ArgumentParser()
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
        raise FileExistsError(f"refusing prior predictor smoke: {output_root}")
    output_root.mkdir(parents=True)
    assert sha256(resnet_weight) == EXPECTED_RESNET_SHA

    seed = 0
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    config_path, config = configure(input_root, cxr_root)

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
    assert {key: len(value) for key, value in data.datasets.items()} == {
        "train": 5142,
        "validation": 861,
        "test": 1483,
    }
    train_batch = one_batch(data.train_dataloader())
    validation_batch = one_batch(data.val_dataloader())
    test_batch = one_batch(data.test_dataloader())
    assert len(train_batch) == len(validation_batch) == len(test_batch) == 64

    device = torch.device("cuda")
    model.train().to(device)
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad()
    ehr, label, latent, sample_id, image = model.get_input(train_batch)
    logits = model(ehr=ehr, z=latent, x0=image)
    train_loss = model.loss(logits, label)
    assert logits.shape == label.shape == (64,)
    assert torch.isfinite(logits).all() and torch.isfinite(train_loss)
    train_loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert finite_gradients
    optimizer.step()
    scheduler.step()
    assert math.isfinite(float(optimizer.param_groups[0]["lr"]))

    model.eval()
    with torch.inference_mode():
        ehr, val_label, latent, val_sample_id, image = model.get_input(validation_batch)
        val_probability = torch.sigmoid(model(ehr=ehr, z=latent, x0=image))
        ehr, test_label, latent, test_sample_id, image = model.get_input(test_batch)
        test_probability = torch.sigmoid(model(ehr=ehr, z=latent, x0=image))
    assert torch.isfinite(val_probability).all()
    assert torch.isfinite(test_probability).all()
    val_auc = roc_auc_score(val_label.cpu().numpy(), val_probability.cpu().numpy())
    val_ap = average_precision_score(val_label.cpu().numpy(), val_probability.cpu().numpy())
    train_loss_value = float(train_loss.detach().cpu())
    expected_test_label = test_label.cpu()
    expected_test_sample_id = test_sample_id.cpu()
    expected_test_probability = test_probability.cpu()

    checkpoint_path = output_root / "one_step.ckpt"
    checkpoint = {
        "epoch": 0,
        "global_step": 1,
        "optimizer_states": [optimizer.state_dict()],
        "state_dict": model.state_dict(),
        "lr_schedulers": [scheduler.state_dict()],
        "validation_ap": float(val_ap),
    }
    torch.save(checkpoint, checkpoint_path)
    del (
        checkpoint,
        optimizer,
        scheduler,
        model,
        train_loss,
        logits,
        ehr,
        label,
        latent,
        sample_id,
        image,
        val_label,
        val_sample_id,
        val_probability,
        test_label,
        test_sample_id,
        test_probability,
    )
    gc.collect()
    torch.cuda.empty_cache()

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    reloaded = instantiate_from_config(config.model)
    missing, unexpected = reloaded.load_state_dict(payload["state_dict"], strict=True)
    assert not missing and not unexpected
    del payload
    reloaded.eval().to(device)
    with torch.inference_mode():
        ehr, replay_label, latent, replay_sample_id, image = reloaded.get_input(test_batch)
        replay_probability = torch.sigmoid(reloaded(ehr=ehr, z=latent, x0=image))
    assert torch.equal(replay_label.cpu(), expected_test_label)
    assert torch.equal(replay_sample_id.cpu(), expected_test_sample_id)
    assert torch.equal(replay_probability.cpu(), expected_test_probability)

    predictions_path = output_root / "predictions.csv"
    pd.DataFrame(
        {
            "sample_id": replay_sample_id.cpu().numpy().astype(np.int64),
            "target": replay_label.cpu().numpy().astype(np.int64),
            "probability": replay_probability.cpu().numpy().astype(np.float64),
        }
    ).to_csv(predictions_path, index=False)
    replay = pd.read_csv(predictions_path)
    assert replay["sample_id"].tolist() == replay_sample_id.cpu().tolist()
    assert replay["target"].tolist() == replay_label.cpu().to(torch.int64).tolist()
    assert np.array_equal(
        replay["probability"].to_numpy(dtype=np.float32),
        replay_probability.cpu().numpy(),
    )
    torch.cuda.synchronize()

    report = {
        "batch_size": 64,
        "checkpoint": {
            "bytes": checkpoint_path.stat().st_size,
            "missing_keys": len(missing),
            "sha256": sha256(checkpoint_path),
            "unexpected_keys": len(unexpected),
            "weights_only_reload": "pass",
        },
        "configuration": {
            "effective_learning_rate": 0.000128,
            "optimizer": "released AdamW defaults",
            "scheduler": "released CosineAnnealingLR(T_max=100, eta_min=1e-7)",
            "yaml_sha256": sha256(config_path),
        },
        "environment": {
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "finite_gradients": finite_gradients,
        "input_verification_sha256": sha256(input_root / "verification.json"),
        "max_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "model": {
            "parameters": sum(parameter.numel() for parameter in reloaded.parameters()),
            "source_class": type(reloaded).__name__,
        },
        "one_batch_validation": {"auprc": float(val_ap), "auroc": float(val_auc)},
        "prediction_serialization": {
            "rows": len(replay),
            "sha256": sha256(predictions_path),
            "status": "exact",
        },
        "resnet_weight_sha256": EXPECTED_RESNET_SHA,
        "runtime_seconds": time.perf_counter() - started,
        "seed": seed,
        "status": "pass",
        "train_loss": train_loss_value,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output_root / "result.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
