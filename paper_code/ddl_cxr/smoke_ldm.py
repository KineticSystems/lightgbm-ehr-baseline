#!/usr/bin/env python3
"""One exact-loader, one-DDIM-step smoke of the released public DDL-CXR LDM."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from ldm.data.dynamic_data import CollateFunc, GenerateZ1
from ldm.util import instantiate_from_config


PAPER_ROOT = Path(__file__).resolve().parent
EXPECTED_LDM_SHA = "bdd11363779e6136a409c2e94d90913e59243ec438e9e6b3bf2e97885e823b22"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cxr-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ddim-steps", type=int, default=1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite smoke result: {args.output}")
    checkpoint = args.checkpoint.resolve(strict=True)
    cxr_root = args.cxr_root.resolve(strict=True)
    metadata_root = args.metadata_root.resolve(strict=True)
    assert sha256(checkpoint) == EXPECTED_LDM_SHA

    seed = 23
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)

    dataset = GenerateZ1(
        partition="train",
        mimic_cxr_jpg_dir=str(cxr_root),
        metadata_path=str(metadata_root),
    )
    assert len(dataset) == 5142
    assert 1 <= args.batch_size <= 48
    assert 1 <= args.ddim_steps <= 200
    batch = CollateFunc(dataset_mode="generate_z1")(
        [dataset[index] for index in range(args.batch_size)]
    )
    ehr, mask, cxr, sample_id = batch
    assert ehr.ndim == 3 and ehr.shape[0] == args.batch_size and ehr.shape[2] == 79
    assert mask.shape == (args.batch_size, ehr.shape[1] + 1)
    assert cxr.shape == (args.batch_size, 3, 224, 224)
    assert sample_id.shape == (args.batch_size,)
    assert torch.isfinite(ehr).all() and torch.isfinite(cxr).all()

    config = OmegaConf.load(PAPER_ROOT / "configs/latent-diffusion/LDM_generate_z1.yaml")
    model_config = OmegaConf.create(OmegaConf.to_container(config.model, resolve=True))
    model_config.params.ckpt_path = None
    model_config.params.metadata_path = str(args.output.parent.resolve())
    model_config.params.offline_partition = "train"
    model_config.params.ddim_step = args.ddim_steps
    model = instantiate_from_config(model_config)

    with torch.serialization.safe_globals([ModelCheckpoint]):
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    state = payload["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, missing
    assert not unexpected, unexpected
    del state, payload
    gc.collect()

    model.eval().to("cuda")
    batch = tuple(value.to("cuda") for value in batch)
    torch.cuda.reset_peak_memory_stats()
    gpu_start = time.perf_counter()
    with torch.inference_mode():
        z0, encoded_ehr, loaded_sample_id = model.get_input_generation(batch)
        x_t = model.global_noise.repeat(z0.shape[0], 1, 1, 1).to("cuda")
        with model.ema_scope("one-step smoke"):
            generated = model.predict(encoded_ehr=encoded_ehr, z0=z0, x_T=x_t)
    torch.cuda.synchronize()
    gpu_runtime_seconds = time.perf_counter() - gpu_start

    assert torch.equal(loaded_sample_id, sample_id.to("cuda"))
    assert z0.shape == (args.batch_size, 4, 28, 28), z0.shape
    assert generated.shape == (args.batch_size, 4, 28, 28), generated.shape
    assert torch.isfinite(z0).all() and torch.isfinite(encoded_ehr).all()
    assert torch.isfinite(generated).all()

    report = {
        "checkpoint_sha256": EXPECTED_LDM_SHA,
        "batch_size": args.batch_size,
        "ddim_steps": args.ddim_steps,
        "environment": {
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "exact_loader": {
            "cxr_shape": list(cxr.shape),
            "ehr_shape": list(ehr.shape),
            "mask_shape": list(mask.shape),
            "partition_rows": len(dataset),
        },
        "generated": {
            "finite": True,
            "mean": float(generated.float().mean().cpu()),
            "shape": list(generated.shape),
            "std": float(generated.float().std().cpu()),
        },
        "max_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_encode_and_sample_seconds": gpu_runtime_seconds,
        "model_state": {
            "missing_keys": len(missing),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "unexpected_keys": len(unexpected),
        },
        "seed": seed,
        "status": "pass",
        "weights_only_load": "pass",
        "z0_shape": list(z0.shape),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
