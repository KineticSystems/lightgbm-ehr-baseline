#!/usr/bin/env python3
"""Generate one DDL-CXR latent partition through the exact released model methods."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader

from ldm.data.dynamic_data import CollateFunc, GenerateZ1
from ldm.util import instantiate_from_config


PAPER_ROOT = Path(__file__).resolve().parent
EXPECTED_LDM_SHA = "bdd11363779e6136a409c2e94d90913e59243ec438e9e6b3bf2e97885e823b22"
EXPECTED_ROWS = {"train": 5142, "validate": 861, "test": 1483}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=sorted(EXPECTED_ROWS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cxr-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    assert args.batch_size == 48
    assert args.num_workers == 4
    checkpoint = args.checkpoint.resolve(strict=True)
    cxr_root = args.cxr_root.resolve(strict=True)
    input_root = args.input_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    h5_path = output_root / f"{args.partition}_z1_for_pred.h5"
    result_path = output_root / "result.json"
    if h5_path.exists() or result_path.exists():
        raise FileExistsError(f"refusing prior generation output in {output_root}")
    assert sha256(checkpoint) == EXPECTED_LDM_SHA

    seed = 23
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)

    dataset = GenerateZ1(
        partition=args.partition,
        mimic_cxr_jpg_dir=str(cxr_root),
        metadata_path=str(input_root),
    )
    expected_rows = EXPECTED_ROWS[args.partition]
    assert len(dataset) == expected_rows
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=CollateFunc(dataset_mode="generate_z1"),
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    assert len(loader) == math.ceil(expected_rows / args.batch_size)

    config_path = PAPER_ROOT / "configs/latent-diffusion/LDM_generate_z1.yaml"
    config = OmegaConf.load(config_path)
    model_config = OmegaConf.create(OmegaConf.to_container(config.model, resolve=True))
    model_config.params.ckpt_path = None
    model_config.params.metadata_path = str(output_root)
    model_config.params.offline_partition = args.partition
    assert model_config.params.ddim_step == 200
    assert model_config.params.get("eta", 1.0) == 1.0
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
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows_written = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = tuple(value.to("cuda") for value in batch)
            model.test_step(batch, batch_index)
            rows_written += int(batch[0].shape[0])
    torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - started
    assert rows_written == expected_rows
    assert h5_path.is_file()

    expected_keys = {str(index) for index in range(expected_rows)}
    dtype_counts: dict[str, int] = {}
    finite = 0
    with h5py.File(h5_path, "r") as handle:
        assert set(handle.keys()) == expected_keys
        for key in handle.keys():
            data = handle[key][...]
            assert data.shape == (4, 28, 28), (key, data.shape)
            assert np.isfinite(data).all(), key
            dtype = str(data.dtype)
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            finite += 1
    assert finite == expected_rows

    report = {
        "batch_size": args.batch_size,
        "batches": len(loader),
        "checkpoint_sha256": EXPECTED_LDM_SHA,
        "configuration": {
            "ddim_steps": 200,
            "eta": "released constructor default 1.0",
            "global_noise": "one seed-23 tensor repeated across every example",
            "yaml_sha256": sha256(config_path),
        },
        "environment": {
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "hdf5": {
            "bytes": h5_path.stat().st_size,
            "dtype_counts": dict(sorted(dtype_counts.items())),
            "keys": expected_rows,
            "sha256": sha256(h5_path),
            "shape": [4, 28, 28],
        },
        "input": {
            "generation_ehr_report_sha256": sha256(
                input_root / "released_generation_ehr.json"
            ),
            "policy": "released cross-row local index with missing cha schema repair",
        },
        "max_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "model_state": {
            "missing_keys": len(missing),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "unexpected_keys": len(unexpected),
            "weights_only_load": "pass",
        },
        "num_workers": args.num_workers,
        "partition": args.partition,
        "rows": expected_rows,
        "runtime_seconds": runtime_seconds,
        "seed": seed,
        "status": "complete",
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    result_path.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
