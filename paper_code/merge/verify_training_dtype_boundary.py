#!/usr/bin/env python3
"""Prove that consuming MERGE arrays during Dataset conversion is exact."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT / "mimiciv"))

from mimiciv_rus_multimodal import align_multimodal_irg_ts  # noqa: E402
from train_mimiciv_multimodal import MultimodalMIMICIVDataset  # noqa: E402


MODALITY_DIMS = {"labs_vitals": 30, "cxr": 1024, "notes": 768}
MODALITIES = sorted(MODALITY_DIMS)
CHUNK_SIZE = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_tensor_hash(digest, name: str, tensor: torch.Tensor) -> None:
    contiguous = tensor.detach().cpu().contiguous()
    digest.update(name.encode())
    digest.update(str(contiguous.numpy().dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.numpy().tobytes())


def processed_stay(stay: dict, quiet: io.StringIO) -> dict:
    irregular = {
        "notes": [
            (stay["text_time"][index], stay["text_embeddings"][index])
            for index in range(len(stay["text_time"]))
        ],
        "cxr": [
            (stay["cxr_time"][index], stay["cxr_feats"][index])
            for index in range(len(stay["cxr_time"]))
        ],
    }
    with contextlib.redirect_stdout(quiet):
        result = align_multimodal_irg_ts(
            stay["reg_ts"], irregular, MODALITY_DIMS, linear_interpolation=False
        )
    quiet.seek(0)
    quiet.truncate(0)
    result["labs_vitals"] = (
        stay["reg_ts"],
        np.ones(len(stay["reg_ts"]), dtype=bool),
    )
    return result


def verify_split(name: str, path: Path, input_report: dict) -> dict:
    expected_sha256 = input_report["artifacts"]["pickles"][name]
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise AssertionError(f"{name} pickle differs from the verified input")

    with path.open("rb") as handle:
        stays = pickle.load(handle)
    eligible = [stay for stay in stays if len(stay["ts_tt"]) > 12]
    expected_count = int(input_report["splits"][name]["postfilter_count"])
    if len(eligible) != expected_count:
        raise AssertionError(f"{name} eligible count differs: {len(eligible)} != {expected_count}")

    expected_digest = hashlib.sha256()
    repaired_digest = hashlib.sha256()
    source_array_bytes = 0
    dataset_tensor_bytes = 0
    positives = 0
    consumed_samples = 0
    quiet = io.StringIO()
    rus_data = {
        "U": torch.zeros(3, 48),
        "R": torch.zeros(3, 3, 48),
        "S": torch.zeros(3, 3, 48),
    }

    for chunk_start in range(0, len(eligible), CHUNK_SIZE):
        source_samples = []
        labels = []
        for stay in eligible[chunk_start : chunk_start + CHUNK_SIZE]:
            sample = processed_stay(stay, quiet)
            for modality in MODALITIES:
                values, mask = sample[modality]
                if values.shape[0] != 48 or mask.shape[0] != 48:
                    raise AssertionError(f"{name} has a non-48-step {modality} tensor")
                expected_values = torch.from_numpy(values).float()
                expected_mask = torch.from_numpy(mask).bool()
                update_tensor_hash(expected_digest, modality, expected_values)
                update_tensor_hash(expected_digest, f"{modality}_mask", expected_mask)
                source_array_bytes += values.nbytes + mask.nbytes
            label = int(stay["label"])
            update_tensor_hash(expected_digest, "label", torch.tensor(label, dtype=torch.long))
            labels.append(label)
            positives += label
            source_samples.append(sample)

        with contextlib.redirect_stdout(quiet):
            dataset = MultimodalMIMICIVDataset(
                source_samples,
                labels,
                rus_data,
                MODALITIES,
                MODALITY_DIMS,
                max_seq_len=48,
                truncate_from_end=True,
            )
        quiet.seek(0)
        quiet.truncate(0)

        # The compatibility repair must consume each source dictionary so its
        # float64 arrays can be released as soon as the float32 tensor exists.
        if any(sample for sample in source_samples):
            raise AssertionError(
                "Dataset did not consume source modality arrays during conversion"
            )
        for index in range(len(dataset)):
            modalities, _, label = dataset[index]
            for modality in MODALITIES:
                values, mask = modalities[modality]
                if values.dtype != torch.float32 or mask.dtype != torch.bool:
                    raise AssertionError(f"{name} Dataset dtype changed for {modality}")
                update_tensor_hash(repaired_digest, modality, values)
                update_tensor_hash(repaired_digest, f"{modality}_mask", mask)
                dataset_tensor_bytes += (
                    values.numel() * values.element_size()
                    + mask.numel() * mask.element_size()
                )
            update_tensor_hash(
                repaired_digest, "label", torch.tensor(label, dtype=torch.long)
            )
        consumed_samples += len(dataset)
        del dataset, labels, source_samples
        gc.collect()
        print(
            f"verified {name}: {min(chunk_start + CHUNK_SIZE, len(eligible))}/{len(eligible)}",
            file=sys.stderr,
        )

    expected_hash = expected_digest.hexdigest()
    repaired_hash = repaired_digest.hexdigest()
    if repaired_hash != expected_hash:
        raise AssertionError(f"{name} aggregate model-tensor hash changed")
    if consumed_samples != expected_count:
        raise AssertionError(f"{name} Dataset did not consume every eligible stay")
    result = {
        "pickle_bytes": path.stat().st_size,
        "pickle_sha256": actual_sha256,
        "eligible_stays": len(eligible),
        "positives": positives,
        "conversion_chunks": (len(eligible) + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "source_array_bytes": source_array_bytes,
        "dataset_tensor_bytes": dataset_tensor_bytes,
        "bytes_released_by_float32_boundary": source_array_bytes - dataset_tensor_bytes,
        "model_tensor_sha256": repaired_hash,
    }
    del eligible, stays
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_report = json.loads(args.input_report.read_text())
    if input_report.get("status") != "exact":
        raise AssertionError("IHM input verification is not exact")
    splits = {
        "train": verify_split("train", args.train, input_report),
        "validation": verify_split("validation", args.validation, input_report),
        "test": verify_split("test", args.test, input_report),
    }
    report = {
        "status": "exact",
        "repair": (
            "consume each source modality array as the released Dataset converts "
            "it to its existing float32/bool tensor boundary"
        ),
        "checks": [
            "all three verified author pickles and every eligible stay",
            "unchanged released aligner and preprocessing values",
            "released pre-repair Dataset tensor boundary independently hashed",
            "repaired Dataset tensors, masks, labels, order, and dtypes exact",
            "source modality dictionaries empty after conversion",
        ],
        "splits": splits,
        "input_report_sha256": sha256_file(args.input_report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
