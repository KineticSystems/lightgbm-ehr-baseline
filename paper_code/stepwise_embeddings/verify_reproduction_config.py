#!/usr/bin/env python3
"""Verify the frozen reconstructed MIMIC FTT config without fitting or testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gin
import numpy as np

from icu_benchmarks.data.loader import ICUVariableLengthDataset
from icu_benchmarks.models import train as _train  # noqa: F401; registers train_common
from icu_benchmarks.models.wrappers import DLWrapper


EXPECTED_HDF_BYTES = 482_886_932
EXPECTED_TRAIN_STAYS = 29_030
EXPECTED_TRAIN_LABELS = 2_377_738
EXPECTED_TRAIN_POSITIVES = 49_260


def _query(name: str):
    value = gin.query_parameter(name)
    if isinstance(value, gin.config.ConfigurableReference) and value.evaluate:
        value = value.scoped_configurable_fn()
    if isinstance(value, tuple):
        return list(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/reproduce/mimic/embeddings/ftt-no-group/"
            "transformer_ftt_decomp.gin"
        ),
    )
    parser.add_argument(
        "--output",
        default=".export_assets/reproduction/audit/config_preflight.json",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    output_path = Path(args.output)
    gin.parse_config_file(str(config_path))

    data_path = Path(_query("train_common.data_path"))
    if not data_path.is_file():
        raise FileNotFoundError("paper-local author-HDF link is missing")
    if data_path.stat().st_size != EXPECTED_HDF_BYTES:
        raise AssertionError("author HDF size differs from the frozen manifest")

    expected_bindings = {
        "Adam.lr": 1e-4,
        "DLWrapper.train.batch_size": 64,
        "DLWrapper.train.epochs": 1000,
        "DLWrapper.train.min_delta": 1e-4,
        "DLWrapper.train.patience": 10,
        "FeatureTokenizer_Transformer.token_dim": 64,
        "ICUVariableLengthDataset.maxlen": -1,
        "ICUVariableLengthLoaderTables.data_resampling": 1,
        "ICUVariableLengthLoaderTables.label_resampling": 1,
        "ICUVariableLengthLoaderTables.on_RAM": False,
        "ICUVariableLengthLoaderTables.task": "decomp_24Hours",
        "Transformer.depth": 1,
        "Transformer.dropout": 0.4,
        "Transformer.dropout_att": 0.3,
        "Transformer.emb": 42,
        "Transformer.heads": 2,
        "Transformer.hidden": 64,
        "aggregation/StackedTransformerBlocks.depth": 2,
        "aggregation/StackedTransformerBlocks.heads": 3,
        "embedding/StackedTransformerBlocks.depth": 2,
        "embedding/StackedTransformerBlocks.heads": 3,
        "embedding/StackedTransformerBlocks.hidden": 64,
    }
    resolved_bindings = {key: _query(key) for key in expected_bindings}
    if resolved_bindings != expected_bindings:
        raise AssertionError(
            f"operative bindings differ: {resolved_bindings!r} != {expected_bindings!r}"
        )

    modalities = _query("Splitted_Embedding.reindex_modalities")
    if modalities != [list(range(42))]:
        raise AssertionError("FTT configuration is not the selected ungrouped boundary")
    if _query("Splitted_Embedding.merge") != "attention_cls":
        raise AssertionError("paper-selected attention aggregation is not active")

    dataset = ICUVariableLengthDataset(str(data_path), split="train")
    loader = dataset.h5_loader
    if len(dataset) != EXPECTED_TRAIN_STAYS:
        raise AssertionError("unexpected number of target-bearing training stays")
    if loader.num_labels["train"] != EXPECTED_TRAIN_LABELS:
        raise AssertionError("unexpected number of training landmarks")
    train_labels = loader.labels["train"]
    train_positives = int(np.nansum(train_labels))
    if train_positives != EXPECTED_TRAIN_POSITIVES:
        raise AssertionError("unexpected number of positive training landmarks")
    sample, labels, pad_mask = dataset[0]
    sample_summary = {
        "data_dtype": str(sample.dtype),
        "data_shape": list(sample.shape),
        "label_dtype": str(labels.dtype),
        "label_shape": list(labels.shape),
        "mask_dtype": str(pad_mask.dtype),
        "mask_shape": list(pad_mask.shape),
        "finite_data": bool(np.isfinite(sample).all()),
        "finite_labels": bool(np.isfinite(labels).all()),
    }

    model = DLWrapper()
    encoder = model.encoder
    parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
    )
    optimizer_group = model.optimizer.param_groups[0]

    report = {
        "status": "exact_reconstructed_config",
        "config": str(config_path),
        "author_hdf": {
            "bytes": data_path.stat().st_size,
            "channels": int(sample.shape[-1]),
            "maximum_sequence_length": int(loader.maxlen),
            "train_target_bearing_stays": len(dataset),
            "train_landmarks": loader.num_labels["train"],
            "train_positives": train_positives,
        },
        "bindings": resolved_bindings,
        "ungrouped_channel_indices": modalities[0],
        "aggregation": _query("Splitted_Embedding.merge"),
        "first_train_sample": sample_summary,
        "model": {
            "class": type(encoder).__name__,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "optimizer": type(model.optimizer).__name__,
            "optimizer_lr": optimizer_group["lr"],
            "optimizer_weight_decay": optimizer_group["weight_decay"],
        },
        "test_accesses": 0,
    }

    loader.data_h5._v_file.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
