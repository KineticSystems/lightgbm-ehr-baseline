"""Persist SMART test logits and aggregate the three released paper seeds."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import random
import re
from argparse import Namespace
from pathlib import Path
from statistics import mean, pstdev, stdev

import numpy as np
import polars as pl
import torch
from sklearn import metrics
from torch.utils.data import DataLoader, SequentialSampler

from data.dataloader import collate_fn
from data.mimiciii import load_mimic_iii_mortality
from models.smart import Classifier, Encoder
from cohort_source import load_cohort
from utils.metrics import print_metrics_binary
from utils.utils import set_seed


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
SEED_TO_REPLICA = {1: 1, 42: 2, 3407: 3}
PUBLISHED = {
    "raw_auroc": 0.8561,
    "raw_auprc": 0.5330,
    "minpse": 0.5129,
    "positive_f1": 0.4423,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("numpy", "polars", "pyarrow", "scikit-learn", "torch")
        },
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "gpu": {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        },
        "torch_cuda_arch_list": torch.cuda.get_arch_list(),
    }


def model_args(seed: int, batch_size: int, epochs: int) -> Namespace:
    return Namespace(
        dataset="mimic_mortality",
        data_dropout=0.0,
        epochs=epochs,
        freeze_epochs=5,
        lr=1e-3,
        d_model=32,
        seed=seed,
        batch_size=batch_size,
        dropout=0.1,
        local_rank=0,
        e_layers=2,
        n_heads=4,
        input_dim=17,
        demo_dim=0,
        num_class=2,
        max_len=48,
    )


def parse_epoch_log(path: Path, expected_epochs: int) -> list[dict]:
    pattern = re.compile(
        r"Epoch (\d+): Train Loss ([0-9.eE+-]+), Valid Loss ([0-9.eE+-]+)"
    )
    values = [
        {
            "epoch": int(match.group(1)),
            "train_loss_rounded": float(match.group(2)),
            "validation_loss_rounded": float(match.group(3)),
        }
        for match in pattern.finditer(path.read_text())
    ]
    if len(values) != expected_epochs:
        raise AssertionError(
            f"{path} has {len(values)} epoch records, expected {expected_epochs}"
        )
    return values


def raw_and_corrected_metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    released = print_metrics_binary(
        torch.from_numpy(labels), torch.from_numpy(logits), verbose=False
    )
    margin = logits[:, 1] - logits[:, 0]
    precision, recall, _ = metrics.precision_recall_curve(labels, margin)
    return {
        "released_raw_logit": {
            "auroc": float(released["auroc"]),
            "auprc": float(released["auprc"]),
            "minpse": float(released["minpse"]),
            "positive_f1": float(released["f1_score"]),
        },
        "corrected_margin": {
            "auroc": float(metrics.roc_auc_score(labels, margin)),
            "auprc": float(metrics.auc(recall, precision)),
        },
    }


def aggregate_if_complete(output_root: Path, epochs: int, batch_size: int) -> None:
    results = []
    for seed in SEED_TO_REPLICA:
        path = output_root / f"seed_{seed}" / "result.json"
        if not path.is_file():
            return
        result = json.loads(path.read_text())
        if result.get("status") != "complete":
            return
        results.append(result)

    metric_paths = {
        "raw_auroc": ("released_raw_logit", "auroc"),
        "raw_auprc": ("released_raw_logit", "auprc"),
        "minpse": ("released_raw_logit", "minpse"),
        "positive_f1": ("released_raw_logit", "positive_f1"),
        "corrected_margin_auroc": ("corrected_margin", "auroc"),
        "corrected_margin_auprc": ("corrected_margin", "auprc"),
    }
    aggregate = {}
    for name, path in metric_paths.items():
        values = [result["test_metrics"][path[0]][path[1]] for result in results]
        entry = {
            "values": values,
            "mean": mean(values),
            "population_std": pstdev(values),
            "sample_std": stdev(values),
        }
        if name in PUBLISHED:
            entry["published"] = PUBLISHED[name]
            entry["mean_minus_published"] = entry["mean"] - PUBLISHED[name]
        aggregate[name] = entry
    summary = {
        "status": "complete",
        "fidelity": "faithful_with_deviations",
        "seeds": list(SEED_TO_REPLICA),
        "epochs_per_stage": epochs,
        "single_gpu_batch_size": batch_size,
        "headline_std_convention": "population (ddof=0); sample SD also reported",
        "per_seed": [
            {
                "seed": result["seed"],
                "replica": result["replica"],
                "selected_pretrain_epoch": result["selected_pretrain_epoch"],
                "selected_finetune_epoch": result["selected_finetune_epoch"],
                "test_metrics": result["test_metrics"],
            }
            for result in results
        ],
        "aggregate": aggregate,
        "published": PUBLISHED,
        "known_deviations": [
            "historical derived-pickle row order is unrecoverable",
            "single RTX 5090 batch 256 replaces four-GPU DDP batch 64 per rank",
            "Blackwell-compatible PyTorch/CUDA replaces Torch 2.1.2/CUDA 12.1",
        ],
        "preserved_source_defects": [
            "example-level split with cross-split patient overlap",
            "normalization fitted on the pooled cohort",
            "validation checkpoint selection and headline AUROC/AUPRC use raw class-1 logit",
        ],
    }
    write_json(output_root / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=tuple(SEED_TO_REPLICA), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=PAPER_ROOT / ".export_assets/source")
    parser.add_argument("--task-root", type=Path, default=WORKSPACE_ROOT / "paper_tasks/smart")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output_root = args.output_root.resolve()
    replica = SEED_TO_REPLICA[args.seed]
    model_dir = run_root / "mimic_mortality/smart"
    pretrain_checkpoint = model_dir / "checkpoint-mse.pth"
    finetune_checkpoint = model_dir / "checkpoint-prc.pth"
    pretrain_log = model_dir / "pretraining.log"
    finetune_log = model_dir / "finetuning.log"
    for path in (pretrain_checkpoint, finetune_checkpoint, pretrain_log, finetune_log):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("SMART evaluation requires CUDA")

    set_seed(args.seed)
    train_dataset, validation_dataset, test_dataset = load_mimic_iii_mortality()
    if (len(train_dataset), len(validation_dataset), len(test_dataset)) != (16911, 2113, 2115):
        raise AssertionError("SMART released split counts changed")

    cohort = load_cohort(args.cache_root.resolve())
    indices = list(range(cohort.height))
    random.Random(args.seed).shuffle(indices)
    test_indices = indices[16911 + 2113 :]
    test_rows = cohort[test_indices]
    task_index = pl.read_parquet(
        args.task_root.resolve()
        / f"mortality_48h_replica_{replica:02d}/audit/task_index.parquet"
    )
    expected_test_ids = set(
        task_index.filter(pl.col("split") == "test")["sample_id"].to_list()
    )
    if set(test_rows["sample_id"].to_list()) != expected_test_ids:
        raise AssertionError("runtime test membership differs from export replica")

    model_configuration = model_args(args.seed, args.batch_size, args.epochs)
    encoder = Encoder(model_configuration).cuda()
    classifier = Classifier(model_configuration).cuda()
    checkpoint = torch.load(finetune_checkpoint, map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder"])
    classifier.load_state_dict(checkpoint["classifier"])
    encoder.eval()
    classifier.eval()
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(test_dataset),
        collate_fn=collate_fn,
    )
    criterion = torch.nn.CrossEntropyLoss()
    logits_parts = []
    label_parts = []
    loss_sum = 0.0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch in loader:
            for key in batch:
                batch[key] = batch[key].cuda()
            if batch["x"].dtype != torch.float32 or tuple(batch["x"].shape[1:]) != (48, 17):
                raise AssertionError("SMART evaluation loader x boundary changed")
            if batch["mask"].dtype != torch.float32 or batch["labels"].dtype != torch.int64:
                raise AssertionError("SMART evaluation loader dtype changed")
            logits = classifier(encoder(**batch), **batch)
            loss_sum += float(criterion(logits, batch["labels"]).item()) * len(logits)
            logits_parts.append(logits.cpu().numpy())
            label_parts.append(batch["labels"].cpu().numpy())
    logits = np.concatenate(logits_parts).astype(np.float32)
    labels = np.concatenate(label_parts).astype(np.int64)
    if not np.array_equal(labels, test_rows["label"].to_numpy().astype(np.int64)):
        raise AssertionError("SMART test labels/order differ from frozen source replica")
    test_metrics = raw_and_corrected_metrics(labels, logits)
    test_metrics["cross_entropy"] = loss_sum / len(labels)

    seed_output = output_root / f"seed_{args.seed}"
    seed_output.mkdir(parents=True, exist_ok=True)
    predictions = test_rows.select(
        "subject_id", "sample_id", "hadm_id", "source_name", "source_position"
    ).with_columns(
        pl.Series("source_row", np.arange(len(labels), dtype=np.int32)),
        pl.Series("label", labels),
        pl.Series("logit_0", logits[:, 0]),
        pl.Series("logit_1", logits[:, 1]),
        pl.Series("margin", logits[:, 1] - logits[:, 0]),
    ).select(
        "source_row",
        "subject_id",
        "sample_id",
        "hadm_id",
        "source_name",
        "source_position",
        "label",
        "logit_0",
        "logit_1",
        "margin",
    )
    prediction_path = seed_output / "test_predictions.parquet"
    temporary_prediction = prediction_path.with_suffix(".parquet.tmp")
    predictions.write_parquet(temporary_prediction)
    temporary_prediction.replace(prediction_path)

    pretrain_state = torch.load(pretrain_checkpoint, map_location="cpu")
    result = {
        "status": "complete",
        "seed": args.seed,
        "replica": replica,
        "configuration": vars(model_configuration),
        "selected_pretrain_epoch": int(pretrain_state["epoch"]),
        "selected_finetune_epoch": int(checkpoint["epoch"]),
        "pretrain_history": parse_epoch_log(pretrain_log, args.epochs),
        "finetune_history": parse_epoch_log(finetune_log, args.epochs),
        "split": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
            "test_positive": int(labels.sum()),
            "test_membership": "exact export replica",
        },
        "test_metrics": test_metrics,
        "environment": environment(),
        "evaluation_peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "artifacts": {
            "pretrain_checkpoint": {
                "file": str(pretrain_checkpoint),
                "sha256": sha256(pretrain_checkpoint),
            },
            "finetune_checkpoint": {
                "file": str(finetune_checkpoint),
                "sha256": sha256(finetune_checkpoint),
            },
            "pretrain_log": {"file": str(pretrain_log), "sha256": sha256(pretrain_log)},
            "finetune_log": {"file": str(finetune_log), "sha256": sha256(finetune_log)},
            "test_predictions": {
                "file": str(prediction_path),
                "sha256": sha256(prediction_path),
            },
        },
    }
    write_json(seed_output / "result.json", result)
    aggregate_if_complete(output_root, args.epochs, args.batch_size)
    print(json.dumps({"seed": args.seed, "metrics": test_metrics}, indent=2, sort_keys=True))
    del encoder, classifier, loader
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
