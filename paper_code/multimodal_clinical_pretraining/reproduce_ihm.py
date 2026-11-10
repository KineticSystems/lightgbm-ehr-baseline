"""Reproduce the released-checkpoint full-label IHM experiment.

This is an adjacent execution harness.  It imports the released dataset,
model, checkpoint loader, optimizer, and scheduler without modifying their
source files.  The harness adds only the controls absent from the release:
declared seeds, no-network metrics, validation-only epoch selection, and
machine-readable predictions/results.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
import time
from argparse import Namespace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmimic.data import IHMDataset
from torchmimic.data.preprocessing import Normalizer
from torchmimic.utils import pad_colalte


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
sys.path.insert(0, str(PAPER_ROOT))

from multimodal_clinical_pretraining.models import create_model  # noqa: E402
from multimodal_clinical_pretraining.scheduler import create_scheduler  # noqa: E402
from multimodal_clinical_pretraining.utils import (  # noqa: E402
    load_pretrained_model,
)


CHECKPOINT_SHA256 = (
    "de3a2a54ab244daa411cc6dc51c0aca3e34c93773c5c4bd168601be9fcd8771e"
)
LISTFILE_SHA256 = {
    "train": "802bf8e91912c195ce1800b248f967c4cb0d77b55c38358deaddcaf265ed88d5",
    "validation": "7b0fa183e44b5dde12f081cb3578b9228f6cbfe27786ead478f6252890815d32",
    "test": "828c0915b34c41d2fa231fe1f0a443e5b2e71e712225eab631727b902c5604d3",
}
EXPECTED_SPLITS = {
    "train": {"rows": 14_681, "positive": 1_987},
    "validation": {"rows": 3_222, "positive": 436},
    "test": {"rows": 3_236, "positive": 374},
}
PUBLISHED = {"auroc": 0.856, "auprc": 0.495}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_inventory() -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: distribution_version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "torch",
                "torchmimic",
                "torchvision",
            )
        },
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "torch_cuda_arch_list": torch.cuda.get_arch_list()
        if torch.cuda.is_available()
        else [],
        "gpu": gpu,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metric_values(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return {
        # A one-batch smoke subset can contain only one class. Full split runs
        # always contain both classes; represent the intentionally partial
        # smoke AUROC as null instead of emitting non-standard JSON NaN.
        "auroc": float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else None,
        # This is the released torchmimic metric, not average precision.
        "auprc": float(auc(recall, precision)),
    }


def read_listfile(path: Path) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    labels: list[int] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            names.append(row["stay"])
            labels.append(int(row["y_true"]))
    return names, np.asarray(labels, dtype=np.float32)


def verify_dataset(
    split: str, dataset: IHMDataset, listfile: Path
) -> dict[str, Any]:
    expected_names, expected_labels = read_listfile(listfile)
    if dataset.names != expected_names:
        raise AssertionError(f"{split} IHMDataset changed released list order")
    labels = np.asarray(dataset.labels, dtype=np.float32)
    if not np.array_equal(labels, expected_labels):
        raise AssertionError(f"{split} IHMDataset changed released labels")
    expected = EXPECTED_SPLITS[split]
    if len(dataset) != expected["rows"] or int(labels.sum()) != expected["positive"]:
        raise AssertionError(f"{split} cohort count differs from the frozen contract")
    first_values, _, first_length, _ = dataset[0]
    if first_values.dtype != torch.float32 or tuple(first_values.shape) != (48, 76):
        raise AssertionError(f"{split} loader boundary is not float32 [48,76]")
    if first_length != 48:
        raise AssertionError(f"{split} sequence length changed")
    return {
        "rows": len(dataset),
        "positive": int(labels.sum()),
        "listfile_sha256": sha256(listfile),
        "ordered_name_sha256": hashlib.sha256(
            "\n".join(expected_names).encode()
        ).hexdigest(),
        "ordered_label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "first_shape": list(first_values.shape),
        "dtype": str(first_values.dtype),
    }


def load_datasets(
    source_root: Path, include_test: bool
) -> tuple[dict[str, IHMDataset], dict[str, Any]]:
    root = source_root / "in-hospital-mortality"
    declarations = {
        "train": ("train_listfile.csv", True),
        "validation": ("val_listfile.csv", True),
    }
    if include_test:
        declarations["test"] = ("test_listfile.csv", False)

    for split, (filename, _) in declarations.items():
        actual = sha256(root / filename)
        if actual != LISTFILE_SHA256[split]:
            raise AssertionError(
                f"{split} listfile hash changed: expected {LISTFILE_SHA256[split]}, "
                f"found {actual}"
            )

    datasets = {
        split: IHMDataset(
            str(root),
            customListFile=str(root / filename),
            train=train,
        )
        for split, (filename, train) in declarations.items()
    }

    # Preserve the released entrypoint's post-construction reassignment.  The
    # datasets eagerly materialize self.data in __init__, so this has no effect
    # on any tensor; omitting the operation would nevertheless hide a source
    # behavior that the audit deliberately records.
    header = datasets["train"].discretizer.transform(
        datasets["train"].reader.read_example(0)["X"]
    )[1].split(",")
    continuous = [index for index, name in enumerate(header) if "->" not in name]
    resource = PAPER_ROOT / "multimodal_clinical_pretraining/resources/normalizer_params"
    for dataset in datasets.values():
        dataset.normalizer = Normalizer(fields=continuous)
        dataset.normalizer.load_params(str(resource))

    audit = {
        split: verify_dataset(split, dataset, root / declarations[split][0])
        for split, dataset in datasets.items()
    }
    return datasets, audit


def model_arguments(cli: argparse.Namespace, epochs: int, seed: int) -> Namespace:
    return Namespace(
        experiment="full_eval",
        task="IHM",
        pretrained_path=str(cli.checkpoint),
        batch_size=cli.batch_size,
        epochs=epochs,
        lr=cli.lr,
        linear_lr=0.001,  # released default; unreachable for full_eval
        min_lr=1e-8,
        weight_decay=cli.weight_decay,
        opt="adamW",
        sched="cosine",
        decay_rate=1,
        warmup=True,
        warmup_lr=1e-5,
        warmup_epochs=0,
        cooldown_epochs=0,
        grad_clip=5,  # released argument; released loop does not apply it
        use_measurements=True,
        use_notes=False,
        use_projector=False,
        use_cls_token=False,
        measurement_model="Transformer",
        measurement_emb_size=128,
        measurement_num_heads=8,
        measurement_num_layers=8,
        measurement_activation=nn.GELU(),
        measurement_mask_rate=0.1,
        measurement_max_seq_len=256,
        measurement_dropout=0.0,
        notes_mask_rate=0.4,
        notes_emb_size=128,
        notes_num_heads=8,
        notes_num_layers=8,
        notes_dropout=0.0,
        notes_max_seq_len=256,
        text_model="BERT",
        mlp="128",
        use_pos_emb=True,
        n=EXPECTED_SPLITS["train"]["rows"],
        n_features=76,
        n_classes=1,
        world_size=1,
        local_rank=-1,
        seed=seed,
        device="cuda",
    )


def load_model(args: Namespace) -> tuple[nn.Sequential, dict[str, Any]]:
    backbone = create_model(args)
    state_dict = torch.load(args.pretrained_path, map_location="cpu")
    checkpoint_backbone = {
        key.replace("module.models.", ""): value
        for key, value in state_dict.items()
        if "models.0" in key
    }
    model_keys = set(backbone.models.state_dict())
    checkpoint_keys = set(checkpoint_backbone)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    if missing:
        raise AssertionError(f"checkpoint is missing measurement keys: {missing}")

    # Call the released loader, then establish that every expected measurement
    # tensor really was copied. strict=False intentionally ignores the two
    # obsolete measurement linear.* tensors shipped in the checkpoint.
    backbone = load_pretrained_model(backbone, args)
    loaded_state = backbone.models.state_dict()
    unequal = [
        key
        for key in sorted(model_keys)
        if not torch.equal(loaded_state[key].cpu(), checkpoint_backbone[key])
    ]
    if unequal:
        raise AssertionError(f"released checkpoint loader failed for keys: {unequal}")

    model = nn.Sequential(backbone, nn.Linear(args.measurement_emb_size, 1)).cuda()
    model[-1].weight.data.normal_(mean=0.0, std=0.01)
    model[-1].bias.data.zero_()
    return model, {
        "checkpoint_measurement_keys": len(checkpoint_keys),
        "model_measurement_keys": len(model_keys),
        "missing_keys": missing,
        "ignored_unexpected_keys": unexpected,
    }


def make_loader(
    dataset: IHMDataset, batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=pad_colalte,
        pin_memory=False,
        shuffle=shuffle,
    )


def run_loader(
    model: nn.Sequential,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    max_batches: int | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (values, labels, seq_lengths, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if values.dtype != torch.float32 or values.ndim != 3:
                raise AssertionError("collate boundary dtype/rank changed")
            if tuple(values.shape[1:]) != (48, 76):
                raise AssertionError(f"collate boundary shape changed: {values.shape}")
            if any(int(length) != 48 for length in seq_lengths):
                raise AssertionError("collate boundary sequence length changed")
            if training:
                optimizer.zero_grad()
            values = values.cuda()
            labels = labels.cuda()
            logits = model([{"x": values}])[:, 0, :].contiguous()
            probabilities = F.sigmoid(logits)[:, 0]
            loss = criterion(probabilities, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_count = labels.numel()
            loss_sum += float(loss.item()) * batch_count
            count += batch_count
            all_labels.append(labels.detach().cpu().numpy())
            all_probabilities.append(probabilities.detach().cpu().numpy())

    labels_array = np.concatenate(all_labels)
    probability_array = np.concatenate(all_probabilities)
    metrics = metric_values(labels_array, probability_array)
    metrics["loss"] = loss_sum / count
    metrics["rows"] = int(count)
    return metrics, labels_array, probability_array


def run_declaration(
    cli: argparse.Namespace,
    purpose: str,
    seed: int,
    epochs: int,
    max_batches: int | None,
) -> dict[str, Any]:
    return {
        "purpose": purpose,
        "seed": seed,
        "epochs": epochs,
        "batch_size": cli.batch_size,
        "validation_batch_size": cli.batch_size * 2,
        "learning_rate": cli.lr,
        "linear_head_learning_rate": 0.001,
        "linear_head_learning_rate_effective": False,
        "weight_decay": cli.weight_decay,
        "optimizer": "torch.optim.AdamW",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_epsilon": 1e-8,
        "scheduler": "released cosine scheduler",
        "warmup_epochs": 0,
        "architecture": {
            "input_features": 76,
            "embedding_size": 128,
            "attention_heads": 8,
            "transformer_layers": 8,
            "measurement_dropout": 0.0,
            "positional_embedding": True,
        },
        "max_batches_per_split": max_batches,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "test_access": purpose in ("evaluation", "smoke"),
    }


def completed_result(path: Path, declaration: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("status") != "complete" or value.get("run") != declaration:
        raise RuntimeError(f"existing result conflicts with this run: {path}")
    for artifact in value.get("artifacts", {}).values():
        artifact_path = path.parent / artifact["file"]
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"completed artifact no longer verifies: {artifact_path}")
    print(f"reusing verified completed run {path.parent}", flush=True)
    return value


def prediction_identifiers(names: list[str]) -> tuple[list[int], list[int]]:
    subjects: list[int] = []
    stays: list[int] = []
    for name in names:
        match = re.match(r"^(\d+)_episode(\d+)_", name)
        if match is None:
            raise AssertionError(f"cannot parse released IHM filename: {name}")
        subjects.append(int(match.group(1)))
        stays.append(int(match.group(2)))
    return subjects, stays


def train_one(
    cli: argparse.Namespace,
    datasets: dict[str, IHMDataset],
    dataset_audit: dict[str, Any],
    purpose: str,
    seed: int,
    epochs: int,
    output_dir: Path,
    max_batches: int | None = None,
) -> dict[str, Any]:
    declaration = run_declaration(cli, purpose, seed, epochs, max_batches)
    result_path = output_dir / "result.json"
    previous = completed_result(result_path, declaration)
    if previous is not None:
        return previous

    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(datasets["train"], cli.batch_size, shuffle=True)
    validation_loader = make_loader(
        datasets["validation"], cli.batch_size * 2, shuffle=False
    )
    test_loader = (
        make_loader(datasets["test"], cli.batch_size * 2, shuffle=False)
        if "test" in datasets
        else None
    )

    # The release constructs DataLoaders first, then seeds immediately before
    # model construction and first iteration. Preserve that RNG ordering.
    set_seed(seed)
    args = model_arguments(cli, epochs, seed)
    model, checkpoint_load = load_model(args)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler, _ = create_scheduler(args, optimizer)
    criterion = nn.BCELoss()

    started = time.time()
    history = []
    for epoch in range(epochs):
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics, _, _ = run_loader(
            model, train_loader, criterion, optimizer, max_batches
        )
        scheduler.step(epoch + 1)
        validation_metrics, _, _ = run_loader(
            model, validation_loader, criterion, None, max_batches
        )
        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": epoch_lr,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        validation_auroc = validation_metrics["auroc"]
        validation_auroc_text = (
            "null" if validation_auroc is None else f"{validation_auroc:.6f}"
        )
        print(
            f"{purpose} seed={seed} epoch={epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_auroc={validation_auroc_text} "
            f"val_auprc={validation_metrics['auprc']:.6f}",
            flush=True,
        )

    test_metrics = None
    artifacts: dict[str, dict[str, str]] = {}
    if test_loader is not None:
        test_metrics, labels, probabilities = run_loader(
            model, test_loader, criterion, None, max_batches
        )
        names = datasets["test"].names[: len(labels)]
        expected_labels = np.asarray(datasets["test"].labels[: len(labels)])
        if not np.array_equal(labels, expected_labels):
            raise AssertionError("test prediction order differs from released list order")
        subject_ids, stay_ids = prediction_identifiers(names)
        predictions_path = output_dir / "test_predictions.parquet"
        prediction_frame = pd.DataFrame(
            {
                "source_row": np.arange(len(labels), dtype=np.int32),
                "subject_id": np.asarray(subject_ids, dtype=np.int64),
                "sample_id": np.asarray(stay_ids, dtype=np.int64),
                "source_file": names,
                "label": labels.astype(np.float32),
                "probability": probabilities.astype(np.float32),
            }
        )
        temporary_predictions = predictions_path.with_suffix(".parquet.tmp")
        prediction_frame.to_parquet(temporary_predictions, index=False)
        temporary_predictions.replace(predictions_path)
        artifacts["test_predictions"] = {
            "file": predictions_path.name,
            "sha256": sha256(predictions_path),
        }
        if purpose == "evaluation":
            model_path = output_dir / "model_state.pt"
            temporary_model = model_path.with_suffix(".pt.tmp")
            torch.save(model.state_dict(), temporary_model)
            temporary_model.replace(model_path)
            artifacts["model_state"] = {
                "file": model_path.name,
                "sha256": sha256(model_path),
            }
        test_auroc = test_metrics["auroc"]
        test_auroc_text = "null" if test_auroc is None else f"{test_auroc:.6f}"
        print(
            f"{purpose} seed={seed} test_loss={test_metrics['loss']:.6f} "
            f"test_auroc={test_auroc_text} "
            f"test_auprc={test_metrics['auprc']:.6f}",
            flush=True,
        )

    torch.cuda.synchronize()
    result = {
        "status": "complete",
        "run": declaration,
        "checkpoint_load": checkpoint_load,
        "dataset_audit": dataset_audit,
        "history": history,
        "final_validation": history[-1]["validation"],
        "test": test_metrics,
        "published": PUBLISHED,
        "elapsed_seconds": time.time() - started,
        "artifacts": artifacts,
    }
    write_json(result_path, result)

    del model, optimizer, scheduler, train_loader, validation_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


def fixed_protocol(cli: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch_size": cli.batch_size,
        "learning_rate": cli.lr,
        "weight_decay": cli.weight_decay,
        "epoch_candidates": cli.epoch_candidates,
        "selection_seed": cli.selection_seed,
        "selection_metric": "minimum final validation BCE",
        "evaluation_seeds": cli.seeds,
        "published": PUBLISHED,
        "fidelity": "faithful_with_deviations",
        "known_deviations": [
            "paper does not release its winning downstream hyperparameters",
            "paper does not release its five seeds",
            "modern Blackwell-compatible PyTorch/CUDA is required",
            "adjacent harness replaces W&B with persisted local metrics",
        ],
        "confirmed_source_defect": (
            "public pretrained checkpoint was selected with IHM test labels"
        ),
    }


def select_epochs(cli: argparse.Namespace) -> dict[str, Any]:
    selection_path = cli.output_root / "selection.json"
    protocol = fixed_protocol(cli)
    if selection_path.exists():
        existing = json.loads(selection_path.read_text())
        if existing.get("protocol") != protocol:
            raise RuntimeError("existing selection.json conflicts with this protocol")
        print(f"reusing frozen selection {selection_path}", flush=True)
        return existing

    # Deliberately omit the test dataset and its labels in selection phase.
    datasets, audit = load_datasets(cli.source_root, include_test=False)
    candidates = []
    for epochs in cli.epoch_candidates:
        result = train_one(
            cli,
            datasets,
            audit,
            purpose="selection",
            seed=cli.selection_seed,
            epochs=epochs,
            output_dir=cli.output_root / "selection" / f"epochs_{epochs}",
        )
        candidates.append(
            {
                "epochs": epochs,
                "final_validation": result["final_validation"],
                "result_file": str(
                    Path("selection") / f"epochs_{epochs}" / "result.json"
                ),
            }
        )
    winner = min(
        candidates,
        key=lambda candidate: (candidate["final_validation"]["loss"], candidate["epochs"]),
    )
    selection = {
        "status": "complete",
        "protocol": protocol,
        "test_data_loaded": False,
        "candidates": candidates,
        "selected_epochs": winner["epochs"],
        "selected_final_validation": winner["final_validation"],
    }
    write_json(selection_path, selection)
    return selection


def evaluate(cli: argparse.Namespace) -> dict[str, Any]:
    selection_path = cli.output_root / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("run --phase select before any test evaluation")
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "complete" or selection.get("protocol") != fixed_protocol(cli):
        raise RuntimeError("selection.json is incomplete or conflicts with this protocol")
    epochs = int(selection["selected_epochs"])
    datasets, audit = load_datasets(cli.source_root, include_test=True)
    results = []
    for seed in cli.seeds:
        results.append(
            train_one(
                cli,
                datasets,
                audit,
                purpose="evaluation",
                seed=seed,
                epochs=epochs,
                output_dir=cli.output_root / "evaluation" / f"seed_{seed}",
            )
        )

    per_seed = [
        {"seed": seed, **result["test"]}
        for seed, result in zip(cli.seeds, results)
    ]
    aggregate = {}
    for metric in ("auroc", "auprc"):
        values = [result["test"][metric] for result in results]
        aggregate[metric] = {
            "mean": mean(values),
            "sample_std": stdev(values) if len(values) > 1 else 0.0,
            "published": PUBLISHED[metric],
            "mean_minus_published": mean(values) - PUBLISHED[metric],
        }
    summary = {
        "status": "complete",
        "protocol": fixed_protocol(cli),
        "selected_epochs": epochs,
        "selection_file": "selection.json",
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    write_json(cli.output_root / "summary.json", summary)
    return summary


def smoke(cli: argparse.Namespace) -> dict[str, Any]:
    datasets, audit = load_datasets(cli.source_root, include_test=True)
    result = train_one(
        cli,
        datasets,
        audit,
        purpose="smoke",
        seed=cli.selection_seed,
        epochs=1,
        output_dir=cli.output_root / "smoke",
        max_batches=1,
    )
    write_json(cli.output_root / "smoke_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "select", "evaluate"), required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PAPER_ROOT / ".export_assets/author_source",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PAPER_ROOT
            / ".export_assets/google_drive_1wB-4kUrNB9cHqD1qvR5fFEOaIUXmXTxI/pretrained_model.pth"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "paper_results/multimodal_clinical_pretraining/reproduction"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--epoch-candidates", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_root = args.output_root.resolve()
    if args.batch_size <= 0 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("batch size and learning rate must be positive; decay nonnegative")
    if not args.epoch_candidates or any(epoch <= 0 for epoch in args.epoch_candidates):
        raise ValueError("epoch candidates must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("evaluation seeds must be nonempty and distinct")
    return args


def main() -> None:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the reproduction requires an available CUDA GPU")
    if sha256(cli.checkpoint) != CHECKPOINT_SHA256:
        raise AssertionError("released checkpoint hash changed")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    cli.output_root.mkdir(parents=True, exist_ok=True)
    write_json(cli.output_root / "environment.json", environment_inventory())
    write_json(cli.output_root / "protocol.json", fixed_protocol(cli))
    if cli.phase == "smoke":
        smoke(cli)
    elif cli.phase == "select":
        select_epochs(cli)
    else:
        evaluate(cli)


if __name__ == "__main__":
    main()
