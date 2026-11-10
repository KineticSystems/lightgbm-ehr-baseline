"""Run DPMM's prespecified clean primary or released-loop sensitivity."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from dataset_mimic3.fusion import my_collate
from models.ehr_models import LSTM
from models.mimic3.fusion_dp import DP_Fusion
from models.note_models import BertForRepresentation
from reproduction_data import (
    CACHE_ROOT,
    COHORT_REPORT,
    FULL_BATCH_REPORT,
    PAPER_ROOT,
    TASK_ROOT,
    TOKENIZER,
    WORKSPACE_ROOT,
    build_author_bundle,
    released_loaders,
    sha256,
)


SEED = 1002
BATCH_SIZE = 16
MAX_EPOCHS = 100
PATIENCE = 15
BOOTSTRAP_SEED = 1002
PUBLISHED = {
    "auroc": {"point": 0.860, "lower": 0.840, "upper": 0.878},
    "auprc": {"point": 0.511, "lower": 0.462, "upper": 0.566},
    "f1": {"point": 0.498, "lower": 0.456, "upper": 0.543},
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class AccessCountingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.accesses = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        self.accesses += 1
        return self.dataset[index]

    def __getattr__(self, name):
        if name in {"dataset", "accesses"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fixed_args(output_root: Path, epochs: int) -> Namespace:
    return Namespace(
        task="in-hospital-mortality",
        labels_set="mortality",
        num_classes=1,
        vision_num_classes=1,
        batch_size=BATCH_SIZE,
        num_workers=0,
        data_pairs="partial_ehr_note",
        data_ratio=1.0,
        missing_token=None,
        timestep=2.0,
        imputation="previous",
        normalizer_state=str(CACHE_ROOT / "readm_ts.normalizer"),
        pretrained=False,
        mode="train",
        eval=False,
        network=None,
        epochs=epochs,
        lr=1e-4,
        beta_1=0.9,
        patience=PATIENCE,
        load_state=None,
        load_state_cxr=None,
        load_state_ehr=None,
        load_state_note=None,
        resume=False,
        dropout=0.0,
        rec_dropout=0.0,
        fusion_type="dp",
        layers=2,
        dim=256,
        depth=1,
        bert_type=str(TOKENIZER),
        bert_model_id="huawei-noah/TinyBERT_General_4L_312D",
        bert_revision="34707a33cd59a94ecde241ac209bf35103691b43",
        orig_d_txt=312,
        d_txt=512,
        fusion="joint",
        align=0.0,
        dp=1e-6,
        dp_fuse_type="lstm",
        dp_normalize_feats=True,
        dp_resample=True,
        replace_w_align=None,
        K=3,
        rho_scale=-3.0,
        eta=1.0,
        temperature=1e-4,
        save_dir=str(output_root),
    )


def config_dict(args: Namespace) -> dict:
    return {key: value for key, value in vars(args).items() if key != "sep_dp"}


def environment() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in (
                "numpy",
                "pandas",
                "polars",
                "pyarrow",
                "scikit-learn",
                "scipy",
                "tokenizers",
                "torch",
                "transformers",
                "wandb",
            )
        },
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "torch_cuda_arch_list": torch.cuda.get_arch_list(),
        "gpu": {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        },
    }


def file_record(path: Path) -> dict:
    return {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def code_records() -> dict:
    relative = (
        "models/loss.py",
        "models/mimic3/fusion_dp.py",
        "models/ehr_models.py",
        "models/note_models.py",
        "trainers/mimic3/dp_trainer.py",
        "trainers/trainer.py",
        "trainers/utils.py",
        "dataset_mimic3/ehr_dataset.py",
        "dataset_mimic3/fusion.py",
        "reproduction_data.py",
        "reproduce_dpmm.py",
    )
    return {name: sha256(PAPER_ROOT / name) for name in relative}


def assert_preflight() -> dict:
    check_path = WORKSPACE_ROOT / "paper_results/dpmm/runtime/reproduction_data_check.json"
    check = json.loads(check_path.read_text())
    if check.get("status") != "exact_adapter_boundary":
        raise AssertionError("DPMM reproduction adapter has not passed its exhaustive gate")
    required = {
        "token_ids",
        "attention_mask",
        "label_note",
        "sequence_length",
        "pair_presence",
    }
    if set(check.get("fields_matching_frozen_diagnostic_exactly", [])) != required:
        raise AssertionError("DPMM reproduction adapter exact-field gate changed")
    if sha256(FULL_BATCH_REPORT) != "dc7d9ba85484350272a40ab278206f41a5116597baf3ff70ad1540db7bc90d11":
        raise AssertionError("DPMM frozen full-batch report hash changed")
    if sha256(COHORT_REPORT) != "6e0f98ae7073bbdc748501727084ef955bd60fff2575483652daabf1b1d9f233":
        raise AssertionError("DPMM frozen cohort report hash changed")
    if check["frozen_full_batch_report"]["sha256"] != sha256(FULL_BATCH_REPORT):
        raise AssertionError("DPMM adapter check points to a different batch report")
    loss_source = (PAPER_ROOT / "models/loss.py").read_text()
    repaired = (
        "MultivariateNormal(self.mu[k], scale_tril=torch.diag(cov[k]))"
        in loss_source
    )
    if not repaired:
        raise AssertionError("DPMM missing-note sampler repair is absent")
    return {"adapter_check": file_record(check_path), "adapter": check}


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    _, _, thresholds = metrics.roc_curve(labels, probabilities)
    thresholds = thresholds[1:]
    if len(thresholds) == 0:
        raise AssertionError("DPMM validation produced no finite thresholds")
    scores = np.asarray(
        [metrics.f1_score(labels, probabilities > threshold) for threshold in thresholds]
    )
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def point_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | None = None,
    compute_optimal: bool = True,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if compute_optimal:
        selected_threshold, optimal_f1 = best_f1_threshold(labels, probabilities)
    else:
        selected_threshold, optimal_f1 = None, None
    if threshold is None:
        applied_threshold = 0.5 if selected_threshold is None else selected_threshold
    else:
        applied_threshold = float(threshold)
    return {
        "auroc": float(metrics.roc_auc_score(labels, probabilities)),
        "auprc": float(metrics.average_precision_score(labels, probabilities)),
        "f1": float(metrics.f1_score(labels, probabilities > applied_threshold)),
        "threshold": applied_threshold,
        "within_split_optimal_threshold": selected_threshold,
        "within_split_optimal_f1": optimal_f1,
    }


def basic_interval(point: float, values: np.ndarray) -> dict:
    delta = point - values
    return {
        "lower": float(point - np.percentile(delta, 97.5)),
        "upper": float(point - np.percentile(delta, 2.5)),
    }


def bootstrap_metrics(
    frame: pd.DataFrame, threshold: float, output: Path, seed: int
) -> dict:
    labels = frame["label"].to_numpy(np.int64)
    probabilities = frame["probability"].to_numpy(np.float64)
    subjects = frame["subject_id"].to_numpy(np.int64)
    point = point_metrics(labels, probabilities, threshold)
    rows = []
    row_rng = np.random.default_rng(seed)
    for replicate in range(1000):
        indices = row_rng.integers(0, len(frame), size=len(frame))
        value = point_metrics(
            labels[indices], probabilities[indices], threshold, compute_optimal=False
        )
        rows.append(
            {
                "procedure": "released_row_basic",
                "replicate": replicate,
                "auroc": value["auroc"],
                "auprc": value["auprc"],
                "f1": value["f1"],
            }
        )
    unique_subjects = np.unique(subjects)
    subject_groups = [np.flatnonzero(subjects == subject) for subject in unique_subjects]
    cluster_rng = np.random.default_rng(seed)
    for replicate in range(1000):
        selected = cluster_rng.integers(0, len(subject_groups), size=len(subject_groups))
        indices = np.concatenate([subject_groups[index] for index in selected])
        value = point_metrics(
            labels[indices], probabilities[indices], threshold, compute_optimal=False
        )
        rows.append(
            {
                "procedure": "subject_cluster_basic",
                "replicate": replicate,
                "auroc": value["auroc"],
                "auprc": value["auprc"],
                "f1": value["f1"],
            }
        )
    distributions = pd.DataFrame(rows)
    temporary = output.with_suffix(output.suffix + ".tmp")
    distributions.to_parquet(temporary, index=False)
    temporary.replace(output)
    result = {"seed": seed, "replicates": 1000, "procedures": {}}
    for procedure, values in distributions.groupby("procedure"):
        entry = {}
        for name in ("auroc", "auprc", "f1"):
            entry[name] = {
                "point": point[name],
                **basic_interval(point[name], values[name].to_numpy(np.float64)),
            }
        result["procedures"][str(procedure)] = entry
    return result


def non_buffer_state(model: DP_Fusion) -> dict:
    loss = model.dp_loss
    return {
        "gamma_1": loss.gamma_1.detach().cpu().clone(),
        "gamma_2": loss.gamma_2.detach().cpu().clone(),
        "phi": loss.phi.detach().cpu().clone(),
    }


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


class Engine:
    def __init__(self, args: Namespace):
        self.args = args
        self.device = torch.device("cuda")
        ehr = LSTM(
            input_dim=76,
            num_classes=args.num_classes,
            hidden_dim=args.dim,
            dropout=args.dropout,
            layers=args.layers,
        ).to(self.device)
        note = BertForRepresentation(args).to(self.device)
        self.model = DP_Fusion(args, ehr, note).to(self.device)
        self.loss = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), args.lr, betas=(0.9, args.beta_1)
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=10, mode="min"
        )

    def _move(self, batch):
        x, token, mask, label_ehr, _, lengths, pairs = batch
        x = torch.from_numpy(x).float().to(self.device)
        token = token.to(self.device)
        mask = mask.to(self.device)
        labels = torch.from_numpy(label_ehr).float().to(self.device).unsqueeze(1)
        return x, token, mask, labels, lengths, pairs

    def train_batch(self, batch) -> dict:
        x, token, mask, labels, lengths, pairs = self._move(batch)
        output = self.model(x, lengths, token, mask, pairs)
        probabilities = output["dp"]
        bce = self.loss(probabilities, labels)
        dp_term = self.args.dp * output["dp_loss"]
        total = bce + dp_term
        if not torch.isfinite(total):
            raise FloatingPointError("DPMM training loss is not finite")
        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()
        return {
            "bce_sum": float(bce.item()) * len(labels),
            "dp_sum": float(dp_term.item()) * len(labels),
            "total_sum": float(total.item()) * len(labels),
            "labels": labels.detach().cpu().numpy().reshape(-1),
            "probabilities": probabilities.detach().cpu().numpy().reshape(-1),
            "examples": len(labels),
        }

    def train_epoch(self, loader) -> dict:
        self.model.train()
        pieces = []
        totals = {"bce_sum": 0.0, "dp_sum": 0.0, "total_sum": 0.0, "examples": 0}
        started = time.monotonic()
        for batch_index, batch in enumerate(loader, start=1):
            value = self.train_batch(batch)
            pieces.append(value)
            for name in totals:
                totals[name] += value[name]
            if batch_index % 100 == 0:
                print(
                    f"train batch {batch_index}/{len(loader)} "
                    f"mean_bce={totals['bce_sum']/totals['examples']:.6f}",
                    flush=True,
                )
        labels = np.concatenate([value["labels"] for value in pieces])
        probabilities = np.concatenate([value["probabilities"] for value in pieces])
        point = point_metrics(labels, probabilities, compute_optimal=False)
        return {
            "examples": int(totals["examples"]),
            "bce": totals["bce_sum"] / totals["examples"],
            "dp_weighted": totals["dp_sum"] / totals["examples"],
            "total": totals["total_sum"] / totals["examples"],
            "auroc": point["auroc"],
            "auprc": point["auprc"],
            "seconds": time.monotonic() - started,
        }

    def evaluate(self, loader, threshold: float | None = None) -> dict:
        self.model.eval()
        labels_parts = []
        probability_parts = []
        bce_sum = 0.0
        dp_sum = 0.0
        examples = 0
        started = time.monotonic()
        with torch.no_grad():
            for batch in loader:
                x, token, mask, labels, lengths, pairs = self._move(batch)
                output = self.model(x, lengths, token, mask, pairs)
                probabilities = output["dp"]
                bce = self.loss(probabilities, labels)
                dp_term = self.args.dp * output["dp_loss"]
                if not torch.isfinite(bce + dp_term):
                    raise FloatingPointError("DPMM evaluation loss is not finite")
                labels_parts.append(labels.cpu().numpy().reshape(-1))
                probability_parts.append(probabilities.cpu().numpy().reshape(-1))
                bce_sum += float(bce.item()) * len(labels)
                dp_sum += float(dp_term.item()) * len(labels)
                examples += len(labels)
        labels = np.concatenate(labels_parts).astype(np.int64)
        probabilities = np.concatenate(probability_parts).astype(np.float32)
        point = point_metrics(labels, probabilities, threshold)
        return {
            "examples": examples,
            "bce": bce_sum / examples,
            "dp_weighted": dp_sum / examples,
            "seconds": time.monotonic() - started,
            "labels": labels,
            "probabilities": probabilities,
            "metrics": point,
        }


def counted_test_loader(test_loader):
    counting = AccessCountingDataset(test_loader.dataset)
    loader = DataLoader(
        counting,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=my_collate,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )
    return loader, counting


def save_predictions(
    output: Path,
    rows: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    if len(rows) != len(labels):
        raise AssertionError("DPMM prediction rows and labels differ in length")
    expected_labels = rows["label"].to_numpy(np.int64)
    if not np.array_equal(labels, expected_labels):
        raise AssertionError("DPMM test labels/order differ from frozen cohort")
    frame = pd.DataFrame(
        {
            "source_row": np.arange(len(rows), dtype=np.int32),
            "subject_id": rows["SUBJECT_ID"].to_numpy(np.int64),
            "sample_id": rows["stay_id"].to_numpy(np.int64),
            "hadm_id": rows["HADM_ID"].to_numpy(np.int64),
            "source_name": rows["source_name"].astype(str).to_numpy(),
            "fusion_position": rows["fusion_position"].to_numpy(np.int64),
            "note_present": rows["note_present"].to_numpy(bool),
            "label": labels,
            "probability": probabilities,
            "threshold_prediction": probabilities > threshold,
        }
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output)
    return frame


def checkpoint_payload(
    engine: Engine, epoch: int, best_auroc: float, threshold: float, history: list
) -> dict:
    return {
        "epoch": epoch,
        "state_dict": engine.model.state_dict(),
        "best_auroc": best_auroc,
        "threshold": threshold,
        "optimizer": engine.optimizer.state_dict(),
        "scheduler": engine.scheduler.state_dict(),
        "history": history,
        "non_buffer_dp_state_at_selection": non_buffer_state(engine.model),
    }


def run_clean(output_root: Path, epochs: int, preflight: dict) -> dict:
    started = time.monotonic()
    args = fixed_args(output_root, epochs)
    seed_all(SEED)
    bundle = build_author_bundle(args)
    train_loader, validation_loader, raw_test_loader = released_loaders(args, bundle)
    test_loader, test_counter = counted_test_loader(raw_test_loader)
    engine = Engine(args)
    torch.cuda.reset_peak_memory_stats()
    history = []
    best_auroc = -float("inf")
    best_threshold = None
    best_epoch = None
    patience = 0
    checkpoint_path = output_root / "best_checkpoint.pth.tar"
    for epoch in range(1, epochs + 1):
        train = engine.train_epoch(train_loader)
        validation = engine.evaluate(validation_loader)
        engine.scheduler.step(validation["bce"])
        record = {
            "epoch": epoch,
            "learning_rate": engine.optimizer.param_groups[0]["lr"],
            "train": train,
            "validation": {
                key: value
                for key, value in validation.items()
                if key not in {"labels", "probabilities"}
            },
        }
        history.append(record)
        improved = validation["metrics"]["auroc"] > best_auroc
        if improved:
            best_auroc = validation["metrics"]["auroc"]
            best_threshold = validation["metrics"]["threshold"]
            best_epoch = epoch
            patience = 0
            torch.save(
                checkpoint_payload(
                    engine, epoch, best_auroc, best_threshold, copy.deepcopy(history)
                ),
                checkpoint_path,
            )
        else:
            patience += 1
        write_json(
            output_root / "progress.json",
            {
                "status": "training",
                "mode": "clean",
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_auroc": best_auroc,
                "patience": patience,
                "test_dataset_accesses": test_counter.accesses,
                "history": history,
            },
        )
        print(
            f"epoch={epoch} train_auroc={train['auroc']:.6f} "
            f"val_auroc={validation['metrics']['auroc']:.6f} "
            f"val_auprc={validation['metrics']['auprc']:.6f} "
            f"best_epoch={best_epoch} patience={patience}",
            flush=True,
        )
        if patience >= PATIENCE:
            print(f"early stopping after epoch {epoch}", flush=True)
            break
    if test_counter.accesses != 0:
        raise AssertionError("DPMM clean primary accessed test data during selection")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    engine.model.load_state_dict(checkpoint["state_dict"])
    best_threshold = float(checkpoint["threshold"])
    # Match the released Trainer.load_state behavior: only registered state_dict
    # tensors are restored. gamma_1/gamma_2/phi remain at end-of-training state.
    evaluation_state = output_root / "evaluation_state_before_test.pth.tar"
    torch.save(
        {
            "state_dict": engine.model.state_dict(),
            "non_buffer_dp_state": non_buffer_state(engine.model),
            "rng_state": rng_state(),
            "threshold": best_threshold,
            "selected_epoch": int(checkpoint["epoch"]),
            "source_semantics": "best registered weights plus final-loop non-buffer DP state",
        },
        evaluation_state,
    )
    test = engine.evaluate(test_loader, best_threshold)
    if test_counter.accesses != len(test_loader.dataset):
        raise AssertionError("DPMM clean primary did not make exactly one test pass")
    predictions_path = output_root / "test_predictions.parquet"
    prediction_frame = save_predictions(
        predictions_path,
        bundle.ordered_rows("test"),
        test["labels"],
        test["probabilities"],
        best_threshold,
    )
    bootstrap_path = output_root / "test_bootstrap.parquet"
    bootstrap = bootstrap_metrics(
        prediction_frame, best_threshold, bootstrap_path, BOOTSTRAP_SEED
    )
    result = {
        "status": "complete",
        "fidelity": "faithful_with_documented_deviations",
        "mode": "paper_declared_clean_primary",
        "seed": SEED,
        "configuration": config_dict(args),
        "selection": {
            "criterion": "maximum validation AUROC",
            "selected_epoch": int(checkpoint["epoch"]),
            "selected_validation_auroc": float(checkpoint["best_auroc"]),
            "selected_validation_threshold": best_threshold,
            "epochs_completed": len(history),
            "early_stopping_patience": PATIENCE,
            "test_dataset_accesses_before_selection": 0,
        },
        "test": {
            "examples": test["examples"],
            "positives": int(test["labels"].sum()),
            "dataset_passes": 1,
            "bce": test["bce"],
            "dp_weighted": test["dp_weighted"],
            "metrics": test["metrics"],
            "bootstrap": bootstrap,
            "auroc_minus_published": test["metrics"]["auroc"] - PUBLISHED["auroc"]["point"],
            "auroc_inside_published_interval": (
                PUBLISHED["auroc"]["lower"]
                < test["metrics"]["auroc"]
                < PUBLISHED["auroc"]["upper"]
            ),
        },
        "published": PUBLISHED,
        "history": history,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "environment": environment(),
        "preflight": preflight,
        "code_sha256": code_records(),
        "known_deviations": [
            "reconstructed author-source normalizer and note artifacts are not historical byte-identical files",
            "one RTX 5090 and Blackwell-compatible Torch/CUDA replace the paper runtime",
            "clean train-then-validation loop replaces the defective validation-before-training/test-exposed released loop",
            "fatal DirichletProcessLossSep self.mu_x reference repaired to defined self.mu",
            "DP-only adjacent entrypoint avoids unrelated deleted Copula imports",
        ],
        "preserved_source_behaviors": [
            "validation-leaking reconstructed normalizer",
            "exactly-five-note exclusion and shifted/no-lower-bound note cutoff",
            "stochastic missing-note resampling during validation and test",
            "one-layer fusion LSTM despite paper text saying two layers",
            "best checkpoint restores registered weights but not non-buffer gamma/phi state",
        ],
        "artifacts": {},
    }
    for name, path in {
        "best_checkpoint": checkpoint_path,
        "evaluation_state_before_test": evaluation_state,
        "test_predictions": predictions_path,
        "test_bootstrap": bootstrap_path,
    }.items():
        result["artifacts"][name] = file_record(path)
    write_json(output_root / "result.json", result)
    return result


def run_smoke(output_root: Path, preflight: dict) -> dict:
    started = time.monotonic()
    args = fixed_args(output_root, 1)
    seed_all(SEED)
    bundle = build_author_bundle(args)
    train_loader, validation_loader, raw_test_loader = released_loaders(args, bundle)
    _, test_counter = counted_test_loader(raw_test_loader)
    engine = Engine(args)
    torch.cuda.reset_peak_memory_stats()
    paired_batch = my_collate([bundle.fusion["train"][index] for index in range(16)])
    boundary = len(bundle.fusion["train"].files_paired)
    mixed_indices = list(range(8)) + list(range(boundary, boundary + 8))
    mixed_batch = my_collate(
        [bundle.fusion["train"][index] for index in mixed_indices]
    )
    paired = engine.train_batch(paired_batch)
    mixed = engine.train_batch(mixed_batch)
    if sum(mixed_batch[-1]) != 8:
        raise AssertionError("DPMM smoke mixed batch did not contain 8 paired rows")
    checkpoint_path = output_root / "batch_checkpoint.pth.tar"
    torch.save({"state_dict": engine.model.state_dict()}, checkpoint_path)
    before = hashlib.sha256(
        b"".join(
            value.detach().cpu().contiguous().numpy().tobytes()
            for value in engine.model.state_dict().values()
        )
    ).hexdigest()
    with torch.no_grad():
        next(engine.model.parameters()).zero_()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    engine.model.load_state_dict(checkpoint["state_dict"])
    after = hashlib.sha256(
        b"".join(
            value.detach().cpu().contiguous().numpy().tobytes()
            for value in engine.model.state_dict().values()
        )
    ).hexdigest()
    if before != after:
        raise AssertionError("DPMM smoke checkpoint round trip changed model state")
    del engine, train_loader, validation_loader, raw_test_loader, bundle
    gc.collect()
    torch.cuda.empty_cache()

    # Re-seed and rebuild so the one-epoch gate starts from the primary state.
    seed_all(SEED)
    bundle = build_author_bundle(args)
    train_loader, validation_loader, raw_test_loader = released_loaders(args, bundle)
    _, test_counter = counted_test_loader(raw_test_loader)
    engine = Engine(args)
    train = engine.train_epoch(train_loader)
    validation = engine.evaluate(validation_loader)
    engine.scheduler.step(validation["bce"])
    if test_counter.accesses != 0:
        raise AssertionError("DPMM one-epoch smoke accessed test data")
    epoch_checkpoint = output_root / "one_epoch_checkpoint.pth.tar"
    torch.save(
        checkpoint_payload(
            engine,
            1,
            validation["metrics"]["auroc"],
            validation["metrics"]["threshold"],
            [],
        ),
        epoch_checkpoint,
    )
    result = {
        "status": "smoke_complete",
        "seed": SEED,
        "configuration": config_dict(args),
        "paired_batch": {
            "examples": paired["examples"],
            "bce": paired["bce_sum"] / paired["examples"],
            "total": paired["total_sum"] / paired["examples"],
        },
        "mixed_batch": {
            "examples": mixed["examples"],
            "paired": 8,
            "unpaired": 8,
            "bce": mixed["bce_sum"] / mixed["examples"],
            "total": mixed["total_sum"] / mixed["examples"],
        },
        "checkpoint_round_trip_sha256": before,
        "one_epoch": {
            "train": train,
            "validation": {
                key: value
                for key, value in validation.items()
                if key not in {"labels", "probabilities"}
            },
            "test_dataset_accesses": test_counter.accesses,
        },
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "environment": environment(),
        "preflight": preflight,
        "code_sha256": code_records(),
        "artifacts": {
            "batch_checkpoint": file_record(checkpoint_path),
            "one_epoch_checkpoint": file_record(epoch_checkpoint),
        },
    }
    write_json(output_root / "result.json", result)
    return result


def numpy_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: numpy_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [numpy_value(item) for item in value]
    return value


def run_released_loop(output_root: Path, preflight: dict) -> dict:
    from trainers.mimic3.dp_trainer import DirichletProcessTrainer

    started = time.monotonic()
    args = fixed_args(output_root, MAX_EPOCHS)
    seed_all(SEED)
    bundle = build_author_bundle(args)
    train_loader, validation_loader, raw_test_loader = released_loaders(args, bundle)
    test_loader, test_counter = counted_test_loader(raw_test_loader)
    trainer = DirichletProcessTrainer(
        train_loader, validation_loader, args, test_dl=test_loader
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    quick_test_calls = len(trainer.epochs_stats["auroc test"])
    accesses_before_final = test_counter.accesses
    if accesses_before_final != quick_test_calls * len(test_loader.dataset):
        raise AssertionError("DPMM released-loop test access count changed")
    checkpoint_path = output_root / "best_checkpoint.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trainer.args.load_state = str(checkpoint_path)
    trainer.model.load_state_dict(checkpoint["state_dict"])
    returned = trainer.eval()
    if test_counter.accesses != (quick_test_calls + 1) * len(test_loader.dataset):
        raise AssertionError("DPMM released-loop final test pass count changed")
    labels = np.load(output_root / "gt.npy").reshape(-1).astype(np.int64)
    probabilities = np.load(output_root / "pred.npy").reshape(-1).astype(np.float32)
    threshold = float(trainer.best_threshold[0])
    independent = point_metrics(labels, probabilities, threshold)
    if abs(independent["auroc"] - float(returned["auroc_mean"])) > 1e-12:
        raise AssertionError("DPMM released-loop saved predictions changed AUROC")
    predictions_path = output_root / "test_predictions.parquet"
    prediction_frame = save_predictions(
        predictions_path,
        bundle.ordered_rows("test"),
        labels,
        probabilities,
        threshold,
    )
    bootstrap_path = output_root / "test_bootstrap_deterministic.parquet"
    bootstrap = bootstrap_metrics(
        prediction_frame, threshold, bootstrap_path, BOOTSTRAP_SEED
    )
    result = {
        "status": "complete",
        "fidelity": "faithful_with_documented_deviations",
        "mode": "minimally_repaired_released_loop_test_exposed",
        "seed": SEED,
        "configuration": config_dict(args),
        "selection": {
            "criterion": "maximum validation AUROC in released validation-before-training loop",
            "selected_epoch": int(checkpoint["epoch"]),
            "selected_validation_auroc": float(checkpoint["best_auroc"]),
            "selected_validation_threshold": threshold,
            "loop_iterations": quick_test_calls,
            "trained_epochs": len(trainer.epochs_stats["loss train"]),
            "test_passes_before_final": quick_test_calls,
        },
        "test": {
            "examples": len(labels),
            "positives": int(labels.sum()),
            "dataset_passes": quick_test_calls + 1,
            "metrics": independent,
            "released_return": numpy_value(returned),
            "deterministic_bootstrap": bootstrap,
            "auroc_minus_published": independent["auroc"] - PUBLISHED["auroc"]["point"],
            "auroc_inside_published_interval": (
                PUBLISHED["auroc"]["lower"]
                < independent["auroc"]
                < PUBLISHED["auroc"]["upper"]
            ),
        },
        "published": PUBLISHED,
        "history": numpy_value(trainer.epochs_stats),
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "environment": environment(),
        "preflight": preflight,
        "code_sha256": code_records(),
        "known_test_exposure": True,
        "artifacts": {},
    }
    for name, path in {
        "best_checkpoint": checkpoint_path,
        "last_checkpoint": output_root / "last_checkpoint.pth.tar",
        "test_predictions": predictions_path,
        "deterministic_bootstrap": bootstrap_path,
        "source_pred_npy": output_root / "pred.npy",
        "source_gt_npy": output_root / "gt.npy",
    }.items():
        result["artifacts"][name] = file_record(path)
    write_json(output_root / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "clean", "released-loop"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DPMM reproduction requires CUDA")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("DPMM reproduction expected the RTX 5090 sm_120 device")
    output_root = args.output_root.resolve()
    result_path = output_root / "result.json"
    if result_path.is_file():
        print(f"completed result already exists: {result_path}")
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete DPMM run: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "run.log"
    with log_path.open("w", buffering=1) as log, contextlib.redirect_stdout(
        Tee(sys.__stdout__, log)
    ), contextlib.redirect_stderr(Tee(sys.__stderr__, log)):
        print(f"starting DPMM mode={args.mode} output={output_root}", flush=True)
        preflight = assert_preflight()
        if args.mode == "smoke":
            result = run_smoke(output_root, preflight)
        elif args.mode == "clean":
            result = run_clean(output_root, MAX_EPOCHS, preflight)
        else:
            result = run_released_loop(output_root, preflight)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "mode": result.get("mode", "smoke"),
                    "test": result.get("test"),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    result = json.loads(result_path.read_text())
    result.setdefault("artifacts", {})["run_log"] = file_record(log_path)
    write_json(result_path, result)


if __name__ == "__main__":
    main()
