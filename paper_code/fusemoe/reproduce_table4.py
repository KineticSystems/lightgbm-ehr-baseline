#!/usr/bin/env python3
"""Run one prespecified FuseMoE Table 4 seed using the released training path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from accelerate import Accelerator
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from transformers import AutoModel, AutoTokenizer, set_seed


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT / "src"))

import core.train as author_train  # noqa: E402
from core.model import MULTCrossModel  # noqa: E402
from preprocessing.data_mimiciv import data_perpare  # noqa: E402


PAPER_SEEDS = (32, 42, 52, 62, 72)
EXPECTED_SPLITS = {
    "train": (6139, 974),
    "validation": (1315, 226),
    "test": (1316, 224),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_author_args(seed: int, input_dir: Path, run_dir: Path):
    """Return the complete author-argument namespace for the frozen primary."""

    return SimpleNamespace(
        task="ihm-48-table4",
        file_path=str(input_dir),
        output_dir=str(run_dir),
        tensorboard_dir=None,
        seed=seed,
        mode="train",
        modeltype="TS_Text",
        eval_score=["auc", "auprc", "f1"],
        num_labels=2,
        max_length=1024,
        pad_to_max_length=False,
        model_path=None,
        train_batch_size=2,
        eval_batch_size=8,
        num_update_bert_epochs=2,
        num_train_epochs=8,
        txt_learning_rate=2e-5,
        ts_learning_rate=4e-4,
        gradient_accumulation_steps=16,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        pt_mask_ratio=0.15,
        mean_mask_length=3,
        chunk=False,
        chunk_type="sent_doc_pos",
        warmup_proportion=0.10,
        kernel_size=1,
        num_heads=8,
        layers=3,
        cross_layers=3,
        embed_dim=128,
        irregular_learn_emb_ts=True,
        irregular_learn_emb_text=True,
        irregular_learn_emb_cxr=False,
        irregular_learn_emb_ecg=False,
        reg_ts=True,
        tt_max=48,
        embed_time=64,
        ts_to_txt=False,
        txt_to_ts=False,
        dropout=0.10,
        model_name="bioLongformer",
        num_of_notes=5,
        notes_order="Last",
        ratio_notes_order=None,
        bertcount=0,
        first_n_item=3,
        fine_tune=False,
        self_cross=False,
        TS_mixup=True,
        mixup_level="batch",
        fp16=True,
        debug=False,
        generate_data=False,
        FTLSTM=False,
        Interp=False,
        cpu=False,
        datagereate_seed=42,
        TS_model="Atten",
        cross_method="moe",
        hidden_size=512,
        gating_function=["laplace"],
        num_of_experts=[16],
        top_k=[4],
        disjoint_top_k=2,
        num_modalities=2,
        use_pt_text_embeddings=True,
        router_type="joint",
        use_balance_loss=True,
        balance_loss_coef=0.01,
        ck_file_path=str(run_dir / "checkpoint") + "/",
        mixed_precision="fp16",
    )


def config_dict(args) -> dict:
    values = dict(vars(args))
    values["paper_target"] = {
        "auroc": 0.8103,
        "f1": 0.4653,
        "paper_seeds": list(PAPER_SEEDS),
    }
    values["checkpoint_rule"] = "maximum validation F1; strict greater-than replacement"
    values["test_rule"] = "one ordered test pass after training from selected checkpoint"
    return values


def create_stack(args, model_root: Path):
    accelerator = Accelerator(mixed_precision="fp16", cpu=False)
    device = accelerator.device
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, use_fast=False, local_files_only=True
    )
    encoder = AutoModel.from_pretrained(model_root, local_files_only=True)
    encoder = encoder.to(device)

    train_dataset, _train_sampler, train_loader = data_perpare(
        args, "train", tokenizer
    )
    validation_dataset, _validation_sampler, validation_loader = data_perpare(
        args, "val", tokenizer
    )
    test_dataset, _test_sampler, test_loader = data_perpare(args, "test", tokenizer)
    for name, dataset in (
        ("train", train_dataset),
        ("validation", validation_dataset),
        ("test", test_dataset),
    ):
        expected_rows, expected_positive = EXPECTED_SPLITS[name]
        if len(dataset) != expected_rows:
            raise AssertionError(f"unexpected {name} rows: {len(dataset)}")
        positive = sum(int(row["label"]) for row in dataset.data)
        if positive != expected_positive:
            raise AssertionError(f"unexpected {name} positives: {positive}")
        if any(len(row["irg_ts"]) >= 1000 for row in dataset.data):
            raise AssertionError(f"{name} contains a collator-filtered row")

    model = MULTCrossModel(
        args=args,
        device=device,
        orig_d_ts=30,
        orig_reg_d_ts=60,
        orig_d_txt=768,
        ts_seq_num=args.tt_max,
        text_seq_num=args.num_of_notes,
        Biobert=encoder,
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [p for n, p in model.named_parameters() if "bert" not in n]},
            {
                "params": [p for n, p in model.named_parameters() if "bert" in n],
                "lr": args.txt_learning_rate,
            },
        ],
        lr=args.ts_learning_rate,
    )
    model, optimizer, train_loader, validation_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, test_loader
    )
    return {
        "accelerator": accelerator,
        "device": device,
        "model": model,
        "optimizer": optimizer,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "test_loader": test_loader,
    }


def ts_text_loss(model, batch, args):
    (
        ts,
        ts_mask,
        ts_tt,
        reg_ts,
        input_ids,
        attention_mask,
        text_embeddings,
        note_time,
        note_time_mask,
        _cxr_feats,
        _cxr_time,
        _cxr_time_mask,
        _ecg_feats,
        _ecg_time,
        _ecg_time_mask,
        labels,
        _cxr_missing,
        _text_missing,
        _ecg_missing,
    ) = batch
    task_loss, balance_loss = model(
        x_ts=ts,
        x_ts_mask=ts_mask,
        ts_tt_list=ts_tt,
        input_ids_sequences=input_ids,
        attn_mask_sequences=attention_mask,
        text_emb=text_embeddings,
        note_time_list=note_time,
        note_time_mask_list=note_time_mask,
        labels=labels,
        reg_ts=reg_ts,
    )
    total_loss = task_loss + args.balance_loss_coef * balance_loss
    return task_loss, balance_loss, total_loss


def evaluate_test_once(args, model, data_loader, dataset):
    model.eval()
    cursor = 0
    batches = 0
    scores = []
    labels = []
    stay_ids = []
    subject_ids = []
    hadm_ids = []
    for batch in data_loader:
        if batch is None:
            raise AssertionError("test collator returned None")
        (
            ts,
            ts_mask,
            ts_tt,
            reg_ts,
            input_ids,
            attention_mask,
            text_embeddings,
            note_time,
            note_time_mask,
            _cxr_feats,
            _cxr_time,
            _cxr_time_mask,
            _ecg_feats,
            _ecg_time,
            _ecg_time_mask,
            label,
            _cxr_missing,
            _text_missing,
            _ecg_missing,
        ) = batch
        with torch.no_grad():
            logits = model(
                x_ts=ts,
                x_ts_mask=ts_mask,
                ts_tt_list=ts_tt,
                input_ids_sequences=input_ids,
                attn_mask_sequences=attention_mask,
                text_emb=text_embeddings,
                note_time_list=note_time,
                note_time_mask_list=note_time_mask,
                reg_ts=reg_ts,
            )
        if logits is None or not torch.isfinite(logits).all():
            raise AssertionError("test logits are missing or nonfinite")
        batch_scores = logits.detach().float().cpu().numpy()
        batch_labels = label.detach().cpu().numpy().astype(np.int64)
        rows = dataset.data[cursor : cursor + len(batch_labels)]
        expected_labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        np.testing.assert_array_equal(batch_labels, expected_labels)
        scores.append(batch_scores)
        labels.append(batch_labels)
        stay_ids.extend(int(row["stay_id"]) for row in rows)
        subject_ids.extend(int(row["name"]) for row in rows)
        hadm_ids.extend(int(row["hadm_id"]) for row in rows)
        cursor += len(batch_labels)
        batches += 1
    if cursor != EXPECTED_SPLITS["test"][0]:
        raise AssertionError(f"test denominator changed: {cursor}")
    if batches != math.ceil(EXPECTED_SPLITS["test"][0] / args.eval_batch_size):
        raise AssertionError(f"test batch count changed: {batches}")

    y_score = np.concatenate(scores).astype(np.float64)
    y_true = np.concatenate(labels).astype(np.int64)
    y_pred = (y_score > 0.5).astype(np.int64)
    precision, recall, _threshold = precision_recall_curve(y_true, y_score)
    metrics = {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(auc(recall, precision)),
        "f1": float(f1_score(y_true, y_pred)),
        "rows": int(len(y_true)),
        "positive": int(y_true.sum()),
        "batches": batches,
        "nan_logits": 0,
        "filtered_examples": 0,
    }
    arrays = {
        "stay_id": np.asarray(stay_ids, dtype=np.int64),
        "subject_id": np.asarray(subject_ids, dtype=np.int64),
        "hadm_id": np.asarray(hadm_ids, dtype=np.int64),
        "y_true": y_true,
        "y_score": y_score,
    }
    return metrics, arrays


def install_checkpoint_tracker(args):
    original = author_train.check_point
    events = []

    def tracked(all_val, model, all_logits, tracked_args, eval_score=None):
        epoch = len(events)
        previous = max(
            (event["validation"][eval_score] for event in events), default=None
        )
        current = float(all_val[eval_score])
        selected = previous is None or current > previous
        original(all_val, model, all_logits, tracked_args, eval_score)
        events.append(
            {
                "zero_based_epoch": epoch,
                "validation": {
                    key: float(value)
                    for key, value in all_val.items()
                    if key != "auc_scores"
                },
                "selected": selected,
            }
        )

    author_train.check_point = tracked
    return events, original


def package_versions() -> dict:
    names = (
        "accelerate",
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorboardX",
        "tokenizers",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
    )
    return {name: importlib.metadata.version(name) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=PAPER_SEEDS)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args_cli = parser.parse_args()

    if not (args_cli.input_dir / "verification.json").is_file():
        raise FileNotFoundError("input verification.json is required")
    if args_cli.run_dir.exists():
        existing = sorted(path.name for path in args_cli.run_dir.iterdir())
        # run_table4_seed.sh opens the tee target before this process starts.
        # Its own preflight still refuses every pre-existing nonempty run.
        if existing not in ([], ["console.log"]):
            raise FileExistsError(f"refusing to mix with existing run: {args_cli.run_dir}")
    args_cli.run_dir.mkdir(parents=True, exist_ok=True)
    author_args = build_author_args(args_cli.seed, args_cli.input_dir, args_cli.run_dir)
    atomic_json(args_cli.run_dir / "config.json", config_dict(author_args))

    set_seed(author_args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    stack = create_stack(author_args, args_cli.model_root)
    parameter_count = sum(parameter.numel() for parameter in stack["model"].parameters())
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in stack["model"].parameters()
        if parameter.requires_grad
    )
    events, original_checkpoint = install_checkpoint_tracker(author_args)
    try:
        author_train.trainer_irg(
            model=stack["model"],
            args=author_args,
            accelerator=stack["accelerator"],
            train_dataloader=stack["train_loader"],
            dev_dataloader=stack["validation_loader"],
            test_data_loader=stack["test_loader"],
            device=stack["device"],
            optimizer=stack["optimizer"],
            writer=None,
        )
    finally:
        author_train.check_point = original_checkpoint
    if len(events) != author_args.num_train_epochs:
        raise AssertionError(f"expected 8 validation events, found {len(events)}")

    checkpoint_path = (
        Path(author_args.ck_file_path) / "f1" / f"{author_args.seed}.pth.tar"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=stack["device"])
    stack["model"].load_state_dict(checkpoint["network"])
    selected_events = [event for event in events if event["selected"]]
    selected_epoch = selected_events[-1]["zero_based_epoch"]
    expected_best_f1 = selected_events[-1]["validation"]["f1"]
    if float(checkpoint["best_val"]["f1"]) != expected_best_f1:
        raise AssertionError("tracked validation-F1 checkpoint differs from saved checkpoint")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    test_started = time.time()
    test_metrics, prediction_arrays = evaluate_test_once(
        author_args, stack["model"], stack["test_loader"], stack["test_dataset"]
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    test_seconds = time.time() - test_started
    prediction_path = args_cli.run_dir / "predictions.npz"
    np.savez_compressed(prediction_path, **prediction_arrays)
    events_path = args_cli.run_dir / "validation_history.json"
    atomic_json(events_path, events)

    input_manifest_path = args_cli.input_dir / "manifest.json"
    input_verification_path = args_cli.input_dir / "verification.json"
    result = {
        "status": "complete",
        "paper": "fusemoe",
        "target": "Table 4 MIMIC-IV 48-IHM vital signs + notes Laplace FuseMoE",
        "seed": author_args.seed,
        "published": {"auroc": 0.8103, "f1": 0.4653},
        "selected_zero_based_epoch": selected_epoch,
        "checkpoint_validation": {
            key: float(value)
            for key, value in checkpoint["best_val"].items()
            if key != "auc_scores"
        },
        "test": test_metrics,
        "parameter_count": int(parameter_count),
        "trainable_parameter_count_at_construction": int(trainable_parameter_count),
        "runtime_seconds": float(time.time() - started),
        "test_runtime_seconds": float(test_seconds),
        "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "environment": {
            "python": platform.python_version(),
            "packages": package_versions(),
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
        },
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint_path.relative_to(args_cli.run_dir)),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": sha256_file(checkpoint_path),
            },
            "predictions": {
                "path": prediction_path.name,
                "bytes": prediction_path.stat().st_size,
                "sha256": sha256_file(prediction_path),
            },
            "validation_history": {
                "path": events_path.name,
                "sha256": sha256_file(events_path),
            },
            "input_manifest_sha256": sha256_file(input_manifest_path),
            "input_verification_sha256": sha256_file(input_verification_path),
        },
        "evaluation_rule": {
            "checkpoint": "strict maximum validation F1",
            "test": "one ordered pass after training; positive-class softmax probability",
            "threshold": 0.5,
        },
    }
    atomic_json(args_cli.run_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
