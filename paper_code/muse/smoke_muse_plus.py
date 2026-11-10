"""Run one mixed MUSE+ optimization batch through unmodified author code."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT / "src" / "core"))

import src.dataset.mimic4_dataset as author_dataset_module  # noqa: E402
from model import MMLBackbone  # noqa: E402
from reproduce_muse import (  # noqa: E402
    AUTHOR_MISSING_LAB_WIDTH,
    LazyAdmissionMap,
    build_code_embeddings,
    build_vocabulary,
    install,
)
from src.dataset.utils import mimic4_collate_fn  # noqa: E402
from src.utils import set_seed  # noqa: E402


WORK = PAPER_ROOT / ".export_assets" / "repro"
BATCH_SIZE = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    vocabulary = build_vocabulary()
    embeddings = build_code_embeddings(vocabulary)
    admissions = LazyAdmissionMap(AUTHOR_MISSING_LAB_WIDTH)
    admissions.prepack()
    install(vocabulary, embeddings, admissions)

    train_set = author_dataset_module.MIMIC4Dataset(
        split="train", task="mortality", load_no_label=True
    )
    validation_set = author_dataset_module.MIMIC4Dataset(
        split="val", task="mortality", load_no_label=False
    )
    labeled_count = len(train_set) - len(train_set.no_label_admission_ids)
    mixed_indices = list(range(BATCH_SIZE // 2)) + list(
        range(labeled_count, labeled_count + BATCH_SIZE // 2)
    )
    train_loader = DataLoader(
        Subset(train_set, mixed_indices),
        batch_size=BATCH_SIZE,
        collate_fn=mimic4_collate_fn,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )
    validation_loader = DataLoader(
        Subset(validation_set, range(BATCH_SIZE)),
        batch_size=BATCH_SIZE,
        collate_fn=mimic4_collate_fn,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )

    args = argparse.Namespace(
        dataset="mimic4",
        num_classes=1,
        embedding_size=128,
        code_pretrained_embedding=True,
        code_layers=2,
        code_heads=2,
        bert_type="prajjwal1/bert-tiny",
        rnn_layers=1,
        rnn_type="GRU",
        rnn_bidirectional=False,
        ffn_layers=2,
        gnn_layers=2,
        gnn_norm=None,
        dropout=0.25,
        epochs=100,
        batch_size=BATCH_SIZE,
        lr=2e-4,
        weight_decay=0,
        seed=42,
        device=torch.device("cuda"),
    )
    set_seed(args.seed)
    model = MMLBackbone(args, train_set.tokenizer)
    model.to(args.device)
    torch.cuda.reset_peak_memory_stats()
    train_scores = model.train_epoch(train_loader)
    if not np.isfinite(train_scores["loss"]):
        raise AssertionError("one-batch MUSE+ loss is not finite")

    smoke_root = WORK / "smoke" / "muse_plus_one_batch"
    smoke_root.mkdir(parents=True, exist_ok=True)
    checkpoint = smoke_root / "checkpoint.pt"
    predictions_path = smoke_root / "validation_predictions.txt"
    torch.save(model.state_dict(), checkpoint)
    expected_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    with torch.no_grad():
        next(model.parameters()).add_(1)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    for name, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), expected_state[name]):
            raise AssertionError(f"checkpoint reload changed parameter {name}")

    validation_scores, predictions = model.eval_epoch(
        validation_loader, bootstrap=False
    )
    if predictions.shape != (BATCH_SIZE, 3):
        raise AssertionError("released validation prediction shape changed")
    np.savetxt(predictions_path, predictions, delimiter=",", fmt="%s")

    report = {
        "status": "passed",
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "configuration": vars(args) | {"device": str(args.device)},
        "batch": {
            "examples": BATCH_SIZE,
            "labeled": BATCH_SIZE // 2,
            "auxiliary": BATCH_SIZE // 2,
        },
        "train_loss": float(train_scores["loss"]),
        "validation_metrics": {
            name: float(value) for name, value in validation_scores.items()
        },
        "max_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "artifacts": {
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint),
            "predictions_bytes": predictions_path.stat().st_size,
            "predictions_sha256": sha256_file(predictions_path),
        },
    }
    report_path = smoke_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
