#!/usr/bin/env python3
"""Train the released GraphCare BAT model on the exact cohort.

The paper's complete KG boundary is private.  This runner therefore exposes a
strictly named direct-node approximation based only on released cluster maps;
it never presents regenerated topology as the paper's missing KG.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq
import sklearn
import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.metrics import average_precision_score, roc_auc_score


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
DEFAULT_TASK_DIR = (
    WORKSPACE_ROOT / "paper_tasks/graphcare/subsequent_visit_mortality"
)
MAP_ROOT = PAPER_ROOT / "clustering/ccscm_ccsproc_atc3"
MODEL_PATH = PAPER_ROOT / "graphcare_/model.py"
EXPECTED_ROWS = {"train": 7730, "validation": 991, "test": 996}
EXPECTED_POSITIVES = {"train": 531, "validation": 77, "test": 54}
EXPECTED_EVENTS = 833_355
EXPECTED_HISTORY = 28
EXPECTED_VOCABULARY = 785
PUBLISHED_AUROC = 0.703
PUBLISHED_AUPRC = 0.167


@dataclasses.dataclass(frozen=True)
class Sample:
    sample_id: int
    subject_id: int
    label: int
    split: str
    source_split_position: int
    visits: tuple[tuple[int, ...], ...]  # newest first; zero padding is implicit
    nodes: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Cohort:
    splits: dict[str, tuple[Sample, ...]]
    max_history: int
    num_nodes: int
    released_num_nodes: int
    declared_codes: int
    direct_released_nodes: int
    event_count: int
    profile: str
    source_hashes: dict[str, str]


@dataclasses.dataclass
class TensorBatch:
    node_ids: torch.Tensor
    rel_ids: torch.Tensor
    edge_index: torch.Tensor
    graph_batch: torch.Tensor
    visit_node: torch.Tensor
    ehr_nodes: torch.Tensor
    labels: torch.Tensor
    sample_ids: np.ndarray
    subject_ids: np.ndarray
    history_lengths: np.ndarray

    def to(self, device: torch.device) -> "TensorBatch":
        return TensorBatch(
            node_ids=self.node_ids.to(device, non_blocking=True),
            rel_ids=self.rel_ids.to(device, non_blocking=True),
            edge_index=self.edge_index.to(device, non_blocking=True),
            graph_batch=self.graph_batch.to(device, non_blocking=True),
            visit_node=self.visit_node.to(device, non_blocking=True),
            ehr_nodes=self.ehr_nodes.to(device, non_blocking=True),
            labels=self.labels.to(device, non_blocking=True),
            sample_ids=self.sample_ids,
            subject_ids=self.subject_ids,
            history_lengths=self.history_lengths,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_graphcare_class():
    spec = importlib.util.spec_from_file_location("graphcare_released_model", MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load released model from {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    # PyG 2.6 resolves the released MessagePassing type annotations through
    # the defining module at layer construction time.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GraphCare


def code_maps(profile: str):
    filenames = {
        "CONDITION": "ccscm_id2clus.json",
        "PROCEDURE": "ccsproc_id2clus.json",
        "DRUG": "atc3_id2clus.json",
    }
    raw_maps: dict[str, dict[str, int]] = {}
    all_released: set[int] = set()
    hashes: dict[str, str] = {}
    for namespace, filename in filenames.items():
        path = MAP_ROOT / filename
        mapping = {key: int(value) for key, value in load_json(path).items()}
        raw_maps[namespace] = mapping
        all_released.update(mapping.values())
        hashes[str(path.relative_to(WORKSPACE_ROOT))] = sha256_file(path)

    if sum(len(mapping) for mapping in raw_maps.values()) != EXPECTED_VOCABULARY:
        raise AssertionError("released GraphCare code maps do not contain 785 codes")
    if len(all_released) != 489:
        raise AssertionError("released direct EHR codes do not map to 489 clusters")

    inverse_path = MAP_ROOT / "clusters_inv_th015.json"
    cluster_inverse = load_json(inverse_path)
    released_num_nodes = max(int(value) for value in cluster_inverse.values()) + 1
    hashes[str(inverse_path.relative_to(WORKSPACE_ROOT))] = sha256_file(inverse_path)
    if released_num_nodes != 4599:
        raise AssertionError("released threshold-0.15 node space is not 4,599")

    if profile == "direct_compact":
        released_to_model = {
            released: compact for compact, released in enumerate(sorted(all_released))
        }
        num_nodes = len(released_to_model)
    elif profile == "direct_released_space":
        released_to_model = {released: released for released in all_released}
        num_nodes = released_num_nodes
    else:  # protected by argparse, retained as an invariant
        raise ValueError(profile)

    mapped = {
        namespace: {
            code: released_to_model[released] for code, released in mapping.items()
        }
        for namespace, mapping in raw_maps.items()
    }
    return mapped, num_nodes, released_num_nodes, len(all_released), hashes


def load_cohort(task_dir: Path, profile: str) -> Cohort:
    index_path = task_dir / "audit/task_index.parquet"
    events_path = task_dir / "data/events.parquet"
    codes_path = task_dir / "metadata/codes.parquet"
    manifest_path = task_dir / "manifest.json"
    for path in (index_path, events_path, codes_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    maps, num_nodes, released_num_nodes, direct_nodes, map_hashes = code_maps(profile)
    index = pq.read_table(
        index_path,
        columns=[
            "subject_id",
            "sample_id",
            "boolean_value",
            "split",
            "source_split_position",
            "history_length",
        ],
    ).to_pydict()
    row_count = len(index["sample_id"])
    if row_count != sum(EXPECTED_ROWS.values()):
        raise AssertionError(f"unexpected task index size: {row_count}")

    metadata: dict[int, tuple[int, int, str, int, int]] = {}
    for position in range(row_count):
        sample_id = int(index["sample_id"][position])
        if sample_id in metadata:
            raise AssertionError(f"duplicate sample_id: {sample_id}")
        metadata[sample_id] = (
            int(index["subject_id"][position]),
            int(bool(index["boolean_value"][position])),
            str(index["split"][position]),
            int(index["source_split_position"][position]),
            int(index["history_length"][position]),
        )

    visits_by_sample: dict[int, dict[int, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    unknown_codes: set[str] = set()
    event_count = 0
    parquet = pq.ParquetFile(events_path)
    for batch in parquet.iter_batches(
        batch_size=131_072, columns=["sample_id", "sequence_row", "code"]
    ):
        columns = batch.to_pydict()
        for sample_id, sequence_row, namespaced_code in zip(
            columns["sample_id"], columns["sequence_row"], columns["code"]
        ):
            event_count += 1
            namespace, separator, raw_code = str(namespaced_code).partition("//")
            mapping = maps.get(namespace)
            if not separator or mapping is None or raw_code not in mapping:
                unknown_codes.add(str(namespaced_code))
                continue
            visits_by_sample[int(sample_id)][int(sequence_row)].add(mapping[raw_code])

    if unknown_codes:
        preview = sorted(unknown_codes)[:10]
        raise AssertionError(f"codes missing released cluster mappings: {preview}")
    if event_count != EXPECTED_EVENTS:
        raise AssertionError(f"unexpected event count: {event_count}")

    max_history = max(row[4] for row in metadata.values())
    if max_history != EXPECTED_HISTORY:
        raise AssertionError(f"unexpected maximum history: {max_history}")

    by_split: dict[str, list[Sample]] = {name: [] for name in EXPECTED_ROWS}
    for sample_id, (subject_id, label, split, split_position, history_length) in (
        metadata.items()
    ):
        visit_map = visits_by_sample.get(sample_id)
        if not visit_map:
            raise AssertionError(f"sample has no events: {sample_id}")
        chronological_rows = sorted(visit_map)
        if len(chronological_rows) != history_length:
            raise AssertionError(
                f"sample {sample_id} has {len(chronological_rows)} visits, "
                f"expected {history_length}"
            )
        # data_prepare.pad_and_convert walks the released chronological visits
        # backwards, then appends all-zero rows to max_history.
        visits = tuple(
            tuple(sorted(visit_map[row])) for row in reversed(chronological_rows)
        )
        nodes = tuple(sorted({node for visit in visits for node in visit}))
        if not nodes:
            raise AssertionError(f"sample has no mapped direct nodes: {sample_id}")
        by_split[split].append(
            Sample(
                sample_id=sample_id,
                subject_id=subject_id,
                label=label,
                split=split,
                source_split_position=split_position,
                visits=visits,
                nodes=nodes,
            )
        )

    frozen_splits: dict[str, tuple[Sample, ...]] = {}
    for split, expected_rows in EXPECTED_ROWS.items():
        rows = sorted(by_split[split], key=lambda sample: sample.source_split_position)
        if len(rows) != expected_rows:
            raise AssertionError(f"{split} row count differs: {len(rows)}")
        positives = sum(sample.label for sample in rows)
        if positives != EXPECTED_POSITIVES[split]:
            raise AssertionError(f"{split} positive count differs: {positives}")
        if [sample.source_split_position for sample in rows] != list(
            range(expected_rows)
        ):
            raise AssertionError(f"{split} source positions are not contiguous")
        frozen_splits[split] = tuple(rows)

    declared_codes = pq.read_table(codes_path, columns=["code"]).num_rows
    if declared_codes != EXPECTED_VOCABULARY:
        raise AssertionError(f"unexpected declared vocabulary: {declared_codes}")

    source_hashes = {
        str(index_path.relative_to(WORKSPACE_ROOT)): sha256_file(index_path),
        str(events_path.relative_to(WORKSPACE_ROOT)): sha256_file(events_path),
        str(codes_path.relative_to(WORKSPACE_ROOT)): sha256_file(codes_path),
        str(manifest_path.relative_to(WORKSPACE_ROOT)): sha256_file(manifest_path),
        str(MODEL_PATH.relative_to(WORKSPACE_ROOT)): sha256_file(MODEL_PATH),
        str(Path(__file__).resolve().relative_to(WORKSPACE_ROOT)): sha256_file(
            Path(__file__).resolve()
        ),
        **map_hashes,
    }
    return Cohort(
        splits=frozen_splits,
        max_history=max_history,
        num_nodes=num_nodes,
        released_num_nodes=released_num_nodes,
        declared_codes=declared_codes,
        direct_released_nodes=direct_nodes,
        event_count=event_count,
        profile=profile,
        source_hashes=source_hashes,
    )


def collate(samples: Sequence[Sample], cohort: Cohort) -> TensorBatch:
    batch_size = len(samples)
    if batch_size == 0:
        raise ValueError("cannot collate an empty batch")
    visit_node = torch.zeros(
        (batch_size, cohort.max_history, cohort.num_nodes), dtype=torch.float32
    )
    ehr_nodes = torch.zeros((batch_size, cohort.num_nodes), dtype=torch.float32)
    all_node_ids: list[int] = []
    graph_batch: list[int] = []
    labels: list[float] = []
    sample_ids: list[int] = []
    subject_ids: list[int] = []
    history_lengths: list[int] = []

    for batch_index, sample in enumerate(samples):
        for visit_index, nodes in enumerate(sample.visits):
            visit_node[batch_index, visit_index, list(nodes)] = 1.0
        ehr_nodes[batch_index, list(sample.nodes)] = 1.0
        all_node_ids.extend(sample.nodes)
        graph_batch.extend([batch_index] * len(sample.nodes))
        labels.append(float(sample.label))
        sample_ids.append(sample.sample_id)
        subject_ids.append(sample.subject_id)
        history_lengths.append(len(sample.visits))

    node_ids = torch.tensor(all_node_ids, dtype=torch.long)
    # The public-artifact approximation has no attributable external edges.
    # Self edges keep the released BAT message/edge-attention path executable
    # without inventing medical knowledge.
    local_node = torch.arange(len(all_node_ids), dtype=torch.long)
    edge_index = torch.stack((local_node, local_node), dim=0)
    return TensorBatch(
        node_ids=node_ids,
        rel_ids=torch.zeros(len(all_node_ids), dtype=torch.long),
        edge_index=edge_index,
        graph_batch=torch.tensor(graph_batch, dtype=torch.long),
        visit_node=visit_node,
        ehr_nodes=ehr_nodes,
        labels=torch.tensor(labels, dtype=torch.float32).reshape(-1, 1),
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        subject_ids=np.asarray(subject_ids, dtype=np.int64),
        history_lengths=np.asarray(history_lengths, dtype=np.int16),
    )


def batches(
    rows: Sequence[Sample],
    cohort: Cohort,
    batch_size: int,
    order: Sequence[int] | None = None,
    drop_last: bool = False,
    max_batches: int | None = None,
) -> Iterator[TensorBatch]:
    indices: Sequence[int] = range(len(rows)) if order is None else order
    emitted = 0
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        if len(selected) < batch_size and drop_last:
            break
        yield collate([rows[index] for index in selected], cohort)
        emitted += 1
        if max_batches is not None and emitted >= max_batches:
            break


def binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict[str, float | None]:
    if len(np.unique(y_true)) < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }


def stratified_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, history_lengths: np.ndarray
) -> dict[str, dict[str, float | int | None]]:
    masks = {
        "one_visit": history_lengths == 1,
        "two_visits": history_lengths == 2,
        "two_or_more_visits": history_lengths >= 2,
        "three_or_more_visits": history_lengths >= 3,
        "five_or_more_visits": history_lengths >= 5,
    }
    result: dict[str, dict[str, float | int | None]] = {}
    for name, mask in masks.items():
        metrics = binary_metrics(y_true[mask], y_prob[mask])
        metrics["rows"] = int(mask.sum())
        metrics["positives"] = int(y_true[mask].sum())
        result[name] = metrics
    return result


def evaluate(
    model: torch.nn.Module,
    rows: Sequence[Sample],
    cohort: Cohort,
    batch_size: int,
    device: torch.device,
    drop_last: bool,
    max_batches: int | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    sample_ids: list[np.ndarray] = []
    subject_ids: list[np.ndarray] = []
    history_lengths: list[np.ndarray] = []
    with torch.no_grad():
        for cpu_batch in batches(
            rows,
            cohort,
            batch_size,
            drop_last=drop_last,
            max_batches=max_batches,
        ):
            batch = cpu_batch.to(device)
            logits = model(
                node_ids=batch.node_ids,
                rel_ids=batch.rel_ids,
                edge_index=batch.edge_index,
                batch=batch.graph_batch,
                visit_node=batch.visit_node,
                ehr_nodes=batch.ehr_nodes,
            )
            labels.append(batch.labels.detach().cpu().numpy().reshape(-1))
            probabilities.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            sample_ids.append(cpu_batch.sample_ids)
            subject_ids.append(cpu_batch.subject_ids)
            history_lengths.append(cpu_batch.history_lengths)
    prediction = {
        "y_true": np.concatenate(labels),
        "y_prob": np.concatenate(probabilities),
        "sample_id": np.concatenate(sample_ids),
        "subject_id": np.concatenate(subject_ids),
        "history_length": np.concatenate(history_lengths),
    }
    metrics = binary_metrics(prediction["y_true"], prediction["y_prob"])
    metrics["rows"] = int(len(prediction["y_true"]))
    metrics["positives"] = int(prediction["y_true"].sum())
    metrics["by_history_length"] = stratified_metrics(
        prediction["y_true"], prediction["y_prob"], prediction["history_length"]
    )
    return metrics, prediction


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalized_random(rows: int, columns: int) -> torch.Tensor:
    values = torch.randn(rows, columns, dtype=torch.float32)
    return F.normalize(values, p=2, dim=1)


def build_model(args: argparse.Namespace, cohort: Cohort, device: torch.device):
    GraphCare = load_graphcare_class()
    node_embedding = normalized_random(cohort.num_nodes, args.embedding_dim)
    relation_embedding = normalized_random(1, args.embedding_dim)
    model = GraphCare(
        num_nodes=cohort.num_nodes,
        num_rels=1,
        max_visit=cohort.max_history,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        out_channels=1,
        layers=args.num_layers,
        dropout=args.dropout,
        decay_rate=args.decay_rate,
        node_emb=node_embedding,
        rel_emb=relation_embedding,
        freeze=False,
        patient_mode=args.patient_mode,
        use_alpha=not args.disable_alpha,
        use_beta=not args.disable_beta,
        use_edge_attn=not args.disable_edge_attention,
        self_attn=0.0,
        gnn="BAT",
        attn_init=None,
        drop_rate=0.0,
        mask_padded_visits=args.mask_padded_visits,
    )
    return model.to(device)


def checkpoint_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def run_seed(
    args: argparse.Namespace,
    cohort: Cohort,
    seed: int,
    output_dir: Path,
    device: torch.device,
) -> dict:
    seed_everything(seed)
    seed_dir = output_dir / f"seed_{seed}"
    if seed_dir.exists() and any(seed_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {seed_dir}; pass --overwrite")
    seed_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, cohort, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_rows = cohort.splits["train"]
    validation_rows = cohort.splits["validation"]
    generator = torch.Generator().manual_seed(seed)
    best_auroc = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict] = []
    start_time = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        losses: list[float] = []
        for cpu_batch in batches(
            train_rows,
            cohort,
            args.batch_size,
            order=order,
            drop_last=True,
            max_batches=args.max_train_batches,
        ):
            batch = cpu_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                node_ids=batch.node_ids,
                rel_ids=batch.rel_ids,
                edge_index=batch.edge_index,
                batch=batch.graph_batch,
                visit_node=batch.visit_node,
                ehr_nodes=batch.ehr_nodes,
                in_drop=True,
            )
            loss = F.binary_cross_entropy_with_logits(logits, batch.labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_metrics, _ = evaluate(
            model,
            validation_rows,
            cohort,
            args.batch_size,
            device,
            drop_last=True,
            max_batches=args.max_eval_batches,
        )
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "validation_drop_last": validation_metrics,
        }
        history.append(row)
        current = validation_metrics["auroc"]
        improved = current is not None and math.isfinite(current) and current >= best_auroc
        if improved:
            best_auroc = current
            best_epoch = epoch
            best_state = checkpoint_to_cpu(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"seed={seed} epoch={epoch} loss={row['training_loss']:.6f} "
            f"val_auroc={current if current is None else f'{current:.6f}'} "
            f"best={best_auroc:.6f}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("no finite validation AUROC was produced")
    model.load_state_dict(best_state)
    checkpoint_path = seed_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": best_state,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_drop_last_auroc": best_auroc,
        },
        checkpoint_path,
    )

    validation_full, validation_prediction = evaluate(
        model,
        validation_rows,
        cohort,
        args.batch_size,
        device,
        drop_last=False,
        max_batches=args.max_eval_batches,
    )
    test_full, test_prediction = evaluate(
        model,
        cohort.splits["test"],
        cohort,
        args.batch_size,
        device,
        drop_last=False,
        max_batches=args.max_eval_batches,
    )
    np.savez_compressed(seed_dir / "validation_predictions.npz", **validation_prediction)
    np.savez_compressed(seed_dir / "test_predictions.npz", **test_prediction)

    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_drop_last_auroc": best_auroc,
        "validation_full": validation_full,
        "test_full": test_full,
        "elapsed_seconds": time.monotonic() - start_time,
        "epochs_completed": len(history),
        "history": history,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    (seed_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def runtime_manifest(args: argparse.Namespace, cohort: Cohort, device: torch.device) -> dict:
    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "--id=0",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = "unavailable"
    all_samples = [sample for rows in cohort.splits.values() for sample in rows]
    history_length_baseline = {}
    for split, rows in cohort.splits.items():
        labels = np.asarray([sample.label for sample in rows], dtype=np.int8)
        lengths = np.asarray([len(sample.visits) for sample in rows], dtype=np.int16)
        history_length_baseline[split] = binary_metrics(labels, -lengths)

    return {
        "schema_version": 1,
        "paper": "graphcare",
        "task": "MIMIC-III subsequent-visit mortality",
        "published_auroc": PUBLISHED_AUROC,
        "published_auprc": PUBLISHED_AUPRC,
        "fidelity": {
            "profile": cohort.profile,
            "status": "public-artifact direct-node approximation",
            "exact": [
                "export cohort, labels, split, event order, and direct code occurrences",
                "released CCS-CM/CCS-PROC/ATC3-to-cluster assignments",
                (
                    "released GraphCare BAT forward equations"
                    if not args.mask_padded_visits
                    else "released GraphCare BAT equations plus the explicit padding-mask ablation"
                ),
                "paper Table 10 mortality hyperparameters unless overridden",
            ],
            "approximated": [
                "unreleased concept-specific KG and global graph are replaced by self edges",
                "unreleased GPT node/relation embeddings are random unit-norm trainable weights",
                "unreleased attention initialization is disabled",
            ],
        },
        "cohort": {
            "rows": {name: len(rows) for name, rows in cohort.splits.items()},
            "positives": {
                name: sum(sample.label for sample in rows)
                for name, rows in cohort.splits.items()
            },
            "events": cohort.event_count,
            "declared_codes": cohort.declared_codes,
            "direct_released_nodes": cohort.direct_released_nodes,
            "model_nodes": cohort.num_nodes,
            "released_total_nodes": cohort.released_num_nodes,
            "maximum_history": cohort.max_history,
            "unique_subjects": len({sample.subject_id for sample in all_samples}),
            "subjects_contributing_prefix_at_least_length": {
                str(length): len(
                    {
                        sample.subject_id
                        for sample in all_samples
                        if len(sample.visits) >= length
                    }
                )
                for length in (1, 2, 3, 5, 10)
            },
            "target_semantics": (
                "each cumulative observed-visit prefix predicts mortality on "
                "the immediately subsequent admission"
            ),
            "history_length_distribution": {
                "mean": float(np.mean([len(sample.visits) for sample in all_samples])),
                "exact_counts": {
                    str(length): sum(
                        len(sample.visits) == length
                        for sample in all_samples
                    )
                    for length in range(1, cohort.max_history + 1)
                },
                "exact_positive_counts": {
                    str(length): sum(
                        sample.label
                        for sample in all_samples
                        if len(sample.visits) == length
                    )
                    for length in range(1, cohort.max_history + 1)
                },
            },
            "negative_history_length_only_baseline": history_length_baseline,
        },
        "configuration": {
            "batch_size": args.batch_size,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "num_layers": args.num_layers,
            "decay_rate": args.decay_rate,
            "use_alpha": not args.disable_alpha,
            "use_beta": not args.disable_beta,
            "use_edge_attention": not args.disable_edge_attention,
            "mask_padded_visits": args.mask_padded_visits,
            "patient_mode": args.patient_mode,
            "gnn": "BAT",
            "drop_last_train": True,
            "drop_last_validation_selection": True,
            "drop_last_independent_test": False,
            "max_train_batches": args.max_train_batches,
            "max_eval_batches": args.max_eval_batches,
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pyarrow": pq.__version__ if hasattr(pq, "__version__") else "22.0.0",
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_geometric": torch_geometric.__version__,
            "nvidia_driver": driver,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
        },
        "source_sha256": cohort.source_hashes,
    }


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def construct_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument(
        "--profile",
        choices=("direct_compact", "direct_released_space"),
        default="direct_compact",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=1536)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--decay-rate", type=float, default=0.01)
    parser.add_argument(
        "--patient-mode", choices=("joint", "graph", "node"), default="joint"
    )
    parser.add_argument("--disable-alpha", action="store_true")
    parser.add_argument("--disable-beta", action="store_true")
    parser.add_argument("--disable-edge-attention", action="store_true")
    parser.add_argument("--mask-padded-visits", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.embedding_dim < 1 or args.hidden_dim < 1:
        parser.error("batch and model dimensions must be positive")
    if args.epochs < 1 or args.patience < 1 or args.num_layers < 1:
        parser.error("epochs, patience, and layers must be positive")
    if not 0.0 <= args.dropout <= 1.0:
        parser.error("dropout must be in [0, 1]")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("max train batches must be positive")
    if args.max_eval_batches is not None and args.max_eval_batches < 1:
        parser.error("max eval batches must be positive")
    if args.output_dir is None:
        args.output_dir = (
            WORKSPACE_ROOT
            / "paper_results/graphcare/gpu_reproduction"
            / args.profile
        )
    return args


def main() -> None:
    args = construct_args()
    cohort = load_cohort(args.task_dir.resolve(), args.profile)
    inspection = {
        "profile": cohort.profile,
        "rows": {name: len(rows) for name, rows in cohort.splits.items()},
        "positives": {
            name: sum(sample.label for sample in rows)
            for name, rows in cohort.splits.items()
        },
        "events": cohort.event_count,
        "declared_codes": cohort.declared_codes,
        "direct_released_nodes": cohort.direct_released_nodes,
        "model_nodes": cohort.num_nodes,
        "released_total_nodes": cohort.released_num_nodes,
        "maximum_history": cohort.max_history,
    }
    if args.dry_run:
        print(json.dumps(inspection, indent=2, sort_keys=True))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GraphCare GPU reproduction")
    device = torch.device("cuda:0")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = runtime_manifest(args, cohort, device)
    manifest["seeds"] = args.seeds
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    results = [run_seed(args, cohort, seed, output_dir, device) for seed in args.seeds]
    test_aurocs = np.asarray([row["test_full"]["auroc"] for row in results])
    test_auprcs = np.asarray([row["test_full"]["auprc"] for row in results])
    validation_aurocs = np.asarray([row["validation_full"]["auroc"] for row in results])
    validation_auprcs = np.asarray([row["validation_full"]["auprc"] for row in results])

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }

    summary = {
        "seeds": args.seeds,
        "runs": len(results),
        "published_auroc": PUBLISHED_AUROC,
        "published_auprc": PUBLISHED_AUPRC,
        "validation_full": {
            "auroc": summarize(validation_aurocs),
            "auprc": summarize(validation_auprcs),
        },
        "independent_test": {
            "auroc": summarize(test_aurocs),
            "auprc": summarize(test_auprcs),
        },
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["independent_test"], sort_keys=True))


if __name__ == "__main__":
    main()
