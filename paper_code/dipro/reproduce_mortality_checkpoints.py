#!/usr/bin/env python3
"""Safely evaluate the three released DiPro mortality checkpoints.

``smoke`` performs one validation-only forward pass.  ``evaluate`` performs
exactly one ordered, size-four test pass for each frozen checkpoint.  There are
no model, checkpoint, seed, batch-size, split, or metric tuning arguments.
"""

import argparse
import ast
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.utils.data import DataLoader
import torchvision


PAPER_DIR = Path(__file__).resolve().parent
ASSETS = PAPER_DIR / ".export_assets"
PREPARED = ASSETS / "multimodal_reproduction"
SOURCE = ASSETS / "source_cache"
CHECKPOINT_DIR = ASSETS / "external_artifacts" / "huggingface_DiPro_model" / "mortality"
GOLD = (
    ASSETS
    / "external_artifacts"
    / "physionet_chest_imagenome_1.0.0"
    / "gold_dataset"
    / "gold_object_comparison_with_coordinates.txt"
)
DEFAULT_RUN_DIR = ASSETS / "checkpoint_reproduction"
TORCHVISION_CACHE = ASSETS / "torch_home" / "hub" / "checkpoints" / "resnet50-0676ba61.pth"
SEEDS = (23, 29, 66)
BATCH_SIZE = 4
EXPECTED_TEST_ROWS = 546
EXPECTED_TEST_POSITIVE = 110
EXPECTED_UNSAFE_GLOBALS = {
    "pytorch_lightning.callbacks.early_stopping.EarlyStopping",
    "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
}
EXPECTED_CHECKPOINT_SHA256 = {
    23: "80a4415eb8abe751e221542484e71436e1b8cbabd96f0e2a2441fdb0a73df810",
    29: "96191a3acc911116ed39ad56e59e0b091be2d5981d7ee31609ccc804566f9942",
    66: "c5b3446e7222211931d4d312ac1e1a3efcef1157a8598365b97519799d6ca7f8",
}
EXPECTED_AUTHOR_SHA256 = {
    "dataset_module/dipro_dataset.py": "3a11c545da29818471e6a99907f877a91679fb8e3294d77ff26139bf48b8888d",
    "models/model.py": "122a1fe18aa8ced70c479e0ab19968221b9b8ab92aa654a5d3cbcc6744b82166",
    "models/Base_Predictor.py": "a227a9954e9ba2447f425637098ff31bac83a6070e71158a0cb8e3d1b42aca6a",
    "models/Component.py": "bbc0ea283dafda1e831783d157011fb719cc0c53bd545ae4bb69613e16c32375",
    "mimic_data_extract/mimic3benchmark/constants.py": "c15ade778cca12c748fe863350704fcbe12a7d45496fc3b6ac906bd32f28055e",
    "configs/DiPro.yaml": "c59803545349cbf7c2cdaae74464e4bb648cb228278759fc3a7a79005ebb7849",
    "scripts/test.sh": "01040cc5783f5f66bc5e49b024da1443bceb194bf63182616010fbb07e8273ac",
}
EXPECTED_RESNET_SHA256 = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
EXPECTED_PREPARED_MANIFEST_SHA256 = "53d702eedc8c5030805c053d12dbe10e39f0c210eeafebf4fcc6e5d7aeec36d0"
EXPECTED_REAL_LOADER_SHA256 = "d5db96e461955ccb788b4cca08557c2ea34aef60efbd16c35c8c4bd7aac0c141"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_frozen_sources() -> Dict[str, str]:
    observed: Dict[str, str] = {}
    for relative, expected in EXPECTED_AUTHOR_SHA256.items():
        path = PAPER_DIR / relative
        require(path.is_file(), "missing author source: {}".format(relative))
        actual = sha256(path)
        require(actual == expected, "author source changed: {}".format(relative))
        observed[relative] = actual
    require(sha256(TORCHVISION_CACHE) == EXPECTED_RESNET_SHA256, "ResNet-50 cache hash mismatch")
    require(
        sha256(PREPARED / "PREPARATION_COMPLETE.json") == EXPECTED_PREPARED_MANIFEST_SHA256,
        "prepared multimodal manifest changed",
    )
    require(
        sha256(PREPARED / "REAL_LOADER_CHECK.json") == EXPECTED_REAL_LOADER_SHA256,
        "real multimodal loader gate changed or is absent",
    )
    for seed, expected in EXPECTED_CHECKPOINT_SHA256.items():
        require(sha256(CHECKPOINT_DIR / ("seed_{}.ckpt".format(seed))) == expected, "checkpoint hash mismatch")
    return observed


def import_author_support_functions(save_path: Path):
    mimic_extract = PAPER_DIR / "mimic_data_extract"
    sys.path.insert(0, str(PAPER_DIR))
    sys.path.insert(0, str(mimic_extract))
    original_argv = sys.argv[:]
    dummy = save_path
    sys.argv = [
        original_argv[0],
        "--scene_graph", str(PREPARED / "scene_graph"),
        "--save_path", str(save_path),
        "--progression_task_label_dir", str(dummy / "unused_progression"),
        "--golden_dir", str(GOLD),
        "--all_stay_csv_path", str(SOURCE / "all_stays.csv"),
        "--cxr_stay", str(SOURCE / "cxr_stay.csv"),
        "--split_save_dir", str(dummy),
        "--all_comparison_path", str(PREPARED / "all_comparison.csv"),
    ]
    try:
        from mimic3benchmark.constants import PROGRESSION_DISEASE, PROGRESSION_DICT
        from mimic3benchmark.scripts import process_ImaGenome
        from mimic3benchmark.scripts.split_train_and_test import (
            get_organ_progression_labels_mort_los,
            merge_progression_task,
            split_dataset_for_each_task,
        )
    finally:
        sys.argv = original_argv
    process_ImaGenome.args.save_path = str(save_path)
    return {
        "PROGRESSION_DISEASE": PROGRESSION_DISEASE,
        "PROGRESSION_DICT": PROGRESSION_DICT,
        "get_neighbor_cxr_labels": process_ImaGenome.get_neighbor_cxr_labels,
        "correct_using_golden_labels": process_ImaGenome.correct_using_golden_labels,
        "pair_progression_statistic": process_ImaGenome.pair_progression_statistic,
        "split_dataset_for_each_task": split_dataset_for_each_task,
        "merge_progression_task": merge_progression_task,
        "get_organ_progression_labels_mort_los": get_organ_progression_labels_mort_los,
    }


def verify_support_manifest(support: Path) -> Dict[str, Any]:
    manifest_path = support / "SUPPORT_COMPLETE.json"
    require(manifest_path.is_file(), "label-support manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["output_sha256"].items():
        path = support / relative
        require(path.is_file() and sha256(path) == expected, "label-support file changed: {}".format(relative))
    return manifest


def prepare_label_support(run_dir: Path) -> Dict[str, Any]:
    support = run_dir / "label_support"
    if support.exists():
        return verify_support_manifest(support)

    temporary = Path(tempfile.mkdtemp(prefix=".label_support.", dir=str(run_dir)))
    try:
        progression_work = temporary / "progression_work"
        progression_work.mkdir()
        author = import_author_support_functions(progression_work)
        diseases = author["PROGRESSION_DISEASE"]
        progression_dict = author["PROGRESSION_DICT"]

        # Resume the released preprocessing exactly where the mortality-only
        # the adapter had intentionally stopped.
        neighbor_path = progression_work / "neighbor_progression.csv"
        corrected_path = progression_work / "golden_corrected_neighbor_progression.csv"
        author["get_neighbor_cxr_labels"](str(PREPARED / "all_comparison.csv"), str(progression_work))
        author["correct_using_golden_labels"](str(GOLD), str(neighbor_path))
        require(corrected_path.is_file(), "author golden-label correction output is absent")
        author["pair_progression_statistic"](str(corrected_path), str(SOURCE / "cxr_stay.csv"), str(progression_work))

        author["split_dataset_for_each_task"](
            str(SOURCE / "all_stays.csv"),
            str(SOURCE / "cxr_stay.csv"),
            str(GOLD),
            str(temporary),
            str(progression_work / "disease_progressions"),
        )
        mortality_dir = temporary / "mortality_los"
        base_mortality_sha256: Dict[str, str] = {}
        for split in ("train", "val", "test"):
            generated = mortality_dir / (split + ".csv")
            frozen = SOURCE / "split" / "mortality_los" / (split + ".csv")
            generated_sha = sha256(generated)
            require(generated_sha == sha256(frozen), "real progression rebuild changed frozen {} mortality split".format(split))
            base_mortality_sha256[split] = generated_sha

        progression_root = temporary / "disease_progression"
        author["merge_progression_task"](str(temporary))
        author["get_organ_progression_labels_mort_los"](
            str(GOLD), str(PREPARED / "all_comparison.csv"), str(temporary), str(temporary)
        )
        for split in ("train", "val", "test"):
            generated = mortality_dir / (split + ".csv")
            prepared = PREPARED / "split" / "mortality_los" / (split + ".csv")
            require(sha256(generated) == sha256(prepared), "support enrichment differs from prepared {} split".format(split))

        merged_summary: Dict[str, Any] = {}
        for split in ("train", "val", "test"):
            path = progression_root / (split + ".csv")
            frame = pd.read_csv(path)
            require(set(diseases).issubset(frame.columns), "merged progression columns are incomplete")
            merged_summary[split] = {"rows": len(frame), "sha256": sha256(path)}

        train = pd.read_csv(progression_root / "train.csv")
        loss_weights: Dict[str, List[float]] = {}
        for disease in diseases:
            mapped = train[disease].map(progression_dict)
            count = int(mapped.count())
            weights = []
            for class_index in range(3):
                class_count = int((mapped == class_index).sum())
                require(class_count > 0, "zero auxiliary class count for {}".format(disease))
                weights.append(count / (class_count * 3.0))
            loss_weights[disease] = weights

        output_paths = sorted(path for path in temporary.rglob("*") if path.is_file())
        manifest = {
            "status": "complete",
            "purpose": "Released auxiliary loss-weight support required by DiPro constructor",
            "method": (
                "authors' unchanged get_neighbor_cxr_labels, correct_using_golden_labels, "
                "pair_progression_statistic, split_dataset_for_each_task, merge_progression_task, "
                "and get_organ_progression_labels_mort_los"
            ),
            "base_mortality_sha256": base_mortality_sha256,
            "merged": merged_summary,
            "auxiliary_loss_weights": loss_weights,
            "output_sha256": {
                str(path.relative_to(temporary)): sha256(path) for path in output_paths
            },
        }
        (temporary / "SUPPORT_COMPLETE.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.rename(support)
        return verify_support_manifest(support)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def resolve_metadata(mimic_root: Path) -> Path:
    candidates = (
        mimic_root / "mimic-cxr-2.0.0-metadata.csv",
        mimic_root / "mimic-cxr-2.0.0-metadata.csv.gz",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    require(path is not None, "missing exact MIMIC-CXR-JPG 2.0.0 metadata")
    require((mimic_root / "files").is_dir(), "missing exact MIMIC-CXR-JPG 2.0.0 image tree")
    return path


def import_author_model_and_loader():
    sys.path.insert(0, str(PAPER_DIR))
    from dataset_module.dipro_dataset import CollateFunc, DiproDataset
    from models.model import DiPro
    return CollateFunc, DiproDataset, DiPro


def make_loader(
    partition: str,
    mimic_root: Path,
    metadata_path: Path,
    support: Path,
    drop_last: bool,
):
    CollateFunc, DiproDataset, _ = import_author_model_and_loader()
    dataset = DiproDataset(
        partition=partition,
        task_name="mortality",
        label_path=str(support),
        demographic_path=str(SOURCE / "demographic_processed.csv"),
        ehr_time_series_path=str(SOURCE / "ehr_preprocessed"),
        bbox_csv=str(PREPARED / "cxr_bbox.csv"),
        mimic_cxr_path=str(mimic_root / "files"),
        cxr_meta_path=str(metadata_path),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=CollateFunc(dim=0),
        pin_memory=True,
        drop_last=drop_last,
    )
    return dataset, loader


def set_released_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch 1.9/CUDA 11.1 enabled TF32 on Ampere by default.  Preserve that
    # released runtime behavior explicitly under torch 2.7/Blackwell.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def safe_state_dict(seed: int) -> Tuple[Mapping[str, torch.Tensor], Dict[str, Any]]:
    path = CHECKPOINT_DIR / ("seed_{}.ckpt".format(seed))
    unsafe_globals = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    require(unsafe_globals == EXPECTED_UNSAFE_GLOBALS, "unexpected checkpoint pickle globals")
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

    with torch.serialization.safe_globals([EarlyStopping, ModelCheckpoint]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(checkpoint, dict), "checkpoint is not a mapping")
    state = checkpoint.get("state_dict")
    require(isinstance(state, Mapping), "checkpoint state_dict is absent")
    require(len(state) == 455, "unexpected checkpoint state key count")
    require(all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()), "non-tensor state entry")
    schema_lines = [
        "{}\t{}\t{}".format(key, tuple(value.shape), value.dtype) for key, value in state.items()
    ]
    schema_sha = hashlib.sha256(("\n".join(schema_lines) + "\n").encode()).hexdigest()
    metadata = {
        "seed": seed,
        "checkpoint_sha256": sha256(path),
        "unsafe_globals_reviewed": sorted(unsafe_globals),
        "top_level_keys": sorted(checkpoint.keys()),
        "pytorch_lightning_version": checkpoint.get("pytorch-lightning_version"),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "state_keys": len(state),
        "state_numel": int(sum(value.numel() for value in state.values())),
        "state_schema_sha256": schema_sha,
    }
    # Keep only the tensor mapping alive; optimizer and callback payloads are
    # discarded before constructing the model.
    del checkpoint
    gc.collect()
    return state, metadata


def build_strict_model(seed: int, support: Path):
    _, _, DiPro = import_author_model_and_loader()
    state, checkpoint_metadata = safe_state_dict(seed)
    model = DiPro(
        task_name="mortality",
        label_path=str(support),
        lambda_consistency=0,
        lambda_orthogonal=2.0,
        lambda_static_order=0,
        lambda_dynamic_order=0,
        lambda_CE_loss=2.0,
        num_transformer_layers=4,
        group_num=8,
        d_model=256,
        dropout_rate=0.1,
        num_fusion_layers=1,
        freeze_backbone=False,
        mode="max",
        max_epochs=100,
        ckpt_path=None,
    )
    incompatible = model.load_state_dict(state, strict=True)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys, "strict state load mismatch")
    checkpoint_metadata["strict_missing_keys"] = list(incompatible.missing_keys)
    checkpoint_metadata["strict_unexpected_keys"] = list(incompatible.unexpected_keys)
    checkpoint_metadata["model_parameter_count"] = int(sum(parameter.numel() for parameter in model.parameters()))
    del state
    gc.collect()
    model.eval()
    model.to("cuda:0")
    return model, checkpoint_metadata


def forward_batch(model: Any, batch: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    (
        ehr,
        ehr_mask,
        regions,
        region_mask,
        demographics,
        targets,
        y_aux,
        _y_aux_mask,
        ehr_times,
        interval_times,
    ) = batch
    inputs = [ehr, ehr_mask, regions, region_mask, demographics, y_aux, ehr_times, interval_times]
    inputs = [tensor.to("cuda:0", non_blocking=True) for tensor in inputs]
    targets_gpu = targets.to("cuda:0", non_blocking=True)
    with torch.no_grad():
        outputs = model(*inputs)
        logits = outputs[0]
        test_loss = model.loss(logits[:, 0].squeeze(1), targets_gpu)
    require(logits.shape == (len(targets), 1, 1), "unexpected mortality output shape")
    require(bool(torch.isfinite(logits).all()), "non-finite mortality logits")
    require(bool(torch.isfinite(test_loss)), "non-finite released test loss")
    losses: Dict[str, float] = {"test_loss": float(test_loss.detach().cpu())}
    names = ("consistency_loss", "orthogonal_loss", "static_loss", "dynamic_loss")
    for name, value in zip(names, outputs[1:]):
        require(bool(torch.isfinite(value).all()), "non-finite {}".format(name))
        losses[name] = float(value.detach().cpu())
    return logits[:, 0, 0].detach().float().cpu(), losses


def environment_record() -> Dict[str, Any]:
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }


def validate_run_dir(path: Path) -> Path:
    resolved = path.resolve()
    require(resolved.parent == ASSETS.resolve(), "run directory must be an immediate child of .export_assets")
    resolved.mkdir(exist_ok=True)
    return resolved


def smoke(mimic_root: Path, run_dir: Path) -> None:
    smoke_path = run_dir / "SMOKE.json"
    require(not smoke_path.exists(), "refusing to overwrite completed smoke result")
    sources = verify_frozen_sources()
    metadata_path = resolve_metadata(mimic_root)
    support_manifest = prepare_label_support(run_dir)
    support = run_dir / "label_support"
    set_released_seed(23)
    require(torch.cuda.is_available(), "CUDA is not available")
    require(torch.cuda.get_device_capability(0) == (12, 0), "expected RTX 5090 sm_120")
    dataset, loader = make_loader("val", mimic_root, metadata_path, support, drop_last=True)
    require(len(dataset) == 285, "unexpected validation dataset length")
    batch = next(iter(loader))
    model, checkpoint_metadata = build_strict_model(23, support)
    torch.cuda.reset_peak_memory_stats()
    logits, losses = forward_batch(model, batch)
    torch.cuda.synchronize()
    result = {
        "status": "pass",
        "partition": "validation",
        "checkpoint_seed": 23,
        "batch_size": BATCH_SIZE,
        "batch_indices": [0, 1, 2, 3],
        "logits": [float(value) for value in logits.tolist()],
        "losses": losses,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": checkpoint_metadata,
        "support_manifest_sha256": sha256(support / "SUPPORT_COMPLETE.json"),
        "prepared_manifest_sha256": sha256(PREPARED / "PREPARATION_COMPLETE.json"),
        "real_loader_check_sha256": sha256(PREPARED / "REAL_LOADER_CHECK.json"),
        "author_source_sha256": sources,
        "environment": environment_record(),
    }
    smoke_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "smoke_complete", "sha256": sha256(smoke_path)}, sort_keys=True), flush=True)
    del model, batch, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_seed(
    seed: int,
    loader: DataLoader,
    labels: pd.DataFrame,
    support: Path,
    run_dir: Path,
) -> Tuple[Dict[str, Any], Path, Path]:
    prediction_path = run_dir / ("predictions_seed_{}.csv".format(seed))
    metric_path = run_dir / ("metrics_seed_{}.json".format(seed))
    require(not prediction_path.exists() and not metric_path.exists(), "refusing to overwrite seed output")
    set_released_seed(seed)
    model, checkpoint_metadata = build_strict_model(seed, support)
    logits_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []
    batch_losses: List[Dict[str, float]] = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    offset = 0
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader):
        logits, losses = forward_batch(model, batch)
        targets = batch[5].detach().cpu().numpy().astype(np.int64)
        batch_size = len(targets)
        expected = labels.iloc[offset : offset + batch_size]["mortality_inhospital"].to_numpy(dtype=np.int64)
        require(np.array_equal(targets, expected), "test target/order mismatch at batch {}".format(batch_index))
        logits_all.append(logits.numpy())
        targets_all.append(targets)
        batch_losses.append(losses)
        offset += batch_size
        if (batch_index + 1) % 10 == 0 or batch_index + 1 == total_batches:
            print(json.dumps({
                "event": "progress",
                "seed": seed,
                "batch": batch_index + 1,
                "batches": total_batches,
                "rows": offset,
            }, sort_keys=True), flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    logits_np = np.concatenate(logits_all).astype(np.float32, copy=False)
    targets_np = np.concatenate(targets_all).astype(np.int64, copy=False)
    require(len(logits_np) == EXPECTED_TEST_ROWS and offset == EXPECTED_TEST_ROWS, "test row count mismatch")
    require(int(targets_np.sum()) == EXPECTED_TEST_POSITIVE, "test positive count mismatch")
    probabilities = torch.sigmoid(torch.from_numpy(logits_np)).numpy().astype(np.float32, copy=False)
    auroc = float(roc_auc_score(targets_np, logits_np))
    auprc = float(average_precision_score(targets_np, logits_np))
    probability_auroc = float(roc_auc_score(targets_np, probabilities))
    probability_auprc = float(average_precision_score(targets_np, probabilities))

    predictions = pd.DataFrame({
        "row_index": np.arange(EXPECTED_TEST_ROWS, dtype=np.int64),
        "subject_id": labels["subject_id"].to_numpy(dtype=np.int64),
        "stay_id": labels["stay_id"].to_numpy(dtype=np.int64),
        "target": targets_np,
        "logit": logits_np.astype(np.float64),
        "probability": probabilities.astype(np.float64),
    })
    predictions.to_csv(prediction_path, index=False, float_format="%.17g")
    metric = {
        "seed": seed,
        "rows": len(predictions),
        "positive": int(targets_np.sum()),
        "batches": total_batches,
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
        "test_passes": 1,
        "released_raw_logit_auroc": auroc,
        "released_raw_logit_auprc": auprc,
        "sigmoid_probability_auroc_diagnostic": probability_auroc,
        "sigmoid_probability_auprc_diagnostic": probability_auprc,
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": checkpoint_metadata,
        "prediction_sha256": sha256(prediction_path),
        "mean_batch_losses": {
            name: float(np.mean([batch[name] for batch in batch_losses])) for name in batch_losses[0]
        },
    }
    metric_path.write_text(json.dumps(metric, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event": "seed_complete",
        "seed": seed,
        "auroc": auroc,
        "auprc": auprc,
        "prediction_sha256": metric["prediction_sha256"],
        "metric_sha256": sha256(metric_path),
    }, sort_keys=True), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metric, prediction_path, metric_path


def evaluate(mimic_root: Path, run_dir: Path) -> None:
    complete_path = run_dir / "RUN_COMPLETE.json"
    require(not complete_path.exists(), "refusing to overwrite completed reproduction")
    sources = verify_frozen_sources()
    smoke_path = run_dir / "SMOKE.json"
    require(smoke_path.is_file(), "validation-only GPU smoke must pass before test evaluation")
    smoke_record = json.loads(smoke_path.read_text())
    require(smoke_record.get("status") == "pass", "GPU smoke did not pass")
    metadata_path = resolve_metadata(mimic_root)
    support_manifest = prepare_label_support(run_dir)
    support = run_dir / "label_support"
    require(torch.cuda.is_available(), "CUDA is not available")
    labels = pd.read_csv(support / "mortality_los" / "test.csv")
    require(len(labels) == EXPECTED_TEST_ROWS, "unexpected exact test rows")
    require(int(labels["mortality_inhospital"].sum()) == EXPECTED_TEST_POSITIVE, "unexpected test positives")
    dataset, loader = make_loader("test", mimic_root, metadata_path, support, drop_last=False)
    require(len(dataset) == EXPECTED_TEST_ROWS and len(loader) == math.ceil(EXPECTED_TEST_ROWS / BATCH_SIZE), "test loader cardinality mismatch")

    metrics = []
    output_paths: List[Path] = [smoke_path, support / "SUPPORT_COMPLETE.json"]
    for seed in SEEDS:
        metric, prediction_path, metric_path = evaluate_seed(seed, loader, labels, support, run_dir)
        metrics.append(metric)
        output_paths.extend([prediction_path, metric_path])

    aurocs = np.array([metric["released_raw_logit_auroc"] for metric in metrics], dtype=np.float64)
    auprcs = np.array([metric["released_raw_logit_auprc"] for metric in metrics], dtype=np.float64)
    aggregate = {
        "seeds": list(SEEDS),
        "auroc_by_seed": {str(metric["seed"]): metric["released_raw_logit_auroc"] for metric in metrics},
        "auprc_by_seed": {str(metric["seed"]): metric["released_raw_logit_auprc"] for metric in metrics},
        "auroc_mean": float(aurocs.mean()),
        "auroc_sample_std": float(aurocs.std(ddof=1)),
        "auprc_mean": float(auprcs.mean()),
        "auprc_sample_std": float(auprcs.std(ddof=1)),
        "paper_auroc_mean": 0.897,
        "paper_auroc_reported_sd": 0.002,
        "paper_auprc_mean": 0.742,
        "paper_auprc_reported_sd": 0.003,
        "frozen_auroc_match_interval": [0.895, 0.899],
        "frozen_auprc_match_interval": [0.739, 0.745],
        "auroc_numerically_consistent": bool(0.895 <= aurocs.mean() <= 0.899),
        "auprc_numerically_consistent": bool(0.739 <= auprcs.mean() <= 0.745),
        "jointly_numerically_consistent": bool(
            0.895 <= aurocs.mean() <= 0.899 and 0.739 <= auprcs.mean() <= 0.745
        ),
        "auroc_delta_from_paper": float(aurocs.mean() - 0.897),
        "auprc_delta_from_paper": float(auprcs.mean() - 0.742),
        "test_passes_per_checkpoint": 1,
    }
    aggregate_path = run_dir / "aggregate_metrics.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    output_paths.append(aggregate_path)
    set_released_seed(66)
    completion = {
        "status": "complete",
        "fidelity": "bounded_checkpoint_approximation",
        "target": "DiPro longitudinal-CXR first-48-hour in-hospital mortality",
        "aggregate": aggregate,
        "author_source_sha256": sources,
        "prepared_manifest_sha256": sha256(PREPARED / "PREPARATION_COMPLETE.json"),
        "real_loader_check_sha256": sha256(PREPARED / "REAL_LOADER_CHECK.json"),
        "support_manifest_sha256": sha256(support / "SUPPORT_COMPLETE.json"),
        "smoke_sha256": sha256(smoke_path),
        "output_sha256": {
            path.name: sha256(path) for path in output_paths if path.parent == run_dir
        },
        "environment": environment_record(),
    }
    complete_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event": "run_complete",
        "auroc_mean": aggregate["auroc_mean"],
        "auprc_mean": aggregate["auprc_mean"],
        "jointly_numerically_consistent": aggregate["jointly_numerically_consistent"],
        "sha256": sha256(complete_path),
    }, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "evaluate"))
    parser.add_argument("--mimic-cxr-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    run_dir = validate_run_dir(args.run_dir)
    mimic_root = args.mimic_cxr_root.resolve()
    if args.mode == "smoke":
        smoke(mimic_root, run_dir)
    else:
        evaluate(mimic_root, run_dir)


if __name__ == "__main__":
    main()
