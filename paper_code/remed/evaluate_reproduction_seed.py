#!/usr/bin/env python3
"""Independently capture and verify one terminal REMed seed."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from accelerate import Accelerator
from h5pickle import File
from sklearn.metrics import average_precision_score, roc_auc_score

from main import get_parser
from src.models import REMed
from src.trainer.remed import REMedTrainer
from src.utils.trainer_utils import PredLoss, PredMetric


EXPECTED_TEST_SAMPLES = 3_243
EXPECTED_TEST_POSITIVES = 631


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def require_report(path: Path, status: str) -> dict:
    report = json.loads(path.read_text())
    if report.get("status") != status:
        raise AssertionError((path, report.get("status"), status))
    return report


def source_test_metric_dicts(path: Path) -> list[dict[str, float]]:
    found = []
    for line in path.read_text(errors="replace").splitlines():
        position = line.find("{'epoch':")
        if position < 0:
            continue
        try:
            candidate = ast.literal_eval(line[position:])
        except (SyntaxError, ValueError):
            continue
        if isinstance(candidate, dict) and "test" in candidate:
            found.append(candidate["test"])
    return [{key: float(value) for key, value in metrics.items()} for metrics in found]


def resolved_args(seed: int, input_root: Path, save_root: Path):
    command = [
        "--input_path",
        str(input_root),
        "--save_dir",
        str(save_root),
        "--src_data",
        "mimiciv",
        "--train_type",
        "remed",
        "--time",
        "-99999",
        "--pred_time",
        "48",
        "--lr",
        "1e-5",
        "--scorer",
        "--scorer_use_time",
        "--pretrained",
        f"encoder_{seed}",
        "--no_pretrained_checkpoint",
        "--resume_name",
        f"remed_{seed}",
        "--test_only",
    ]
    args = get_parser().parse_args(command)
    args.seed = seed
    return args


def load_selected_model(args, checkpoint: Path, device_loader):
    accelerator = Accelerator(mixed_precision="bf16", split_batches=True)
    if accelerator.num_processes != 1:
        raise AssertionError("prediction capture requires exactly one process")
    args.local_batch_size = args.batch_size // accelerator.num_processes
    if args.local_batch_size != 8:
        raise AssertionError(args.local_batch_size)

    # Match Trainer.test()/load_model(): construct the model after Accelerator,
    # drop the released RoFormer position table, and tolerate exactly that one
    # missing key under the authors' strict=False load.
    model = REMed(args)
    state = dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    omitted_key = "pred_model.model.embed_positions.weight"
    if omitted_key not in state:
        raise AssertionError(f"selected checkpoint lacks {omitted_key}")
    del state[omitted_key]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys != [omitted_key] or incompatible.unexpected_keys:
        raise AssertionError(incompatible)
    model, loader = accelerator.prepare(model, device_loader)
    model.eval()
    model.set_mode("predictor")
    return accelerator, model, loader


def clustered_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    subjects: np.ndarray,
    seed: int,
    repetitions: int,
) -> dict:
    unique_subjects = np.unique(subjects)
    groups = {subject: np.flatnonzero(subjects == subject) for subject in unique_subjects}
    generator = np.random.default_rng(seed)
    aurocs, auprcs = [], []
    for _ in range(repetitions):
        sampled = generator.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate([groups[subject] for subject in sampled])
        y, score = labels[indices], probabilities[indices]
        if np.unique(y).size != 2:
            raise AssertionError("patient bootstrap produced one class")
        aurocs.append(roc_auc_score(y, score))
        auprcs.append(average_precision_score(y, score))
    return {
        "repetitions": repetitions,
        "seed": seed,
        "unique_subjects": len(unique_subjects),
        "auroc_percentile_95": [float(x) for x in np.percentile(aurocs, (2.5, 97.5))],
        "auprc_percentile_95": [float(x) for x in np.percentile(auprcs, (2.5, 97.5))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(2020, 2021, 2022), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--encoder-console-log", type=Path, required=True)
    parser.add_argument("--source-console-log", type=Path, required=True)
    parser.add_argument("--environment-report", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1_000)
    args_cli = parser.parse_args()

    if args_cli.output_dir.exists():
        raise FileExistsError(args_cli.output_dir)
    require_report(args_cli.environment_report, "passed")
    require_report(args_cli.input_report, "exact_paper_local_input_shim")
    if not args_cli.encoder_console_log.is_file():
        raise FileNotFoundError(args_cli.encoder_console_log)
    if not args_cli.source_console_log.is_file():
        raise FileNotFoundError(args_cli.source_console_log)
    encoder_source_metrics = source_test_metric_dicts(args_cli.encoder_console_log)
    remed_source_metrics = source_test_metric_dicts(args_cli.source_console_log)
    if not encoder_source_metrics:
        raise AssertionError("encoder stage log contains no source test metric dictionary")
    if not remed_source_metrics:
        raise AssertionError(
            "REMed source log contains no test metric dictionary"
        )
    source_metrics = remed_source_metrics[-1]

    args = resolved_args(args_cli.seed, args_cli.input_root, args_cli.save_root)
    run_dir = args_cli.save_root / f"remed_{args_cli.seed}"
    encoder_hdf = args_cli.save_root / f"encoder_{args_cli.seed}" / "mimiciv_encoded.h5"
    checkpoint = run_dir / "checkpoint_best.pt"
    for path in (encoder_hdf, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    # REMedTrainer constructs the exact released max length, cohort frame, and
    # ReprDataset class without calling its train/test controller.
    trainer = REMedTrainer(args)
    trainer.data = File(trainer.data_path, "r")
    dataset = trainer.dataset(args, trainer.test_subset, trainer.data, trainer.df)
    ordered_keys = np.asarray(dataset.keys)
    if len(ordered_keys) != EXPECTED_TEST_SAMPLES:
        raise AssertionError(len(ordered_keys))
    source_loader = trainer.dataloader_set(trainer.test_subset)
    accelerator, model, loader = load_selected_model(args, checkpoint, source_loader)
    criterion = PredLoss(args)
    metric = PredMetric(args)

    captured_keys, captured_labels, captured_probabilities = [], [], []
    cursor = 0
    with torch.no_grad():
        for sample in loader:
            output, representations = model(**sample)
            _, logging_outputs = criterion(output, representations)
            metric(logging_outputs, accelerator)
            target = output["target"]["los_7"].detach().float().cpu().reshape(-1).numpy()
            probability = output["pred"]["los_7"].detach().float().cpu().reshape(-1).numpy()
            count = len(target)
            batch_keys = ordered_keys[cursor : cursor + count]
            if len(batch_keys) != count or np.any(target == -1):
                raise AssertionError((cursor, count, len(batch_keys), target))
            captured_keys.extend(batch_keys.tolist())
            captured_labels.extend(target.tolist())
            captured_probabilities.extend(probability.tolist())
            cursor += count

    released_metrics = {key: float(value) for key, value in metric.get_metrics().items()}
    trainer.data.close()
    if cursor != EXPECTED_TEST_SAMPLES or captured_keys != ordered_keys.tolist():
        raise AssertionError((cursor, len(captured_keys), len(ordered_keys)))

    labels = np.asarray(captured_labels, dtype=np.float64)
    probabilities = np.asarray(captured_probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all() or int(labels.sum()) != EXPECTED_TEST_POSITIVES:
        raise AssertionError((np.isfinite(probabilities).all(), labels.sum()))

    test_frame = trainer.df.loc[
        trainer.df["split_2020"] == "test", ["stay_id", "subject_id", "los_7"]
    ].copy()
    frame_keys = test_frame["stay_id"].astype(str).to_numpy()
    if not np.array_equal(frame_keys, ordered_keys):
        raise AssertionError("dataset key order differs from cohort test order")
    frame_labels = test_frame["los_7"].to_numpy(dtype=np.float64)
    if not np.array_equal(frame_labels, labels):
        raise AssertionError("captured labels differ from cohort labels")
    stay_ids = test_frame["stay_id"].to_numpy(dtype=np.int64)
    subject_ids = test_frame["subject_id"].to_numpy(dtype=np.int64)

    independent = {
        "los_7_auroc": float(roc_auc_score(labels, probabilities)),
        "los_7_auprc": float(average_precision_score(labels, probabilities)),
    }
    for key, value in independent.items():
        if abs(value - released_metrics[key]) > 1e-12:
            raise AssertionError((key, value, released_metrics[key]))
    if set(source_metrics) != set(released_metrics):
        raise AssertionError((set(source_metrics), set(released_metrics)))
    for key in source_metrics:
        if abs(source_metrics[key] - released_metrics[key]) > 1e-10:
            raise AssertionError((key, source_metrics[key], released_metrics[key]))

    args_cli.output_dir.mkdir(parents=True)
    predictions_path = args_cli.output_dir / "los_7_test_predictions.parquet"
    prediction_frame = pl.DataFrame(
        {
            "seed": np.full(EXPECTED_TEST_SAMPLES, args_cli.seed, dtype=np.int64),
            "source_order": np.arange(EXPECTED_TEST_SAMPLES, dtype=np.int64),
            "stay_id": stay_ids,
            "subject_id": subject_ids,
            "label": labels,
            "probability": probabilities,
        },
        schema={
            "seed": pl.Int64,
            "source_order": pl.Int64,
            "stay_id": pl.Int64,
            "subject_id": pl.Int64,
            "label": pl.Float64,
            "probability": pl.Float64,
        },
    )
    prediction_frame.write_parquet(predictions_path, compression="zstd", statistics=True)

    report = {
        "audit_test_accesses": 1,
        "bootstrap": clustered_bootstrap(
            labels,
            probabilities,
            subject_ids,
            seed=2_026_090_100 + args_cli.seed,
            repetitions=args_cli.bootstrap_repetitions,
        ),
        "checkpoint": {
            "bytes": checkpoint.stat().st_size,
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
        },
        "encoded_hdf": {"bytes": encoder_hdf.stat().st_size, "path": str(encoder_hdf)},
        "frozen_inputs": {
            "environment_report": {
                "path": str(args_cli.environment_report),
                "sha256": sha256(args_cli.environment_report),
            },
            "input_report": {
                "path": str(args_cli.input_report),
                "sha256": sha256(args_cli.input_report),
            },
            "encoder_console_log": {
                "bytes": args_cli.encoder_console_log.stat().st_size,
                "path": str(args_cli.encoder_console_log),
                "sha256": sha256(args_cli.encoder_console_log),
            },
            "source_console_log": {
                "bytes": args_cli.source_console_log.stat().st_size,
                "path": str(args_cli.source_console_log),
                "sha256": sha256(args_cli.source_console_log),
            },
        },
        "identity_hashes": {
            "ordered_labels_float64": array_sha256(labels),
            "ordered_stay_ids_int64": array_sha256(stay_ids),
            "ordered_subject_ids_int64": array_sha256(subject_ids),
        },
        "independent_los_7_metrics": independent,
        "encoder_source_metrics": encoder_source_metrics,
        "predictions": {
            "bytes": predictions_path.stat().st_size,
            "path": str(predictions_path),
            "sha256": sha256(predictions_path),
        },
        "released_metrics": released_metrics,
        "samples": EXPECTED_TEST_SAMPLES,
        "seed": args_cli.seed,
        "split_column": "split_2020",
        "source_metrics": source_metrics,
        "source_test_accesses": len(encoder_source_metrics) + len(remed_source_metrics),
        "status": "passed",
        "test_accesses_total": (
            len(encoder_source_metrics) + len(remed_source_metrics) + 1
        ),
        "test_positives": EXPECTED_TEST_POSITIVES,
    }
    report_path = args_cli.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
