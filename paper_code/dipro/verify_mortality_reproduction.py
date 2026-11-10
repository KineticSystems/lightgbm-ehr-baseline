#!/usr/bin/env python3
"""Independent, CPU-only verifier for the DiPro checkpoint reproduction."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PAPER_DIR = Path(__file__).resolve().parent
ASSETS = PAPER_DIR / ".export_assets"
DEFAULT_RUN_DIR = ASSETS / "checkpoint_reproduction"
PREPARED = ASSETS / "multimodal_reproduction"
CHECKPOINT_DIR = ASSETS / "external_artifacts" / "huggingface_DiPro_model" / "mortality"
SEEDS = (23, 29, 66)
EXPECTED_CHECKPOINT_SHA256 = {
    23: "80a4415eb8abe751e221542484e71436e1b8cbabd96f0e2a2441fdb0a73df810",
    29: "96191a3acc911116ed39ad56e59e0b091be2d5981d7ee31609ccc804566f9942",
    66: "c5b3446e7222211931d4d312ac1e1a3efcef1157a8598365b97519799d6ca7f8",
}
EXPECTED_TEST_LABEL_SHA256 = "9646270846908881cdfe87d5199cf93542e0375512583c56da5a680f6ee98fcd"
EXPECTED_PREPARATION_SHA256 = "53d702eedc8c5030805c053d12dbe10e39f0c210eeafebf4fcc6e5d7aeec36d0"
EXPECTED_LOADER_SHA256 = "d5db96e461955ccb788b4cca08557c2ea34aef60efbd16c35c8c4bd7aac0c141"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(left: float, right: float, tolerance: float = 1e-14) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    require(run_dir.parent == ASSETS.resolve(), "run directory must be paper-local")
    verification_path = run_dir / "VERIFICATION.json"
    require(not verification_path.exists(), "refusing to overwrite existing verification")

    complete_path = run_dir / "RUN_COMPLETE.json"
    aggregate_path = run_dir / "aggregate_metrics.json"
    smoke_path = run_dir / "SMOKE.json"
    support_path = run_dir / "label_support" / "SUPPORT_COMPLETE.json"
    for path in (complete_path, aggregate_path, smoke_path, support_path):
        require(path.is_file(), "missing required run artifact: {}".format(path.name))

    complete = json.loads(complete_path.read_text())
    aggregate = json.loads(aggregate_path.read_text())
    smoke = json.loads(smoke_path.read_text())
    support = json.loads(support_path.read_text())
    require(complete.get("status") == "complete", "run is not marked complete")
    require(smoke.get("status") == "pass" and smoke.get("partition") == "validation", "smoke is not validation-only/pass")
    require(support.get("status") == "complete", "support preprocessing is incomplete")

    test_labels_path = PREPARED / "split" / "mortality_los" / "test.csv"
    require(sha256(test_labels_path) == EXPECTED_TEST_LABEL_SHA256, "exact test labels changed")
    require(sha256(PREPARED / "PREPARATION_COMPLETE.json") == EXPECTED_PREPARATION_SHA256, "preparation manifest changed")
    require(sha256(PREPARED / "REAL_LOADER_CHECK.json") == EXPECTED_LOADER_SHA256, "real-loader check changed")
    labels = pd.read_csv(test_labels_path)
    require(len(labels) == 546 and int(labels["mortality_inhospital"].sum()) == 110, "test cardinality changed")

    recomputed: List[Dict[str, Any]] = []
    file_hashes: Dict[str, str] = {}
    for seed in SEEDS:
        checkpoint_path = CHECKPOINT_DIR / ("seed_{}.ckpt".format(seed))
        require(sha256(checkpoint_path) == EXPECTED_CHECKPOINT_SHA256[seed], "checkpoint changed: {}".format(seed))
        prediction_path = run_dir / ("predictions_seed_{}.csv".format(seed))
        metric_path = run_dir / ("metrics_seed_{}.json".format(seed))
        require(prediction_path.is_file() and metric_path.is_file(), "missing seed output: {}".format(seed))
        metric = json.loads(metric_path.read_text())
        predictions = pd.read_csv(prediction_path)
        require(
            list(predictions.columns) == ["row_index", "subject_id", "stay_id", "target", "logit", "probability"],
            "prediction schema changed",
        )
        require(len(predictions) == 546, "prediction row count changed")
        require(np.array_equal(predictions["row_index"].to_numpy(), np.arange(546)), "row index changed")
        require(np.array_equal(predictions["subject_id"].to_numpy(dtype=np.int64), labels["subject_id"].to_numpy(dtype=np.int64)), "subject order changed")
        require(np.array_equal(predictions["stay_id"].to_numpy(dtype=np.int64), labels["stay_id"].to_numpy(dtype=np.int64)), "stay order changed")
        require(
            np.array_equal(predictions["target"].to_numpy(dtype=np.int64), labels["mortality_inhospital"].to_numpy(dtype=np.int64)),
            "target order changed",
        )
        require(np.isfinite(predictions[["logit", "probability"]].to_numpy()).all(), "non-finite predictions")
        require(((predictions["probability"] >= 0) & (predictions["probability"] <= 1)).all(), "invalid probabilities")

        target = predictions["target"].to_numpy(dtype=np.int64)
        logit = predictions["logit"].to_numpy(dtype=np.float64)
        probability = predictions["probability"].to_numpy(dtype=np.float64)
        auroc = float(roc_auc_score(target, logit))
        auprc = float(average_precision_score(target, logit))
        probability_auroc = float(roc_auc_score(target, probability))
        probability_auprc = float(average_precision_score(target, probability))
        require(close(auroc, metric["released_raw_logit_auroc"]), "seed AUROC mismatch")
        require(close(auprc, metric["released_raw_logit_auprc"]), "seed AUPRC mismatch")
        require(close(probability_auroc, metric["sigmoid_probability_auroc_diagnostic"]), "probability AUROC mismatch")
        require(close(probability_auprc, metric["sigmoid_probability_auprc_diagnostic"]), "probability AUPRC mismatch")
        require(metric["rows"] == 546 and metric["positive"] == 110, "metric cardinality mismatch")
        require(metric["batches"] == 137 and metric["batch_size"] == 4, "metric batching mismatch")
        require(metric["shuffle"] is False and metric["drop_last"] is False, "metric loader policy mismatch")
        require(metric["test_passes"] == 1, "checkpoint test-pass count mismatch")
        require(metric["checkpoint"]["strict_missing_keys"] == [], "checkpoint has missing state keys")
        require(metric["checkpoint"]["strict_unexpected_keys"] == [], "checkpoint has unexpected state keys")
        require(metric["checkpoint"]["state_keys"] == 455, "checkpoint state key count changed")
        require(metric["checkpoint"]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256[seed], "metric checkpoint identity mismatch")
        require(sha256(prediction_path) == metric["prediction_sha256"], "prediction hash mismatch")
        require(complete["output_sha256"][prediction_path.name] == sha256(prediction_path), "completion prediction hash mismatch")
        require(complete["output_sha256"][metric_path.name] == sha256(metric_path), "completion metric hash mismatch")
        file_hashes[prediction_path.name] = sha256(prediction_path)
        file_hashes[metric_path.name] = sha256(metric_path)
        recomputed.append({"seed": seed, "auroc": auroc, "auprc": auprc})

    aurocs = np.array([row["auroc"] for row in recomputed], dtype=np.float64)
    auprcs = np.array([row["auprc"] for row in recomputed], dtype=np.float64)
    expected_aggregate = {
        "auroc_mean": float(aurocs.mean()),
        "auroc_sample_std": float(aurocs.std(ddof=1)),
        "auprc_mean": float(auprcs.mean()),
        "auprc_sample_std": float(auprcs.std(ddof=1)),
    }
    for key, value in expected_aggregate.items():
        require(close(value, aggregate[key]), "aggregate mismatch: {}".format(key))
        require(close(value, complete["aggregate"][key]), "completion aggregate mismatch: {}".format(key))
    expected_joint = bool(0.895 <= aurocs.mean() <= 0.899 and 0.739 <= auprcs.mean() <= 0.745)
    require(aggregate["jointly_numerically_consistent"] is expected_joint, "frozen match decision mismatch")
    require(aggregate["test_passes_per_checkpoint"] == 1, "aggregate test-pass count mismatch")
    require(complete["output_sha256"][aggregate_path.name] == sha256(aggregate_path), "aggregate hash mismatch")
    require(complete["smoke_sha256"] == sha256(smoke_path), "smoke hash mismatch")
    require(complete["support_manifest_sha256"] == sha256(support_path), "support hash mismatch")
    file_hashes[aggregate_path.name] = sha256(aggregate_path)
    file_hashes["SMOKE.json"] = sha256(smoke_path)
    file_hashes["SUPPORT_COMPLETE.json"] = sha256(support_path)
    file_hashes["RUN_COMPLETE.json"] = sha256(complete_path)

    verification = {
        "status": "pass",
        "independent_recomputation": recomputed,
        "aggregate": expected_aggregate,
        "jointly_numerically_consistent": expected_joint,
        "test_rows_per_checkpoint": 546,
        "test_passes_per_checkpoint": 1,
        "checkpoint_sha256": {str(seed): EXPECTED_CHECKPOINT_SHA256[seed] for seed in SEEDS},
        "artifact_sha256": dict(sorted(file_hashes.items())),
    }
    verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "pass",
        "auroc_mean": expected_aggregate["auroc_mean"],
        "auroc_sample_std": expected_aggregate["auroc_sample_std"],
        "auprc_mean": expected_aggregate["auprc_mean"],
        "auprc_sample_std": expected_aggregate["auprc_sample_std"],
        "jointly_numerically_consistent": expected_joint,
        "verification_sha256": sha256(verification_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
