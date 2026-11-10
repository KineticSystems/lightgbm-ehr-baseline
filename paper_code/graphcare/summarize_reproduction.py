#!/usr/bin/env python3
"""Consolidate GraphCare reproduction and LightGBM sensitivity evidence."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
RESULTS_ROOT = WORKSPACE_ROOT / "paper_results/graphcare"
TASK_DIR = WORKSPACE_ROOT / "paper_tasks/graphcare/subsequent_visit_mortality"
MAP_ROOT = PAPER_ROOT / "clustering/ccscm_ccsproc_atc3"
PUBLISHED = {"auroc": 0.703, "auprc": 0.167}
PLAIN_LIGHTGBM_REPRESENTATIONS = tuple(
    f"raw_icd_{icd_encoding}__{drug_encoding}"
    for icd_encoding in ("full", "full_plus_prefix")
    for drug_encoding in ("none", "atc3", "gsn")
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def stratified_metrics(
    labels: np.ndarray, probabilities: np.ndarray, history_lengths: np.ndarray
) -> dict[str, dict[str, float | int | None]]:
    masks = {
        "one_visit": history_lengths == 1,
        "two_visits": history_lengths == 2,
        "two_or_more_visits": history_lengths >= 2,
        "three_or_more_visits": history_lengths >= 3,
        "five_or_more_visits": history_lengths >= 5,
    }
    result = {}
    for name, mask in masks.items():
        subset = labels[mask]
        row: dict[str, float | int | None] = {
            "rows": int(mask.sum()),
            "positives": int(subset.sum()),
            "auroc": None,
            "auprc": None,
        }
        if np.unique(subset).size == 2:
            row.update(metrics(subset, probabilities[mask]))
        result[name] = row
    return result


def clustered_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    subject_ids: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    positions: dict[int, np.ndarray] = {}
    for subject_id in np.unique(subject_ids):
        positions[int(subject_id)] = np.flatnonzero(subject_ids == subject_id)
    subjects = np.asarray(sorted(positions), dtype=np.int64)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"auroc": [], "auprc": []}
    for _ in range(replicates):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([positions[int(subject_id)] for subject_id in sampled])
        subset = labels[indices]
        if np.unique(subset).size != 2:
            continue
        row = metrics(subset, probabilities[indices])
        for name in values:
            values[name].append(row[name])
    return {
        name: {
            "lower": float(np.quantile(metric_values, 0.025)),
            "upper": float(np.quantile(metric_values, 0.975)),
            "valid_replicates": len(metric_values),
        }
        for name, metric_values in values.items()
    }


def paired_clustered_bootstrap(
    labels: np.ndarray,
    candidate_probability: np.ndarray,
    reference_probability: np.ndarray,
    subject_ids: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    positions = {
        int(subject_id): np.flatnonzero(subject_ids == subject_id)
        for subject_id in np.unique(subject_ids)
    }
    subjects = np.asarray(sorted(positions), dtype=np.int64)
    rng = np.random.default_rng(seed)
    candidate_point = metrics(labels, candidate_probability)
    reference_point = metrics(labels, reference_probability)
    values: dict[str, list[float]] = {"auroc": [], "auprc": []}
    for _ in range(replicates):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([positions[int(subject_id)] for subject_id in sampled])
        subset = labels[indices]
        if np.unique(subset).size != 2:
            continue
        candidate = metrics(subset, candidate_probability[indices])
        reference = metrics(subset, reference_probability[indices])
        for name in values:
            values[name].append(candidate[name] - reference[name])
    return {
        name: {
            "point_difference": candidate_point[name] - reference_point[name],
            "lower": float(np.quantile(metric_values, 0.025)),
            "upper": float(np.quantile(metric_values, 0.975)),
            "probability_difference_above_zero": float(
                np.mean(np.asarray(metric_values) > 0)
            ),
            "valid_replicates": len(metric_values),
        }
        for name, metric_values in values.items()
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def gpu_mean_prediction(directory: Path) -> dict[str, np.ndarray]:
    summary = load_json(directory / "summary.json")
    rows = []
    for seed in summary["seeds"]:
        with np.load(directory / f"seed_{seed}/test_predictions.npz") as payload:
            rows.append({name: payload[name].copy() for name in payload.files})
    reference = rows[0]
    for row in rows[1:]:
        for name in ("y_true", "sample_id", "subject_id", "history_length"):
            if not np.array_equal(reference[name], row[name]):
                raise AssertionError(f"GPU seed predictions disagree on {name}")
    return {
        "labels": reference["y_true"],
        "probabilities": np.mean(
            np.stack([row["y_prob"] for row in rows], axis=0), axis=0
        ),
        "sample_ids": reference["sample_id"],
        "subject_ids": reference["subject_id"],
        "history_lengths": reference["history_length"],
    }


def gpu_ensemble(
    directory: Path, bootstrap_replicates: int, bootstrap_seed: int
) -> dict[str, Any]:
    summary = load_json(directory / "summary.json")
    prediction = gpu_mean_prediction(directory)
    mean_probability = prediction["probabilities"]
    labels = prediction["labels"]
    return {
        "seeds": summary["seeds"],
        "seedwise_validation": summary["validation_full"],
        "seedwise_test": summary["independent_test"],
        "mean_probability_ensemble": {
            **metrics(labels, mean_probability),
            "by_history_length": stratified_metrics(
                labels, mean_probability, prediction["history_lengths"]
            ),
            "patient_clustered_95pct_ci": clustered_bootstrap(
                labels,
                mean_probability,
                prediction["subject_ids"],
                bootstrap_replicates,
                bootstrap_seed,
            ),
        },
    }


def raw_icd_mean_prediction(directory: Path, representation: str) -> dict[str, np.ndarray]:
    result = load_json(directory / "result.json")
    rows = []
    for tuning_seed in result["tuning_seeds"]:
        path = directory / "predictions" / f"{representation}_seed_{tuning_seed}.npz"
        with np.load(path) as payload:
            rows.append({name: payload[name].copy() for name in payload.files})
    reference = rows[0]
    for row in rows[1:]:
        for name in (
            "test_labels",
            "test_sample_id",
            "test_subject_id",
            "test_history_length",
        ):
            if not np.array_equal(reference[name], row[name]):
                raise AssertionError(
                    f"raw-ICD tuning-seed predictions disagree on {name}"
                )
    return {
        "labels": reference["test_labels"],
        "probabilities": np.mean(
            np.stack([row["test_probability"] for row in rows], axis=0), axis=0
        ),
        "sample_ids": reference["test_sample_id"],
        "subject_ids": reference["test_subject_id"],
        "history_lengths": reference["test_history_length"],
    }


def raw_icd_summary(directory: Path) -> dict[str, Any]:
    result = load_json(directory / "result.json")
    return {
        "status": result["status"],
        "time_representation": result["time_representation"],
        "vocabulary_fit": result["vocabulary_fit"],
        "tuning_seeds": result["tuning_seeds"],
        "trials_per_representation_per_seed": result[
            "trials_per_representation_per_seed"
        ],
        "validation_selected_representation_counts": result[
            "validation_selected_representation_counts"
        ],
        "representations": {
            name: {
                key: value[key]
                for key in (
                    "description",
                    "training_vocabulary_size",
                    "feature_count",
                    "unknown_token_occurrences",
                    "validation_auroc",
                    "test_auroc",
                    "test_auprc",
                    "mean_probability_ensemble",
                )
            }
            for name, value in result["representations"].items()
        },
        "paired_ensemble_differences_vs_mapped_ccs_atc": result[
            "paired_ensemble_differences_vs_mapped_ccs_atc"
        ],
        "raw_icd_audit": result["raw_icd_audit"],
    }


def plain_lightgbm_mean_prediction(
    directory: Path, representation: str
) -> dict[str, np.ndarray]:
    result = load_json(directory / "result.json")
    rows = []
    for tuning_seed in result["tuning_seeds"]:
        path = directory / "predictions" / f"{representation}_seed_{tuning_seed}.npz"
        with np.load(path) as payload:
            rows.append({name: payload[name].copy() for name in payload.files})
    reference = rows[0]
    for row in rows[1:]:
        for name in (
            "test_labels",
            "test_sample_id",
            "test_subject_id",
            "test_history_length",
        ):
            if not np.array_equal(reference[name], row[name]):
                raise AssertionError(
                    f"plain LightGBM tuning-seed predictions disagree on {name}"
                )
    return {
        "labels": reference["test_labels"],
        "probabilities": np.mean(
            np.stack([row["test_probability"] for row in rows], axis=0), axis=0
        ),
        "sample_ids": reference["test_sample_id"],
        "subject_ids": reference["test_subject_id"],
        "history_lengths": reference["test_history_length"],
    }


def plain_lightgbm_summary(directory: Path) -> dict[str, Any]:
    result = load_json(directory / "result.json")
    return {
        "status": result["status"],
        "time_representation": result["time_representation"],
        "vocabulary_fit": result["vocabulary_fit"],
        "design": result["design"],
        "tuning_seeds": result["tuning_seeds"],
        "trials_per_representation_per_seed": result[
            "trials_per_representation_per_seed"
        ],
        "validation_selected_representation_counts": result[
            "validation_selected_representation_counts"
        ],
        "representations": {
            name: {
                key: value[key]
                for key in (
                    "description",
                    "training_vocabulary_size",
                    "feature_count",
                    "unknown_token_occurrences",
                    "validation_auroc",
                    "test_auroc",
                    "test_auprc",
                    "mean_probability_ensemble",
                )
            }
            for name, value in result["representations"].items()
        },
        "paired_ensemble_contrasts": result["paired_ensemble_contrasts"],
        "raw_input_audit": result["raw_input_audit"],
    }


def raw_atc_replication_check(raw_directory: Path, plain_directory: Path) -> dict:
    raw_trials = [
        row
        for row in load_json(raw_directory / "trials.json")
        if row["representation"] == "raw_icd_full_atc"
    ]
    plain_trials = [
        row
        for row in load_json(plain_directory / "trials.json")
        if row["representation"] == "raw_icd_full__atc3"
    ]
    key = lambda row: (row["tuning_seed"], row["trial_within_representation"])
    raw_trials.sort(key=key)
    plain_trials.sort(key=key)
    if [key(row) for row in raw_trials] != [key(row) for row in plain_trials]:
        raise AssertionError("raw ICD + ATC paired trial keys differ")
    parameters_exact = all(
        raw["parameters"] == plain["parameters"]
        for raw, plain in zip(raw_trials, plain_trials, strict=True)
    )
    validation_auroc_max_absolute_difference = max(
        abs(raw["validation_auroc"] - plain["validation_auroc"])
        for raw, plain in zip(raw_trials, plain_trials, strict=True)
    )

    raw_result = load_json(raw_directory / "result.json")
    plain_result = load_json(plain_directory / "result.json")
    raw_selected_by_seed = {
        row["tuning_seed"]: row["selected_trial"]
        for row in raw_result["representations"]["raw_icd_full_atc"][
            "per_tuning_seed"
        ]
    }
    plain_selected_by_seed = {
        row["tuning_seed"]: row["selected_trial"]
        for row in plain_result["representations"]["raw_icd_full__atc3"][
            "per_tuning_seed"
        ]
    }
    maximum_prediction_difference = 0.0
    selected_trials_exact = True
    for tuning_seed in raw_result["tuning_seeds"]:
        with np.load(
            raw_directory / "predictions" / f"raw_icd_full_atc_seed_{tuning_seed}.npz"
        ) as raw_payload, np.load(
            plain_directory
            / "predictions"
            / f"raw_icd_full__atc3_seed_{tuning_seed}.npz"
        ) as plain_payload:
            for name in (
                "test_labels",
                "test_sample_id",
                "test_subject_id",
                "test_history_length",
            ):
                if not np.array_equal(raw_payload[name], plain_payload[name]):
                    raise AssertionError(f"raw ICD + ATC replication differs on {name}")
            difference = float(
                np.max(
                    np.abs(
                        raw_payload["test_probability"]
                        - plain_payload["test_probability"]
                    )
                )
            )
            maximum_prediction_difference = max(
                maximum_prediction_difference, difference
            )
        selected_trials_exact &= (
            raw_selected_by_seed[tuning_seed]
            == plain_selected_by_seed[tuning_seed]
        )
    status = (
        "exact"
        if parameters_exact
        and selected_trials_exact
        and validation_auroc_max_absolute_difference == 0.0
        and maximum_prediction_difference == 0.0
        else "discrepancy"
    )
    return {
        "status": status,
        "paired_trial_count": len(raw_trials),
        "parameters_exact": parameters_exact,
        "selected_trials_exact": selected_trials_exact,
        "validation_auroc_max_absolute_difference": (
            validation_auroc_max_absolute_difference
        ),
        "test_probability_max_absolute_difference": maximum_prediction_difference,
    }


def align_candidate_to_reference(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> np.ndarray:
    candidate_position = {
        int(sample_id): position
        for position, sample_id in enumerate(candidate["sample_ids"])
    }
    try:
        order = np.asarray(
            [candidate_position[int(sample_id)] for sample_id in reference["sample_ids"]]
        )
    except KeyError as error:
        raise AssertionError("prediction sample IDs differ") from error
    if len(candidate_position) != len(reference["sample_ids"]):
        raise AssertionError("prediction sample counts differ")
    if not np.array_equal(reference["labels"], candidate["labels"][order]):
        raise AssertionError("prediction labels differ")
    if not np.array_equal(reference["subject_ids"], candidate["subject_ids"][order]):
        raise AssertionError("prediction subject IDs differ")
    return candidate["probabilities"][order]


def lightgbm_primary(
    directory: Path,
    task_dir: Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    result = load_json(directory / "result.json")
    with np.load(directory / "predictions.npz") as payload:
        probabilities = payload["test_probability"].copy()
        labels = payload["test_labels"].copy()
        sample_ids = payload["test_sample_id"].copy()
        subject_ids = payload["test_group_id"].copy()
    index = pq.read_table(
        task_dir / "audit/task_index.parquet",
        columns=["sample_id", "history_length"],
    ).to_pydict()
    length_by_sample = {
        int(sample_id): int(length)
        for sample_id, length in zip(
            index["sample_id"], index["history_length"], strict=True
        )
    }
    history_lengths = np.asarray(
        [length_by_sample[int(sample_id)] for sample_id in sample_ids], dtype=np.int16
    )
    return {
        "selected_representation": result["selected_representation"],
        "feature_count": 5113,
        "validation_auroc": result["validation_auroc"],
        "test": {
            **metrics(labels, probabilities),
            "by_history_length": stratified_metrics(
                labels, probabilities, history_lengths
            ),
            "patient_clustered_95pct_ci": clustered_bootstrap(
                labels,
                probabilities,
                subject_ids,
                bootstrap_replicates,
                bootstrap_seed,
            ),
        },
    }


def lightgbm_simple(result_paths: list[Path]) -> dict[str, Any]:
    rows = [load_json(path) for path in result_paths]
    if len(rows) < 2:
        raise ValueError("at least two simple LightGBM tuning-seed results are required")
    representations = ("current_only", "current_and_any_prior")
    selected = Counter(row["selected_representation"] for row in rows)
    by_representation = {}
    for representation in representations:
        candidates = [row["best_by_representation"][representation] for row in rows]
        history_names = candidates[0]["test"]["by_history_length"]
        by_history_length = {}
        for history_name in history_names:
            history_rows = [
                candidate["test"]["by_history_length"][history_name]
                for candidate in candidates
            ]
            by_history_length[history_name] = {
                "rows": history_rows[0]["rows"],
                "positives": history_rows[0]["positives"],
                "auroc": summarize_values(
                    [row["auroc"] for row in history_rows if row["auroc"] is not None]
                ),
                "auprc": summarize_values(
                    [row["auprc"] for row in history_rows if row["auprc"] is not None]
                ),
            }
        by_representation[representation] = {
            "validation_auroc": summarize_values(
                [candidate["validation"]["auroc"] for candidate in candidates]
            ),
            "test_auroc": summarize_values(
                [candidate["test"]["auroc"] for candidate in candidates]
            ),
            "test_auprc": summarize_values(
                [candidate["test"]["auprc"] for candidate in candidates]
            ),
            "test_by_history_length": by_history_length,
            "per_tuning_seed": [
                {
                    "tuning_seed": row["tuning_seed"],
                    "selected_trial": candidate["selected_trial"],
                    "validation": candidate["validation"],
                    "test": {
                        "auroc": candidate["test"]["auroc"],
                        "auprc": candidate["test"]["auprc"],
                    },
                }
                for row, candidate in zip(rows, candidates, strict=True)
            ],
        }
    return {
        "tuning_seeds": [row["tuning_seed"] for row in rows],
        "trials_per_representation_per_seed": rows[0]["trials_per_representation"],
        "validation_selected_representation_counts": dict(selected),
        "by_representation": by_representation,
    }


def cluster_collapse(task_dir: Path) -> dict[str, Any]:
    filenames = {
        "CONDITION": "ccscm_id2clus.json",
        "PROCEDURE": "ccsproc_id2clus.json",
        "DRUG": "atc3_id2clus.json",
    }
    mappings = {
        namespace: {
            code: int(cluster)
            for code, cluster in load_json(MAP_ROOT / filename).items()
        }
        for namespace, filename in filenames.items()
    }
    cluster_namespaces: dict[int, set[str]] = defaultdict(set)
    collision_clusters = {}
    for namespace, mapping in mappings.items():
        counts = Counter(mapping.values())
        collision_clusters[namespace] = {
            cluster for cluster, count in counts.items() if count > 1
        }
        for cluster in mapping.values():
            cluster_namespaces[cluster].add(namespace)

    totals = Counter()
    collisions = Counter()
    observed_codes: dict[str, set[str]] = defaultdict(set)
    parquet = pq.ParquetFile(task_dir / "data/events.parquet")
    for batch in parquet.iter_batches(columns=["code"]):
        for namespaced_code in batch.column(0).to_pylist():
            namespace, code = str(namespaced_code).split("//", 1)
            totals[namespace] += 1
            observed_codes[namespace].add(code)
            if mappings[namespace][code] in collision_clusters[namespace]:
                collisions[namespace] += 1

    return {
        "released_codes": {
            namespace: len(mapping) for namespace, mapping in mappings.items()
        },
        "observed_codes": {
            namespace: len(codes) for namespace, codes in observed_codes.items()
        },
        "direct_clusters": len(cluster_namespaces),
        "clusters_shared_across_namespaces": sum(
            len(namespaces) > 1 for namespaces in cluster_namespaces.values()
        ),
        "clusters_shared_across_all_three_namespaces": sum(
            len(namespaces) == 3 for namespaces in cluster_namespaces.values()
        ),
        "events_in_within_namespace_collision_clusters": {
            namespace: {
                "events": collisions[namespace],
                "total_events": totals[namespace],
                "fraction": collisions[namespace] / totals[namespace],
            }
            for namespace in mappings
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intended-dir",
        type=Path,
        default=RESULTS_ROOT / "gpu_reproduction/intended_attention_10seeds",
    )
    parser.add_argument(
        "--released-driver-dir",
        type=Path,
        default=RESULTS_ROOT / "gpu_reproduction/released_driver_10seeds",
    )
    parser.add_argument(
        "--released-space-dir",
        type=Path,
        default=RESULTS_ROOT / "gpu_reproduction/released_space_intended_seed0",
    )
    parser.add_argument(
        "--primary-lightgbm-dir",
        type=Path,
        default=RESULTS_ROOT / "subsequent_visit_mortality",
    )
    parser.add_argument("--task-dir", type=Path, default=TASK_DIR)
    parser.add_argument(
        "--simple-lightgbm-glob",
        default=str(RESULTS_ROOT / "current_prior_sensitivity_seed_*/result.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--crosswalk-audit",
        type=Path,
        default=RESULTS_ROOT / "crosswalk_audit.json",
    )
    parser.add_argument(
        "--raw-icd-dir",
        type=Path,
        default=RESULTS_ROOT / "raw_icd_sensitivity",
    )
    parser.add_argument(
        "--plain-lightgbm-dir",
        type=Path,
        default=RESULTS_ROOT / "plain_raw_lightgbm",
    )
    parser.add_argument(
        "--raw-roundtrip-audit",
        type=Path,
        default=RESULTS_ROOT / "raw_roundtrip_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "reproduction_summary.json",
    )
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("bootstrap replicates must be positive")

    simple_paths = [Path(path) for path in sorted(glob.glob(args.simple_lightgbm_glob))]
    gpu_prediction = gpu_mean_prediction(args.intended_dir)
    raw_icd_predictions = {
        representation: raw_icd_mean_prediction(args.raw_icd_dir, representation)
        for representation in (
            "mapped_ccs_atc",
            "raw_icd_prefix_atc",
            "mapped_plus_icd_prefix",
            "raw_icd_full_atc",
        )
    }
    plain_lightgbm_predictions = {
        representation: plain_lightgbm_mean_prediction(
            args.plain_lightgbm_dir, representation
        )
        for representation in PLAIN_LIGHTGBM_REPRESENTATIONS
    }
    paired_neural_vs_lightgbm = {
        representation: paired_clustered_bootstrap(
            gpu_prediction["labels"],
            gpu_prediction["probabilities"],
            align_candidate_to_reference(gpu_prediction, prediction),
            gpu_prediction["subject_ids"],
            args.bootstrap_replicates,
            args.bootstrap_seed,
        )
        for representation, prediction in raw_icd_predictions.items()
    }
    paired_neural_vs_plain_lightgbm = {
        representation: paired_clustered_bootstrap(
            gpu_prediction["labels"],
            gpu_prediction["probabilities"],
            align_candidate_to_reference(gpu_prediction, prediction),
            gpu_prediction["subject_ids"],
            args.bootstrap_replicates,
            args.bootstrap_seed,
        )
        for representation, prediction in plain_lightgbm_predictions.items()
    }
    result = {
        "paper": "graphcare",
        "task": "MIMIC-III subsequent-visit mortality",
        "published": PUBLISHED,
        "gpu_public_artifact_approximation": {
            "paper_intended_attention": gpu_ensemble(
                args.intended_dir,
                args.bootstrap_replicates,
                args.bootstrap_seed,
            ),
            "released_driver_attention_disabled": gpu_ensemble(
                args.released_driver_dir,
                args.bootstrap_replicates,
                args.bootstrap_seed,
            ),
            "released_4599_node_width_sensitivity": gpu_ensemble(
                args.released_space_dir,
                args.bootstrap_replicates,
                args.bootstrap_seed,
            ),
        },
        "lightgbm": {
            "existing_full_representation": lightgbm_primary(
                args.primary_lightgbm_dir,
                args.task_dir,
                args.bootstrap_replicates,
                args.bootstrap_seed,
            ),
            "simple_time_sensitivity": lightgbm_simple(simple_paths),
            "raw_icd_sensitivity": raw_icd_summary(args.raw_icd_dir),
            "plain_raw_code_factorial": plain_lightgbm_summary(
                args.plain_lightgbm_dir
            ),
            "raw_icd_atc_replication_check": raw_atc_replication_check(
                args.raw_icd_dir, args.plain_lightgbm_dir
            ),
        },
        "paired_neural_minus_lightgbm_mean_probability_ensembles": (
            paired_neural_vs_lightgbm
        ),
        "paired_neural_minus_plain_lightgbm_mean_probability_ensembles": (
            paired_neural_vs_plain_lightgbm
        ),
        "mapping_audit": {
            "raw_crosswalks": load_json(args.crosswalk_audit),
            "exact_raw_mimic_roundtrip": load_json(args.raw_roundtrip_audit),
            "released_cluster_collapse": cluster_collapse(args.task_dir),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
