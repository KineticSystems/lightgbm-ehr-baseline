"""Bridge to the Rust trial executor.

Python plans trials, opens tasks, and keeps selection/ensembling/metrics;
this module hands one representation's execution — featurization, native
Dataset construction, the trial loop, and test prediction — to the
featurizer binary, which drives the exact ``lib_lightgbm.so`` installed in
this environment through LightGBM's public C API. Parameter strings are
serialized here with lightgbm's own ``_param_dict_to_str`` so both executors
feed LightGBM byte-identical configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

import lightgbm as lgb
from lightgbm.basic import _ConfigAliases, _param_dict_to_str
import numpy as np

from .split_contract import SplitContract, _binary, featurizer_threads


RUST_EXECUTOR = "rust-lightgbm-v1"


def _booster_parameters(
    parameters: dict[str, Any],
    num_boost_round: int,
    dataset_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Replicate lgb.train's parameter resolution for the Booster.

    The engine strips num_iterations aliases, injects the explicit boosting
    round budget, and merges the Dataset's parameters over the trial's at
    Booster construction; without this the fitted trees are identical but the
    model text's resolved-parameter section differs between executors.
    """
    merged = dict(parameters)
    for alias in _ConfigAliases.get("num_iterations"):
        merged.pop(alias, None)
    merged["num_iterations"] = num_boost_round
    merged.update(dict(dataset_parameters))
    return merged


def lightgbm_library_path() -> Path:
    path = Path(lgb.__file__).resolve().parent / "lib" / "lib_lightgbm.so"
    if not path.is_file():
        raise RuntimeError(f"lib_lightgbm.so is missing: {path}")
    return path


@dataclass(frozen=True)
class RustTrialResult:
    number: int
    best_iteration: int | None
    validation_prediction: np.ndarray | None
    model_path: Path | None
    failure: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _launch(kind: str, config: dict[str, Any], spill_dir: Path) -> dict[str, Any]:
    config_path = spill_dir / f"{kind}_config.json"
    config_path.write_bytes(json.dumps(config, sort_keys=True, indent=1).encode("utf-8"))
    completed = subprocess.run(
        [str(_binary()), f"--{kind}-config", str(config_path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=os.environ.copy(),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Rust {kind} executor exited {completed.returncode}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    results_path = spill_dir / "results.json"
    if not results_path.is_file():
        raise RuntimeError(f"Rust {kind} executor produced no results.json")
    return json.loads(results_path.read_text())


def _read_verified_predictions(
    path: Path, expected_sha256: str, expected_rows: int
) -> np.ndarray:
    if _sha256_file(path) != expected_sha256:
        raise RuntimeError(f"spilled prediction digest differs: {path}")
    values = np.fromfile(path, dtype="<f8")
    if values.shape != (expected_rows,):
        raise RuntimeError(
            f"spilled prediction has {values.shape} values, expected {expected_rows}"
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"spilled prediction contains non-finite values: {path}")
    return values


def run_trials(
    *,
    representations: Sequence[tuple[str, SplitContract, SplitContract, dict[str, Any], Path]],
    trials: Sequence[tuple[int, int, dict[str, Any]]],
    num_boost_round: int,
    early_stopping_rounds: int,
    parallel_trials: int,
    spill_root: Path,
) -> tuple[dict[str, list[RustTrialResult]], dict[str, dict[str, Any]]]:
    """Run a study's planned trials, across every representation at once.

    ``representations`` gives each searched representation as its name, train
    and validation contracts, Dataset parameters and spill directory.
    ``trials`` is the whole study in the order the executor should hand work
    out: each entry pairs the representation it needs with its Optuna trial
    number and complete native parameter map.

    Every representation is constructed before any trial starts, so a worker
    reaching the end of one representation's trials crosses into the next
    rather than idling through the ragged tail. Ordering the list by
    representation keeps a worker on one Dataset while its trials last, so the
    overlap is confined to the boundary. Results come back grouped by
    representation name, with validation predictions in ascending-sample-ID
    order.
    """
    spill_root.mkdir(parents=True, exist_ok=True)
    threads = featurizer_threads()
    by_index = list(representations)
    for _name, _train, _validation, _parameters, rep_spill in by_index:
        rep_spill.mkdir(parents=True, exist_ok=True)
    config = {
        "lightgbm_lib": str(lightgbm_library_path()),
        "representations": [
            {
                "name": name,
                "train_stream": train_contract._config(),
                "validation_stream": validation_contract._config(),
                # Construction-only thread count: LightGBM's bin mappers are
                # computed per feature and rows land by index, so this cannot
                # change the Dataset; the trial parameters keep their own
                # pinned num_threads.
                "dataset_parameters_str": _param_dict_to_str(
                    {**dict(parameters), "num_threads": threads}
                ),
                "spill_dir": str(rep_spill),
            }
            for name, train_contract, validation_contract, parameters, rep_spill
            in by_index
        ],
        "featurizer_threads": threads,
        "trials": [
            {
                "number": number,
                "representation": index,
                "parameters_str": _param_dict_to_str(
                    _booster_parameters(
                        parameters,
                        num_boost_round,
                        by_index[index][3],
                    )
                ),
            }
            for index, number, parameters in trials
        ],
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "parallel_trials": parallel_trials,
        "spill_dir": str(spill_root),
    }
    report = _launch("train", config, spill_root)
    if report.get("executor") != RUST_EXECUTOR:
        raise RuntimeError("Rust executor reported an unexpected identity")

    reported = report.get("representations", [])
    if len(reported) != len(by_index):
        raise RuntimeError("Rust executor returned a different representation set")
    constructions: dict[str, dict[str, Any]] = {}
    for record, (name, train_contract, validation_contract, _p, _s) in zip(
        reported, by_index
    ):
        if record.get("name") != name:
            raise RuntimeError("Rust executor returned representations out of order")
        if record.get("feature_count") != train_contract.feature_count:
            raise RuntimeError(
                f"Rust executor feature width differs from the {name} contract"
            )
        if record.get("train_rows") != train_contract.row_count:
            raise RuntimeError(f"Rust executor train rows differ for {name}")
        if record.get("validation_rows") != validation_contract.row_count:
            raise RuntimeError(f"Rust executor validation rows differ for {name}")
        constructions[name] = {
            "executor": RUST_EXECUTOR,
            "feature_count": record["feature_count"],
            "train_rows": record["train_rows"],
            "validation_rows": record["validation_rows"],
            "sampled_rows": record["sampled_rows"],
            "bin_sample_staging_bytes": record["bin_sample_staging_bytes"],
            "construction_seconds": record.get("construction_seconds"),
            "featurizer_threads": report.get("featurizer_threads"),
        }

    outcomes = report.get("trials", [])
    if {record["number"] for record in outcomes} != {
        number for _index, number, _parameters in trials
    }:
        raise RuntimeError("Rust executor returned a different trial set")
    results: dict[str, list[RustTrialResult]] = {name: [] for name, *_ in by_index}
    for record in outcomes:
        if "representation" in record:
            index = int(record["representation"])
        else:
            raise RuntimeError(
                f"Rust executor returned trial {record.get('number')} without a "
                "representation; every result, including a recoverable failure, "
                "must name the representation it belongs to"
            )
        name, _train, validation_contract, _p, _s = by_index[index]
        if "failure" in record:
            results[name].append(
                RustTrialResult(
                    number=int(record["number"]),
                    best_iteration=None,
                    validation_prediction=None,
                    model_path=None,
                    failure=str(record["failure"]),
                )
            )
            continue
        prediction = _read_verified_predictions(
            Path(record["prediction_path"]),
            record["prediction_sha256"],
            validation_contract.row_count,
        )
        model_path = Path(record["model_path"])
        if _sha256_file(model_path) != record["model_sha256"]:
            raise RuntimeError(f"spilled model digest differs: {model_path}")
        results[name].append(
            RustTrialResult(
                number=int(record["number"]),
                best_iteration=int(record["best_iteration"]),
                validation_prediction=prediction,
                model_path=model_path,
            )
        )
    for group in results.values():
        group.sort(key=lambda result: result.number)
    return results, constructions


def run_representation_trials(
    *,
    train_contract: SplitContract,
    validation_contract: SplitContract,
    trials: Sequence[tuple[int, dict[str, Any]]],
    dataset_parameters: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    parallel_trials: int,
    spill_dir: Path,
) -> tuple[list[RustTrialResult], dict[str, Any]]:
    """Run one representation's planned trials. Thin wrapper over run_trials."""
    name = spill_dir.name
    results, constructions = run_trials(
        representations=[
            (name, train_contract, validation_contract, dataset_parameters, spill_dir)
        ],
        trials=[(0, number, parameters) for number, parameters in trials],
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        parallel_trials=parallel_trials,
        spill_root=spill_dir,
    )
    return results[name], constructions[name]


def predict_members(
    *,
    contract: SplitContract,
    model_paths: Sequence[Path],
    spill_dir: Path,
) -> list[np.ndarray]:
    """Predict the split for each member model, ascending-sample-ID order."""
    spill_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "lightgbm_lib": str(lightgbm_library_path()),
        "stream": contract._config(),
        "model_paths": [str(path) for path in model_paths],
        "spill_dir": str(spill_dir),
        "featurizer_threads": featurizer_threads(),
    }
    report = _launch("predict", config, spill_dir)
    if report.get("rows") != contract.row_count:
        raise RuntimeError("Rust predict rows differ from the task index")
    members = report.get("members", [])
    if len(members) != len(model_paths):
        raise RuntimeError("Rust predict returned a different member count")
    return [
        _read_verified_predictions(
            Path(record["prediction_path"]),
            record["prediction_sha256"],
            contract.row_count,
        )
        for record in sorted(members, key=lambda record: record["rank"])
    ]
