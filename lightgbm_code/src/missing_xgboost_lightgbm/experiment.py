"""Run the shared native-sparse Optuna/LightGBM binary experiment."""

from __future__ import annotations

import argparse
import ctypes
import shutil
from dataclasses import dataclass
import gc
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time
from typing import Any

import lightgbm
import numpy as np
import optuna
import pyarrow
import sklearn

from .common_format import open_task, sha256_file, sha256_strings, write_json
from .modeling import (
    HISTOGRAM_POOL_PER_WORKER_MB,
    ensemble_probability,
    recoverable_lightgbm_trial_failure as _recoverable_lightgbm_trial_failure,
    native_fixed_parameters,
    native_trial_parameters,
    patient_clustered_bootstrap,
    point_metrics,
    suggest_parameters,
)
from .split_contract import SplitContract
from .python_executor import (
    PYTHON_EXECUTOR,
    predict_members as python_predict_members,
    run_representation_trials as python_run_representation_trials,
)
from .rust_trainer import (
    RUST_EXECUTOR,
    predict_members,
    run_trials,
)


PROTOCOL_VERSION = "lightgbm-binary-v10-ensemble"
DEFAULT_TRIALS = 50
DEFAULT_TUNING_SEED = 0
MODEL_SEED = 0
BOOTSTRAP_REPLICATES = 2_000
# Two LightGBM threads per trial, with trials running in parallel worker
# threads over one shared native Dataset per representation.
# Trials are fully planned before any training, so the sampler sequence and
# every fitted model are independent of worker count and completion order.
DEFAULT_THREADS = 2
DEFAULT_PARALLEL_TRIALS = 4
# The shipped model is the uniform logit-averaged ensemble of the top-5
# validation trials within the selected representation (Caruana et al., ICML
# 2004 sorted initialization; auto-sklearn). The representation itself is
# chosen by the single best validation-AUROC trial.
ENSEMBLE_SIZE = 5
MAX_BOOST_ROUNDS = 5_000
EARLY_STOPPING_ROUNDS = 100
MAX_BIN = 255
DEFAULT_BIN_SAMPLE_ROWS = 200_000
MAX_STREAM_CHUNK_BYTES = 64 << 20
# Execution memory guard, not a protocol parameter: the largest connected
# row-group component the Rust featurizer may hold at once. A 24-byte Event
# plus grouping overhead runs 40-50 bytes per row, so this bound costs about
# 2.4-3 GiB. Raise it if a cohort's largest component is rejected.
MAX_COMPONENT_EVENT_ROWS = 60_000_000
RUST_RECORD_BATCH_ROWS = 65_536
# Every task offers `summary`; a task whose temporal profile admits one also
# offers a three-segment representation. See MedsTask.representations.


def _release_native_memory() -> None:
    """Release completed candidate/trial native heaps before the next one."""
    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
    except AttributeError:
        return
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


@dataclass(frozen=True)
class _PlannedTrial:
    trial: optuna.trial.Trial
    representation: str
    search_parameters: dict[str, Any]
    model_parameters: dict[str, Any]


@dataclass(frozen=True)
class EnsembleMember:
    """One retained trial: the argmax member is index 0 of the ensemble."""

    trial_number: int
    parameters: dict[str, Any]
    best_iteration: int
    validation_auroc: float
    model_string: str
    validation_prediction: np.ndarray


@dataclass(frozen=True)
class TunedTask:
    representation: str
    members: tuple[EnsembleMember, ...]
    validation_auroc: float
    validation_probability: np.ndarray
    feature_names: tuple[str, ...]
    trials: tuple[dict[str, Any], ...]
    construction: dict[str, Any]


def _sha256_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "optuna": optuna.__version__,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
    }


def _audit_inventory(source) -> dict[str, dict[str, str]]:
    """Record the exact-source cohort and batch checks, and say what is missing.

    These checks are what establish that a task directory holds the paper's own
    cohort rather than an approximation of it, so a fit without them is not
    evidence about the paper. It is still a legitimate thing to run -- on your
    own MEDS data, or on a cohort you are still iterating on -- so a missing or
    inexact report is reported here and recorded in ``result.json`` rather than
    stopping the run. Set ``MXGB_REQUIRE_AUDITS=1`` to refuse instead, which is
    what the campaign protocol does.
    """
    audit_dir = source.root / "audit"
    paths = {
        "batch": audit_dir / "full_batch_recreation.json",
        "cohort": audit_dir / "cohort_reconstruction.json",
    }
    strict = os.environ.get("MXGB_REQUIRE_AUDITS", "").strip() not in ("", "0")
    result: dict[str, dict[str, str]] = {}
    unverified: list[str] = []
    for name, path in paths.items():
        command = "check_batch.sh" if name == "batch" else "check_cohort.sh"
        if not path.is_file():
            if strict:
                raise ValueError(f"{command} must pass before shared model training")
            unverified.append(f"{name}: no report; {command} has not been run")
            result[name] = {"status": "missing", "command": command}
            continue
        report = json.loads(path.read_text())
        status = report.get("status")
        exact = (
            status == "exact"
            or (name == "batch" and status == "exact_structured_source_boundary")
            or (
                isinstance(status, str)
                and status.startswith(
                    ("exact_source_with_", "exact_at_released_")
                )
            )
        )
        if not exact:
            if strict:
                raise ValueError(f"{name} audit is not exact-source: {path}")
            unverified.append(f"{name}: report is not exact-source ({status!r})")
        result[name] = {
            "file": str(path.relative_to(source.root)),
            "sha256": sha256_file(path),
            "status": str(status),
            "exact_source": "true" if exact else "false",
        }
    if unverified:
        print(
            "warning: this cohort has not been verified against its paper:",
            *(f"  {line}" for line in unverified),
            "  Numbers from it describe the data as exported, not the paper's"
            " reported cohort.",
            "  result.json records this under export_audits.",
            sep="\n",
            file=sys.stderr,
        )
    return result


def _log(message: str, start: float) -> None:
    print(f"[{time.monotonic() - start:8.1f}s] {message}", flush=True)


def _validate_partitions(
    task_id: str, partitions: dict[str, dict[str, np.ndarray]]
) -> None:
    for split, data in partitions.items():
        if np.unique(data["y"]).size != 2:
            raise ValueError(
                f"task {task_id!r} {split} split does not contain both classes"
            )


def _cross_split_group_overlap(
    partitions: dict[str, dict[str, np.ndarray]],
) -> dict[str, int]:
    pairs = [
        (left, right)
        for index, left in enumerate(partitions)
        for right in tuple(partitions)[index + 1 :]
    ]
    return {
        f"{left}_{right}": int(
            np.intersect1d(
                partitions[left]["group_id"], partitions[right]["group_id"]
            ).size
        )
        for left, right in pairs
    }


def _configured_threads() -> int:
    raw = os.environ.get("MXGB_THREADS", str(DEFAULT_THREADS))
    try:
        threads = int(raw)
    except ValueError as error:
        raise ValueError(f"MXGB_THREADS must be an integer, got {raw!r}") from error
    if threads < 1:
        raise ValueError("MXGB_THREADS must be positive")
    return threads


def _configured_executor() -> str:
    executor = os.environ.get("MXGB_EXECUTOR", "rust")
    if executor not in ("rust", "python"):
        raise ValueError(f"MXGB_EXECUTOR must be 'rust' or 'python', got {executor!r}")
    return executor


def _configured_parallel_trials() -> int:
    raw = os.environ.get("MXGB_PARALLEL_TRIALS", str(DEFAULT_PARALLEL_TRIALS))
    try:
        workers = int(raw)
    except ValueError as error:
        raise ValueError(
            f"MXGB_PARALLEL_TRIALS must be an integer, got {raw!r}"
        ) from error
    if workers < 1:
        raise ValueError("MXGB_PARALLEL_TRIALS must be positive")
    return workers


def _retain_member(
    members: list[EnsembleMember], candidate: EnsembleMember
) -> None:
    """Keep the top ENSEMBLE_SIZE by validation AUROC; earliest trial wins ties.

    The winning-trial selection in tune_task applies the same rule, so the two
    cannot disagree regardless of the order trials complete in.
    """
    members.append(candidate)
    members.sort(key=lambda item: (-item.validation_auroc, item.trial_number))
    while len(members) > ENSEMBLE_SIZE:
        del members[-1]


def _stream_batch_rows(feature_count: int) -> int:
    return max(1, (MAX_STREAM_CHUNK_BYTES - 8) // (16 + 4 * feature_count))


def _bin_sample_rows(row_count: int, feature_count: int) -> int:
    """Use LightGBM's documented bin-sample default, bounded by the split.

    Feature width does not enter: the sample is collected as columns rather
    than a dense matrix, so a wide representation is sampled as deeply as a
    narrow one. LightGBM's documentation warns that small samples give
    "unexpected errors and poor accuracy", which bites hardest on rare sparse
    features -- they land in a single bin and become unsplittable.
    """
    del feature_count
    return min(row_count, DEFAULT_BIN_SAMPLE_ROWS)


def _dataset_parameters(bin_sample_rows: int) -> dict[str, Any]:
    return {
        "max_bin": MAX_BIN,
        "bin_construct_sample_cnt": bin_sample_rows,
        "data_random_seed": MODEL_SEED,
        "feature_pre_filter": False,
        "use_missing": True,
        # See fixed_parameters for why an unshown entry must read as missing.
        "zero_as_missing": True,
        "is_enable_sparse": True,
        "enable_bundle": True,
        "verbosity": -1,
    }


def _contract(
    source,
    split: str,
    representation: str,
):
    return SplitContract(
        events_path=source._path("events"),
        index_path=source._path("task_index"),
        split=split,
        representation=representation,
        codes=source._structured_codes(),
        occurrence_codes=source.occurrence_codes,
        ignored_codes=source.note_codes,
        observation_window_hours=source.observation_window_hours,
        halves_split_hours=source.halves_split_hours,
        timed=source.timed,
        max_component_rows=MAX_COMPONENT_EVENT_ROWS,
        max_chunk_bytes=MAX_STREAM_CHUNK_BYTES,
        record_batch_rows=RUST_RECORD_BATCH_ROWS,
    )


def _assert_contract_partition(contract, partition: dict[str, np.ndarray]) -> None:
    if contract.row_count != len(partition["y"]):
        raise ValueError("Rust stream and task partition have different row counts")
    if not np.array_equal(
        contract.restore_prediction_order(contract.labels),
        partition["y"].astype(np.float32),
    ):
        raise ValueError("Rust stream labels disagree with the task partition")
    if not np.array_equal(
        contract.restore_prediction_order(contract.sample_ids),
        partition["sample_id"],
    ):
        raise ValueError("Rust stream sample IDs disagree with the task partition")


def _plan_trials(
    *,
    study: optuna.study.Study,
    representations: tuple[str, ...],
    trials: int,
    negative_to_positive_ratio: float,
    threads: int,
) -> tuple[_PlannedTrial, ...]:
    # Enqueue every eligible representation once; remaining categorical choices
    # are sampled normally by RandomSampler.
    if trials >= len(representations):
        for representation in representations:
            study.enqueue_trial({"representation": representation})
    planned = []
    for _ in range(trials):
        trial = study.ask()
        representation, search = suggest_parameters(
            trial,
            representations,
            negative_to_positive_ratio,
        )
        model_parameters = {
            **native_fixed_parameters(MODEL_SEED, threads),
            **native_trial_parameters(search),
        }
        planned.append(
            _PlannedTrial(
                trial=trial,
                representation=representation,
                search_parameters=search,
                model_parameters=model_parameters,
            )
        )
    return tuple(planned)


def tune_task(
    source,
    task_id: str,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    output_dir: Path,
    trials: int,
    tuning_seed: int,
    threads: int,
    start: float,
    parallel_trials: int | None = None,
) -> TunedTask:
    """Search with native Dataset reuse and retain the top-5 ensemble."""
    parallel_trials = (
        _configured_parallel_trials() if parallel_trials is None else parallel_trials
    )
    if parallel_trials < 1:
        raise ValueError("parallel_trials must be positive")
    executor_kind = _configured_executor()
    positive_weight = float(train["y"].sum())
    total_weight = float(len(train["y"]))
    ratio = (total_weight - positive_weight) / positive_weight
    representations = source.representations
    _log(
        f"{task_id}: train={len(train['y']):,}, validation={len(validation['y']):,}, "
        f"Optuna trials={trials} over {'/'.join(representations)}, "
        f"{parallel_trials} parallel x {threads} threads",
        start,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=tuning_seed),
    )
    planned = _plan_trials(
        study=study,
        representations=representations,
        trials=trials,
        negative_to_positive_ratio=ratio,
        threads=threads,
    )
    retained: dict[str, list[EnsembleMember]] = {}
    constructions: dict[str, dict[str, Any]] = {}
    validation_contracts: dict[str, Any] = {}
    feature_names_by_representation: dict[str, tuple[str, ...]] = {}

    # Plan every representation up front -- contracts, trial order, sizing --
    # so the Rust executor receives the whole study in one call. A worker that
    # exhausts one representation's trials then crosses into the next instead
    # of idling through its ragged tail, which is bounded below by the single
    # slowest trial and so cannot be shortened by adding workers.
    prepared: list[dict[str, Any]] = []
    for representation in representations:
        # Longest-processing-time-first: a low learning rate needs more boosting
        # rounds before early stopping, so those trials are the slowest. Workers
        # claim from this list in order, so starting with the slowest keeps the
        # ragged tail short. Ordering by representation first keeps a worker on
        # one Dataset while its trials last, so overlap between representations
        # is confined to the boundary. Trial parameters and identities are
        # untouched, only the order they are handed out in.
        representation_trials = sorted(
            (item for item in planned if item.representation == representation),
            key=lambda item: (
                item.search_parameters["learning_rate"],
                item.trial.number,
            ),
        )
        if not representation_trials:
            continue
        train_contract = _contract(source, "train", representation)
        validation_contract = _contract(source, "validation", representation)
        _assert_contract_partition(train_contract, train)
        _assert_contract_partition(validation_contract, validation)
        feature_count = train_contract.feature_count
        if validation_contract.feature_names != train_contract.feature_names:
            raise ValueError("train and validation feature names differ")
        prepared.append(
            {
                "representation": representation,
                "trials": representation_trials,
                "train_contract": train_contract,
                "validation_contract": validation_contract,
                "feature_count": feature_count,
                "sample_rows": _bin_sample_rows(
                    train_contract.row_count, feature_count
                ),
            }
        )

    # The same principle one level up: the widest representation holds the
    # slowest trials, so running it last strands the tail with nothing cheaper
    # to fall back on. First means the tail is narrow-representation trials,
    # short enough to keep finished workers busy.
    prepared.sort(key=lambda entry: -entry["feature_count"])

    rust_results: dict[str, list[Any]] = {}
    if executor_kind == "rust":
        for entry in prepared:
            _log(
                f"{task_id}: Rust executor building {entry['representation']} "
                f"({entry['feature_count']:,} features, "
                f"{entry['sample_rows']:,} bin-sample rows, "
                f"{len(entry['trials'])} trials)",
                start,
            )
        rust_results, rust_constructions = run_trials(
            representations=[
                (
                    entry["representation"],
                    entry["train_contract"],
                    entry["validation_contract"],
                    _dataset_parameters(entry["sample_rows"]),
                    output_dir / "rust_spill" / entry["representation"],
                )
                for entry in prepared
            ],
            trials=[
                (
                    index,
                    item.trial.number,
                    {
                        **item.model_parameters,
                        "histogram_pool_size": HISTOGRAM_POOL_PER_WORKER_MB,
                    },
                )
                for index, entry in enumerate(prepared)
                for item in entry["trials"]
            ],
            num_boost_round=MAX_BOOST_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            parallel_trials=parallel_trials,
            spill_root=output_dir / "rust_spill",
        )
        for entry in prepared:
            construction = rust_constructions[entry["representation"]]
            constructions[entry["representation"]] = construction
            _log(
                f"{task_id}: {entry['representation']} constructed in "
                f"{construction.get('construction_seconds') or float('nan'):.0f}s "
                f"({construction.get('featurizer_threads') or 'all'} "
                "featurizer threads)",
                start,
            )

    for entry in prepared:
        representation = entry["representation"]
        representation_trials = entry["trials"]
        train_contract = entry["train_contract"]
        validation_contract = entry["validation_contract"]
        feature_count = entry["feature_count"]
        retained[representation] = []
        validation_contracts[representation] = validation_contract
        feature_names_by_representation[representation] = (
            train_contract.feature_names
        )
        if executor_kind == "rust":
            results = rust_results[representation]
        else:
            _log(
                f"{task_id}: Python reference executor building {representation} "
                f"({feature_count:,} features, "
                f"{len(representation_trials)} trials)",
                start,
            )
            results, python_construction = python_run_representation_trials(
                source=source,
                train_contract=train_contract,
                validation_contract=validation_contract,
                representation=representation,
                trials=[
                    (
                        item.trial.number,
                        {
                            **item.model_parameters,
                            "histogram_pool_size": HISTOGRAM_POOL_PER_WORKER_MB,
                        },
                    )
                    for item in representation_trials
                ],
                dataset_parameters=_dataset_parameters(entry["sample_rows"]),
                num_boost_round=MAX_BOOST_ROUNDS,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )
            constructions[representation] = python_construction

        by_number = {item.trial.number: item for item in representation_trials}
        for result in results:
            item = by_number[result.number]
            item.trial.set_user_attr("feature_count", feature_count)
            item.trial.set_user_attr("model_parameters", item.search_parameters)
            if result.failure is not None:
                failure = "lightgbm_sampled_split_empty_child"
                item.trial.set_user_attr("failure", failure)
                study.tell(item.trial, state=optuna.trial.TrialState.FAIL)
                _log(
                    f"{task_id} trial {item.trial.number + 1:03d}/{trials}: "
                    f"representation={representation}, failed={failure}",
                    start,
                )
                continue
            auroc = point_metrics(
                validation_contract.labels, result.validation_prediction
            )["auroc"]
            item.trial.set_user_attr("best_iteration", result.best_iteration)
            study.tell(item.trial, auroc)
            _log(
                f"{task_id} trial {item.trial.number + 1:03d}/{trials}: "
                f"representation={representation}, AUROC={auroc:.4f}, "
                f"rounds={result.best_iteration}",
                start,
            )
            # lightgbm's model_to_string appends this wrapper metadata line
            # after the C string. The Rust executor spills only the C string,
            # so append it there to keep both executors byte-identical.
            model_string = (
                result.model_path.read_text() + "\npandas_categorical:null\n"
                if executor_kind == "rust"
                else result.model_string
            )
            _retain_member(
                retained[representation],
                EnsembleMember(
                    trial_number=result.number,
                    parameters=item.search_parameters,
                    best_iteration=result.best_iteration,
                    validation_auroc=auroc,
                    model_string=model_string,
                    validation_prediction=result.validation_prediction.astype(
                        np.float64
                    ),
                ),
            )
        # The spilled models have been read, so the train contract is dead
        # once its trials are absorbed.
        entry["train_contract"] = None
        del train_contract
        _release_native_memory()

    if executor_kind == "rust":
        shutil.rmtree(output_dir / "rust_spill", ignore_errors=True)

    records = tuple(
        {
            "number": trial.number,
            "state": trial.state.name,
            "validation_auroc": trial.value,
            "representation": trial.params.get("representation"),
            "parameters": trial.user_attrs.get("model_parameters"),
            "best_iteration": trial.user_attrs.get("best_iteration"),
            "feature_count": trial.user_attrs.get("feature_count"),
            "failure": trial.user_attrs.get("failure"),
        }
        for trial in study.trials
    )
    write_json(output_dir / "trials.json", records)
    if not any(retained.values()):
        raise RuntimeError("every planned LightGBM trial failed")
    # Highest validation AUROC, earliest trial number breaking ties. Optuna's
    # study.best_trial would resolve a tie by whichever trial finished first,
    # which is a scheduling accident here because trials are handed out widest
    # representation first. This is also the rule _retain_member applies, so
    # selection and ensemble retention cannot disagree.
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]
    if not completed:
        raise RuntimeError("every planned LightGBM trial failed")
    winning_trial = min(completed, key=lambda trial: (-trial.value, trial.number))
    winning_representation = str(winning_trial.params["representation"])
    members = tuple(retained[winning_representation])
    if not members or members[0].trial_number != winning_trial.number:
        raise RuntimeError("retained members and Optuna best trial disagree")
    validation_contract = validation_contracts[winning_representation]
    ensemble_validation = ensemble_probability(
        [member.validation_prediction for member in members]
    )
    ensemble_validation_auroc = point_metrics(
        validation_contract.labels, ensemble_validation
    )["auroc"]
    return TunedTask(
        representation=winning_representation,
        members=members,
        validation_auroc=ensemble_validation_auroc,
        validation_probability=validation_contract.restore_prediction_order(
            ensemble_validation
        ),
        feature_names=feature_names_by_representation[winning_representation],
        trials=records,
        construction=constructions[winning_representation],
    )


def _representation_summary(tuned: TunedTask) -> dict[str, dict[str, Any]]:
    complete = [
        record
        for record in tuned.trials
        if record["representation"] is not None
        and record["validation_auroc"] is not None
    ]
    names = sorted({str(record["representation"]) for record in complete})
    return {
        name: {
            "trials": sum(record["representation"] == name for record in complete),
            "best_validation_auroc": max(
                record["validation_auroc"]
                for record in complete
                if record["representation"] == name
            ),
        }
        for name in names
    }


def evaluate_task(
    source,
    export_audits: dict[str, dict[str, str]],
    partitions: dict[str, dict[str, np.ndarray]],
    tuned: TunedTask,
    output_dir: Path,
    trials: int,
    tuning_seed: int,
    threads: int,
    bootstrap_replicates: int,
    start: float,
) -> dict[str, Any]:
    """Evaluate the selected validation-trained ensemble once on sealed test."""
    task_id = source.task_id
    train = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]
    test_contract = _contract(source, "test", tuned.representation)
    _assert_contract_partition(test_contract, test)
    if test_contract.feature_names != tuned.feature_names:
        raise ValueError("test feature names differ from the selected candidate")
    model_files = []
    for rank, member in enumerate(tuned.members):
        name = f"model_member_{rank}.txt"
        (output_dir / name).write_text(member.model_string)
        model_files.append(name)
    executor_kind = _configured_executor()
    if executor_kind == "rust":
        spill_dir = output_dir / "rust_spill" / "test"
        member_test_predictions = predict_members(
            contract=test_contract,
            model_paths=[output_dir / name for name in model_files],
            spill_dir=spill_dir,
        )
        shutil.rmtree(spill_dir)
    else:
        member_test_predictions = python_predict_members(
            source=source,
            contract=test_contract,
            representation=tuned.representation,
            model_strings=[member.model_string for member in tuned.members],
        )
    test_probability = test_contract.restore_prediction_order(
        ensemble_probability(member_test_predictions)
    )
    test_metrics = point_metrics(test["y"], test_probability)
    confidence_intervals = patient_clustered_bootstrap(
        test["y"],
        test_probability,
        test["group_id"],
        replicates=bootstrap_replicates,
        seed=0,
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        validation_probability=tuned.validation_probability,
        validation_labels=validation["y"],
        validation_group_id=validation["group_id"],
        validation_sample_id=validation["sample_id"],
        test_probability=test_probability,
        test_labels=test["y"],
        test_group_id=test["group_id"],
        test_sample_id=test["sample_id"],
    )
    task = source.task
    manifest = {
        "result_schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "paper": source.manifest["paper"],
        "task_id": task_id,
        "task_name": task["name"],
        "source_manifest_sha256": sha256_file(source.root / "manifest.json"),
        "export_audits": export_audits,
        "primary_selection_metric": "AUROC",
        "selection": {
            "sampler": "optuna.samplers.RandomSampler",
            "sampler_seed": tuning_seed,
            "trials": trials,
            "joint_search": False,
            "selected_trial": tuned.members[0].trial_number,
            "selected_representation": tuned.representation,
            "selected_parameters": tuned.members[0].parameters,
            "selected_best_iteration": tuned.members[0].best_iteration,
            "selected_validation_auroc": tuned.members[0].validation_auroc,
            "by_representation": _representation_summary(tuned),
            "ensemble": {
                "size": len(tuned.members),
                "weighting": "uniform_logit_average",
                "validation_auroc": tuned.validation_auroc,
                "members": [
                    {
                        "rank": rank,
                        "trial": member.trial_number,
                        "validation_auroc": member.validation_auroc,
                        "best_iteration": member.best_iteration,
                        "parameters": member.parameters,
                    }
                    for rank, member in enumerate(tuned.members)
                ],
            },
        },
        "feature_count": len(tuned.feature_names),
        "feature_name_sha256": sha256_strings(tuned.feature_names),
        "temporal_profile": source.temporal_profile,
        "searched_representations": list(source.representations),
        "halves_split_hours": source.halves_split_hours,
        "execution": {
            "executor": (
                RUST_EXECUTOR if executor_kind == "rust" else PYTHON_EXECUTOR
            ),
            "exact_full_feature_representation": True,
            "native_missing_values": True,
            "bin_sample_rows_requested": DEFAULT_BIN_SAMPLE_ROWS,
            "zero_as_missing_with_vacating_offset": True,
            "maximum_component_event_rows": MAX_COMPONENT_EVENT_ROWS,
            "construction": tuned.construction,
        },
        "split": {
            split: {
                "rows": len(data["y"]),
                "groups": len(np.unique(data["group_id"])),
                "positive_rows": int(data["y"].sum()),
                "group_id_sha256": _sha256_array(
                    np.sort(np.unique(data["group_id"]))
                ),
            }
            for split, data in partitions.items()
        },
        "cross_split_group_overlap": _cross_split_group_overlap(partitions),
        "training": {
            "fit_split": "train",
            "early_stopping_split": "validation",
            "training_rows": len(train["y"]),
            "model_seed": MODEL_SEED,
            "member_iterations": [
                member.best_iteration for member in tuned.members
            ],
            "validation_auroc": tuned.validation_auroc,
            "trial_models_reused_without_final_refit": True,
        },
        "test": test_metrics,
        "patient_clustered_95pct_ci": confidence_intervals,
        "published_auroc": source.published_auroc,
        "contains_restricted_derivatives": True,
        "restricted_local_files": ["predictions.npz", *model_files],
        "versions": _versions(),
    }
    write_json(output_dir / "manifest.json", manifest)
    _log(f"{task_id}: test AUROC={test_metrics['auroc']:.4f}", start)
    return manifest


def run(
    task_dir: Path,
    output_dir: Path,
    *,
    trials: int = DEFAULT_TRIALS,
    tuning_seed: int = DEFAULT_TUNING_SEED,
    threads: int | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Run the fixed protocol; optional overrides exist only for tests."""
    if trials < 1 or bootstrap_replicates < 1:
        raise ValueError("trials and bootstrap_replicates must be positive")
    threads = _configured_threads() if threads is None else threads
    if threads < 1:
        raise ValueError("threads must be positive")
    start = time.monotonic()
    source = open_task(task_dir)
    export_audits = _audit_inventory(source)
    task_id = source.task_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    partitions = {
        split: source.load(split) for split in ("train", "validation")
    }
    _validate_partitions(task_id, partitions)
    tuned = tune_task(
        source,
        task_id,
        partitions["train"],
        partitions["validation"],
        output_dir,
        trials=trials,
        tuning_seed=tuning_seed,
        threads=threads,
        start=start,
    )
    _log(
        f"{task_id}: selected {tuned.representation} at validation AUROC "
        f"{tuned.validation_auroc:.4f}",
        start,
    )
    _log(f"{task_id}: selection complete; loading sealed test split", start)
    partitions["test"] = source.load("test")
    _validate_partitions(task_id, partitions)
    result = evaluate_task(
        source,
        export_audits,
        partitions,
        tuned,
        output_dir,
        trials=trials,
        tuning_seed=tuning_seed,
        threads=threads,
        bootstrap_replicates=bootstrap_replicates,
        start=start,
    )
    disclosure_safe = {
        "task_id": result["task_id"],
        "task_name": result["task_name"],
        "selected_representation": result["selection"]["selected_representation"],
        "validation_auroc": result["selection"]["selected_validation_auroc"],
        "ensemble_size": result["selection"]["ensemble"]["size"],
        "ensemble_validation_auroc": result["selection"]["ensemble"][
            "validation_auroc"
        ],
        "auroc": result["test"]["auroc"],
        "auroc_ci_lower": result["patient_clustered_95pct_ci"]["auroc"]["lower"],
        "auroc_ci_upper": result["patient_clustered_95pct_ci"]["auroc"]["upper"],
        "published_auroc": result["published_auroc"],
    }
    write_json(output_dir / "result.json", disclosure_safe)
    del partitions, tuned
    _release_native_memory()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.task_dir, args.output_dir)


if __name__ == "__main__":
    main()
