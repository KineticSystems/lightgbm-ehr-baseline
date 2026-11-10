"""Fully-Python reference executor: trial list to results and models.

This is the deliberately simple counterpart of the Rust executor
(`rust_trainer.py`): events are featurized by the literal Python oracle in
``reference_features``, matrices live in ordinary NumPy arrays, Datasets are
plain in-memory ``lgb.Dataset`` objects, and trials run sequentially through
``lgb.train``. No streaming, no private LightGBM APIs, no staging machinery.
It is slow and memory-hungry by design — obviousness is the point — so it
serves fixtures and small tasks as the independent verification and fallback
path; ultra-wide tasks are Rust-only. Cross-path parity is float-tolerance
equality (the oracle's arithmetic is not bit-pinned to Rust's); the parity
suite additionally proves the training half bit-identical on shared
matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .modeling import recoverable_lightgbm_trial_failure
from .reference_features import build_reference_event_representations
from .split_contract import SplitContract, lightgbm_feature_names


PYTHON_EXECUTOR = "python-reference-v1"
_ZERO_VACATING_OFFSET = 0.1
_OFFSET_EXEMPT_STATISTICS = frozenset(
    {"count", "event_count", "code_count", "distinct_time_count", "window_hours"}
)


@dataclass(frozen=True)
class PythonTrialResult:
    number: int
    best_iteration: int | None
    validation_prediction: np.ndarray | None
    model_string: str | None
    failure: str | None = None


def _load_split_events(source, split: str, sample_ids: np.ndarray) -> dict[str, Any]:
    """Read one split's events into the oracle's literal input format."""
    index_columns = ["sample_id", "observation_start", "prediction_time", "split"]
    index = pq.read_table(source._path("task_index"), columns=index_columns)
    index = index.filter(pc.equal(index["split"], split))
    start_by_sample = {
        sample: np.datetime64(start, "us")
        for sample, start in zip(
            index["sample_id"].to_pylist(), index["observation_start"].to_pylist()
        )
    }
    window_by_sample = {
        sample: (np.datetime64(stop, "us") - np.datetime64(start, "us"))
        / np.timedelta64(1, "h")
        for sample, start, stop in zip(
            index["sample_id"].to_pylist(),
            index["observation_start"].to_pylist(),
            index["prediction_time"].to_pylist(),
        )
    }
    event_columns = ["sample_id", "time", "code", "numeric_value", "source_row", "split"]
    events = pq.read_table(source._path("events"), columns=event_columns)
    events = events.filter(pc.equal(events["split"], split))
    ignored = set(source.note_codes)
    rows = {
        "sample_id": [],
        "time_hours": [],
        "code": [],
        "value": [],
        "source_row": [],
    }
    for sample, time, code, value, source_row in zip(
        events["sample_id"].to_pylist(),
        events["time"].to_pylist(),
        events["code"].to_pylist(),
        events["numeric_value"].to_pylist(),
        events["source_row"].to_pylist(),
    ):
        if code in ignored:
            continue
        rows["sample_id"].append(sample)
        rows["time_hours"].append(
            (np.datetime64(time, "us") - start_by_sample[sample])
            / np.timedelta64(1, "h")
        )
        rows["code"].append(code)
        rows["value"].append(np.nan if value is None else float(value))
        rows["source_row"].append(source_row)
    loaded_events = {name: np.asarray(values) for name, values in rows.items()}
    return {
        "events": loaded_events,
        "sample_window_hours": np.asarray(
            [window_by_sample[int(sample)] for sample in sample_ids],
            dtype=np.float64,
        ),
    }


def _representation_matrix(
    source, contract: SplitContract, representation: str
) -> np.ndarray:
    """One split's oracle feature matrix, rows in ascending-sample-ID order."""
    loaded = _load_split_events(source, contract.split, contract.sample_ids)
    built = build_reference_event_representations(
        loaded["events"],
        contract.sample_ids,
        source._structured_codes(),
        source.observation_window_hours,
        occurrence_codes=source.occurrence_codes,
        sample_window_hours=loaded["sample_window_hours"],
        temporal_profile=source.temporal_profile,
        halves_split_hours=source.halves_split_hours,
    )
    matrix = built[representation]
    if tuple(matrix.names) != contract.feature_names:
        raise RuntimeError("oracle feature names differ from the shared contract")
    return matrix.values


def apply_zero_vacating_offset(
    values: np.ndarray, offset_columns: np.ndarray
) -> np.ndarray:
    """The executor-side missing-value encoding, identical to the contract."""
    values = np.array(values, copy=True)
    if offset_columns.any():
        block = values[:, offset_columns]
        np.add(
            block,
            values.dtype.type(_ZERO_VACATING_OFFSET),
            out=block,
            where=block >= 0.0,
        )
        np.copyto(block, 0.0, where=np.isnan(block))
        values[:, offset_columns] = block
    return values




def run_representation_trials(
    *,
    source,
    train_contract: SplitContract,
    validation_contract: SplitContract,
    representation: str,
    trials: Sequence[tuple[int, dict[str, Any]]],
    dataset_parameters: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[list[PythonTrialResult], dict[str, Any]]:
    """Run one representation's planned trials fully in Python, sequentially."""
    train_x = apply_zero_vacating_offset(
        _representation_matrix(source, train_contract, representation),
        train_contract.offset_columns,
    )
    validation_x = apply_zero_vacating_offset(
        _representation_matrix(source, validation_contract, representation),
        validation_contract.offset_columns,
    )
    train_dataset = lgb.Dataset(
        train_x,
        label=train_contract.labels,
        params=dict(dataset_parameters),
        feature_name=lightgbm_feature_names(list(train_contract.feature_names)),
        free_raw_data=False,
    )
    validation_dataset = lgb.Dataset(
        validation_x,
        label=validation_contract.labels,
        reference=train_dataset,
        params=dict(dataset_parameters),
        free_raw_data=False,
    )
    results = []
    for number, parameters in trials:
        trial_parameters = dict(parameters)
        try:
            booster = lgb.train(
                trial_parameters,
                train_dataset,
                num_boost_round=num_boost_round,
                valid_sets=[validation_dataset],
                valid_names=["validation"],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=early_stopping_rounds,
                        first_metric_only=True,
                        verbose=False,
                        min_delta=0.0,
                    )
                ],
                keep_training_booster=True,
            )
        except lgb.basic.LightGBMError as error:
            if not recoverable_lightgbm_trial_failure(error):
                raise
            failure = str(error)
            results.append(
                PythonTrialResult(
                    number=number,
                    best_iteration=None,
                    validation_prediction=None,
                    model_string=None,
                    failure=failure,
                )
            )
            continue
        best_iteration = (
            int(booster.best_iteration)
            if booster.best_iteration > 0
            else int(booster.current_iteration())
        )
        while booster.current_iteration() > best_iteration:
            booster.rollback_one_iter()
        prediction = np.asarray(
            booster.predict(validation_x, num_iteration=-1), dtype=np.float64
        )
        model_string = booster.model_to_string(num_iteration=-1)
        booster.free_dataset()
        results.append(
            PythonTrialResult(
                number=number,
                best_iteration=best_iteration,
                validation_prediction=prediction,
                model_string=model_string,
            )
        )
    construction = {
        "executor": PYTHON_EXECUTOR,
        "feature_count": train_contract.feature_count,
        "train_rows": train_contract.row_count,
        "validation_rows": validation_contract.row_count,
    }
    return results, construction


def predict_members(
    *,
    source,
    contract: SplitContract,
    representation: str,
    model_strings: Sequence[str],
) -> list[np.ndarray]:
    """Predict the split for each member model, ascending-sample-ID order."""
    features = apply_zero_vacating_offset(
        _representation_matrix(source, contract, representation),
        contract.offset_columns,
    )
    predictions = []
    for model_string in model_strings:
        booster = lgb.Booster(model_str=model_string)
        predictions.append(
            np.asarray(booster.predict(features, num_iteration=-1), dtype=np.float64)
        )
    return predictions
