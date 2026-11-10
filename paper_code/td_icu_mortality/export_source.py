"""Export the selected internal 28-day TD-ICU mortality task to MEDS."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from missing_xgboost_lightgbm.landmark_sampling import (  # noqa: E402
    LANDMARK_SAMPLE_CAPS,
    LANDMARK_SAMPLE_SEED,
    membership_mask,
    select_uniform_landmarks,
)
from cohort_source import (  # noqa: E402
    CONTEXT_LENGTH,
    CURRENT_BUILD_PATH,
    INPUT_WINDOW_HOURS,
    LABEL_COLUMNS,
    MIMIC_RELEASE,
    NEXT_STATE_DELAY_HOURS,
    NEXT_STATE_WINDOW_HOURS,
    PAPER_ROOT,
    SAMPLES_PATH,
    SOURCE_BUILD_PATH,
    SOURCE_CONVERSION_PATH,
    UPSTREAM_COMMIT,
    build_author_source,
    build_current_boundary,
    current_files,
    read_json,
    sha256_file,
    source_feature_names,
)


TASK_ID = "mortality_28d"
DELTA_TIME_PREFIX = "DELTA_TIME//"
DELTA_VALUE_PREFIX = "DELTA_VALUE//"
MINIMUM_INITIAL_FREE_BYTES = 70 * 1024**3
MINIMUM_RUNTIME_FREE_BYTES = 30 * 1024**3


def delta_source_features(features: tuple[str, ...]) -> tuple[str, ...]:
    """Features for which the released author tensor defines delta channels."""
    return tuple(
        name
        for name in features
        if name not in {"age", "gender", "patientweight"}
        and ("rate" in name or "bolus" not in name)
    )


def meds_feature_names(features: tuple[str, ...]) -> tuple[str, ...]:
    delta_features = delta_source_features(features)
    return (
        features
        + tuple(f"{DELTA_TIME_PREFIX}{name}" for name in delta_features)
        + tuple(f"{DELTA_VALUE_PREFIX}{name}" for name in delta_features)
    )


def events_from_current(
    frame: pl.DataFrame, features: tuple[str, ...]
) -> pl.DataFrame:
    feature_map = {index: name for index, name in enumerate(features)}
    delta_features = set(delta_source_features(features))
    sequence = list(range(CONTEXT_LENGTH))
    exploded = (
        frame
        # Select before exploding.  The author boundary also contains the two
        # 400-element delta lists and audit-only labels; retaining those here
        # would replicate unused list objects once per exploded token.
        .select(
            "subject_id",
            "sample_id",
            "source_sample_row",
            "split",
            "labeltime",
            "encoded_feature",
            "featuretime",
            "value",
            "delta_time",
            "delta_value",
        )
        .with_columns(
            pl.lit(sequence, dtype=pl.List(pl.Int32)).alias("sequence_row")
        )
        .explode(
            "encoded_feature",
            "featuretime",
            "value",
            "delta_time",
            "delta_value",
            "sequence_row",
        )
        .filter(pl.col("encoded_feature") >= 0)
        .with_columns(
            pl.col("encoded_feature")
            .replace_strict(feature_map, return_dtype=pl.String)
            .alias("code"),
            (
                pl.col("labeltime")
                - pl.duration(minutes=pl.col("featuretime").cast(pl.Int64))
            ).cast(pl.Datetime("us")).alias("time"),
            pl.col("value").cast(pl.Float32).alias("numeric_value"),
            (
                (
                    pl.col("source_sample_row") * CONTEXT_LENGTH
                    + pl.col("sequence_row").cast(pl.Int64)
                )
                * 3
            ).alias("token_source_row"),
            pl.col("featuretime").cast(pl.Float64).alias("source_time"),
        )
    )
    shared = (
        "subject_id",
        "time",
        "sample_id",
        "sequence_row",
        "source_time",
        "split",
    )
    primary = exploded.select(
        *shared,
        "code",
        "numeric_value",
        pl.col("token_source_row").alias("source_row"),
    )
    delta_time = (
        exploded
        .filter(
            pl.col("code").is_in(delta_features)
            & (pl.col("delta_time") >= 0)
        )
        .select(
            *shared,
            pl.concat_str(pl.lit(DELTA_TIME_PREFIX), pl.col("code")).alias("code"),
            pl.col("delta_time").cast(pl.Float32).alias("numeric_value"),
            (pl.col("token_source_row") + 1).alias("source_row"),
        )
    )
    delta_value = (
        exploded
        .filter(
            pl.col("code").is_in(delta_features)
            & pl.col("delta_value").is_finite()
        )
        .select(
            *shared,
            pl.concat_str(pl.lit(DELTA_VALUE_PREFIX), pl.col("code")).alias("code"),
            pl.col("delta_value").cast(pl.Float32).alias("numeric_value"),
            (pl.col("token_source_row") + 2).alias("source_row"),
        )
    )
    events = (
        pl.concat((primary, delta_time, delta_value))
        .sort("source_row")
        .select(
            pl.col("subject_id").cast(pl.Int64),
            "time",
            "code",
            "numeric_value",
            pl.col("sample_id").cast(pl.Int64),
            pl.col("source_row").cast(pl.Int64),
            pl.col("sequence_row").cast(pl.Int32),
            "source_time",
            "split",
        )
    )
    if (
        events["numeric_value"].is_nan().any()
        or events["numeric_value"].is_infinite().any()
        or events["numeric_value"].is_null().any()
    ):
        raise AssertionError("a non-padding author token has no finite value")
    return events


def event_batches(
    features: tuple[str, ...], selected_sample_ids: np.ndarray
) -> Iterator[pl.DataFrame]:
    selected_ids = np.sort(np.asarray(selected_sample_ids, dtype=np.int64))
    for path in current_files():
        if shutil.disk_usage(WORKSPACE_ROOT).free < MINIMUM_RUNTIME_FREE_BYTES:
            raise RuntimeError(
                "TD-ICU export stopped before exhausting the workspace filesystem"
            )
        frame = pl.read_parquet(path)
        keep = membership_mask(
            frame["sample_id"].to_numpy().astype(np.int64, copy=False),
            selected_ids,
        )
        frame = frame.filter(pl.Series("__keep", keep))
        if frame.is_empty():
            continue
        # The shared writer attaches split from the immutable sample index.
        yield events_from_current(frame, features).drop("split")


def export(mimic_root: Path, output_root: Path) -> dict:
    available = shutil.disk_usage(WORKSPACE_ROOT).free
    if available < MINIMUM_INITIAL_FREE_BYTES:
        raise RuntimeError(
            "TD-ICU rolling-landmark construction requires at least 70 GiB free "
            f"at start; found {available / 1024**3:.2f} GiB"
        )
    build_author_source(mimic_root)
    build_current_boundary()
    features = source_feature_names()
    meds_codes = meds_feature_names(features)
    full_samples = pl.read_parquet(SAMPLES_PATH)
    samples, sampling_audit = select_uniform_landmarks(full_samples)
    selected_sample_ids = np.sort(
        samples["sample_id"].to_numpy().astype(np.int64, copy=False)
    )
    task_dir = output_root / TASK_ID
    source_build = read_json(SOURCE_BUILD_PATH)
    current_build = read_json(CURRENT_BUILD_PATH)
    source = {
        "repository_url": "https://github.com/tdgfrost/td-icu-mortality.git",
        "commit": UPSTREAM_COMMIT,
        "paper_experiment": "TD-24hr internal MIMIC-IV evaluation, 28-day mortality cell",
        "paper_table": "Table 2, TD row/28-day column",
        "source_entrypoint": "load_mimic -> clean_combined_data -> create_labels_for_mimic -> encode_combined_data_for_mimic -> create_final_dataframe",
        "mimic_release": MIMIC_RELEASE,
        "configuration": {
            "input_window_hours": INPUT_WINDOW_HOURS,
            "context_length": CONTEXT_LENGTH,
            "next_state_delay_hours": NEXT_STATE_DELAY_HOURS,
            "next_state_window_hours": NEXT_STATE_WINDOW_HOURS,
            "split_seed": 42,
        },
        "source_artifacts": {
            "source_conversion.json": sha256_file(SOURCE_CONVERSION_PATH),
            "source_build.json": sha256_file(SOURCE_BUILD_PATH),
            "current_boundary.json": sha256_file(CURRENT_BUILD_PATH),
            "export/samples.parquet": sha256_file(SAMPLES_PATH),
            **source_build["artifact_sha256"],
        },
        "source_status": source_build["status"],
        "current_boundary_status": current_build["status"],
        "table1_cohort_reconciliation": {
            "status": "inferred_exact_duration_marginals",
            "scope": "descriptive encounter counting only; not a filter on the released model cohort",
            "hospital_duration_seconds_exclusive": 6_660,
            "icu_duration_seconds_exclusive": 7_227,
            "reconstructed_hospital_admissions": 84_659,
            "reconstructed_icu_admissions": 93_508,
            "model_task_membership_changed": False,
            "reason_membership_is_unchanged": "applying both encounter filters to model events would reduce the exact paper-matching 65,050-patient cohort and would not preserve either published encounter marginal jointly",
        },
        "resource_limited_reproduction": True,
        "sampling": sampling_audit,
        "sampling_contract": {
            "seed": LANDMARK_SAMPLE_SEED,
            "caps": LANDMARK_SAMPLE_CAPS,
        },
        "retained_boundary": "all non-padding tokens and finite model-visible delta channels at the exact current-state author boundary, including released truncation and ordering",
        "channel_encoding": {
            "primary": "released feature code at the token event time",
            "delta_time": f"{DELTA_TIME_PREFIX}<released feature code> at the same token time; absent where the author channel is missing",
            "delta_value": f"{DELTA_VALUE_PREFIX}<released feature code> at the same token time; absent where the author channel is missing",
            "source_row": "(source_sample_row * 400 + sequence_row) * 3 + channel, where primary/delta_time/delta_value channels are 0/1/2",
        },
        "excluded": [
            "padding tokens",
            "aligned next-state arrays used only as privileged TD training targets",
            "SICdb external-validation arm",
            "all labels other than selected 28-day mortality as model events",
        ],
        "all_non_note_codes_are_intended_structured_inputs": True,
    }
    return write_meds_task(
        task_dir,
        paper={
            "key": "td_icu_mortality",
            "title": "Robust Real-Time Mortality Prediction in the Intensive Care Unit using Temporal Difference Learning",
        },
        task={
            "id": TASK_ID,
            "name": "28-day mortality at rolling ICU measurement landmarks",
            "primary_metric": "AUROC",
            "published_auroc": 0.839,
            "positive_class": "death timestamp no later than 28 days after the source state marker",
            "observation_window_hours": float(INPUT_WINDOW_HOURS),
            "evaluation_unit": "released rolling measurement-time state landmark",
        },
        codes=meds_codes,
        note_codes=(),
        events=event_batches(features, selected_sample_ids),
        samples=samples,
        source=source,
        dataset_name="MIMIC-IV",
        dataset_version=MIMIC_RELEASE,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    export(args.mimic_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
