"""Export CLUDA Table 1's first MIMIC age-shift mortality cell to MEDS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from cohort_source import (  # noqa: E402
    CODES,
    HISTORY_HOURS,
    MIN_HOURS,
    PUBLISHED_AUROC,
    SOURCE_AGE_GROUP,
    STATIC_FEATURES,
    TARGET_AGE_GROUP,
    TASK_ID,
    TS_FEATURES,
    UPSTREAM_COMMIT,
    UPSTREAM_URL,
    iter_sample_rows,
    load_author_source,
    require_source,
    selected_splits,
    sha256,
)


def samples_frame(selected) -> pl.DataFrame:
    rows = list(iter_sample_rows(selected))
    return pl.DataFrame(
        {
            "subject_id": pl.Series([row["subject_id"] for row in rows], dtype=pl.Int64),
            "sample_id": pl.Series([row["sample_id"] for row in rows], dtype=pl.Int64),
            "observation_start": pl.Series(
                np.asarray(
                    [row["observation_start"] for row in rows], dtype="datetime64[us]"
                ),
                dtype=pl.Datetime("us"),
            ),
            "prediction_time": pl.Series(
                np.asarray(
                    [row["prediction_time"] for row in rows], dtype="datetime64[us]"
                ),
                dtype=pl.Datetime("us"),
            ),
            "boolean_value": pl.Series(
                [row["boolean_value"] for row in rows], dtype=pl.Boolean
            ),
            "split": pl.Series([row["split"] for row in rows], dtype=pl.String),
            "stay_id": pl.Series([row["stay_id"] for row in rows], dtype=pl.Int64),
            "stay_hour": pl.Series([row["stay_hour"] for row in rows], dtype=pl.Int32),
            "loader_patient_id": pl.Series(
                [row["loader_patient_id"] for row in rows], dtype=pl.Int64
            ),
            "source_original_split": pl.Series(
                [row["source_original_split"] for row in rows], dtype=pl.String
            ),
            "loader_age_group": pl.Series(
                [row["loader_age_group"] for row in rows], dtype=pl.Int8
            ),
            "paper_age_group": pl.Series(
                [row["paper_age_group"] for row in rows], dtype=pl.Int8
            ),
        }
    )


def _empty_columns() -> dict[str, list]:
    return {
        "subject_id": [],
        "time": [],
        "code": [],
        "numeric_value": [],
        "sample_id": [],
        "source_row": [],
        "sequence_row": [],
        "source_time": [],
    }


def _frame(columns: dict[str, list]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": pl.Series(columns["subject_id"], dtype=pl.Int64),
            "time": pl.Series(
                np.asarray(columns["time"], dtype="datetime64[us]"),
                dtype=pl.Datetime("us"),
            ),
            "code": pl.Series(columns["code"], dtype=pl.String),
            "numeric_value": pl.Series(columns["numeric_value"], dtype=pl.Float32),
            "sample_id": pl.Series(columns["sample_id"], dtype=pl.Int64),
            "source_row": pl.Series(columns["source_row"], dtype=pl.Int64),
            "sequence_row": pl.Series(columns["sequence_row"], dtype=pl.Int32),
            "source_time": pl.Series(columns["source_time"], dtype=pl.Float64),
        }
    )


def event_batches(selected, *, maximum_rows: int = 200_000):
    """Losslessly run-length encode every duplicated released loader prefix."""
    columns = _empty_columns()

    def append(
        subject_id: int,
        time,
        code: str,
        value: float,
        sid: int,
        source_row: int,
        sequence_row: int,
        source_time: float,
    ) -> None:
        columns["subject_id"].append(subject_id)
        columns["time"].append(time)
        columns["code"].append(code)
        columns["numeric_value"].append(np.float32(value))
        columns["sample_id"].append(sid)
        columns["source_row"].append(source_row)
        columns["sequence_row"].append(sequence_row)
        columns["source_time"].append(float(source_time))

    for split_name in ("train", "validation", "test"):
        arrays = selected[split_name]
        for stay_id, subject_id, intime, source_charttime, sequence, mask, static in zip(
            arrays.stay_ids,
            arrays.subject_ids,
            arrays.intimes,
            arrays.source_charttime,
            arrays.sequence,
            arrays.sequence_mask,
            arrays.static,
        ):
            source_charttime = np.asarray(source_charttime, dtype=np.int64)
            sequence = np.asarray(sequence, dtype=np.float32)
            mask = np.asarray(mask, dtype=np.int64)
            for end_index in range(MIN_HOURS, len(sequence) + 1):
                start_index = max(0, end_index - HISTORY_HOURS)
                width = end_index - start_index
                source_start_hour = int(source_charttime[start_index])
                observation_start = intime + np.timedelta64(source_start_hour, "h")
                sid = int(stay_id) * 100_000 + end_index
                pad = HISTORY_HOURS - (end_index - start_index)

                # Static inputs are supplied unchanged to every released
                # prediction window, including the source's future-looking
                # height/weight/BMI aggregation.
                for feature_index, (name, value) in enumerate(zip(STATIC_FEATURES, static)):
                    append(
                        int(subject_id),
                        observation_start,
                        f"STATIC/{name}",
                        value,
                        sid,
                        feature_index,
                        -1,
                        source_start_hour,
                    )

                boundary_values = sequence[start_index]
                boundary_masks = mask[start_index]
                if end_index < HISTORY_HOURS:
                    # The released loader discards the true mask for these
                    # endpoints.  Persist only the value tensor's minimal
                    # run-length changes: exposing TS occurrences here would
                    # give LightGBM information the published method lost.
                    changed = np.zeros((width, len(TS_FEATURES)), dtype=bool)
                    changed[0] = boundary_values != 0
                    if width > 1:
                        changed[1:] = sequence[start_index + 1 : end_index] != sequence[
                            start_index : end_index - 1
                        ]
                    change_rows, change_features = np.nonzero(changed)
                    for local_row, feature_index in zip(change_rows, change_features):
                        source_index = start_index + int(local_row)
                        source_hour = int(source_charttime[source_index])
                        append(
                            int(subject_id),
                            intime + np.timedelta64(source_hour, "h"),
                            f"VALUE/{TS_FEATURES[int(feature_index)]}",
                            sequence[source_index][feature_index],
                            sid,
                            500 + source_index * len(TS_FEATURES) + int(feature_index),
                            pad + int(local_row),
                            source_hour,
                        )
                else:
                    # At endpoints >=48 the source really consumes the binary
                    # mask.  A boundary carry plus observed TS occurrences is
                    # lossless for both its value and mask tensors.
                    carry_features = np.flatnonzero(
                        np.logical_and(boundary_masks == 0, boundary_values != 0)
                    )
                    for feature_index in carry_features:
                        append(
                            int(subject_id),
                            observation_start,
                            f"CARRY/{TS_FEATURES[int(feature_index)]}",
                            boundary_values[feature_index],
                            sid,
                            100 + int(feature_index),
                            pad,
                            source_start_hour,
                        )

                    observed_rows, observed_features = np.nonzero(mask[start_index:end_index])
                    for local_row, feature_index in zip(observed_rows, observed_features):
                        source_index = start_index + int(local_row)
                        source_hour = int(source_charttime[source_index])
                        append(
                            int(subject_id),
                            intime + np.timedelta64(source_hour, "h"),
                            f"TS/{TS_FEATURES[int(feature_index)]}",
                            sequence[source_index][feature_index],
                            sid,
                            1_000 + source_index * len(TS_FEATURES) + int(feature_index),
                            pad + int(local_row),
                            source_hour,
                        )

                if len(columns["sample_id"]) >= maximum_rows:
                    yield _frame(columns)
                    columns = _empty_columns()
    if columns["sample_id"]:
        yield _frame(columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-synthetic", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    source_manifest = require_source(args.source_root, allow_synthetic=args.allow_synthetic)
    source = load_author_source(args.source_root, allow_synthetic=args.allow_synthetic)
    selected = selected_splits(source)
    samples = samples_frame(selected)
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "cluda",
            "title": "Contrastive Learning for Unsupervised Domain Adaptation of Time Series",
        },
        task={
            "id": TASK_ID,
            "name": "MIMIC-IV mortality under age-domain shift, Group 1 to Group 2",
            "positive_class": "hospital_expire_flag == 1 (death during the hospital admission)",
            "primary_metric": "AUROC",
            "published_auroc": PUBLISHED_AUROC,
            "observation_window_hours": float(HISTORY_HOURS),
            "temporal_profile": "single_visit_timed",
        },
        codes=CODES,
        events=event_batches(selected),
        samples=samples,
        source={
            "repository": UPSTREAM_URL,
            "commit": UPSTREAM_COMMIT,
            "source_manifest_sha256": sha256(args.source_root / "source_manifest.json"),
            "upstream_file_sha256": source_manifest.get("upstream_file_sha256", {}),
            "source_reconstruction_status": source_manifest["status"],
            "dataset_version": source_manifest["dataset_version"],
            "selected_paper_cell": "Main Table 1, source Group 1 to target Group 2",
            "selected_source_configuration": {
                "task": "mortality",
                "path_src": "miiv_fullstays",
                "path_trg": "miiv_fullstays",
                "age_src": SOURCE_AGE_GROUP,
                "age_trg": TARGET_AGE_GROUP,
                "use_mask": True,
                "min_hours": MIN_HOURS,
                "history_hours": HISTORY_HOURS,
            },
            "supervised_projection": {
                "train": "source Group 1 released train split",
                "validation": "source Group 1 released validation split",
                "test": "target Group 2 released test split",
                "omitted_from_supervised_task": [
                    "unlabeled target Group 2 train/validation inputs used by CLUDA",
                    "source Group 1 test inputs",
                ],
            },
            "retained_defects": [
                "early endpoints replace the released observation mask with integer-cast values",
                "static height, weight, and BMI can aggregate post-landmark measurements",
                "feature availability was selected on the complete MIMIC/AUMC cohorts",
            ],
            "encoding": {
                "TS/*": "released observed scaled hourly values",
                "CARRY/*": "lossless boundary value already forward-filled at window start",
                "VALUE/*": "run-length value changes for early windows whose true mask is discarded by the released bug",
                "STATIC/*": "released eight-element static model branch",
            },
            "time_coordinates": {
                "sequence_row": "zero-padded released ICUDataset tensor row",
                "source_time": "exact RICU ICU-relative charttime hour",
                "time": "ICU intime plus exact RICU charttime",
                "landmarks": "absolute source clock; tensor slicing remains sequence-relative",
            },
        },
        dataset_name="CLUDA MIMIC-IV v1.0 age-domain mortality",
        dataset_version=source_manifest["dataset_version"],
    )
    print(
        json.dumps(
            {
                "status": "written",
                "task": str(args.output_dir),
                "splits": manifest["splits"],
                "events": manifest["event_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
