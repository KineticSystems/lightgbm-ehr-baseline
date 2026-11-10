"""Export MUSE's selected structured-plus-notes mortality task to MEDS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from cohort_source import (  # noqa: E402
    AUTHOR_BATCH_SIZE,
    AUTHOR_MISSING_LAB_WIDTH,
    CODE_FILE,
    LAB_FILE,
    MAX_SEQUENCE_LENGTH,
    NOTE_CODE,
    NOTE_FILE,
    OBSERVATION_HOURS,
    PAPER_ROOT,
    PUBLISHED_AUROC,
    TASK_ID,
    UPSTREAM_COMMIT,
    ensure_source,
    lab_channels,
    sha256_file,
    source_codes,
)


def _lookup_frames(
    cache_root: Path, selected: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pl.DataFrame]:
    """Attach modality presence to the selected labeled admissions.

    ``selected`` defaults to the mortality-labeled cohort; the readmission task
    passes its own already split labeled frame so that the identical modality
    presence, trajectory-length, and window construction is reused verbatim.
    """
    if selected is None:
        cohort = pd.read_parquet(cache_root / "cohort.parquet")
        selected = cohort.loc[cohort["mortality"].notna()].copy()
    else:
        selected = selected.copy()
    selected["sample_id"] = selected["hadm_id"].astype(np.int64)
    selected_ids = set(selected["hadm_id"].astype(int))
    lab_ids: set[int] = set()
    for batch in pq.ParquetFile(cache_root / LAB_FILE).iter_batches(
        batch_size=100_000, columns=["hadm_id"]
    ):
        lab_ids.update(
            int(value)
            for value in np.unique(batch.column(0).to_numpy())
            if int(value) in selected_ids
        )
    note_ids: set[int] = set()
    for batch in pq.ParquetFile(cache_root / NOTE_FILE).iter_batches(
        batch_size=100_000, columns=["hadm_id"]
    ):
        note_ids.update(
            int(value)
            for value in batch.column(0).to_pylist()
            if int(value) in selected_ids
        )
    code_lengths: dict[int, int] = {}
    for batch in pq.ParquetFile(cache_root / CODE_FILE).iter_batches(
        batch_size=100_000, columns=["hadm_id"]
    ):
        hadm_ids, counts = np.unique(batch.column(0).to_numpy(), return_counts=True)
        for hadm_id, count in zip(hadm_ids, counts):
            hadm_id = int(hadm_id)
            if hadm_id in selected_ids:
                code_lengths[hadm_id] = code_lengths.get(hadm_id, 0) + int(count)
    selected["lab_present"] = selected["hadm_id"].isin(lab_ids)
    selected["note_present"] = selected["hadm_id"].isin(note_ids)
    selected["trajectory_length"] = (
        selected["hadm_id"].map(code_lengths).fillna(0).astype(np.int32)
    )
    lookup = pl.DataFrame(
        {
            "hadm_id": pl.Series(selected["hadm_id"], dtype=pl.Int64),
            "subject_id": pl.Series(selected["subject_id"], dtype=pl.Int64),
            "prediction_time": pl.Series(
                selected["dischtime"].to_numpy(dtype="datetime64[us]"),
                dtype=pl.Datetime("us"),
            ),
            "observation_start": pl.Series(
                selected["meds_observation_start"].to_numpy(dtype="datetime64[us]"),
                dtype=pl.Datetime("us"),
            ),
            "trajectory_length": pl.Series(
                selected["trajectory_length"], dtype=pl.Int32
            ),
        }
    )
    return selected, lookup


def samples_frame(
    selected: pd.DataFrame, label_column: str = "mortality"
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": pl.Series(selected["subject_id"], dtype=pl.Int64),
            "sample_id": pl.Series(selected["hadm_id"], dtype=pl.Int64),
            "observation_start": pl.Series(
                selected["meds_observation_start"].to_numpy(dtype="datetime64[us]"),
                dtype=pl.Datetime("us"),
            ),
            "prediction_time": pl.Series(
                selected["dischtime"].to_numpy(dtype="datetime64[us]"),
                dtype=pl.Datetime("us"),
            ),
            "boolean_value": pl.Series(
                selected[label_column].astype(bool), dtype=pl.Boolean
            ),
            "split": pl.Series(selected["split"], dtype=pl.String),
            "cohort_position": pl.Series(
                selected["cohort_position"], dtype=pl.Int64
            ),
            "split_position": pl.Series(
                selected["split_position"].astype(np.int64), dtype=pl.Int64
            ),
            "age_bucket": pl.Series(selected["age_bucket"], dtype=pl.String),
            "gender": pl.Series(selected["gender"], dtype=pl.String),
            "ethnicity": pl.Series(selected["race"], dtype=pl.String),
            "trajectory_length": pl.Series(
                selected["trajectory_length"], dtype=pl.Int32
            ),
            "lab_present": pl.Series(selected["lab_present"], dtype=pl.Boolean),
            "note_present": pl.Series(selected["note_present"], dtype=pl.Boolean),
        }
    )


def _base_event_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(
        pl.col("subject_id").cast(pl.Int64),
        pl.col("time").cast(pl.Datetime("us")),
        pl.col("code").cast(pl.String),
        pl.col("numeric_value").cast(pl.Float32),
        pl.col("hadm_id").cast(pl.Int64).alias("sample_id"),
        pl.col("source_row").cast(pl.Int64),
        pl.col("sequence_row").cast(pl.Int32),
        pl.col("source_time").cast(pl.Float64),
        pl.col("text_value").cast(pl.String),
    )


def demographic_events(selected: pd.DataFrame) -> pl.DataFrame:
    repeated = pd.concat(
        [
            selected.assign(
                code="MUSE_AGE//" + selected["age_bucket"],
                source_row=0,
            ),
            selected.assign(
                code="MUSE_GENDER//" + selected["gender"],
                source_row=1,
            ),
            selected.assign(
                code="MUSE_ETHNICITY//" + selected["race"],
                source_row=2,
            ),
            selected.assign(
                code=np.where(
                    selected["lab_present"],
                    "MUSE_LAB_PRESENT//true",
                    "MUSE_LAB_PRESENT//false",
                ),
                source_row=3,
            ),
        ],
        ignore_index=True,
    )
    return _base_event_frame(
        pl.DataFrame(
            {
                "subject_id": repeated["subject_id"],
                "hadm_id": repeated["hadm_id"],
                "time": repeated["meds_observation_start"].to_numpy(dtype="datetime64[us]"),
                "code": repeated["code"],
                "numeric_value": [None] * len(repeated),
                "source_row": repeated["source_row"],
                "sequence_row": [0] * len(repeated),
                "source_time": [0.0] * len(repeated),
                "text_value": [None] * len(repeated),
            }
        )
    )


def code_event_batches(cache_root: Path, lookup: pl.DataFrame):
    for record_batch in pq.ParquetFile(cache_root / CODE_FILE).iter_batches(
        batch_size=250_000
    ):
        frame = pl.from_arrow(record_batch).join(
            lookup, on="hadm_id", how="inner", validate="m:1"
        )
        if frame.is_empty():
            continue
        frame = frame.with_columns(
            pl.concat_str(
                [
                    pl.lit("MUSE_CODE//"),
                    pl.col("event_type"),
                    pl.lit("//"),
                    pl.col("code_name"),
                ]
            ).alias("code"),
            pl.max_horizontal(
                pl.col("observation_start"),
                pl.col("prediction_time")
                - pl.duration(
                    microseconds=(
                        pl.col("trajectory_length") - pl.col("sequence_row")
                    )
                ),
            ).alias("time"),
            pl.lit(None, dtype=pl.Float32).alias("numeric_value"),
            (1_000 + pl.col("sequence_row")).cast(pl.Int64).alias("source_row"),
            pl.col("sequence_row").cast(pl.Float64).alias("source_time"),
            pl.lit(None, dtype=pl.String).alias("text_value"),
        )
        yield _base_event_frame(frame)


def lab_event_batches(cache_root: Path, lookup: pl.DataFrame):
    itemids = lab_channels(cache_root)["itemids"]
    value_codes = [
        value
        for itemid in itemids
        for value in (
            f"MUSE_LAB_VALUE//{itemid}",
            f"MUSE_LAB_ABNORMAL//{itemid}",
        )
    ]
    for record_batch in pq.ParquetFile(cache_root / LAB_FILE).iter_batches(
        batch_size=250_000
    ):
        frame = pl.from_arrow(record_batch).join(
            lookup, on="hadm_id", how="inner", validate="m:1"
        )
        if frame.is_empty():
            continue
        frame = frame.with_columns(
            pl.col("channel_index")
            .replace_strict(
                [-1, *range(len(value_codes))],
                ["MUSE_LAB_SEQUENCE_ROW", *value_codes],
            )
            .alias("code"),
            pl.max_horizontal(
                pl.col("observation_start"),
                pl.col("prediction_time")
                - pl.duration(
                    microseconds=(
                        (MAX_SEQUENCE_LENGTH - pl.col("sequence_row")) * 2
                    )
                ),
            ).alias("time"),
            (
                1_000_000
                + pl.col("sequence_row").cast(pl.Int64) * 1_000
                + pl.col("channel_index").cast(pl.Int64)
                + 1
            ).alias("source_row"),
            pl.col("source_time").cast(pl.Float64),
            pl.lit(None, dtype=pl.String).alias("text_value"),
        )
        yield _base_event_frame(frame)


def note_event_batches(cache_root: Path, lookup: pl.DataFrame):
    for record_batch in pq.ParquetFile(cache_root / NOTE_FILE).iter_batches(
        batch_size=5_000
    ):
        notes = pl.from_arrow(record_batch).join(
            lookup, on="hadm_id", how="inner", validate="1:1"
        )
        if notes.is_empty():
            continue
        notes = notes.with_columns(
            pl.col("prediction_time").alias("time"),
            pl.lit(NOTE_CODE).alias("code"),
            pl.lit(None, dtype=pl.Float32).alias("numeric_value"),
            (2_000_000_000 + pl.col("note_source_row")).alias("source_row"),
            pl.lit(0, dtype=pl.Int32).alias("sequence_row"),
            pl.lit(None, dtype=pl.Float64).alias("source_time"),
            pl.col("text").alias("text_value"),
        )
        yield _base_event_frame(notes)


def event_batches(selected: pd.DataFrame, lookup: pl.DataFrame, cache_root: Path):
    yield demographic_events(selected)
    yield from code_event_batches(cache_root, lookup)
    yield from lab_event_batches(cache_root, lookup)
    yield from note_event_batches(cache_root, lookup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--note-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = ensure_source(
        args.mimic_root, args.note_root, args.cache_root
    )
    selected, lookup = _lookup_frames(args.cache_root)
    channels = lab_channels(args.cache_root)
    author_width = 2 * len(channels["itemids"])
    historical_archive = (
        PAPER_ROOT
        / ".export_assets/external_artifacts/github/"
        "MUSE-3596fd3d5d7d79864e26a80d786ba8c6dc2fbd9b.tar.gz"
    )
    codes = source_codes(args.cache_root)
    source = {
        "upstream_commit": UPSTREAM_COMMIT,
        "historical_producer_commit": (
            "3596fd3d5d7d79864e26a80d786ba8c6dc2fbd9b"
        ),
        "historical_producer_archive_sha256": sha256_file(historical_archive),
        "selected_paper_cell": (
            "Table 1, MIMIC-IV mortality, MUSE, AUC-ROC 0.9004"
        ),
        "selected_model_inputs": [
            "demographics_and_medical_codes",
            "lab_values",
            "clinical_notes",
        ],
        "selection_tier": 2,
        "retained_modalities": ["structured", "notes"],
        "source_configuration": {
            "dataset": "mimic4",
            "task": "mortality",
            "load_no_label": False,
            "seed": 42,
            "batch_size": AUTHOR_BATCH_SIZE,
            "split": "released admission shuffle 70/10/20",
            "prediction_cutoff": "hospital discharge",
            "positive_class": "recorded death within 90 days after discharge",
        },
        "capture_boundary": {
            "codes": (
                "released last-50 aligned type/code trajectory before vocabulary "
                "indexing; deterministic synthetic event time retains only position"
            ),
            "labs": (
                "lossless sparse-delta parameterization of the released last-50 "
                "forward-filled float32 value/abnormal-flag tensor, with explicit "
                "sequence-row markers; zeros remain authored values"
            ),
            "notes": (
                "raw released Brief Hospital Course text after exact section "
                "extraction and last-row-wins admission attachment, before TinyBERT"
            ),
            "demographics": "released decade bucket, gender, and ethnicity categories",
        },
        "missing_modality_membership": {
            "lab_presence_event": True,
            "note_presence": "exactly the presence of a raw note row, including empty text",
            "codes_flag": "constant true in the released loader",
        },
        "author_lab_boundary": {
            "selected_dictionary_items": channels["selected_dictionary_items"],
            "retained_itemids": channels["itemids"],
            "nonmissing_width": author_width,
            "missing_fallback_width": AUTHOR_MISSING_LAB_WIDTH,
            "model_gru_width": AUTHOR_MISSING_LAB_WIDTH,
            "exact_width_match": author_width == AUTHOR_MISSING_LAB_WIDTH,
        },
        "released_code_vocabulary_sha256": sha256_file(
            args.cache_root / "author_code_vocabulary.json"
        ),
        "released_code_vocabulary_size": source_manifest["codes"][
            "author_vocabulary_size"
        ],
        "meds_event_vocabulary_size": len(codes),
        "source_reconstruction": source_manifest,
        "known_source_defects_preserved": [
            "admission-level random split permits subjects in multiple partitions",
            "whole-cohort lab normalization and code vocabulary precede splitting",
            "MIMIC-IV 2.0 HOSP is joined to MIMIC-IV-Note 2.2 by hadm_id",
            "note charttime/storetime are discarded by the author model path",
            "paper Table 4 note missingness uses the base Discharge Instructions producer, not the selected v2 Brief Hospital Course input",
            "74 nonpositive-duration admissions are retained by the released one-sided LOS filter",
        ],
    }
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "muse",
            "title": (
                "Multimodal Patient Representation Learning with Missing "
                "Modalities and Labels"
            ),
        },
        task={
            "id": TASK_ID,
            "name": "90-day post-discharge mortality at hospital discharge",
            "primary_metric": "AUROC",
            "published_auroc": PUBLISHED_AUROC,
            "observation_window_hours": float(OBSERVATION_HOURS),
        },
        codes=codes,
        note_codes=(NOTE_CODE,),
        events=event_batches(selected, lookup, args.cache_root),
        samples=samples_frame(selected),
        source=source,
        dataset_name="MUSE MIMIC-IV mortality structured-plus-notes",
        dataset_version="MIMIC-IV 2.0 / MIMIC-IV-Note 2.2 / MUSE released path",
    )
    print(
        json.dumps(
            {
                "task": str(args.output_dir),
                "samples": len(selected),
                "events": manifest["event_count"],
                "splits": manifest["splits"],
                "exact_lab_width_match": author_width
                == AUTHOR_MISSING_LAB_WIDTH,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
