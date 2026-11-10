"""Export DPMM's selected partial EHR-note IHM experiment to MEDS."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from cohort_source import (  # noqa: E402
    DISCRETIZER_CONFIG,
    EXPECTED_COUNTS,
    EXPECTED_PAIRED,
    EXPECTED_POSITIVES,
    HOURS,
    NOTE_CODE,
    PUBLISHED_AUROC,
    PUBLISHED_TOTAL_POSITIVES,
    TASK_ID,
    TIMESTEP,
    TOKENIZER,
    UPSTREAM_COMMIT,
    ensure_source,
    ordered_text_hash,
    sha256,
    source_codes,
)


def samples_frame(cohort: pd.DataFrame) -> pl.DataFrame:
    prediction = pd.to_datetime(cohort["INTIME"]) + pd.Timedelta(hours=HOURS)
    return pl.DataFrame(
        {
            "subject_id": pl.Series(
                cohort["SUBJECT_ID"].to_numpy(np.int64), dtype=pl.Int64
            ),
            "sample_id": pl.Series(
                cohort["stay_id"].to_numpy(np.int64), dtype=pl.Int64
            ),
            "observation_start": pl.Series(
                pd.to_datetime(cohort["INTIME"]).to_numpy(dtype="datetime64[us]"),
                dtype=pl.Datetime("us"),
            ),
            "prediction_time": pl.Series(
                prediction.to_numpy(dtype="datetime64[us]"),
                dtype=pl.Datetime("us"),
            ),
            "boolean_value": pl.Series(
                cohort["label"].astype(bool).to_numpy(), dtype=pl.Boolean
            ),
            "split": pl.Series(cohort["split"].tolist(), dtype=pl.String),
            "source_name": pl.Series(cohort["source_name"].tolist(), dtype=pl.String),
            "stay_id": pl.Series(cohort["stay_id"].to_numpy(np.int64), dtype=pl.Int64),
            "hadm_id": pl.Series(cohort["HADM_ID"].to_numpy(np.int64), dtype=pl.Int64),
            "fusion_position": pl.Series(
                cohort["fusion_position"].to_numpy(np.int64), dtype=pl.Int64
            ),
            "note_present": pl.Series(
                cohort["note_present"].astype(bool).to_numpy(), dtype=pl.Boolean
            ),
        }
    )


def event_batches(
    cohort: pd.DataFrame,
    structured_path: Path,
    notes: pd.DataFrame,
    *,
    batch_size: int = 250_000,
):
    lookup = pl.DataFrame(
        {
            "stay_id": pl.Series(cohort["stay_id"], dtype=pl.Int64),
            "subject_id": pl.Series(cohort["SUBJECT_ID"], dtype=pl.Int64),
            "INTIME": pl.Series(
                pd.to_datetime(cohort["INTIME"]).to_numpy(dtype="datetime64[us]"),
                dtype=pl.Datetime("us"),
            ),
        }
    )
    ordered_lookup = cohort.sort_values("stay_id")
    lookup_ids = ordered_lookup["stay_id"].to_numpy(np.int64)
    lookup_intimes = pd.to_datetime(ordered_lookup["INTIME"]).to_numpy(
        dtype="datetime64[ns]"
    )
    seen_rows: dict[int, int] = {}
    last_time: dict[int, np.datetime64] = {}
    last_sequence: dict[int, int] = {}
    for record_batch in pq.ParquetFile(structured_path).iter_batches(
        batch_size=batch_size
    ):
        part = pl.from_arrow(record_batch)
        stay_ids = part["stay_id"].to_numpy().astype(np.int64, copy=False)
        charttimes = part["CHARTTIME"].to_numpy().astype("datetime64[us]")
        positions = np.searchsorted(lookup_ids, stay_ids)
        if not np.array_equal(lookup_ids[positions], stay_ids):
            raise AssertionError("structured event references unknown cohort stay")
        source_times = (
            charttimes.astype("datetime64[ns]") - lookup_intimes[positions]
        ) / np.timedelta64(1, "h")
        source_rows = np.empty(len(part), dtype=np.int64)
        sequence_rows = np.empty(len(part), dtype=np.int32)
        starts = np.flatnonzero(
            np.r_[True, stay_ids[1:] != stay_ids[:-1]]
        )
        ends = np.r_[starts[1:], len(part)]
        for start, end in zip(starts, ends):
            stay_id = int(stay_ids[start])
            offset = seen_rows.get(stay_id, 0)
            source_rows[start:end] = offset + np.arange(end - start, dtype=np.int64)
            seen_rows[stay_id] = offset + end - start
            times = charttimes[start:end]
            changes = np.empty(end - start, dtype=np.int32)
            changes[0] = int(
                stay_id not in last_time or times[0] != last_time[stay_id]
            )
            if end - start > 1:
                changes[1:] = (times[1:] != times[:-1]).astype(np.int32)
            base = last_sequence.get(stay_id, -1)
            sequence_rows[start:end] = base + np.cumsum(changes, dtype=np.int32)
            last_time[stay_id] = times[-1]
            last_sequence[stay_id] = int(sequence_rows[end - 1])
        part = (
            part.join(lookup, on="stay_id", how="inner", validate="m:1")
            .with_columns(
                pl.when(pl.col("categorical_value").is_not_null())
                .then(
                    pl.concat_str(
                        ["VARIABLE", pl.lit("//"), "categorical_value"]
                    )
                )
                .otherwise(pl.col("VARIABLE"))
                .alias("code"),
                pl.Series(
                    "source_time", source_times.astype(np.float64), dtype=pl.Float64
                ),
                pl.Series("source_row", source_rows, dtype=pl.Int64),
                pl.Series("sequence_row", sequence_rows, dtype=pl.Int32),
                pl.lit(None, dtype=pl.String).alias("text_value"),
            )
            .select(
                "subject_id",
                pl.col("CHARTTIME").cast(pl.Datetime("us")).alias("time"),
                pl.col("code").cast(pl.String),
                pl.col("numeric_value").cast(pl.Float32),
                pl.col("stay_id").alias("sample_id"),
                "source_row",
                "sequence_row",
                pl.col("source_time").cast(pl.Float64),
                "text_value",
            )
        )
        yield part

    note_lookup = cohort[
        ["stay_id", "SUBJECT_ID", "INTIME"]
    ].rename(columns={"SUBJECT_ID": "subject_id"})
    notes = notes.merge(note_lookup, on="stay_id", how="inner", validate="m:1")
    notes["CHARTTIME"] = pd.to_datetime(notes["CHARTTIME"])
    notes["INTIME"] = pd.to_datetime(notes["INTIME"])
    notes["source_time"] = (
        notes["CHARTTIME"] - notes["INTIME"]
    ).dt.total_seconds() / 3600.0
    # DPMM consumes note order/content, not note timestamps.  Its released bug
    # can select notes outside the ICU [0,48] interval.  Clamp only the MEDS
    # ordering coordinate while retaining the exact source coordinate.
    upper = notes["INTIME"] + pd.Timedelta(hours=HOURS)
    notes["event_time"] = notes["CHARTTIME"].clip(
        lower=notes["INTIME"], upper=upper
    )
    notes = notes.sort_values(["stay_id", "slot"], kind="stable")
    note_frame = pl.DataFrame(
        {
            "subject_id": pl.Series(notes["subject_id"], dtype=pl.Int64),
            "time": pl.Series(
                notes["event_time"].to_numpy(dtype="datetime64[us]"),
                dtype=pl.Datetime("us"),
            ),
            "code": pl.Series([NOTE_CODE] * len(notes), dtype=pl.String),
            "numeric_value": pl.Series([None] * len(notes), dtype=pl.Float32),
            "sample_id": pl.Series(notes["stay_id"], dtype=pl.Int64),
            "source_row": pl.Series(
                1_000_000_000 + notes["slot"].to_numpy(np.int64), dtype=pl.Int64
            ),
            "sequence_row": pl.Series(notes["slot"], dtype=pl.Int32),
            "source_time": pl.Series(notes["source_time"], dtype=pl.Float64),
            "text_value": pl.Series(notes["text"].tolist(), dtype=pl.String),
        }
    )
    yield note_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = ensure_source(args.mimic_root, args.cache_root)
    cohort = pd.read_parquet(
        args.cache_root / source_manifest["files"]["cohort"]
    )
    notes = pd.read_parquet(
        args.cache_root / source_manifest["files"]["selected_notes"]
    )
    structured_path = (
        args.cache_root / source_manifest["files"]["structured_events"]
    )
    codes = source_codes()
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "dpmm",
            "title": (
                "Amplifying Prominent Representations in Multimodal Learning "
                "via Variational Dirichlet Process"
            ),
        },
        task={
            "id": TASK_ID,
            "name": "MIMIC-III in-hospital mortality from the first 48 ICU hours",
            "primary_metric": "AUROC",
            "published_auroc": PUBLISHED_AUROC,
            "observation_window_hours": HOURS,
        },
        codes=codes,
        note_codes=(NOTE_CODE,),
        events=event_batches(cohort, structured_path, notes),
        samples=samples_frame(cohort),
        source={
            "upstream_commit": UPSTREAM_COMMIT,
            "selected_paper_cell": "Table 3, MIMIC-III IHM, partial EHR-note DPMM, AUROC 0.860 (0.840, 0.878)",
            "selected_model_inputs": ["structured", "notes"],
            "selection_tier": 2,
            "retained_modalities": ["structured", "notes"],
            "source_configuration": {
                "task": "in-hospital-mortality",
                "data_pairs": "partial_ehr_note",
                "data_ratio": 1.0,
                "timestep": TIMESTEP,
                "store_masks": True,
                "impute_strategy": "previous",
                "start_time": "zero",
                "bert_type": "huawei-noah/TinyBERT_General_4L_312D",
                "batch_size": 16,
            },
            "structured_capture_boundary": (
                "released cleaned episode time series before 2-hour discretization, "
                "previous imputation, and validation-leaked normalization"
            ),
            "note_capture_boundary": (
                "released cleaned raw strings for the last five loader-selected "
                "note timestamps, before TinyBERT tokenization"
            ),
            "positive_class": "death during the hospital admission",
            "counts": EXPECTED_COUNTS,
            "positives": EXPECTED_POSITIVES,
            "published_total_positives": PUBLISHED_TOTAL_POSITIVES,
            "released_source_total_positives": sum(EXPECTED_POSITIVES.values()),
            "paired_note_counts": EXPECTED_PAIRED,
            "partial_cohort_count": len(cohort),
            "note_rows": len(notes),
            "note_text_sha256": ordered_text_hash(notes["text"]),
            "source_name_sha256": ordered_text_hash(cohort["source_name"]),
            "released_code_vocabulary": list(codes),
            "discretizer_config_sha256": sha256(DISCRETIZER_CONFIG),
            "tokenizer_path": str(TOKENIZER.relative_to(PAPER_ROOT)),
            "tokenizer_revision": "34707a33cd59a94ecde241ac209bf35103691b43",
            "source_reconstruction": source_manifest,
            "known_source_defects_preserved": [
                "note loader rejects exactly five eligible notes by using <= 5",
                "note cutoff is anchored to the first retained event at least one full hour after ICU admission",
                "note lower-bound check is disabled and the shifted 48-hour cutoff can include post-ICU-48-hour text",
                "normalizer moments include the later validation partition",
                "published prevalence diagnostic sums a shuffled drop-last training loader rather than the full cohort",
                "released training exposes test AUROC every epoch (not used by this exporter)",
            ],
            "note_time_serialization": (
                "source_time retains exact hours relative to ICU admission; the MEDS "
                "time coordinate is clamped to [0,48] only because the model uses "
                "note slot order/content and the common schema forbids out-of-window events"
            ),
        },
        dataset_name="DPMM MIMIC-III partial EHR-note IHM",
        dataset_version="MIMIC-III 1.4 / DPMM released HEAD",
    )
    print(
        json.dumps(
            {
                "task": str(args.output_dir),
                "samples": len(cohort),
                "events": manifest["event_count"],
                "splits": manifest["splits"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
