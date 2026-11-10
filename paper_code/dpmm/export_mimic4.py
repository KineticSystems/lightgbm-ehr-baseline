"""Export DPMM's MIMIC-IV partial-cohort EHR experiments (IHM and READM).

`partial` for MIMIC-IV is `data_pairs=partial_ehr_cxr` with `data_ratio=1.0`.
`dataset_mimic4/fusion.py:MIMIC_CXR_EHR.__len__` then returns
`len(ehr_files_paired) + int(1.0 * len(ehr_files_unpaired))`, i.e. every ICU
stay in the released EHR list files; the chest radiograph is attached where one
exists and never restricts membership.  The export therefore exports the complete
released EHR cohort and retains no imaging field of any kind.
"""

from __future__ import annotations

import argparse
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
from cohort_source_mimic4 import (  # noqa: E402
    DISCRETIZER_CONFIG,
    EXPECTED_COUNTS,
    EXPECTED_TOTAL,
    HOURS,
    IHM_PUBLISHED_AUROC,
    IHM_PUBLISHED_CELL,
    IHM_TASK_ID,
    PUBLISHED_IHM_TOTAL_POSITIVES,
    PUBLISHED_LOADER_IHM_POSITIVES,
    PUBLISHED_LOADER_READM_POSITIVES,
    PUBLISHED_READM_TOTAL_POSITIVES,
    READM_PUBLISHED_AUROC,
    READM_PUBLISHED_CELL,
    READM_TASK_ID,
    TIMESTEP,
    UPSTREAM_COMMIT,
    ensure_source,
    ordered_text_hash,
    sha256,
    source_codes,
)


TASKS = {
    "in-hospital-mortality": {
        "task_id": IHM_TASK_ID,
        "label_column": "mortality_label",
        "name": "MIMIC-IV in-hospital mortality from the first 48 ICU hours",
        "published_auroc": IHM_PUBLISHED_AUROC,
        "published_cell": IHM_PUBLISHED_CELL,
        "published_interval": [0.847, 0.877],
        "published_aupr": 0.529,
        "published_f1": 0.518,
        "published_total_positives": PUBLISHED_IHM_TOTAL_POSITIVES,
        "published_loader_positives": PUBLISHED_LOADER_IHM_POSITIVES,
        "labels_set": "mortality",
        "launcher": "scripts/mortality/mimic4/partial/dp.sh",
        "label_producer": (
            "mimic4extract/mimic3benchmark/scripts/create_in_hospital_mortality.py"
        ),
        "positive_class": "death during the hospital admission",
        "label_definition": (
            "released `Mortality` column of the episodic label CSV, i.e. "
            "`add_inhospital_mortality_to_icustays`: `dod` or `deathtime` falls "
            "within [admittime, dischtime] of this admission. The stay must have "
            "`24 * Length of Stay >= 48h` and at least one retained measurement "
            "in [0, 48]; events are truncated to -eps < t < 48 + eps."
        ),
        "dataset_name": "DPMM MIMIC-IV partial EHR in-hospital mortality",
    },
    "readmission": {
        "task_id": READM_TASK_ID,
        "label_column": "readmission_label",
        "name": "MIMIC-IV 30-day hospital readmission from the first 48 ICU hours",
        "published_auroc": READM_PUBLISHED_AUROC,
        "published_cell": READM_PUBLISHED_CELL,
        "published_interval": [0.752, 0.788],
        "published_aupr": 0.486,
        "published_f1": 0.472,
        "published_total_positives": PUBLISHED_READM_TOTAL_POSITIVES,
        "published_loader_positives": PUBLISHED_LOADER_READM_POSITIVES,
        "labels_set": "readm",
        "launcher": "scripts/readmission/mimic4/partial/dp.sh",
        "label_producer": (
            "mimic4extract/mimic3benchmark/scripts/create_readmission_30d.py"
        ),
        "positive_class": (
            "next hospital admission within 30 days of discharge, or in-hospital "
            "death during this admission"
        ),
        "label_definition": (
            "released `create_readmission_30d.py`: positive when the subject's "
            "next hospital admission starts strictly fewer than 30 whole days "
            "after this admission's `dischtime` "
            "(`(next_intime - cur_outtime).days < 30`, a pandas Timedelta.days "
            "floor with no lower bound); the subject's last admission is "
            "negative; `Mortality == 1` (in-hospital death) overrides the "
            "outcome to positive. The stay must have `24 * Length of Stay >= "
            "48h` and at least one retained measurement in [0, 48]; events are "
            "truncated to -eps < t < 48 + eps."
        ),
        "dataset_name": "DPMM MIMIC-IV partial EHR 30-day readmission",
    },
}


def samples_frame(cohort: pd.DataFrame, label_column: str) -> pl.DataFrame:
    prediction = pd.to_datetime(cohort["intime"]) + pd.Timedelta(hours=HOURS)
    return pl.DataFrame(
        {
            "subject_id": pl.Series(
                cohort["subject_id"].to_numpy(np.int64), dtype=pl.Int64
            ),
            "sample_id": pl.Series(cohort["stay_id"].to_numpy(np.int64), dtype=pl.Int64),
            "observation_start": pl.Series(
                pd.to_datetime(cohort["intime"]).to_numpy(dtype="datetime64[us]"),
                dtype=pl.Datetime("us"),
            ),
            "prediction_time": pl.Series(
                prediction.to_numpy(dtype="datetime64[us]"), dtype=pl.Datetime("us")
            ),
            "boolean_value": pl.Series(
                cohort[label_column].astype(bool).to_numpy(), dtype=pl.Boolean
            ),
            "split": pl.Series(cohort["split"].tolist(), dtype=pl.String),
            "source_name": pl.Series(cohort["source_name"].tolist(), dtype=pl.String),
            "stay_id": pl.Series(cohort["stay_id"].to_numpy(np.int64), dtype=pl.Int64),
            "hadm_id": pl.Series(cohort["hadm_id"].to_numpy(np.int64), dtype=pl.Int64),
            "episode_number": pl.Series(
                cohort["episode_number"].to_numpy(np.int64), dtype=pl.Int64
            ),
        }
    )


def event_batches(
    cohort: pd.DataFrame, structured_path: Path, *, batch_size: int = 250_000
):
    known_stays = pl.Series("stay_id", cohort["stay_id"].to_numpy(np.int64), dtype=pl.Int64)
    ordered_lookup = cohort.sort_values("stay_id")
    lookup_ids = ordered_lookup["stay_id"].to_numpy(np.int64)
    lookup_intimes = pd.to_datetime(ordered_lookup["intime"]).to_numpy(
        dtype="datetime64[ns]"
    )
    seen_rows: dict[int, int] = {}
    last_time: dict[int, np.datetime64] = {}
    last_sequence: dict[int, int] = {}
    for record_batch in pq.ParquetFile(structured_path).iter_batches(
        batch_size=batch_size
    ):
        part = pl.from_arrow(record_batch)
        if part.is_empty():
            continue
        if not bool(part["stay_id"].is_in(known_stays).all()):
            raise AssertionError("structured event references a non-cohort stay")
        stay_ids = part["stay_id"].to_numpy().astype(np.int64, copy=False)
        charttimes = part["charttime"].to_numpy().astype("datetime64[us]")
        positions = np.searchsorted(lookup_ids, stay_ids)
        if not np.array_equal(lookup_ids[positions], stay_ids):
            raise AssertionError("structured event references unknown cohort stay")
        source_times = (
            charttimes.astype("datetime64[ns]") - lookup_intimes[positions]
        ) / np.timedelta64(1, "h")
        source_rows = np.empty(len(part), dtype=np.int64)
        sequence_rows = np.empty(len(part), dtype=np.int32)
        starts = np.flatnonzero(np.r_[True, stay_ids[1:] != stay_ids[:-1]])
        ends = np.r_[starts[1:], len(part)]
        for start, end in zip(starts, ends):
            stay_id = int(stay_ids[start])
            offset = seen_rows.get(stay_id, 0)
            source_rows[start:end] = offset + np.arange(end - start, dtype=np.int64)
            seen_rows[stay_id] = offset + end - start
            times = charttimes[start:end]
            changes = np.empty(end - start, dtype=np.int32)
            changes[0] = int(stay_id not in last_time or times[0] != last_time[stay_id])
            if end - start > 1:
                changes[1:] = (times[1:] != times[:-1]).astype(np.int32)
            base = last_sequence.get(stay_id, -1)
            sequence_rows[start:end] = base + np.cumsum(changes, dtype=np.int32)
            last_time[stay_id] = times[-1]
            last_sequence[stay_id] = int(sequence_rows[end - 1])
        part = (
            part.with_columns(
                pl.when(pl.col("categorical_value").is_not_null())
                .then(pl.concat_str(["variable", pl.lit("//"), "categorical_value"]))
                .otherwise(pl.col("variable"))
                .alias("code"),
                pl.Series("source_time", source_times.astype(np.float64), dtype=pl.Float64),
                pl.Series("source_row", source_rows, dtype=pl.Int64),
                pl.Series("sequence_row", sequence_rows, dtype=pl.Int32),
                pl.lit(None, dtype=pl.String).alias("text_value"),
            )
            .select(
                "subject_id",
                pl.col("charttime").cast(pl.Datetime("us")).alias("time"),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    spec = TASKS[args.task]
    source_manifest = ensure_source(args.mimic_root, args.cache_root)
    cohort = pd.read_parquet(args.cache_root / source_manifest["files"]["cohort"])
    structured_path = args.cache_root / source_manifest["files"]["structured_events"]

    counts = cohort["split"].value_counts().to_dict()
    positives = (
        cohort.groupby("split")[spec["label_column"]].sum().astype(int).to_dict()
    )
    if counts != EXPECTED_COUNTS or len(cohort) != EXPECTED_TOTAL:
        raise AssertionError(f"DPMM MIMIC-IV cohort counts differ: {counts}")

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
            "id": spec["task_id"],
            "name": spec["name"],
            "primary_metric": "AUROC",
            "published_auroc": spec["published_auroc"],
            "observation_window_hours": HOURS,
        },
        codes=codes,
        note_codes=(),
        events=event_batches(cohort, structured_path),
        samples=samples_frame(cohort, spec["label_column"]),
        source={
            "upstream_commit": UPSTREAM_COMMIT,
            "selected_paper_cell": spec["published_cell"],
            "published_auroc_interval": spec["published_interval"],
            "published_aupr": spec["published_aupr"],
            "published_f1": spec["published_f1"],
            "published_total_positives": spec["published_total_positives"],
            "published_drop_last_loader_positives": spec["published_loader_positives"],
            "released_source_positives": positives,
            "released_source_total_positives": int(sum(positives.values())),
            "selected_model_inputs": ["structured", "chest radiograph"],
            "selection_tier": 4,
            "retained_modalities": ["structured"],
            "omitted_source_modalities": ["chest radiograph (MIMIC-CXR-JPG 2.0.0)"],
            "modality_asymmetry": (
                "The published DPMM MIMIC-IV cell additionally receives a chest "
                "radiograph for the 5,931 stays that have one. The export never "
                "retains imaging, so the shared LightGBM baseline is given the "
                "structured projection only, exactly as the campaign did for "
                "ddl_cxr. The imaging modality does not change cohort membership: "
                "data_ratio=1.0 keeps every unpaired EHR stay."
            ),
            "source_configuration": {
                "task": args.task,
                "labels_set": spec["labels_set"],
                "data_pairs": "partial_ehr_cxr",
                "data_ratio": 1.0,
                "timestep": TIMESTEP,
                "store_masks": True,
                "impute_strategy": "previous",
                "start_time": "zero",
                "batch_size": 32,
                "launcher": spec["launcher"],
            },
            "mimic_iv_version": source_manifest["mimic_iv_version"],
            "mimic_iv_version_evidence": source_manifest["mimic_iv_version_evidence"],
            "label_producer": spec["label_producer"],
            "label_definition": spec["label_definition"],
            "positive_class": spec["positive_class"],
            "prediction_time": "48 hours after ICU admission (released n_hours=48)",
            "structured_capture_boundary": (
                "released cleaned episode time series before 2-hour discretization, "
                "previous imputation, and train-only normalization"
            ),
            "counts": EXPECTED_COUNTS,
            "positives": positives,
            "partial_cohort_count": len(cohort),
            "source_name_sha256": ordered_text_hash(cohort["source_name"]),
            "released_code_vocabulary": list(codes),
            "discretizer_config_sha256": sha256(DISCRETIZER_CONFIG),
            "source_reconstruction": source_manifest,
            "known_source_defects_preserved": [
                "create_readmission_30d.py iterates a sorted DataFrame's index "
                "labels but selects the next stay with .iloc; measured fully "
                "latent on this cohort (0 of 7,249 multi-stay subjects reorder, "
                "0 stays change, 0 negative inter-admission gaps)",
                "in-hospital death overrides the readmission outcome to positive "
                "(supplement definition)",
                "the MIMIC-IV extractor writes VALUENUM into the column the "
                "cleaning rules read as VALUEUOM, so the unit-based Temperature, "
                "Weight, Height and FiO2 branches never see a unit string",
                "clean_temperature tests `'F' in s.lower()`, which is never true; "
                "only the `value >= 79` branch converts Fahrenheit",
                "clean_fio2's Python-2 `np.array(map(...))` collapses to a scalar "
                "True under Python 3, so every non-torr FiO2 value is divided by "
                "100 (identical to the released Python-2 result here because the "
                "per-subject `value` column is text)",
                "the released itemid map is the MIMIC-III one, so MIMIC-III-only "
                "channels can be entirely absent from MIMIC-IV",
                "published prevalence sums a shuffled drop-last training loader "
                "rather than the full cohort",
                "released training exposes test AUROC every epoch (not used here)",
            ],
        },
        dataset_name=spec["dataset_name"],
        dataset_version="MIMIC-IV 2.2 / DPMM released HEAD",
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
