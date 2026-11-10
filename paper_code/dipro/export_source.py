"""Export DiPro's exact released mortality cohort and structured boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from cohort_source import (  # noqa: E402
    DEMOGRAPHIC,
    GCS,
    SELECTED_VAR,
    TASK_ID,
    UPSTREAM_COMMIT,
    sha256_file,
)


VALUE_CODES = tuple(f"DIPRO//EHR//{name}" for name in SELECTED_VAR)
MASK_CODES = tuple(f"DIPRO//OBSERVED_MASK//{name}" for name in SELECTED_VAR)
DEMOGRAPHIC_CODES = tuple(f"DIPRO//DEMOGRAPHIC//{name}" for name in DEMOGRAPHIC)
CODES = VALUE_CODES + MASK_CODES + DEMOGRAPHIC_CODES
SPLIT_FILES = (
    ("train", "train.csv"),
    ("validation", "val.csv"),
    ("test", "test.csv"),
)


def _ordered_hash(rows: list[tuple[int, int, int, str]]) -> str:
    digest = hashlib.sha256()
    for subject_id, stay_id, label, split in rows:
        digest.update(np.asarray([subject_id, stay_id, label], dtype=np.int64).tobytes())
        digest.update(split.encode() + b"\0")
    return digest.hexdigest()


def sample_frame(cache_root: Path) -> tuple[pl.DataFrame, list[dict]]:
    stays = pd.read_csv(
        cache_root / "all_stays.csv",
        usecols=["subject_id", "stay_id", "intime"],
        parse_dates=["intime"],
    ).set_index("stay_id")
    records = []
    for split, filename in SPLIT_FILES:
        source = pd.read_csv(cache_root / "split" / "mortality_los" / filename)
        for source_split_row, row in source.iterrows():
            stay_id = int(row["stay_id"])
            subject_id = int(row["subject_id"])
            stay = stays.loc[stay_id]
            if int(stay["subject_id"]) != subject_id:
                raise AssertionError("DiPro split row disagrees with all_stays")
            intime = pd.Timestamp(stay["intime"])
            records.append(
                {
                    "subject_id": subject_id,
                    "sample_id": stay_id,
                    "observation_start": intime.to_pydatetime(),
                    "prediction_time": (
                        intime + pd.Timedelta(hours=48)
                    ).to_pydatetime(),
                    "boolean_value": bool(int(row["mortality_inhospital"])),
                    "split": split,
                    "source_split_row": int(source_split_row),
                }
            )
    return pl.DataFrame(records), records


def carry_in_source_times(cache_root: Path) -> dict[tuple[int, int], float]:
    """Return valid source observations used to initialize released row zero."""
    result = {}
    connection = sqlite3.connect(cache_root / "source.sqlite")
    rows = connection.execute(
        "SELECT stay_id, variable_index, hours, value "
        "FROM selected_episode_cells WHERE hours < 0 "
        "ORDER BY stay_id, variable_index, hours, episode_row"
    )
    for stay_id, variable_index, hours, value in rows:
        variable_index = int(variable_index)
        variable = SELECTED_VAR[variable_index]
        if variable in GCS:
            value = GCS[variable]["values"].get(value, np.nan)
        value = pd.to_numeric(value, errors="coerce")
        if not pd.isna(value):
            result[(int(stay_id), variable_index)] = float(hours)
    connection.close()
    return result


def event_batches(cache_root: Path, records: list[dict], batch_samples: int = 96):
    demographic = pd.read_csv(cache_root / "demographic_processed.csv").set_index(
        "stay_id"
    )
    carry_in_source_time = carry_in_source_times(cache_root)
    for batch_start in range(0, len(records), batch_samples):
        rows = []
        for record in records[batch_start : batch_start + batch_samples]:
            stay_id = int(record["sample_id"])
            subject_id = int(record["subject_id"])
            start = pd.Timestamp(record["observation_start"])
            episode = pd.read_csv(
                cache_root
                / "ehr_preprocessed"
                / f"New_episode{stay_id}_timeseries.csv"
            )
            episode = episode.loc[episode["regular_interval"] <= 48]
            source_row = 0
            for _, episode_row in episode.iterrows():
                interval = int(episode_row["regular_interval"])
                event_time = start + pd.Timedelta(hours=interval)
                for variable_index, variable in enumerate(SELECTED_VAR):
                    mask = int(episode_row[f"{variable}_mask"])
                    carry_in_time = carry_in_source_time.get(
                        (stay_id, variable_index)
                    )
                    if mask or (interval == 0 and carry_in_time is not None):
                        rows.append(
                            {
                                "subject_id": subject_id,
                                "time": event_time.to_pydatetime(),
                                "code": VALUE_CODES[variable_index],
                                "numeric_value": np.float32(
                                    episode_row[variable]
                                ),
                                "sample_id": stay_id,
                                "source_row": source_row,
                                "sequence_row": interval,
                                "source_time": (
                                    float(interval)
                                    if mask
                                    else float(carry_in_time)
                                ),
                            }
                        )
                        source_row += 1
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "time": event_time.to_pydatetime(),
                            "code": MASK_CODES[variable_index],
                            "numeric_value": np.float32(mask),
                            "sample_id": stay_id,
                            "source_row": source_row,
                            "sequence_row": interval,
                            "source_time": float(interval),
                        }
                    )
                    source_row += 1
            static = demographic.loc[stay_id]
            for code, variable in zip(DEMOGRAPHIC_CODES, DEMOGRAPHIC):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "time": start.to_pydatetime(),
                        "code": code,
                        "numeric_value": np.float32(static[variable]),
                        "sample_id": stay_id,
                        "source_row": source_row,
                        "sequence_row": -1,
                        "source_time": None,
                    }
                )
                source_row += 1
        yield pl.DataFrame(rows)


def build_source_metadata(cache_root: Path, records: list[dict]) -> dict:
    summary = json.loads((cache_root / "cohort_summary.json").read_text())
    lengths = json.loads((cache_root / "episode_lengths.json").read_text())
    connection = sqlite3.connect(cache_root / "source.sqlite")
    scan_states = {
        stage: json.loads(payload)
        for stage, payload in connection.execute(
            "SELECT stage, payload FROM cache_state WHERE stage LIKE 'scan_%'"
        )
    }
    connection.close()
    ordered = [
        (
            int(record["subject_id"]),
            int(record["sample_id"]),
            int(record["boolean_value"]),
            str(record["split"]),
        )
        for record in records
    ]
    derived_paths = (
        cache_root / "all_stays.csv",
        cache_root / "cxr_stay.csv",
        cache_root / "mean_std.csv",
        cache_root / "demographic.csv",
        cache_root / "demographic_processed.csv",
        cache_root / "episode_lengths.json",
    )
    return {
        "upstream_repository": "https://github.com/Chenliu-svg/DiPro.git",
        "upstream_commit": UPSTREAM_COMMIT,
        "loader_entrypoints": [
            "mimic3benchmark.mimic3csv cohort helpers",
            "mimic3benchmark.scripts.get_cxr.align_cxr_with_stay",
            "mimic3benchmark.scripts.split_train_and_test.split_dataset_for_each_task",
            "mimic3benchmark.preprocessing.map_itemids_to_variables",
            "mimic3benchmark.preprocessing.clean_events",
            "mimic3benchmark.subject.convert_events_to_timeseries",
            "mimic3benchmark.scripts.ehr_preprocess.process_demographic",
            "dataset_module.dipro_dataset.DiproDataset.__getitem__",
            "dataset_module.dipro_dataset.CollateFunc",
        ],
        "bounded_instrumentation": {
            "adapter": "cohort_source.py",
            "adapter_sha256": sha256_file(PAPER_ROOT / "cohort_source.py"),
            "reason": (
                "gzip-aware selected-ITEMID streaming replaces only the released "
                "eager expansion/redistribution step; released mappings, cleaning, "
                "cohort alignment, split functions, demographic encoder, Dataset, "
                "and collator remain the executable authorities"
            ),
        },
        "selected_paper_cell": (
            "Table 2 / Table 12, longitudinal DiPro mortality, "
            "AUROC 0.897 +/- 0.002"
        ),
        "selection_tier": 4,
        "selection_adjudication": (
            "Explicit study decision on 2026-08-30 promotes the headline "
            "longitudinal-CXR cell over the available tier-3 EHR-only cell."
        ),
        "selected_model_inputs": ["structured", "longitudinal_regional_cxr_images"],
        "retained_modalities": ["structured"],
        "omitted_modalities": ["imaging"],
        "cohort_restriction": (
            "The selected longitudinal DiPro experiment requires at least two "
            "eligible longitudinal CXRs. The export preserves that exact cohort "
            "and exports only its structured projection."
        ),
        "capture_boundary": (
            "Released normalized hourly EHR observations plus their reliable "
            "pre-imputation masks, and released 17-value demographic vector. "
            "Mask-zero imputed EHR cells are absent from MEDS except when a real "
            "elapsed [-1,0) source observation is carried into row zero; exact "
            "forward-fill reconstruction is asserted by check_batch.sh."
        ),
        "split_policy": (
            "Released seed-0 subject split with Chest ImaGenome gold subjects "
            "forced into test; exact source rows preserved"
        ),
        "ordered_subject_stay_label_split_sha256": _ordered_hash(ordered),
        "source_cohort": summary,
        "episode_lengths": {
            "minimum": min(int(value) for value in lengths.values()),
            "maximum": max(int(value) for value in lengths.values()),
            "count": len(lengths),
        },
        "structured_vocabulary": {
            "values": list(VALUE_CODES),
            "observation_masks": list(MASK_CODES),
            "demographics": list(DEMOGRAPHIC_CODES),
        },
        "derived_artifacts": {
            str(path.relative_to(cache_root)): sha256_file(path)
            for path in derived_paths
        },
        "raw_source_artifacts": summary["source_hashes"],
        "event_source_scans": scan_states,
        "known_source_defects_preserved": [
            "hour bin 48 includes measurements from elapsed [48,49)",
            "EHR and demographic preprocessing is fitted on the full cohort before splitting",
            "Chest ImaGenome gold subjects are forced into test",
        ],
        "information_asymmetry": (
            "No image pixels, image embeddings, image timestamps/masks/flags, "
            "CheXpert fields, bounding boxes, or progression auxiliary targets "
            "are retained. The selected DiPro model receives longitudinal "
            "regional CXR images that the shared LightGBM model does not."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    samples, records = sample_frame(args.cache_root)
    source = build_source_metadata(args.cache_root, records)
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "dipro",
            "title": (
                "Multimodal Disease Progression Modeling via Spatiotemporal "
                "Disentanglement and Multiscale Alignment"
            ),
        },
        task={
            "id": TASK_ID,
            "name": "In-hospital mortality from the first 48 ICU hours",
            "primary_metric": "AUROC",
            "published_auroc": 0.897,
            "observation_window_hours": 48.0,
        },
        codes=CODES,
        note_codes=(),
        events=event_batches(args.cache_root, records),
        samples=samples,
        source=source,
        dataset_name="DiPro released MIMIC-IV mortality cohort",
        dataset_version="MIMIC-IV 2.0 / MIMIC-CXR-JPG 2.0.0 / Chest ImaGenome 1.0.0",
    )
    print(json.dumps({"task": str(args.output_dir), "event_count": manifest["event_count"]}, indent=2))


if __name__ == "__main__":
    main()
