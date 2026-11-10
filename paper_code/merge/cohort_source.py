"""Bounded exact-source reconstruction for MERGE's released MIMIC-IV tasks.

The released extractor eagerly loads the complete multi-gigabyte event tables.
This adapter applies its released item selection and time-mapping functions to
bounded CSV chunks, then materializes the same pivoted irregular table used by
``create_ihm_task.py``.  All later cohort, imputation, scaling, and loader
operations call the released functions directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler


REPOSITORY = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPOSITORY.parents[1]
PREPROCESS_ROOT = REPOSITORY / "mimiciv" / "data_preprocess"
sys.path.insert(0, str(PREPROCESS_ROOT))

from create_ihm_task import fit_and_apply_scaler  # noqa: E402
from create_los_task import fit_and_apply_scaler as los_fit_and_apply_scaler  # noqa: E402
from preprocess_imputed_time_series import (  # noqa: E402
    compute_global_means,
    impute_stays,
)
from preprocess_irg_time_series import (  # noqa: E402
    add_time_delta_vectorized,
    get_labs_of_interest,
    get_vitals_of_interest,
)
from preprocess_notes import add_time_delta_notes_vectorized  # noqa: E402


HOURS = 48
SEED = 42
TASK_ID = "mortality_48h"
LOS_TASK_ID = "los_96h_survival_at_48h"
TASKS = ("ihm", "los")
# create_los_task.py::get_stay_list labels a stay positive when the ICU
# stay is shorter than 96 hours and the admission did not end in death.
LOS_HOURS = 96
NOTE_CODE = "MERGE//RADIOLOGY_NOTE"
NOTE_SOURCE_ROW_OFFSET = 1_000_000_000_000
UPSTREAM_COMMIT = "75de1f059030e23a9629ed9fdd0d31b56e29ffe4"
PUBLISHED_AUROC = 0.8540
LOS_PUBLISHED_AUROC = 0.8188
EXPECTED_PREFILTER_SPLITS = {
    "train": 32435,
    "validation": 6950,
    "test": 6952,
}
EXPECTED_PREFILTER_OVERLAP = {
    "train_validation": 1852,
    "train_test": 1911,
    "validation_test": 592,
    "subjects_in_multiple_splits": 3651,
}
FROZEN_STAGE2_PREFILTER_OVERLAP = {
    "train_validation": 1839,
    "train_test": 1895,
    "validation_test": 599,
    "subjects_in_multiple_splits": 3609,
}

# pandas.pivot_table sorts the released event labels lexicographically. The
# first dictionary match for Potassium is item 50833, which has no rows in
# MIMIC-IV 3.1, so the actual released pivot has these 30—not 31—channels.
FEATURE_NAMES = (
    "Anion Gap",
    "Bicarbonate",
    "Calcium, Total",
    "Chloride",
    "Creatinine",
    "Diastolic BP",
    "GCS - Eye Opening",
    "GCS - Motor Response",
    "GCS - Verbal Response",
    "Glucose",
    "Heart Rate",
    "Hematocrit",
    "Hemoglobin",
    "MCH",
    "MCHC",
    "MCV",
    "Magnesium",
    "Mean BP",
    "Neutrophils",
    "O2 Saturation",
    "Phosphate",
    "Platelet Count",
    "RDW",
    "Red Blood Cells",
    "Respiratory Rate",
    "Sodium",
    "Systolic BP",
    "Urea Nitrogen",
    "Vancomycin",
    "White Blood Cells",
)
KEY_COLUMNS = (
    "subject_id",
    "hadm_id",
    "stay_id",
    "hosp_time_delta",
    "icu_time_delta",
)
RAW_FILES = (
    "hosp/admissions.csv.gz",
    "hosp/labevents.csv.gz",
    "hosp/d_labitems.csv.gz",
    "icu/icustays.csv.gz",
    "icu/chartevents.csv.gz",
    "icu/d_items.csv.gz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def scaler_summary(scaler: StandardScaler) -> dict:
    arrays = {
        name: np.asarray(getattr(scaler, name), dtype=np.float64)
        for name in ("mean_", "var_", "scale_")
    }
    digest = hashlib.sha256()
    for name, values in arrays.items():
        digest.update(name.encode())
        digest.update(values.tobytes())
    return {
        "sha256": digest.hexdigest(),
        "n_features": int(scaler.n_features_in_),
        "mean_sha256": array_sha256(arrays["mean_"]),
        "variance_sha256": array_sha256(arrays["var_"]),
        "scale_sha256": array_sha256(arrays["scale_"]),
    }


def source_inventory(mimic_root: Path) -> dict[str, dict[str, object]]:
    checksum_path = mimic_root / "SHA256SUMS.txt"
    expected = {}
    for line in checksum_path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            expected[parts[1].lstrip("*./")] = parts[0]
    inventory = {}
    for relative in RAW_FILES:
        path = mimic_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if relative not in expected:
            raise AssertionError(f"release checksum manifest omits {relative}")
        inventory[relative] = {
            "sha256": expected[relative],
            "bytes": path.stat().st_size,
            "checksum_source": "MIMIC-IV 3.1 SHA256SUMS.txt",
        }
    inventory["SHA256SUMS.txt"] = {
        "sha256": sha256_file(checksum_path),
        "bytes": checksum_path.stat().st_size,
    }
    return inventory


def _resolve_item_maps(mimic_root: Path) -> tuple[dict[int, str], dict[int, str]]:
    lab_items = pd.read_csv(mimic_root / "hosp" / "d_labitems.csv.gz").dropna()
    chart_items = pd.read_csv(mimic_root / "icu" / "d_items.csv.gz")

    # Invoke the released first-match selection functions on dictionary-shaped
    # inputs. Chunk processing then applies these exact resolved mappings.
    lab_dummy = pd.DataFrame({"itemid": lab_items["itemid"].to_numpy()})
    chart_dummy = pd.DataFrame({"itemid": chart_items["itemid"].to_numpy()})
    selected_labs = get_labs_of_interest(lab_dummy, lab_items)
    selected_chart = get_vitals_of_interest(chart_dummy, chart_items)
    lab_map = {
        int(row.itemid): str(row.event)
        for row in selected_labs[["itemid", "event"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    chart_map = {
        int(row.itemid): str(row.event)
        for row in selected_chart[["itemid", "event"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    if lab_map.get(50833) != "Potassium":
        raise AssertionError("released first-match Potassium item changed")
    requested = set(lab_map.values()) | set(chart_map.values())
    if requested != set(FEATURE_NAMES) | {"Potassium"}:
        raise AssertionError("released labs/vitals item vocabulary changed")
    return lab_map, chart_map


def _open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.create_function("blob_int", 1, _sqlite_integer, deterministic=True)
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-262144;
        CREATE TABLE pivot_values (
            hadm_id INTEGER NOT NULL,
            chart_us INTEGER NOT NULL,
            event TEXT NOT NULL,
            numeric_value REAL NOT NULL,
            PRIMARY KEY (hadm_id, chart_us, event)
        ) WITHOUT ROWID;
        CREATE TABLE row_keys (
            subject_id INTEGER NOT NULL,
            hadm_id INTEGER NOT NULL,
            stay_id INTEGER NOT NULL,
            chart_us INTEGER NOT NULL,
            hosp_time_delta REAL NOT NULL,
            icu_time_delta REAL NOT NULL,
            PRIMARY KEY (subject_id, hadm_id, stay_id, chart_us)
        ) WITHOUT ROWID;
        CREATE TABLE build_state (
            name TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )
    return connection


def _sqlite_integer(value) -> int:
    if isinstance(value, bytes):
        return int.from_bytes(value, byteorder="little", signed=True)
    return int(value)


def _resume_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.create_function("blob_int", 1, _sqlite_integer, deterministic=True)
    state = connection.execute(
        "SELECT value FROM build_state WHERE name = 'source_scan'"
    ).fetchone()
    if state != ("complete",):
        connection.close()
        raise AssertionError("partial MERGE SQLite source scan cannot be resumed")
    return connection


def _insert_keys(connection: sqlite3.Connection, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    chart_us = (frame["charttime"].astype("int64") // 1000).to_numpy(np.int64)
    rows = (
        (int(subject), int(hadm), int(stay), int(chart), float(hosp), float(icu))
        for subject, hadm, stay, chart, hosp, icu in zip(
            frame["subject_id"].to_numpy(np.int64),
            frame["hadm_id"].to_numpy(np.int64),
            frame["stay_id"].to_numpy(np.int64),
            chart_us,
            frame["hosp_time_delta"].to_numpy(np.float64),
            frame["icu_time_delta"].to_numpy(np.float64),
        )
    )
    before = connection.total_changes
    connection.executemany(
        "INSERT OR IGNORE INTO row_keys VALUES (?, ?, ?, ?, ?, ?)", rows
    )
    return connection.total_changes - before


def _insert_values(connection: sqlite3.Connection, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    selected = frame["valuenum"].notna()
    frame = frame.loc[selected]
    if frame.empty:
        return 0
    values = frame["valuenum"].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise AssertionError("released selected events contain a nonfinite value")
    chart_us = (frame["charttime"].astype("int64") // 1000).to_numpy(np.int64)
    rows = (
        (int(hadm), int(chart), str(event), float(value))
        for hadm, chart, event, value in zip(
            frame["hadm_id"].to_numpy(np.int64),
            chart_us,
            frame["event"].astype(str).to_numpy(),
            values,
        )
    )
    before = connection.total_changes
    connection.executemany(
        "INSERT OR IGNORE INTO pivot_values VALUES (?, ?, ?, ?)", rows
    )
    return connection.total_changes - before


def _candidate_chart_values(
    mapped: pd.DataFrame, windows: pd.DataFrame
) -> pd.DataFrame:
    candidate = mapped.merge(windows, on="hadm_id", how="inner", sort=False)
    return candidate.loc[
        (candidate["charttime"] >= candidate["window_start"])
        & (candidate["charttime"] < candidate["window_end"])
    ]


def _materialize_wide(
    connection: sqlite3.Connection, output_path: Path
) -> tuple[int, dict[str, int]]:
    observed = {
        str(row[0]): int(row[1])
        for row in connection.execute(
            "SELECT event, COUNT(*) FROM pivot_values GROUP BY event ORDER BY event"
        )
    }
    if set(observed) != set(FEATURE_NAMES):
        raise AssertionError(
            "actual released pivot vocabulary differs from the expected 30 "
            f"channels: {sorted(observed)}"
        )
    expressions = ",\n".join(
        "MAX(CASE WHEN p.event = "
        + repr(name)
        + " THEN p.numeric_value END) AS \""
        + name.replace('"', '""')
        + "\""
        for name in FEATURE_NAMES
    )
    query = f"""
        SELECT
            blob_int(k.subject_id) AS subject_id,
            blob_int(k.hadm_id) AS hadm_id,
            blob_int(k.stay_id) AS stay_id,
            k.hosp_time_delta,
            k.icu_time_delta,
            {expressions}
        FROM row_keys AS k
        INNER JOIN pivot_values AS p
          ON p.hadm_id = k.hadm_id AND p.chart_us = k.chart_us
        GROUP BY
            k.subject_id, k.hadm_id, k.stay_id, k.chart_us,
            k.hosp_time_delta, k.icu_time_delta
        ORDER BY
            blob_int(k.subject_id), blob_int(k.hadm_id),
            blob_int(k.stay_id), k.hosp_time_delta
    """
    writer = None
    rows = 0
    try:
        for frame in pd.read_sql_query(query, connection, chunksize=100_000):
            frame = frame.loc[:, list(KEY_COLUMNS) + list(FEATURE_NAMES)]
            for name in KEY_COLUMNS[:3]:
                frame[name] = frame[name].astype(np.int64)
            for name in KEY_COLUMNS[3:] + FEATURE_NAMES:
                frame[name] = frame[name].astype(np.float64)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=100_000)
            rows += len(frame)
            if rows % 1_000_000 == 0:
                print(f"materialized {rows:,} author pivot rows", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if rows == 0:
        raise AssertionError("source construction produced no pivot rows")
    return rows, observed


def ensure_source_cache(
    mimic_root: Path, cache_root: Path, chunksize: int = 500_000
) -> dict:
    """Build or validate the bounded copy of the author's pivoted source table."""
    mimic_root = mimic_root.resolve()
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "source_cache_manifest.json"
    wide_path = cache_root / "ts_labs_vitals.parquet"
    if manifest_path.is_file() and wide_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("upstream_commit") == UPSTREAM_COMMIT
            and manifest.get("mimic_version") == "3.1"
            and manifest.get("wide_parquet_sha256") == sha256_file(wide_path)
        ):
            return manifest
        raise AssertionError("existing MERGE source cache failed provenance checks")

    database_path = cache_root / "pivot.sqlite3"
    if wide_path.exists():
        raise FileExistsError(
            "partial MERGE source cache exists; inspect and remove it before retrying"
        )

    inventory = source_inventory(mimic_root)
    admissions = pd.read_csv(
        mimic_root / "hosp" / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "hospital_expire_flag"],
        parse_dates=["admittime"],
    )
    stays = pd.read_csv(
        mimic_root / "icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
    )
    candidates = stays.loc[stays["los"] >= HOURS / 24.0].copy()
    if len(candidates) != 46_337:
        raise AssertionError(f"released LOS>=2 cohort changed: {len(candidates)}")
    candidate_ids = set(candidates["stay_id"].astype(int))
    windows = candidates[["hadm_id", "intime"]].rename(
        columns={"intime": "window_start"}
    )
    windows["window_end"] = windows["window_start"] + pd.Timedelta(hours=HOURS)

    # The author's lab mapper selects the first ICU window for overlapping
    # stays. There are no overlapping source ICU intervals involving this
    # selected long-stay cohort, so bounded windowing cannot alter that choice.
    overlap_pairs = 0
    for _, group in stays.sort_values("intime").groupby("subject_id", sort=False):
        previous_out = None
        previous_long = False
        for row in group.itertuples(index=False):
            current_long = int(row.stay_id) in candidate_ids
            if (
                previous_out is not None
                and row.intime <= previous_out
                and (previous_long or current_long)
            ):
                overlap_pairs += 1
            if previous_out is None or row.outtime > previous_out:
                previous_out = row.outtime
                previous_long = current_long
    if overlap_pairs:
        raise AssertionError("selected ICU intervals overlap; bounded mapping is unsafe")

    lab_map, chart_map = _resolve_item_maps(mimic_root)
    resume_scan = database_path.exists()
    connection = (
        _resume_database(database_path)
        if resume_scan
        else _open_database(database_path)
    )
    inserted_keys = 0
    inserted_values = 0
    try:
        for source_name, relative, item_map, usecols in (() if resume_scan else (
            (
                "labevents",
                "hosp/labevents.csv.gz",
                lab_map,
                ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
            ),
            (
                "chartevents",
                "icu/chartevents.csv.gz",
                chart_map,
                [
                    "subject_id",
                    "hadm_id",
                    "stay_id",
                    "charttime",
                    "itemid",
                    "valuenum",
                ],
            ),
        )):
            reader = pd.read_csv(
                mimic_root / relative,
                usecols=usecols,
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk_number, chunk in enumerate(reader, start=1):
                chunk = chunk.loc[chunk["itemid"].isin(item_map)].copy()
                if source_name == "labevents":
                    chunk = chunk.dropna(subset=["hadm_id", "charttime"])
                else:
                    chunk = chunk.dropna(
                        subset=["hadm_id", "stay_id", "charttime"]
                    )
                if chunk.empty:
                    continue
                chunk["hadm_id"] = chunk["hadm_id"].astype(np.int64)
                if "stay_id" in chunk:
                    chunk["stay_id"] = chunk["stay_id"].astype(np.int64)
                chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="raise")
                chunk["event"] = chunk["itemid"].map(item_map)

                # This is the released time mapper, including its subject-only
                # ICU matching behavior for labevents.
                mapped = add_time_delta_vectorized(chunk, admissions, stays)
                key_rows = mapped.loc[
                    mapped["stay_id"].isin(candidate_ids)
                    & (mapped["icu_time_delta"] >= 0)
                    & (mapped["icu_time_delta"] < HOURS)
                ].copy()
                inserted_keys += _insert_keys(connection, key_rows)
                if source_name == "chartevents":
                    value_rows = _candidate_chart_values(mapped, windows)
                else:
                    value_rows = key_rows
                inserted_values += _insert_values(connection, value_rows)
                connection.commit()
                if chunk_number % 20 == 0:
                    print(
                        f"{source_name}: chunks={chunk_number:,}; "
                        f"keys={inserted_keys:,}; values={inserted_values:,}",
                        flush=True,
                    )
        if not resume_scan:
            connection.execute(
                "INSERT OR REPLACE INTO build_state VALUES ('source_scan', 'complete')"
            )
            connection.commit()
        wide_rows, observed = _materialize_wide(connection, wide_path)
    finally:
        connection.close()

    database_path.unlink()
    manifest = {
        "status": "exact",
        "upstream_commit": UPSTREAM_COMMIT,
        "mimic_version": "3.1",
        "source_files": inventory,
        "selection_entrypoints": [
            "mimiciv/data_preprocess/preprocess_irg_time_series.py::get_labs_of_interest",
            "mimiciv/data_preprocess/preprocess_irg_time_series.py::get_vitals_of_interest",
            "mimiciv/data_preprocess/preprocess_irg_time_series.py::add_time_delta_vectorized",
        ],
        "bounded_equivalence": {
            "candidate_rule": "ICUSTAYS.los >= 2 days",
            "candidate_stays": len(candidates),
            "window": "[0,48) ICU hours",
            "overlapping_intervals_involving_candidate": overlap_pairs,
            "chart_pivot_values_include_every selected event in candidate admission windows": True,
        },
        "requested_item_ids": {
            "labs": {str(key): value for key, value in sorted(lab_map.items())},
            "vitals": {str(key): value for key, value in sorted(chart_map.items())},
        },
        "actual_feature_names": list(FEATURE_NAMES),
        "actual_feature_observations": observed,
        "potassium": {
            "selected_itemid": 50833,
            "observations_in_selected_window": 0,
            "retained_as_feature": False,
        },
        "pivot_rows": wide_rows,
        "wide_parquet": wide_path.name,
        "wide_parquet_sha256": sha256_file(wide_path),
        "chunk_rows": chunksize,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def note_source_inventory(note_root: Path) -> dict[str, object]:
    relative = "note/radiology.csv.gz"
    path = note_root / relative
    checksum_path = note_root / "SHA256SUMS.txt"
    expected = {}
    for line in checksum_path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            expected[parts[1].lstrip("*./")] = parts[0]
    if not path.is_file() or relative not in expected:
        raise FileNotFoundError("exact MIMIC-IV-Note 2.2 radiology source is missing")
    return {
        "file": relative,
        "sha256": expected[relative],
        "bytes": path.stat().st_size,
        "checksum_manifest_sha256": sha256_file(checksum_path),
        "version": "MIMIC-IV-Note 2.2",
    }


def ensure_note_cache(
    mimic_root: Path,
    note_root: Path,
    cache_root: Path,
    chunksize: int = 50_000,
) -> dict:
    """Invoke the released note mapper in bounded chunks and freeze [0,48)."""
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "note_cache_manifest.json"
    note_path = cache_root / "rad_notes_text.parquet"
    if manifest_path.is_file() and note_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("upstream_commit") == UPSTREAM_COMMIT
            and manifest.get("note_parquet_sha256") == sha256_file(note_path)
            and manifest.get("source") == note_source_inventory(note_root)
        ):
            return manifest
        raise AssertionError("existing MERGE note cache failed provenance checks")
    if note_path.exists():
        raise FileExistsError("partial MERGE note cache exists; inspect before retrying")

    inventory = note_source_inventory(note_root)
    admissions = pd.read_csv(
        mimic_root / "hosp" / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        parse_dates=["admittime", "dischtime"],
    )
    stays = pd.read_csv(
        mimic_root / "icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
    )
    candidates = stays.loc[stays["los"] >= HOURS / 24.0]
    candidate_ids = set(candidates["stay_id"].astype(int))
    selected_chunks = []
    reader = pd.read_csv(
        note_root / "note" / "radiology.csv.gz",
        usecols=[
            "note_id",
            "subject_id",
            "hadm_id",
            "note_type",
            "note_seq",
            "charttime",
            "storetime",
            "text",
        ],
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        chunk["source_note_row"] = chunk.index.to_numpy(np.int64)
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="raise")
        chunk["storetime"] = pd.to_datetime(chunk["storetime"], errors="raise")
        mapped = add_time_delta_notes_vectorized(chunk, admissions, stays)
        selected = mapped.loc[
            mapped["stay_id"].isin(candidate_ids)
            & (mapped["icu_time_delta"] >= 0)
            & (mapped["icu_time_delta"] < HOURS)
        ].copy()
        if not selected.empty:
            selected_chunks.append(
                selected[
                    [
                        "note_id",
                        "subject_id",
                        "hadm_id",
                        "stay_id",
                        "note_type",
                        "note_seq",
                        "charttime",
                        "storetime",
                        "hosp_time_delta",
                        "icu_time_delta",
                        "text",
                        "source_note_row",
                    ]
                ]
            )
        if chunk_number % 10 == 0:
            print(
                f"radiology notes: chunks={chunk_number}; "
                f"selected={sum(len(value) for value in selected_chunks):,}",
                flush=True,
            )
    notes = pd.concat(selected_chunks, ignore_index=True)
    notes = notes.sort_values(
        ["subject_id", "hadm_id", "hosp_time_delta", "stay_id", "icu_time_delta"]
    ).reset_index(drop=True)
    notes["subject_id"] = notes["subject_id"].astype(np.int64)
    # The released mapper assigns admissions and ICU stays independently. A
    # small attributable set therefore has a valid stay but null mapped
    # hadm_id; get_stay_list consumes these notes by stay_id and preserves
    # them, so retain the nullable mapper output rather than repairing it.
    notes["hadm_id"] = notes["hadm_id"].astype("Int64")
    notes["stay_id"] = notes["stay_id"].astype(np.int64)
    notes["source_note_row"] = notes["source_note_row"].astype(np.int64)
    notes["note_position"] = notes.groupby("stay_id", sort=False).cumcount().astype(
        np.int64
    )
    if notes["note_id"].duplicated().any():
        raise AssertionError("released mapped radiology note IDs are not unique")
    notes.to_parquet(note_path, index=False, compression="zstd")
    cutoff = candidates[["stay_id", "intime"]].copy()
    cutoff["prediction_time"] = cutoff["intime"] + pd.Timedelta(hours=HOURS)
    audit = notes.merge(cutoff[["stay_id", "prediction_time"]], on="stay_id")
    unavailable = audit["storetime"] >= audit["prediction_time"]
    manifest = {
        "status": "exact",
        "upstream_commit": UPSTREAM_COMMIT,
        "source": inventory,
        "mapper_entrypoint": (
            "mimiciv/data_preprocess/preprocess_notes.py::"
            "add_time_delta_notes_vectorized"
        ),
        "window": "charttime in half-open [0,48) ICU hours",
        "note_rows": len(notes),
        "stays_with_notes": int(notes["stay_id"].nunique()),
        "notes_with_null_mapped_hadm_id": int(notes["hadm_id"].isna().sum()),
        "note_id_sha256": hashlib.sha256(
            "\n".join(notes["note_id"].astype(str)).encode()
        ).hexdigest(),
        "raw_text_sha256": hashlib.sha256(
            b"".join(
                len(value.encode()).to_bytes(8, "little") + value.encode()
                for value in notes["text"].astype(str)
            )
        ).hexdigest(),
        "stored_at_or_after_cutoff_rows": int(unavailable.sum()),
        "stored_at_or_after_cutoff_stays": int(
            audit.loc[unavailable, "stay_id"].nunique()
        ),
        "note_parquet": note_path.name,
        "note_parquet_sha256": sha256_file(note_path),
        "chunk_rows": chunksize,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def retained_notes(cache_root: Path, sample_ids: set[int]) -> pd.DataFrame:
    manifest = json.loads((cache_root / "note_cache_manifest.json").read_text())
    notes = pd.read_parquet(cache_root / manifest["note_parquet"])
    return notes.loc[notes["stay_id"].isin(sample_ids)].copy()


def _split_members(candidate_ids: np.ndarray) -> dict[str, np.ndarray]:
    shuffled = np.asarray(sorted(candidate_ids.tolist()), dtype=np.int64)
    random_state = np.random.RandomState(SEED)
    random_state.shuffle(shuffled)
    train_end = int(len(shuffled) * 0.70)
    validation_end = train_end + int(len(shuffled) * 0.15)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def _overlap(records: list[dict], key: str) -> dict[str, int]:
    names = ("train", "validation", "test")
    members = {
        split: {int(row[key]) for row in records if row["split"] == split}
        for split in names
    }
    result = {
        f"{left}_{right}": len(members[left] & members[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    all_values = set().union(*members.values())
    result[f"{key}s_in_multiple_splits"] = sum(
        sum(value in members[split] for split in names) > 1
        for value in all_values
    )
    return result


def reconstruct_author_boundary(
    mimic_root: Path,
    cache_root: Path,
    *,
    include_irregular_scaler: bool = True,
    task: str = "ihm",
) -> dict:
    """Call the released task preprocessing functions through float32 boundary.

    ``task`` selects the released label constructor.  ``ihm`` is
    ``create_ihm_task.py`` (``ADMISSIONS.hospital_expire_flag``); ``los`` is
    ``create_los_task.py`` (ICU stay shorter than 96 hours *and* hospital
    survival).  Both released scripts share the identical candidate cohort,
    half-open ``[0,48)`` window, seed-42 stay split, imputation, scaling, and
    ``len(ts_tt) > 12`` loader filter, so only the label differs.
    """
    if task not in TASKS:
        raise AssertionError(f"unknown MERGE task: {task}")
    cache_manifest = ensure_source_cache(mimic_root, cache_root)
    wide_path = cache_root / cache_manifest["wide_parquet"]
    irregular = pd.read_parquet(wide_path)
    if tuple(irregular.columns) != KEY_COLUMNS + FEATURE_NAMES:
        raise AssertionError("cached author pivot column order changed")
    irregular = irregular.loc[
        (irregular["icu_time_delta"] >= 0)
        & (irregular["icu_time_delta"] < HOURS)
    ].copy()

    stays = pd.read_csv(
        mimic_root / "icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
    )
    stays = stays.loc[stays["los"] >= HOURS / 24.0].copy()
    raw_candidate_ids = stays["stay_id"].to_numpy(np.int64)
    structured_ids = irregular["stay_id"].unique().astype(np.int64)
    if set(structured_ids.tolist()) != set(raw_candidate_ids.tolist()):
        raise AssertionError(
            "structured availability does not equal the released LOS>=2 cohort"
        )
    split_ids = _split_members(raw_candidate_ids)
    if {name: len(ids) for name, ids in split_ids.items()} != EXPECTED_PREFILTER_SPLITS:
        raise AssertionError("released seed-42 split no longer matches paper counts")

    # These are the released HEAD functions, with exactly the selected source
    # flags: 48 hours, include notes/CXR, include missing, standardized.
    global_means = compute_global_means(
        irregular, split_ids["train"], list(FEATURE_NAMES)
    )
    # Suppress only tqdm rendering; the released function and iteration order
    # are unchanged. This keeps guarded noninteractive runs compact.
    released_tqdm = impute_stays.__globals__["tqdm"]
    impute_stays.__globals__["tqdm"] = lambda values, **_: values
    try:
        imputed = {
            split: impute_stays(
                irregular,
                ids,
                global_means,
                list(FEATURE_NAMES),
                HOURS,
            )
            for split, ids in split_ids.items()
        }
    finally:
        impute_stays.__globals__["tqdm"] = released_tqdm
    irregular_scaler = None
    if include_irregular_scaler:
        train_irregular = irregular.loc[
            irregular["stay_id"].isin(split_ids["train"]), list(FEATURE_NAMES)
        ]
        irregular_scaler = StandardScaler()
        irregular_scaler.fit(train_irregular)
    released_fit_and_apply_scaler = (
        los_fit_and_apply_scaler if task == "los" else fit_and_apply_scaler
    )
    imputed_scaler = released_fit_and_apply_scaler(
        imputed["train"], imputed["validation"], imputed["test"]
    )

    row_counts = irregular.groupby("stay_id", sort=False).size().to_dict()
    first_rows = (
        irregular.groupby("stay_id", sort=False)[["subject_id", "hadm_id"]]
        .first()
        .to_dict("index")
    )
    stay_lookup = {
        int(row.stay_id): row for row in stays.itertuples(index=False)
    }
    admissions = pd.read_csv(
        mimic_root / "hosp" / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "hospital_expire_flag"],
    )
    admission_lookup = {
        int(row.hadm_id): (int(row.subject_id), int(row.hospital_expire_flag))
        for row in admissions.itertuples(index=False)
    }

    records = []
    prefilter_records = []
    boundary_hashes = {}
    ordered_id_hashes = {}
    ordered_label_hashes = {}
    for split, ids in split_ids.items():
        boundary_digest = hashlib.sha256()
        id_digest = hashlib.sha256()
        label_digest = hashlib.sha256()
        for source_position, stay_id_value in enumerate(ids):
            stay_id = int(stay_id_value)
            stay = stay_lookup[stay_id]
            first = first_rows[stay_id]
            subject_id = int(first["subject_id"])
            hadm_id = int(first["hadm_id"])
            if subject_id != int(stay.subject_id) or hadm_id != int(stay.hadm_id):
                raise AssertionError("author pivot identifiers disagree with ICUSTAYS")
            admission_subject, died = admission_lookup[hadm_id]
            if admission_subject != subject_id:
                raise AssertionError("ADMISSIONS subject disagrees with author stay")
            # create_los_task.py::get_stay_list recomputes the ICU duration from
            # ICUSTAYS intime/outtime rather than reusing the `los` column.
            icu_hours = (stay.outtime - stay.intime).total_seconds() / 3600
            if task == "los":
                label = int((icu_hours < LOS_HOURS) and (died == 0))
            else:
                label = died
            base = {
                "split": split,
                "source_position": source_position,
                "stay_id": stay_id,
                "sample_id": stay_id,
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "intime": stay.intime,
                "label": label,
                "irregular_rows": int(row_counts[stay_id]),
            }
            prefilter_records.append(base)
            if base["irregular_rows"] <= 12:
                continue
            values = np.asarray(imputed[split][stay_id], dtype=np.float32)
            if values.shape != (HOURS, len(FEATURE_NAMES)):
                raise AssertionError("released imputed structured shape changed")
            if not np.isfinite(values).all():
                raise AssertionError("released imputed structured branch is nonfinite")
            record = {**base, "values": values}
            records.append(record)
            encoded_id = np.asarray([stay_id], dtype=np.int64).tobytes()
            encoded_label = np.asarray([label], dtype=np.int64).tobytes()
            id_digest.update(encoded_id)
            label_digest.update(encoded_label)
            boundary_digest.update(encoded_id)
            boundary_digest.update(encoded_label)
            boundary_digest.update(values.tobytes())
        boundary_hashes[split] = boundary_digest.hexdigest()
        ordered_id_hashes[split] = id_digest.hexdigest()
        ordered_label_hashes[split] = label_digest.hexdigest()

    prefilter_overlap = _overlap(prefilter_records, "subject_id")
    normalized_prefilter_overlap = {
        key.replace("subject_ids", "subjects"): value
        for key, value in prefilter_overlap.items()
    }
    if normalized_prefilter_overlap != EXPECTED_PREFILTER_OVERLAP:
        raise AssertionError(
            f"released patient overlap changed: {normalized_prefilter_overlap}"
        )

    return {
        "records": records,
        "prefilter_records": prefilter_records,
        "split_ids": split_ids,
        "cache_manifest": cache_manifest,
        "global_means": global_means,
        "irregular_scaler": irregular_scaler,
        "imputed_scaler": imputed_scaler,
        "boundary_sha256": boundary_hashes,
        "ordered_id_sha256": ordered_id_hashes,
        "ordered_label_sha256": ordered_label_hashes,
        "prefilter_subject_overlap": normalized_prefilter_overlap,
        "postfilter_subject_overlap": _overlap(records, "subject_id"),
        "postfilter_admission_overlap": _overlap(records, "hadm_id"),
    }


def samples_frame(records: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": pl.Series(
                [row["subject_id"] for row in records], dtype=pl.Int64
            ),
            "sample_id": pl.Series(
                [row["sample_id"] for row in records], dtype=pl.Int64
            ),
            "observation_start": pl.Series(
                [row["intime"] for row in records], dtype=pl.Datetime("us")
            ),
            "prediction_time": pl.Series(
                [row["intime"] + pd.Timedelta(hours=HOURS) for row in records],
                dtype=pl.Datetime("us"),
            ),
            "boolean_value": pl.Series(
                [bool(row["label"]) for row in records], dtype=pl.Boolean
            ),
            "split": pl.Series([row["split"] for row in records], dtype=pl.String),
            "stay_id": pl.Series(
                [row["stay_id"] for row in records], dtype=pl.Int64
            ),
            "hadm_id": pl.Series(
                [row["hadm_id"] for row in records], dtype=pl.Int64
            ),
            "source_position": pl.Series(
                [row["source_position"] for row in records], dtype=pl.Int64
            ),
            "irregular_rows": pl.Series(
                [row["irregular_rows"] for row in records], dtype=pl.Int64
            ),
        }
    )


def event_batches(
    records: list[dict],
    notes: pd.DataFrame,
    batch_stays: int = 256,
    batch_notes: int = 100_000,
) -> Iterator[pl.DataFrame]:
    """Yield the exact structured boundary and selected raw-note projection."""
    feature_count = len(FEATURE_NAMES)
    row_grid = np.arange(HOURS, dtype=np.int32)
    feature_grid = np.arange(feature_count, dtype=np.int64)
    for start in range(0, len(records), batch_stays):
        selected = records[start : start + batch_stays]
        subjects = []
        times = []
        codes = []
        values = []
        sample_ids = []
        source_rows = []
        sequence_rows = []
        source_times = []
        for record in selected:
            n_rows = HOURS * feature_count
            subjects.extend([record["subject_id"]] * n_rows)
            sample_ids.extend([record["sample_id"]] * n_rows)
            sequence_rows.extend(np.repeat(row_grid, feature_count).tolist())
            source_times.extend(np.repeat(row_grid.astype(np.float64), feature_count))
            codes.extend(list(FEATURE_NAMES) * HOURS)
            values.extend(record["values"].reshape(-1).tolist())
            start_time = np.datetime64(record["intime"], "us")
            hour_times = start_time + row_grid.astype("timedelta64[h]")
            times.extend(np.repeat(hour_times, feature_count).tolist())
            source_rows.extend(
                (
                    int(record["source_position"]) * HOURS * feature_count
                    + np.repeat(row_grid.astype(np.int64), feature_count) * feature_count
                    + np.tile(feature_grid, HOURS)
                ).tolist()
            )
        yield pl.DataFrame(
            {
                "subject_id": pl.Series(subjects, dtype=pl.Int64),
                "time": pl.Series(times, dtype=pl.Datetime("us")),
                "code": pl.Series(codes, dtype=pl.String),
                "numeric_value": pl.Series(values, dtype=pl.Float32),
                "sample_id": pl.Series(sample_ids, dtype=pl.Int64),
                "source_row": pl.Series(source_rows, dtype=pl.Int64),
                "sequence_row": pl.Series(sequence_rows, dtype=pl.Int32),
                "source_time": pl.Series(source_times, dtype=pl.Float64),
                "text_value": pl.Series([None] * len(values), dtype=pl.String),
            }
        )

    # The released stay loader takes note text and ICU-relative times in the
    # order produced by preprocess_notes.py.  ``note_position`` freezes that
    # order per stay; the common note view subsequently orders on source_row.
    for start in range(0, len(notes), batch_notes):
        selected = notes.iloc[start : start + batch_notes]
        if selected.empty:
            continue
        note_positions = selected["note_position"].to_numpy(np.int64)
        yield pl.DataFrame(
            {
                "subject_id": pl.Series(
                    selected["subject_id"].to_numpy(np.int64), dtype=pl.Int64
                ),
                "time": pl.Series(
                    selected["charttime"].to_numpy(dtype="datetime64[us]"),
                    dtype=pl.Datetime("us"),
                ),
                "code": pl.Series([NOTE_CODE] * len(selected), dtype=pl.String),
                "numeric_value": pl.Series(
                    [None] * len(selected), dtype=pl.Float32
                ),
                "sample_id": pl.Series(
                    selected["stay_id"].to_numpy(np.int64), dtype=pl.Int64
                ),
                "source_row": pl.Series(
                    NOTE_SOURCE_ROW_OFFSET + note_positions, dtype=pl.Int64
                ),
                "sequence_row": pl.Series(
                    np.floor(selected["icu_time_delta"].to_numpy(np.float64)).astype(
                        np.int32
                    ),
                    dtype=pl.Int32,
                ),
                "source_time": pl.Series(
                    selected["icu_time_delta"].to_numpy(np.float64),
                    dtype=pl.Float64,
                ),
                "text_value": pl.Series(
                    selected["text"].astype(str).tolist(), dtype=pl.String
                ),
            }
        )
