"""Bounded reconstruction of MUSE's released MIMIC-IV source boundary.

The released notebook builds very large in-memory CSV intermediates and the
released preprocessor then retains every raw lab tuple alongside dense,
forward-filled arrays.  That path exceeds this workspace's guarded memory
budget.  This adapter applies the same released operations in bounded stages,
retains the last deterministic author inputs, and caches only attributable
source facts under the ignored ``.export_assets`` directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
UPSTREAM = json.loads((PAPER_ROOT / "UPSTREAM.json").read_text())
UPSTREAM_COMMIT = UPSTREAM["repositories"][0]["commit"]

TASK_ID = "mortality_90d_post_discharge"
NOTE_CODE = "MUSE_NOTE//BRIEF_HOSPITAL_COURSE"
# Comparator is MUSE+ (paper Table 1, full-data 90-day mortality), the
# paper's final recommended model, per explicit 2026-08-31 adjudication;
# the plain MUSE row is 0.9004. See paper_audits/muse.md for the audited
# MUSE+ training-path defects that this comparison must disclose.
PUBLISHED_AUROC = 0.9201
MAX_STAY_MINUTES = 10 * 24 * 60
OBSERVATION_HOURS = 10 * 24
SPLIT_SEED = 42
AUTHOR_BATCH_SIZE = 128
MAX_SEQUENCE_LENGTH = 50
AUTHOR_MISSING_LAB_WIDTH = 114

SELECTED_LAB_LABELS = (
    "Hematocrit",
    "Platelet",
    "WBC",
    "Bilirubin",
    "pH",
    "Bicarbonate",
    "Creatinine",
    "Lactate",
    "Potassium",
    "Sodium",
)
TYPE_NAMES = ("diagnoses_icd", "procedures_icd", "prescriptions")
AGE_BUCKETS = ("10", "20", "30", "40", "50", "60", "70", "80")
GENDERS = ("F", "M")
ETHNICITIES = (
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
    "HISPANIC OR LATINO",
    "BLACK/AFRICAN",
    "BLACK/CAPE VERDEAN",
    "BLACK/CARIBBEAN ISLAND",
    "HISPANIC/LATINO - HONDURAN",
    "WHITE - OTHER EUROPEAN",
    "WHITE",
    "HISPANIC/LATINO - MEXICAN",
    "OTHER",
    "UNABLE TO OBTAIN",
    "WHITE - RUSSIAN",
    "HISPANIC/LATINO - COLUMBIAN",
    "UNKNOWN",
    "ASIAN - SOUTH EAST ASIAN",
    "ASIAN",
    "PATIENT DECLINED TO ANSWER",
    "HISPANIC/LATINO - CUBAN",
    "WHITE - BRAZILIAN",
    "ASIAN - ASIAN INDIAN",
    "PORTUGUESE",
    "ASIAN - KOREAN",
    "HISPANIC/LATINO - DOMINICAN",
    "HISPANIC/LATINO - SALVADORAN",
    "HISPANIC/LATINO - PUERTO RICAN",
    "SOUTH AMERICAN",
    "WHITE - EASTERN EUROPEAN",
    "ASIAN - CHINESE",
    "AMERICAN INDIAN/ALASKA NATIVE",
    "HISPANIC/LATINO - CENTRAL AMERICAN",
    "MULTIPLE RACE/ETHNICITY",
    "BLACK/AFRICAN AMERICAN",
    "HISPANIC/LATINO - GUATEMALAN",
)
ETHNICITY_GROUP = {
    "BLACK/CARIBBEAN ISLAND": "African American",
    "ASIAN": "Asian",
    "HISPANIC/LATINO - SALVADORAN": "Hispanic",
    "WHITE": "Caucasian",
    "AMERICAN INDIAN/ALASKA NATIVE": "Native American",
    "WHITE - EASTERN EUROPEAN": "Caucasian",
    "WHITE - RUSSIAN": "Caucasian",
    "OTHER": "Other",
    "HISPANIC/LATINO - COLUMBIAN": "Hispanic",
    "ASIAN - SOUTH EAST ASIAN": "Asian",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": "Native American",
    "BLACK/CAPE VERDEAN": "African American",
    "HISPANIC OR LATINO": "Hispanic",
    "HISPANIC/LATINO - DOMINICAN": "Hispanic",
    "HISPANIC/LATINO - PUERTO RICAN": "Hispanic",
    "BLACK/AFRICAN": "African American",
    "ASIAN - CHINESE": "Asian",
    "HISPANIC/LATINO - HONDURAN": "Hispanic",
    "ASIAN - KOREAN": "Asian",
    "ASIAN - ASIAN INDIAN": "Asian",
    "PATIENT DECLINED TO ANSWER": "Other",
    "HISPANIC/LATINO - CENTRAL AMERICAN": "Hispanic",
    "HISPANIC/LATINO - GUATEMALAN": "Hispanic",
    "HISPANIC/LATINO - MEXICAN": "Hispanic",
    "PORTUGUESE": "Caucasian",
    "MULTIPLE RACE/ETHNICITY": "Other",
    "WHITE - BRAZILIAN": "Caucasian",
    "WHITE - OTHER EUROPEAN": "Caucasian",
    "UNABLE TO OBTAIN": "Other",
    "HISPANIC/LATINO - CUBAN": "Hispanic",
    "SOUTH AMERICAN": "Hispanic",
    "BLACK/AFRICAN AMERICAN": "African American",
    "UNKNOWN": "Other",
}

COHORT_FILE = "cohort.parquet"
CODE_FILE = "code_trajectory.parquet"
LAB_FILE = "lab_boundary.parquet"
NOTE_FILE = "brief_hospital_course.parquet"
VOCAB_FILE = "author_code_vocabulary.json"
LAB_CHANNEL_FILE = "lab_channels.json"
SOURCE_MANIFEST_FILE = "source_manifest.json"
PAPER_NOTE_STATS_FILE = "paper_base_note_statistics.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream_distinct_hadm_counts(
    path: Path, group_column: str | None = None
) -> int | dict[str, int]:
    """Count ordered admission groups without materializing millions of IDs.

    Both retained source boundaries are ordered by cohort position, so an
    admission never reappears after the reader advances to the next one. The
    code boundary may interleave its three event types within an admission;
    retaining one last ID per type still counts each admission exactly once.
    """

    columns = ["hadm_id"]
    if group_column is not None:
        columns.append(group_column)
    last_hadm: int | None = None
    total = 0
    last_by_group: dict[str, int] = {}
    counts: dict[str, int] = {}
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=100_000, columns=columns
    ):
        values = batch.to_pydict()
        hadm_ids = values["hadm_id"]
        if group_column is None:
            for hadm_id in hadm_ids:
                hadm_id = int(hadm_id)
                if hadm_id != last_hadm:
                    total += 1
                    last_hadm = hadm_id
            continue
        for hadm_id, group in zip(hadm_ids, values[group_column]):
            hadm_id = int(hadm_id)
            group = str(group)
            if last_by_group.get(group) != hadm_id:
                counts[group] = counts.get(group, 0) + 1
                last_by_group[group] = hadm_id
    return counts if group_column is not None else total


def _stream_unique_hadm_count(path: Path) -> int:
    """Count admission IDs for a boundary whose physical order is not proven."""
    values: set[int] = set()
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=100_000, columns=["hadm_id"]
    ):
        values.update(int(value) for value in batch.column(0).to_pylist())
    return len(values)


def _stream_note_statistics(path: Path) -> dict[str, int | float]:
    parquet = pq.ParquetFile(path)
    rows = int(parquet.metadata.num_rows)
    empty = 0
    chart_after = 0
    stored_after = 0
    word_total = 0
    for batch in parquet.iter_batches(
        batch_size=5_000,
        columns=[
            "text",
            "chart_after_discharge",
            "stored_after_discharge",
        ],
    ):
        values = batch.to_pydict()
        for text in values["text"]:
            empty += int(text == "")
            word_total += len(text.split(" "))
        chart_after += sum(bool(value) for value in values["chart_after_discharge"])
        stored_after += sum(bool(value) for value in values["stored_after_discharge"])
    return {
        "rows": rows,
        "empty_but_present": empty,
        "chart_after_discharge": chart_after,
        "stored_after_discharge": stored_after,
        "average_words": word_total / rows if rows else float("nan"),
    }


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def ordered_text_hash(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def ordered_int_hash(values: Iterable[Any]) -> str:
    return hashlib.sha256(np.asarray(list(values), dtype=np.int64).tobytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _required_raw_files(mimic_root: Path, note_root: Path) -> dict[str, Path]:
    files = {
        "patients": mimic_root / "hosp" / "patients.csv.gz",
        "admissions": mimic_root / "hosp" / "admissions.csv.gz",
        "prescriptions": mimic_root / "hosp" / "prescriptions.csv.gz",
        "diagnoses_icd": mimic_root / "hosp" / "diagnoses_icd.csv.gz",
        "d_icd_diagnoses": mimic_root / "hosp" / "d_icd_diagnoses.csv.gz",
        "procedures_icd": mimic_root / "hosp" / "procedures_icd.csv.gz",
        "d_icd_procedures": mimic_root / "hosp" / "d_icd_procedures.csv.gz",
        "d_labitems": mimic_root / "hosp" / "d_labitems.csv.gz",
        "labevents": mimic_root / "hosp" / "labevents.csv.gz",
        "discharge": note_root / "note" / "discharge.csv.gz",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required exact source files are missing: {missing}")
    return files


def _build_cohort(raw: dict[str, Path], cache_root: Path) -> dict[str, Any]:
    output = cache_root / COHORT_FILE
    if output.is_file():
        frame = pd.read_parquet(output)
        return _cohort_summary(frame)

    patients = pd.read_csv(
        raw["patients"],
        usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
    )
    patients["dod"] = pd.to_datetime(patients["dod"])
    admissions = pd.read_csv(
        raw["admissions"],
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "hospital_expire_flag",
            "race",
        ],
    )
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
    frame = admissions.merge(patients, on="subject_id", how="inner")
    # This vectorized expression is algebraically identical to notebook cell 17.
    frame["age"] = (
        frame["admittime"].dt.year - frame["anchor_year"] + frame["anchor_age"]
    )
    frame["duration_minutes"] = (
        frame["dischtime"] - frame["admittime"]
    ).dt.total_seconds() / 60
    frame = frame.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    frame["readmit_in_days"] = (
        frame.groupby("subject_id")["admittime"].shift(-1) - frame["dischtime"]
    ).dt.days
    frame["die_in_days"] = (frame["dod"] - frame["dischtime"]).dt.days

    excluded_duration = int((frame["duration_minutes"] > MAX_STAY_MINUTES).sum())
    duration_ok = frame["duration_minutes"] <= MAX_STAY_MINUTES
    excluded_age = int(
        (duration_ok & ((frame["age"] < 18) | (frame["age"] > 89))).sum()
    )
    age_ok = frame["age"].between(18, 89)
    excluded_in_hospital = int(
        (duration_ok & age_ok & frame["hospital_expire_flag"].eq(1)).sum()
    )
    frame = frame.loc[
        duration_ok & age_ok & frame["hospital_expire_flag"].eq(0)
    ].copy()
    frame["cohort_position"] = np.arange(len(frame), dtype=np.int64)

    mortality = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    death_known = frame["die_in_days"].notna()
    mortality.loc[death_known] = frame.loc[death_known, "die_in_days"].le(90)
    known_negative = (
        ~death_known
        & frame["readmit_in_days"].notna()
        & frame["readmit_in_days"].gt(90)
    )
    mortality.loc[known_negative] = False
    frame["mortality"] = mortality

    labeled_ids = frame.loc[frame["mortality"].notna(), "hadm_id"].astype(str).tolist()
    generator = random.Random(SPLIT_SEED)
    generator.shuffle(labeled_ids)
    first = int(len(labeled_ids) * 0.7)
    second = int(len(labeled_ids) * 0.8)
    split_ids = {
        "train": labeled_ids[:first],
        "validation": labeled_ids[first:second],
        "test": labeled_ids[second:],
    }
    split_lookup = {
        int(hadm_id): (split, position)
        for split, values in split_ids.items()
        for position, hadm_id in enumerate(values)
    }
    frame["split"] = frame["hadm_id"].map(
        lambda value: split_lookup.get(int(value), (None, None))[0]
    )
    frame["split_position"] = pd.array(
        frame["hadm_id"].map(
            lambda value: split_lookup.get(int(value), (None, None))[1]
        ),
        dtype="Int64",
    )
    frame["age_bucket"] = (frame["age"] // 10 * 10).astype(str)
    # The release keeps nonpositive stays because it checks only duration>10d.
    # MEDS requires observation_start<prediction_time, so retain the exact
    # discharge cutoff and use one microsecond of empty schema window for those
    # defective rows. Original admittime and duration remain audit columns.
    frame["meds_observation_start"] = frame["admittime"]
    nonpositive = frame["admittime"] >= frame["dischtime"]
    frame.loc[nonpositive, "meds_observation_start"] = (
        frame.loc[nonpositive, "dischtime"] - pd.Timedelta(microseconds=1)
    )
    if not set(frame["race"]).issubset(set(ETHNICITIES)):
        raise AssertionError("released race value falls outside the hard-coded tokenizer")
    if not set(frame["gender"]).issubset(set(GENDERS)):
        raise AssertionError("released gender value falls outside the hard-coded tokenizer")

    keep = [
        "subject_id",
        "hadm_id",
        "admittime",
        "meds_observation_start",
        "dischtime",
        "duration_minutes",
        "age",
        "age_bucket",
        "gender",
        "race",
        "mortality",
        "readmit_in_days",
        "die_in_days",
        "cohort_position",
        "split",
        "split_position",
    ]
    frame[keep].to_parquet(output, index=False, compression="zstd")
    for split, values in split_ids.items():
        filename = "val_admission_ids.txt" if split == "validation" else f"{split}_admission_ids.txt"
        (cache_root / filename).write_text("".join(f"{value}\n" for value in values))
    no_label = frame.loc[frame["mortality"].isna(), "hadm_id"].astype(str)
    (cache_root / "no_label_admission_ids.txt").write_text(
        "".join(f"{value}\n" for value in no_label)
    )
    return {
        **_cohort_summary(frame[keep]),
        "excluded": {
            "duration_over_10_days": excluded_duration,
            "age_outside_18_89_after_duration": excluded_age,
            "in_hospital_death_after_duration_age": excluded_in_hospital,
        },
    }


def _cohort_summary(frame: pd.DataFrame) -> dict[str, Any]:
    labeled = frame.loc[frame["mortality"].notna()]
    result = {
        "admissions": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "labeled": int(len(labeled)),
        "missing_label": int(frame["mortality"].isna().sum()),
        "positive": int(labeled["mortality"].sum()),
        "mean_age": float(frame["age"].mean()),
        "nonpositive_duration": int(frame["duration_minutes"].le(0).sum()),
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        selected = labeled.loc[labeled["split"].eq(split)]
        result["splits"][split] = {
            "count": int(len(selected)),
            "subjects": int(selected["subject_id"].nunique()),
            "positive": int(selected["mortality"].sum()),
        }
    return result


def _collect_code_table(
    query: pl.LazyFrame,
    output: Path,
    sort_columns: list[str],
) -> None:
    if output.is_file():
        return
    frame = query.collect(streaming=True).sort(sort_columns)
    frame.write_parquet(output, compression="zstd", row_group_size=250_000)


def _build_prescription_table(
    raw: dict[str, Path], cache_root: Path, output: Path
) -> Path:
    """Execute notebook cells 25--31 without loading all prescriptions."""
    if output.is_file():
        return output
    cohort = pd.read_parquet(
        cache_root / COHORT_FILE,
        columns=["hadm_id", "admittime", "duration_minutes", "cohort_position"],
    ).set_index("hadm_id")
    temporary = cache_root / "prescriptions_selected_chunks_v2.parquet"
    if not temporary.is_file():
        writer = None
        try:
            for chunk in pd.read_csv(
                raw["prescriptions"],
                usecols=["subject_id", "hadm_id", "starttime", "drug"],
                dtype={"drug": str},
                chunksize=1_000_000,
            ):
                chunk["admittime"] = chunk["hadm_id"].map(cohort["admittime"])
                chunk["duration_minutes"] = chunk["hadm_id"].map(
                    cohort["duration_minutes"]
                )
                chunk["cohort_position"] = chunk["hadm_id"].map(
                    cohort["cohort_position"]
                )
                chunk = chunk.dropna()
                if chunk.empty:
                    continue
                chunk["starttime"] = pd.to_datetime(chunk["starttime"])
                chunk["source_time"] = (
                    chunk["starttime"] - chunk["admittime"]
                ).dt.total_seconds() / 60
                chunk = chunk.drop_duplicates()
                timestamp_int = np.trunc(chunk["source_time"]).astype(np.int64)
                chunk = chunk.loc[timestamp_int <= chunk["duration_minutes"]]
                if chunk.empty:
                    continue
                result = pd.DataFrame(
                    {
                        "hadm_id": chunk["hadm_id"].astype(np.int64),
                        "cohort_position": chunk["cohort_position"].astype(np.int64),
                        "kind_order": np.ones(len(chunk), dtype=np.int8),
                        "source_time": chunk["source_time"].astype(np.float64),
                        "event_type": "prescriptions",
                        "code_name": chunk["drug"].str.lower(),
                        "source_order": np.zeros(len(chunk), dtype=np.int64),
                        "dedup_code_name": chunk["drug"].astype(str),
                    }
                )
                table = pa.Table.from_pandas(result, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression="zstd"
                    )
                writer.write_table(table, row_group_size=250_000)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise AssertionError("released prescription path produced no rows")
    return temporary


class _CodeBoundaryWriter:
    def __init__(self, path: Path):
        self.path = path
        self.writer = None
        self.values: dict[str, list[Any]] = {
            "hadm_id": [],
            "cohort_position": [],
            "event_type": [],
            "code_name": [],
            "sequence_row": [],
            "trajectory_length_before_cut": [],
        }
        self.rows = 0

    def append(self, row: tuple[Any, ...], sequence_row: int, length: int) -> None:
        values = (int(row[0]), int(row[1]), str(row[4]), str(row[5]), sequence_row, length)
        for name, value in zip(self.values, values):
            self.values[name].append(value)
        if len(self.values["hadm_id"]) >= 250_000:
            self.flush()

    def flush(self) -> None:
        if not self.values["hadm_id"]:
            return
        table = pa.table(
            {
                "hadm_id": pa.array(self.values["hadm_id"], type=pa.int64()),
                "cohort_position": pa.array(
                    self.values["cohort_position"], type=pa.int64()
                ),
                "event_type": pa.array(self.values["event_type"], type=pa.string()),
                "code_name": pa.array(self.values["code_name"], type=pa.string()),
                "sequence_row": pa.array(
                    self.values["sequence_row"], type=pa.int32()
                ),
                "trajectory_length_before_cut": pa.array(
                    self.values["trajectory_length_before_cut"], type=pa.int32()
                ),
            }
        )
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table, row_group_size=250_000)
        self.rows += len(table)
        for values in self.values.values():
            values.clear()

    def close(self) -> int:
        self.flush()
        if self.writer is not None:
            self.writer.close()
        return self.rows


def _sqlite_insert_parquet(
    connection: sqlite3.Connection, table: str, path: Path, or_ignore: bool
) -> None:
    command = "INSERT OR IGNORE" if or_ignore else "INSERT"
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=100_000):
        columns = pa.Table.from_batches([batch]).to_pydict()
        names = [
            "hadm_id",
            "cohort_position",
            "kind_order",
            "source_time",
            "event_type",
            "code_name",
            "source_order",
        ]
        if table == "prescriptions":
            names.append("dedup_code_name")
        sql = f"{command} INTO {table} VALUES ({','.join('?' for _ in names)})"
        rows = zip(*(columns[name] for name in names))
        connection.executemany(sql, rows)
        connection.commit()


def _merge_code_tables_bounded(
    paths: list[Path], prescription_path: Path, output: Path
) -> dict[str, Any]:
    database = output.with_suffix(".sqlite")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-131072")
    schema = """(
        hadm_id INTEGER NOT NULL,
        cohort_position INTEGER NOT NULL,
        kind_order INTEGER NOT NULL,
        source_time REAL NOT NULL,
        event_type TEXT NOT NULL,
        code_name TEXT NOT NULL,
        source_order INTEGER NOT NULL
    )"""
    connection.execute(f"CREATE TABLE events {schema}")
    connection.execute(
        "CREATE TABLE prescriptions ("
        "hadm_id INTEGER NOT NULL, cohort_position INTEGER NOT NULL, "
        "kind_order INTEGER NOT NULL, source_time REAL NOT NULL, "
        "event_type TEXT NOT NULL, code_name TEXT NOT NULL, "
        "source_order INTEGER NOT NULL, dedup_code_name TEXT NOT NULL, "
        "UNIQUE(hadm_id, cohort_position, kind_order, source_time, "
        "event_type, dedup_code_name, source_order))"
    )
    for path in paths:
        _sqlite_insert_parquet(connection, "events", path, False)
    _sqlite_insert_parquet(connection, "prescriptions", prescription_path, True)
    connection.execute(
        "INSERT INTO events SELECT hadm_id, cohort_position, kind_order, "
        "source_time, event_type, code_name, source_order FROM prescriptions"
    )
    connection.execute("DROP TABLE prescriptions")
    connection.commit()
    connection.execute(
        "CREATE INDEX event_order ON events "
        "(cohort_position, kind_order, source_time, event_type, code_name, source_order)"
    )
    connection.commit()

    cursor = connection.execute(
        "SELECT hadm_id, cohort_position, kind_order, source_time, event_type, "
        "code_name, source_order FROM events ORDER BY cohort_position, kind_order, "
        "source_time, event_type, code_name, source_order"
    )
    writer = _CodeBoundaryWriter(output)
    current_hadm = None
    retained: deque[tuple[Any, ...]] = deque(maxlen=MAX_SEQUENCE_LENGTH)
    length = 0
    cut_admissions = 0
    names: set[str] = set()

    def flush_group() -> None:
        nonlocal cut_admissions
        if current_hadm is None:
            return
        cut_admissions += int(length > MAX_SEQUENCE_LENGTH)
        for sequence_row, row in enumerate(retained):
            writer.append(row, sequence_row, length)
            names.add(str(row[5]))

    for row in cursor:
        hadm_id = int(row[0])
        if current_hadm is not None and hadm_id != current_hadm:
            flush_group()
            retained.clear()
            length = 0
        current_hadm = hadm_id
        retained.append(row)
        length += 1
    flush_group()
    row_count = writer.close()
    connection.close()
    database.unlink()
    return {
        "rows": row_count,
        "cut_admissions": cut_admissions,
        "names": names,
    }


def _build_codes(raw: dict[str, Path], cache_root: Path) -> dict[str, Any]:
    output = cache_root / CODE_FILE
    vocabulary_path = cache_root / VOCAB_FILE
    if output.is_file() and vocabulary_path.is_file():
        component_admissions = {
            "diagnoses_icd": _stream_distinct_hadm_counts(
                cache_root / "diagnoses_boundary.parquet"
            ),
            "procedures_icd": _stream_distinct_hadm_counts(
                cache_root / "procedures_boundary.parquet"
            ),
            "prescriptions": _stream_unique_hadm_count(
                cache_root / "prescriptions_selected_chunks_v2.parquet"
            ),
        }
        return {
            "rows": int(pq.ParquetFile(output).metadata.num_rows),
            "admissions_with_codes": _stream_distinct_hadm_counts(output),
            "component_admissions": component_admissions,
            "author_vocabulary_size": len(json.loads(vocabulary_path.read_text())),
        }

    cohort = pl.scan_parquet(cache_root / COHORT_FILE).select(
        "hadm_id", "admittime", "duration_minutes", "cohort_position"
    )
    diagnoses_dictionary = pl.scan_csv(
        raw["d_icd_diagnoses"], schema_overrides={"icd_code": pl.String}
    ).select(
        "icd_code", "icd_version", "long_title"
    )
    diagnoses = (
        pl.scan_csv(
            raw["diagnoses_icd"], schema_overrides={"icd_code": pl.String}
        )
        .join(diagnoses_dictionary, on=["icd_code", "icd_version"], how="inner")
        .drop_nulls()
        .unique()
        .join(cohort, on="hadm_id", how="inner")
        .filter(pl.col("duration_minutes") >= 0)
        .select(
            "hadm_id",
            "cohort_position",
            pl.lit(0, dtype=pl.Int8).alias("kind_order"),
            pl.lit(0.0, dtype=pl.Float64).alias("source_time"),
            pl.lit("diagnoses_icd").alias("event_type"),
            pl.col("long_title").str.to_lowercase().alias("code_name"),
            pl.col("seq_num").cast(pl.Int64).alias("source_order"),
        )
    )
    diagnoses_path = cache_root / "diagnoses_boundary.parquet"
    _collect_code_table(
        diagnoses,
        diagnoses_path,
        ["cohort_position", "source_order", "code_name"],
    )

    procedure_dictionary = pl.scan_csv(
        raw["d_icd_procedures"], schema_overrides={"icd_code": pl.String}
    ).select(
        "icd_code", "icd_version", "long_title"
    )
    procedures = (
        pl.scan_csv(
            raw["procedures_icd"],
            try_parse_dates=True,
            schema_overrides={"icd_code": pl.String},
        )
        .join(procedure_dictionary, on=["icd_code", "icd_version"], how="inner")
        .drop_nulls()
        .unique()
        .join(cohort, on="hadm_id", how="inner")
        .with_columns(
            (
                (pl.col("chartdate") - pl.col("admittime")).dt.total_milliseconds()
                / 60_000
            ).alias("source_time")
        )
        .filter(pl.col("source_time").cast(pl.Int64) <= pl.col("duration_minutes"))
        .select(
            "hadm_id",
            "cohort_position",
            pl.lit(1, dtype=pl.Int8).alias("kind_order"),
            pl.col("source_time").cast(pl.Float64),
            pl.lit("procedures_icd").alias("event_type"),
            pl.col("long_title").str.to_lowercase().alias("code_name"),
            pl.col("seq_num").cast(pl.Int64).alias("source_order"),
        )
    )
    procedures_path = cache_root / "procedures_boundary.parquet"
    _collect_code_table(
        procedures,
        procedures_path,
        [
            "cohort_position",
            "source_time",
            "event_type",
            "code_name",
            "source_order",
        ],
    )

    prescriptions_path = cache_root / "prescriptions_boundary.parquet"
    prescription_chunks = _build_prescription_table(raw, cache_root, prescriptions_path)
    merged = _merge_code_tables_bounded(
        [diagnoses_path, procedures_path], prescription_chunks, output
    )

    # The release calls list(set(all_codes)).  PYTHONHASHSEED=42 is set by the
    # launcher before interpreter startup, which makes the otherwise-unreleased
    # artifact deterministic under the released Python-3.8 protocol.
    author_vocabulary = list(set(merged["names"]))
    _write_json(vocabulary_path, author_vocabulary)
    component_admissions = {
        "diagnoses_icd": _stream_distinct_hadm_counts(diagnoses_path),
        "procedures_icd": _stream_distinct_hadm_counts(procedures_path),
        "prescriptions": _stream_unique_hadm_count(prescription_chunks),
    }
    frame = pl.scan_parquet(output)
    return {
        "rows": merged["rows"],
        "admissions_with_codes": _stream_distinct_hadm_counts(output),
        "component_admissions": component_admissions,
        "cut_admissions": int(frame.filter(
            pl.col("trajectory_length_before_cut") > MAX_SEQUENCE_LENGTH
        ).select(pl.col("hadm_id").n_unique()).collect().item()),
        "author_vocabulary_size": len(author_vocabulary),
    }


def _selected_lab_item_ids(raw: dict[str, Path]) -> tuple[list[int], int]:
    dictionary = pd.read_csv(raw["d_labitems"])
    dictionary = dictionary.dropna(subset=["label"])
    selected = dictionary.loc[
        dictionary["label"].apply(
            lambda value: any(label in value for label in SELECTED_LAB_LABELS)
        )
    ]
    return selected["itemid"].astype(int).tolist(), len(selected)


def _build_lab_input(raw: dict[str, Path], cache_root: Path) -> Path:
    output = cache_root / "lab_notebook_boundary.parquet"
    if output.is_file():
        return output
    selected_ids, _ = _selected_lab_item_ids(raw)
    admissions = pd.read_csv(
        raw["admissions"], usecols=["hadm_id", "admittime"]
    ).set_index("hadm_id")["admittime"]
    admissions = pd.to_datetime(admissions)
    temporary = cache_root / "lab_selected_chunks.parquet"
    writer = None
    try:
        for chunk in pd.read_csv(
            raw["labevents"],
            usecols=[
                "subject_id",
                "hadm_id",
                "specimen_id",
                "storetime",
                "itemid",
                "valuenum",
                "flag",
            ],
            chunksize=2_000_000,
        ):
            chunk = chunk.loc[chunk["itemid"].isin(selected_ids)].copy()
            if chunk.empty:
                continue
            chunk["admittime"] = chunk["hadm_id"].map(admissions)
            chunk["storetime"] = pd.to_datetime(chunk["storetime"])
            chunk["timestamp"] = (
                chunk["storetime"] - chunk["admittime"]
            ).dt.total_seconds() / 60
            chunk = chunk[
                ["subject_id", "hadm_id", "timestamp", "itemid", "valuenum", "flag"]
            ].dropna(subset=["subject_id", "hadm_id", "timestamp", "itemid", "valuenum"])
            chunk = chunk.drop_duplicates()
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=250_000)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise AssertionError("released selected lab path produced no rows")

    frame = pd.read_parquet(temporary)
    frame = frame.drop_duplicates()
    frame = frame.sort_values(
        ["subject_id", "hadm_id", "timestamp", "itemid"]
    ).reset_index(drop=True)
    frame["lab_source_row"] = np.arange(len(frame), dtype=np.int64)
    frame["normalized_valuenum"] = frame.groupby("itemid")["valuenum"].transform(
        lambda values: (values - values.mean()) / values.std()
    )
    frame = frame.dropna(subset=["normalized_valuenum"])
    frame.to_parquet(output, index=False, compression="zstd")
    temporary.unlink()
    return output


class _LabBoundaryWriter:
    def __init__(self, path: Path):
        self.path = path
        self.writer = None
        self.columns: dict[str, list[Any]] = {
            "hadm_id": [],
            "cohort_position": [],
            "sequence_row": [],
            "source_time": [],
            "channel_index": [],
            "numeric_value": [],
        }
        self.count = 0

    def append(
        self,
        hadm_id: int,
        cohort_position: int,
        sequence_row: int,
        source_time: int,
        channel_index: int,
        numeric_value: float | None,
    ) -> None:
        values = (
            hadm_id,
            cohort_position,
            sequence_row,
            source_time,
            channel_index,
            numeric_value,
        )
        for name, value in zip(self.columns, values):
            self.columns[name].append(value)
        if len(self.columns["hadm_id"]) >= 250_000:
            self.flush()

    def flush(self) -> None:
        if not self.columns["hadm_id"]:
            return
        table = pa.table(
            {
                "hadm_id": pa.array(self.columns["hadm_id"], type=pa.int64()),
                "cohort_position": pa.array(
                    self.columns["cohort_position"], type=pa.int64()
                ),
                "sequence_row": pa.array(
                    self.columns["sequence_row"], type=pa.int32()
                ),
                "source_time": pa.array(self.columns["source_time"], type=pa.int32()),
                "channel_index": pa.array(
                    self.columns["channel_index"], type=pa.int16()
                ),
                "numeric_value": pa.array(
                    self.columns["numeric_value"], type=pa.float32()
                ),
            }
        )
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table, row_group_size=250_000)
        self.count += len(table)
        for values in self.columns.values():
            values.clear()

    def close(self) -> int:
        self.flush()
        if self.writer is not None:
            self.writer.close()
        return self.count


def _write_admission_lab_boundary(
    writer: _LabBoundaryWriter,
    hadm_id: int,
    cohort_position: int,
    rows: pd.DataFrame,
    item_to_channel: dict[int, int],
) -> tuple[int, bool]:
    vector = np.zeros(len(item_to_channel) * 2, dtype=np.float64)
    retained: deque[tuple[int, np.ndarray]] = deque(maxlen=MAX_SEQUENCE_LENGTH)
    for timestamp, group in rows.groupby("timestamp_int", sort=False):
        for row in group.itertuples(index=False):
            channel = item_to_channel[int(row.itemid)]
            vector[channel] = float(row.normalized_valuenum)
            vector[channel + 1] = 1.0 if row.flag == "abnormal" else 0.0
        retained.append((int(timestamp), vector.copy()))
    previous = np.zeros_like(vector, dtype=np.float32)
    for sequence_row, (timestamp, current64) in enumerate(retained):
        current = current64.astype(np.float32)
        # A marker preserves repeated all-unchanged author rows and exact length.
        writer.append(
            hadm_id, cohort_position, sequence_row, timestamp, -1, None
        )
        changed = np.flatnonzero(current != previous)
        for channel in changed:
            writer.append(
                hadm_id,
                cohort_position,
                sequence_row,
                timestamp,
                int(channel),
                float(current[channel]),
            )
        previous = current
    return len(retained), len(rows["timestamp_int"].unique()) > MAX_SEQUENCE_LENGTH


def _build_labs(raw: dict[str, Path], cache_root: Path) -> dict[str, Any]:
    output = cache_root / LAB_FILE
    channels_path = cache_root / LAB_CHANNEL_FILE
    if output.is_file() and channels_path.is_file():
        channels = json.loads(channels_path.read_text())
        parquet = pq.ParquetFile(output)
        return {
            "selected_dictionary_items": channels["selected_dictionary_items"],
            "retained_items": len(channels["itemids"]),
            "author_width": 2 * len(channels["itemids"]),
            "rows": parquet.metadata.num_rows,
            "admissions_with_labs": _stream_distinct_hadm_counts(output),
        }

    notebook_boundary = _build_lab_input(raw, cache_root)
    cohort = pd.read_parquet(
        cache_root / COHORT_FILE,
        columns=["hadm_id", "duration_minutes", "cohort_position"],
    )
    duration_by_hadm = dict(
        zip(cohort["hadm_id"].astype(int), cohort["duration_minutes"].astype(float))
    )
    position_by_hadm = dict(
        zip(cohort["hadm_id"].astype(int), cohort["cohort_position"].astype(int))
    )
    database = cache_root / "lab_boundary.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-131072")
    connection.execute(
        "CREATE TABLE rows (hadm_id INTEGER NOT NULL, cohort_position INTEGER NOT NULL, "
        "lab_source_row INTEGER NOT NULL, timestamp_int INTEGER NOT NULL, "
        "itemid INTEGER NOT NULL, normalized_value REAL NOT NULL, flag TEXT)"
    )
    parquet = pq.ParquetFile(notebook_boundary)
    input_rows = 0
    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=[
            "hadm_id",
            "timestamp",
            "itemid",
            "flag",
            "lab_source_row",
            "normalized_valuenum",
        ],
    ):
        part = pa.Table.from_batches([batch]).to_pandas()
        part["hadm_id"] = part["hadm_id"].astype(np.int64)
        part["cohort_position"] = part["hadm_id"].map(position_by_hadm)
        part["duration_minutes"] = part["hadm_id"].map(duration_by_hadm)
        part = part.loc[part["cohort_position"].notna()]
        if part.empty:
            continue
        part["timestamp_int"] = np.trunc(part["timestamp"]).astype(np.int64)
        part = part.loc[part["timestamp_int"] <= part["duration_minutes"]]
        if part.empty:
            continue
        values = zip(
            part["hadm_id"].astype(int),
            part["cohort_position"].astype(int),
            part["lab_source_row"].astype(int),
            part["timestamp_int"].astype(int),
            part["itemid"].astype(int),
            part["normalized_valuenum"].astype(float),
            part["flag"].where(part["flag"].notna(), None),
        )
        connection.executemany("INSERT INTO rows VALUES (?,?,?,?,?,?,?)", values)
        connection.commit()
        input_rows += len(part)
    connection.execute(
        "CREATE INDEX lab_order ON rows (cohort_position, lab_source_row)"
    )
    connection.commit()

    itemids = []
    seen_items: set[int] = set()
    for (itemid,) in connection.execute(
        "SELECT itemid FROM rows ORDER BY cohort_position, lab_source_row"
    ):
        itemid = int(itemid)
        if itemid not in seen_items:
            seen_items.add(itemid)
            itemids.append(itemid)
    item_to_channel = {itemid: position * 2 for position, itemid in enumerate(itemids)}
    selected_ids, selected_dictionary_items = _selected_lab_item_ids(raw)
    if not set(itemids).issubset(set(selected_ids)):
        raise AssertionError("retained lab channel was not selected by the notebook")
    _write_json(
        channels_path,
        {
            "selected_dictionary_items": selected_dictionary_items,
            "itemids": itemids,
            "layout": "alternating normalized value / abnormal-result flag",
            "normalization": "whole-notebook-table pandas sample std (ddof=1)",
        },
    )

    boundary_writer = _LabBoundaryWriter(output)
    admissions_with_labs = 0
    cut_admissions = 0
    sequence_rows = 0
    current_hadm: int | None = None
    current_position = -1
    current_timestamp: int | None = None
    vector = np.zeros(len(itemids) * 2, dtype=np.float64)
    retained: deque[tuple[int, np.ndarray]] = deque(maxlen=MAX_SEQUENCE_LENGTH)
    original_sequence_rows = 0

    def finish_timestamp() -> None:
        nonlocal original_sequence_rows
        if current_timestamp is not None:
            retained.append((current_timestamp, vector.copy()))
            original_sequence_rows += 1

    def finish_admission() -> None:
        nonlocal admissions_with_labs, cut_admissions, sequence_rows
        if current_hadm is None:
            return
        previous = np.zeros(len(itemids) * 2, dtype=np.float32)
        for sequence_row, (timestamp, current64) in enumerate(retained):
            current = current64.astype(np.float32)
            boundary_writer.append(
                current_hadm,
                current_position,
                sequence_row,
                timestamp,
                -1,
                None,
            )
            for channel in np.flatnonzero(current != previous):
                boundary_writer.append(
                    current_hadm,
                    current_position,
                    sequence_row,
                    timestamp,
                    int(channel),
                    float(current[channel]),
                )
            previous = current
        admissions_with_labs += 1
        cut_admissions += int(original_sequence_rows > MAX_SEQUENCE_LENGTH)
        sequence_rows += len(retained)

    query = connection.execute(
        "SELECT hadm_id, cohort_position, timestamp_int, itemid, "
        "normalized_value, flag FROM rows ORDER BY cohort_position, lab_source_row"
    )
    for hadm_id, cohort_position, timestamp, itemid, normalized, flag in query:
        hadm_id = int(hadm_id)
        timestamp = int(timestamp)
        if current_hadm is not None and hadm_id != current_hadm:
            finish_timestamp()
            finish_admission()
            vector.fill(0)
            retained.clear()
            original_sequence_rows = 0
            current_timestamp = None
        if current_timestamp is not None and timestamp != current_timestamp:
            finish_timestamp()
        current_hadm = hadm_id
        current_position = int(cohort_position)
        current_timestamp = timestamp
        channel = item_to_channel[int(itemid)]
        vector[channel] = float(normalized)
        vector[channel + 1] = 1.0 if flag == "abnormal" else 0.0
    finish_timestamp()
    finish_admission()
    output_rows = boundary_writer.close()
    connection.close()
    database.unlink()
    return {
        "selected_dictionary_items": selected_dictionary_items,
        "retained_items": len(itemids),
        "author_width": 2 * len(itemids),
        "rows": output_rows,
        "retained_source_rows": input_rows,
        "admissions_with_labs": admissions_with_labs,
        "sequence_rows": sequence_rows,
        "cut_admissions": cut_admissions,
    }


def extract_brief_hospital_course(note: str) -> str | float:
    """Released notebook cell 72, kept byte-for-byte in behavior."""
    start_index = note.find("Brief Hospital Course:")
    end_index = note.find("Medications on Admission:")
    if start_index != -1 and end_index != -1:
        instructions = note[
            start_index + len("Brief Hospital Course:") : end_index
        ].strip()
    else:
        return float("nan")
    return instructions.replace("___", "").replace("\n", " ").strip()


def extract_discharge_instructions(note: str) -> str | float:
    """Historical base-notebook section used by the paper statistic path."""
    start_index = note.find("Discharge Instructions:")
    end_index = note.find("Followup Instructions:")
    if start_index != -1 and end_index != -1:
        instructions = note[
            start_index + len("Discharge Instructions:") : end_index
        ].strip()
    else:
        return float("nan")
    return instructions.replace("___", "").replace("\n", " ").strip()


def _build_paper_base_note_statistics(
    raw: dict[str, Path], cache_root: Path
) -> dict[str, Any]:
    """Recreate Table 4's base-pickle note marginal without retaining text.

    The selected Dataset consumes the v2 Brief Hospital Course producer, but
    the recovered ``stat_check_mimic4.py`` opens the base pickle, whose parser
    used Discharge Instructions. Only count and identifier-hash evidence from
    that auxiliary statistics path is persisted.
    """
    output = cache_root / PAPER_NOTE_STATS_FILE
    if output.is_file():
        return json.loads(output.read_text())
    cohort_ids = set(
        pd.read_parquet(cache_root / COHORT_FILE, columns=["hadm_id"])[
            "hadm_id"
        ].astype(int)
    )
    present_ids: set[int] = set()
    extracted_rows = 0
    for chunk in pd.read_csv(
        raw["discharge"], usecols=["hadm_id", "text"], chunksize=5_000
    ):
        extracted = chunk["text"].apply(extract_discharge_instructions)
        for hadm_id, text in zip(chunk["hadm_id"], extracted):
            if isinstance(text, float) and math.isnan(text):
                continue
            extracted_rows += 1
            if not pd.isna(hadm_id) and int(hadm_id) in cohort_ids:
                present_ids.add(int(hadm_id))
    ordered_ids = sorted(present_ids)
    result = {
        "producer": "historical base preprocess_mimic4.py",
        "section": "Discharge Instructions",
        "extracted_rows_before_cohort": extracted_rows,
        "cohort_present": len(ordered_ids),
        "cohort_missing": len(cohort_ids) - len(ordered_ids),
        "cohort_missing_ratio": 1 - len(ordered_ids) / len(cohort_ids),
        "ordered_present_hadm_sha256": ordered_int_hash(ordered_ids),
    }
    _write_json(output, result)
    return result


def _build_notes(raw: dict[str, Path], cache_root: Path) -> dict[str, Any]:
    output = cache_root / NOTE_FILE
    if output.is_file():
        summary = _stream_note_statistics(output)
        summary.pop("average_words")
        return summary

    last_by_admission: dict[int, tuple[Any, ...]] = {}
    source_row = 0
    extracted_rows = 0
    for chunk in pd.read_csv(
        raw["discharge"],
        usecols=["subject_id", "hadm_id", "charttime", "storetime", "text"],
        chunksize=5_000,
    ):
        extracted = chunk["text"].apply(extract_brief_hospital_course)
        for offset, (row, text) in enumerate(zip(chunk.itertuples(index=False), extracted)):
            if isinstance(text, float) and math.isnan(text):
                continue
            if pd.isna(row.hadm_id):
                continue
            extracted_rows += 1
            last_by_admission[int(row.hadm_id)] = (
                int(row.subject_id),
                int(row.hadm_id),
                source_row + offset,
                row.charttime,
                row.storetime,
                str(text),
            )
        source_row += len(chunk)
    notes = pd.DataFrame(
        list(last_by_admission.values()),
        columns=[
            "note_subject_id",
            "hadm_id",
            "note_source_row",
            "charttime",
            "storetime",
            "text",
        ],
    )
    cohort = pd.read_parquet(
        cache_root / COHORT_FILE,
        columns=["subject_id", "hadm_id", "dischtime", "cohort_position"],
    )
    notes = notes.merge(cohort, on="hadm_id", how="inner", validate="1:1")
    if not np.array_equal(notes["note_subject_id"], notes["subject_id"]):
        raise AssertionError("discharge note subject disagrees with admission subject")
    notes["charttime"] = pd.to_datetime(notes["charttime"])
    notes["storetime"] = pd.to_datetime(notes["storetime"])
    notes["chart_after_discharge"] = notes["charttime"] > notes["dischtime"]
    notes["stored_after_discharge"] = notes["storetime"] > notes["dischtime"]
    notes = notes.sort_values("cohort_position").reset_index(drop=True)
    notes[
        [
            "subject_id",
            "hadm_id",
            "cohort_position",
            "note_source_row",
            "charttime",
            "storetime",
            "dischtime",
            "chart_after_discharge",
            "stored_after_discharge",
            "text",
        ]
    ].to_parquet(output, index=False, compression="zstd")
    return {
        "extracted_rows_before_last_write_wins": extracted_rows,
        "rows": int(len(notes)),
        "empty_but_present": int(notes["text"].eq("").sum()),
        "chart_after_discharge": int(notes["chart_after_discharge"].sum()),
        "stored_after_discharge": int(notes["stored_after_discharge"].sum()),
    }


def _source_statistics(cache_root: Path) -> dict[str, Any]:
    cohort = pd.read_parquet(cache_root / COHORT_FILE)
    note_statistics = _stream_note_statistics(cache_root / NOTE_FILE)
    code_counts = {
        "diagnoses_icd": int(
            _stream_distinct_hadm_counts(
                cache_root / "diagnoses_boundary.parquet"
            )
        ),
        "procedures_icd": int(
            _stream_distinct_hadm_counts(
                cache_root / "procedures_boundary.parquet"
            )
        ),
        "prescriptions": _stream_unique_hadm_count(
            cache_root / "prescriptions_selected_chunks_v2.parquet"
        ),
    }
    lab_admissions = _stream_distinct_hadm_counts(cache_root / LAB_FILE)
    assert isinstance(lab_admissions, int)
    total = len(cohort)
    labeled = cohort.loc[cohort["mortality"].notna()]
    gender = cohort["gender"].value_counts().to_dict()
    race_group = cohort["race"].map(ETHNICITY_GROUP).value_counts().to_dict()
    return {
        "admissions": total,
        "subjects": int(cohort["subject_id"].nunique()),
        "labeled": len(labeled),
        "mortality_positive": int(labeled["mortality"].sum()),
        "mortality_positive_ratio": float(labeled["mortality"].mean()),
        "mortality_missing": int(cohort["mortality"].isna().sum()),
        "mortality_missing_ratio": float(cohort["mortality"].isna().mean()),
        "mean_age": float(cohort["age"].mean()),
        "gender_counts": {str(key): int(value) for key, value in gender.items()},
        "race_group_counts": {
            str(key): int(value) for key, value in race_group.items()
        },
        "missing_ratio": {
            "diagnoses_icd": 1 - code_counts["diagnoses_icd"] / total,
            "procedures_icd": 1 - code_counts["procedures_icd"] / total,
            "prescriptions": 1 - code_counts["prescriptions"] / total,
            "labvectors": 1 - lab_admissions / total,
            "discharge": 1 - int(note_statistics["rows"]) / total,
        },
        "average_discharge_words": float(note_statistics["average_words"]),
    }


def ensure_source(
    mimic_root: Path,
    note_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    mimic_root = mimic_root.resolve()
    note_root = note_root.resolve()
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    raw = _required_raw_files(mimic_root, note_root)
    manifest_path = cache_root / SOURCE_MANIFEST_FILE
    adapter_hash = sha256_file(Path(__file__))
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("adapter_sha256") != adapter_hash:
            raise RuntimeError(
                "source cache was built by a different adapter revision; move the "
                "ignored cache aside before rebuilding"
            )
        for record in manifest["cache_files"].values():
            path = cache_root / record["file"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"source cache file is missing or modified: {path}")
        return manifest

    cohort_summary = _build_cohort(raw, cache_root)
    code_summary = _build_codes(raw, cache_root)
    lab_summary = _build_labs(raw, cache_root)
    note_summary = _build_notes(raw, cache_root)
    paper_base_note_statistics = _build_paper_base_note_statistics(raw, cache_root)
    statistics = _source_statistics(cache_root)
    cache_files = {
        name: {
            "file": filename,
            "sha256": sha256_file(cache_root / filename),
        }
        for name, filename in {
            "cohort": COHORT_FILE,
            "codes": CODE_FILE,
            "labs": LAB_FILE,
            "notes": NOTE_FILE,
            "author_vocabulary": VOCAB_FILE,
            "lab_channels": LAB_CHANNEL_FILE,
            "train_ids": "train_admission_ids.txt",
            "validation_ids": "val_admission_ids.txt",
            "test_ids": "test_admission_ids.txt",
            "no_label_ids": "no_label_admission_ids.txt",
            "paper_base_note_statistics": PAPER_NOTE_STATS_FILE,
        }.items()
    }
    manifest = {
        "adapter_sha256": adapter_hash,
        "upstream_commit": UPSTREAM_COMMIT,
        "source_versions": {
            "mimic_iv": "2.0",
            "mimic_iv_note": "2.2",
        },
        "raw_files": {
            name: {
                "relative_path": str(path.relative_to(
                    mimic_root if name != "discharge" else note_root
                )),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in raw.items()
        },
        "cache_files": cache_files,
        "cohort": cohort_summary,
        "codes": code_summary,
        "labs": lab_summary,
        "notes": note_summary,
        "paper_base_note_statistics": paper_base_note_statistics,
        "statistics": statistics,
        "released_transform": {
            "notebook": "src/preprocess/parse_mimic4_remote.ipynb",
            "historical_base_producer_commit": (
                "3596fd3d5d7d79864e26a80d786ba8c6dc2fbd9b"
            ),
            "current_v2_producer": "src/preprocess/preprocess_mimic4.py",
            "splitter": "src/preprocess/data_split_mimic4.py",
            "dataset": "src/dataset/mimic4_dataset.py::MIMIC4Dataset",
            "collate": "src/dataset/utils.py::mimic4_collate_fn",
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def author_vocabulary(cache_root: Path) -> list[str]:
    return json.loads((cache_root / VOCAB_FILE).read_text())


def lab_channels(cache_root: Path) -> dict[str, Any]:
    return json.loads((cache_root / LAB_CHANNEL_FILE).read_text())


def clinical_event_code(event_type: str, code_name: str) -> str:
    return f"MUSE_CODE//{event_type}//{code_name}"


def source_codes(cache_root: Path) -> tuple[str, ...]:
    code_pairs: set[tuple[str, str]] = set()
    for batch in pq.ParquetFile(cache_root / CODE_FILE).iter_batches(
        batch_size=100_000, columns=["event_type", "code_name"]
    ):
        values = batch.to_pydict()
        code_pairs.update(zip(values["event_type"], values["code_name"]))
    channels = lab_channels(cache_root)["itemids"]
    codes = [f"MUSE_AGE//{value}" for value in AGE_BUCKETS]
    codes.extend(f"MUSE_GENDER//{value}" for value in GENDERS)
    codes.extend(f"MUSE_ETHNICITY//{value}" for value in ETHNICITIES)
    codes.extend(
        clinical_event_code(event_type, code_name)
        for event_type, code_name in sorted(code_pairs)
    )
    codes.extend(
        [
            "MUSE_LAB_PRESENT//false",
            "MUSE_LAB_PRESENT//true",
            "MUSE_LAB_SEQUENCE_ROW",
        ]
    )
    for itemid in channels:
        codes.append(f"MUSE_LAB_VALUE//{itemid}")
        codes.append(f"MUSE_LAB_ABNORMAL//{itemid}")
    codes.append(NOTE_CODE)
    if len(codes) != len(set(codes)):
        raise AssertionError("MUSE MEDS code construction collided")
    return tuple(codes)


class ParquetGroupCursor:
    """Traverse a Parquet file grouped by one already-sorted integer key."""

    def __init__(self, path: Path, key: str, columns: list[str] | None = None):
        self.key = key
        self._batches = pq.ParquetFile(path).iter_batches(
            batch_size=100_000, columns=columns
        )
        self._pending: pd.DataFrame | None = None
        self._groups: Iterator[tuple[Any, pd.DataFrame]] = iter(())
        self._next: tuple[int, pd.DataFrame] | None = None
        self._advance()

    def _advance(self) -> None:
        while True:
            try:
                key, frame = next(self._groups)
                self._next = (int(key), frame)
                return
            except StopIteration:
                pass
            try:
                frame = pa.Table.from_batches([next(self._batches)]).to_pandas()
            except StopIteration:
                if self._pending is not None:
                    key = int(self._pending[self.key].iloc[0])
                    self._next = (key, self._pending)
                    self._pending = None
                else:
                    self._next = None
                return
            if self._pending is not None:
                frame = pd.concat([self._pending, frame], ignore_index=True)
                self._pending = None
            last_key = frame[self.key].iloc[-1]
            last = frame.loc[frame[self.key].eq(last_key)]
            body = frame.loc[~frame.index.isin(last.index)]
            self._pending = last.reset_index(drop=True)
            self._groups = iter(body.groupby(self.key, sort=False))

    def take(self, key: int) -> pd.DataFrame:
        while self._next is not None and self._next[0] < key:
            self._advance()
        if self._next is not None and self._next[0] == key:
            result = self._next[1]
            self._advance()
            return result
        return pd.DataFrame()


def reconstruct_labvectors(rows: pd.DataFrame, width: int) -> np.ndarray | None:
    if rows.empty:
        return None
    vector = np.zeros(width, dtype=np.float32)
    result: list[np.ndarray] = []
    for sequence_row, group in rows.groupby("sequence_row", sort=False):
        if int(sequence_row) != len(result):
            raise AssertionError("lab sequence rows are not contiguous")
        for row in group.itertuples(index=False):
            if int(row.channel_index) >= 0:
                vector[int(row.channel_index)] = np.float32(row.numeric_value)
        result.append(vector.copy())
    return np.stack(result)
