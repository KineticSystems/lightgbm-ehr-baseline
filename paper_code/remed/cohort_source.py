"""Exact REMed/snub source boundary used by the adapter.

The released snub producer stores one ICU stay per HDF5 group.  ``hi`` is the
input consumed by the released ``EHRDataset``: ordered event rows with token,
token-type, and digit-place channels.  This module makes that boundary
reversible in MEDS without importing or reimplementing the raw MIMIC loader.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


TASK_ID = "los_7d_at_48h"
TASK_NAME = "ICU length of stay greater than seven days at the 48-hour prediction point"
PUBLISHED_AUROC = 0.836
PREDICTION_HOURS = 48
LABEL_COLUMN = "los_7"
SOURCE_SPLITS = ("train", "valid", "test")
SPLIT_MAP = {"train": "train", "valid": "validation", "test": "test"}
EVENT_PAYLOAD_PREFIX = "SNUB_HI_EVENT/"
EVENT_CODE_PREFIX = "SNUB_HI_SHA256/"
CODE_METADATA_NAME = "export_code_metadata.parquet"
SNUB_COMMIT = "bab6c07acce4a617153fba7da4f68bc32d20652f"
REMED_COMMIT = "eb444fad98b842be3c4d10bb690fd2f85f6ad90e"


@dataclass(frozen=True)
class Sample:
    source_position: int
    subject_id: int
    hadm_id: int
    stay_id: int
    split: str
    label: bool
    prediction_time_us: int
    actual_observation_start_us: int
    event_count: int
    token_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths(source_root: Path) -> dict[str, Path]:
    paths = {
        "h5": source_root / "mimiciv.h5",
        "cohort": source_root / "mimiciv_cohort.csv",
        "build": source_root / "source_build.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "the exact snub 48-hour source build is incomplete: " + ", ".join(missing)
        )
    return paths


def _timestamp_us(value: object) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.to_datetime64().astype("datetime64[us]").astype(np.int64))


def _load_cohort(path: Path) -> pd.DataFrame:
    cohort = pd.read_csv(path)
    required = {
        "subject_id",
        "hadm_id",
        "stay_id",
        "ADMITTIME",
        "INTIME",
        LABEL_COLUMN,
        "split_2020",
    }
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"snub cohort is missing columns: {sorted(missing)}")
    if cohort["stay_id"].duplicated().any():
        raise ValueError("snub cohort stay_id is not unique")
    if set(cohort["split_2020"].dropna().unique()) != set(SOURCE_SPLITS):
        raise ValueError("snub split_2020 does not contain train/valid/test exactly")
    if cohort[LABEL_COLUMN].isna().any():
        raise ValueError(f"{LABEL_COLUMN} unexpectedly contains missing labels")
    if not set(cohort[LABEL_COLUMN].unique()).issubset({0, 1, False, True}):
        raise ValueError(f"{LABEL_COLUMN} is not binary")
    return cohort


def load_samples(source_root: Path) -> tuple[list[Sample], float]:
    """Load the exact cohort order and inspect ragged source shapes/times."""
    paths = source_paths(source_root)
    cohort = _load_cohort(paths["cohort"])
    samples: list[Sample] = []
    earliest_relative_minute = 0
    with h5py.File(paths["h5"], "r") as source:
        ehr = source["ehr"]
        if set(ehr.keys()) != set(cohort["stay_id"].astype(str)):
            raise ValueError("snub HDF5 and cohort contain different stay identifiers")
        for position, row in cohort.iterrows():
            stay_id = int(row["stay_id"])
            group = ehr[str(stay_id)]
            hi = group["hi"]
            times = np.asarray(group["time"][:], dtype=np.int64)
            if hi.ndim != 3 or hi.shape[1:] != (3, 128) or hi.dtype != np.int16:
                raise ValueError(f"stay {stay_id} has unexpected hi shape/dtype")
            if times.ndim != 1 or len(times) != hi.shape[0]:
                raise ValueError(f"stay {stay_id} has inconsistent event times")
            if len(times) == 0 or np.any(np.diff(times) < 0):
                raise ValueError(f"stay {stay_id} has an empty or unordered event stream")
            if int(times[-1]) >= PREDICTION_HOURS * 60:
                raise ValueError(f"stay {stay_id} contains a post-cutoff event")
            earliest_relative_minute = min(earliest_relative_minute, int(times[0]))

            admit_us = _timestamp_us(row["ADMITTIME"])
            icu_us = admit_us + int(round(float(row["INTIME"]))) * 60_000_000
            prediction_us = icu_us + PREDICTION_HOURS * 3_600_000_000
            actual_start_us = icu_us + int(times[0]) * 60_000_000
            fl = group["fl"]
            if fl.ndim != 2 or fl.shape[0] != 3 or fl.dtype != np.int16:
                raise ValueError(f"stay {stay_id} has unexpected fl shape/dtype")
            samples.append(
                Sample(
                    source_position=int(position),
                    subject_id=int(row["subject_id"]),
                    hadm_id=int(row["hadm_id"]),
                    stay_id=stay_id,
                    split=SPLIT_MAP[str(row["split_2020"])],
                    label=bool(row[LABEL_COLUMN]),
                    prediction_time_us=prediction_us,
                    actual_observation_start_us=actual_start_us,
                    event_count=int(hi.shape[0]),
                    token_count=int(fl.shape[1]),
                )
            )
    observation_hours = PREDICTION_HOURS - earliest_relative_minute / 60.0
    if observation_hours <= 0:
        raise ValueError("derived infinite-history observation window is not positive")
    return samples, observation_hours


def encode_hi_event(event: np.ndarray) -> str:
    event = np.asarray(event)
    if event.shape != (3, 128) or event.dtype != np.int16:
        raise ValueError("snub event must have shape [3,128] and int16 dtype")
    nonpadding = np.flatnonzero(np.any(event != 0, axis=0))
    width = int(nonpadding[-1]) + 1 if len(nonpadding) else 0
    payload = bytes((width,)) + event[:, :width].astype("<i2", copy=False).tobytes()
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return EVENT_PAYLOAD_PREFIX + encoded


def decode_hi_event(code: str) -> np.ndarray:
    if not code.startswith(EVENT_PAYLOAD_PREFIX):
        raise ValueError("not a REMed/snub structured event code")
    encoded = code[len(EVENT_PAYLOAD_PREFIX) :]
    payload = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    if not payload:
        raise ValueError("empty REMed/snub event payload")
    width = payload[0]
    expected_bytes = 1 + 3 * width * np.dtype("<i2").itemsize
    if len(payload) != expected_bytes or width > 128:
        raise ValueError("malformed REMed/snub event payload")
    event = np.zeros((3, 128), dtype=np.int16)
    if width:
        event[:, :width] = np.frombuffer(payload[1:], dtype="<i2").reshape(3, width)
    return event


def event_code_from_payload(payload: str) -> str:
    if not payload.startswith(EVENT_PAYLOAD_PREFIX):
        raise ValueError("not a canonical REMed/snub event payload")
    return EVENT_CODE_PREFIX + hashlib.sha256(payload.encode("ascii")).hexdigest()


def event_code(event: np.ndarray) -> str:
    return event_code_from_payload(encode_hi_event(event))


def _write_code_metadata_batch(
    writer: pq.ParquetWriter | None,
    path: Path,
    codes: list[str],
    descriptions: list[str],
) -> pq.ParquetWriter:
    table = pa.table(
        {
            "code": pa.array(codes, type=pa.string()),
            "description": pa.array(descriptions, type=pa.large_string()),
        }
    )
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table, row_group_size=100_000)
    return writer


def collect_codes(
    source_root: Path,
    samples: list[Sample],
    *,
    batch_size: int = 100_000,
) -> tuple[tuple[str, ...], Path]:
    """Build a fixed-width vocabulary plus lossless MEDS code descriptions."""
    paths = source_paths(source_root)
    metadata_path = source_root / CODE_METADATA_NAME
    if metadata_path.is_file():
        codes = tuple(
            pq.read_table(metadata_path, columns=["code"])["code"].to_pylist()
        )
        if not codes or any(not code.startswith(EVENT_CODE_PREFIX) for code in codes):
            raise ValueError("cached REMed code metadata is malformed")
        return codes, metadata_path

    partial_path = metadata_path.with_suffix(".partial.parquet")
    partial_path.unlink(missing_ok=True)
    seen: set[bytes] = set()
    codes: list[str] = []
    pending_codes: list[str] = []
    pending_descriptions: list[str] = []
    writer: pq.ParquetWriter | None = None
    try:
        with h5py.File(paths["h5"], "r") as source:
            ehr = source["ehr"]
            for sample in samples:
                for event in ehr[str(sample.stay_id)]["hi"]:
                    payload = encode_hi_event(np.asarray(event, dtype=np.int16))
                    digest = hashlib.sha256(payload.encode("ascii")).digest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    code = EVENT_CODE_PREFIX + digest.hex()
                    codes.append(code)
                    pending_codes.append(code)
                    pending_descriptions.append(payload)
                    if len(pending_codes) == batch_size:
                        writer = _write_code_metadata_batch(
                            writer,
                            partial_path,
                            pending_codes,
                            pending_descriptions,
                        )
                        pending_codes.clear()
                        pending_descriptions.clear()
        if pending_codes:
            writer = _write_code_metadata_batch(
                writer,
                partial_path,
                pending_codes,
                pending_descriptions,
            )
    finally:
        if writer is not None:
            writer.close()
    if not codes:
        raise ValueError("snub source contains no structured events")
    partial_path.replace(metadata_path)
    return tuple(codes), metadata_path


def sample_frame(samples: list[Sample], observation_hours: float) -> pl.DataFrame:
    window_us = int(round(observation_hours * 3_600_000_000))
    return pl.DataFrame(
        {
            "subject_id": [sample.subject_id for sample in samples],
            "sample_id": [sample.stay_id for sample in samples],
            "observation_start": [
                sample.prediction_time_us - window_us for sample in samples
            ],
            "prediction_time": [sample.prediction_time_us for sample in samples],
            "boolean_value": [sample.label for sample in samples],
            "split": [sample.split for sample in samples],
            "source_position": [sample.source_position for sample in samples],
            "hadm_id": [sample.hadm_id for sample in samples],
            "actual_observation_start": [
                sample.actual_observation_start_us for sample in samples
            ],
            "source_event_count": [sample.event_count for sample in samples],
            "source_token_count": [sample.token_count for sample in samples],
        },
        schema={
            "subject_id": pl.Int64,
            "sample_id": pl.Int64,
            "observation_start": pl.Datetime("us"),
            "prediction_time": pl.Datetime("us"),
            "boolean_value": pl.Boolean,
            "split": pl.String,
            "source_position": pl.Int64,
            "hadm_id": pl.Int64,
            "actual_observation_start": pl.Datetime("us"),
            "source_event_count": pl.Int64,
            "source_token_count": pl.Int64,
        },
    )


def event_batches(
    source_root: Path,
    samples: list[Sample],
    *,
    batch_size: int = 50_000,
):
    paths = source_paths(source_root)
    rows: dict[str, list[object]] = {
        "subject_id": [],
        "time": [],
        "code": [],
        "numeric_value": [],
        "sample_id": [],
        "source_row": [],
        "sequence_row": [],
        "source_time": [],
    }
    source_row = 0
    with h5py.File(paths["h5"], "r") as source:
        ehr = source["ehr"]
        for sample in samples:
            group = ehr[str(sample.stay_id)]
            times = np.asarray(group["time"][:], dtype=np.int64)
            icu_us = sample.prediction_time_us - PREDICTION_HOURS * 3_600_000_000
            for sequence_row, (event, minute) in enumerate(zip(group["hi"], times)):
                rows["subject_id"].append(sample.subject_id)
                rows["time"].append(icu_us + int(minute) * 60_000_000)
                rows["code"].append(event_code(np.asarray(event, dtype=np.int16)))
                rows["numeric_value"].append(None)
                rows["sample_id"].append(sample.stay_id)
                rows["source_row"].append(source_row)
                rows["sequence_row"].append(sequence_row)
                rows["source_time"].append(float(minute))
                source_row += 1
                if len(rows["sample_id"]) == batch_size:
                    yield pl.DataFrame(rows)
                    rows = {name: [] for name in rows}
    if rows["sample_id"]:
        yield pl.DataFrame(rows)


def split_summary(samples: list[Sample]) -> dict[str, dict[str, int]]:
    return {
        split: {
            "count": sum(sample.split == split for sample in samples),
            "subjects": len(
                {sample.subject_id for sample in samples if sample.split == split}
            ),
            "positive": sum(
                sample.split == split and sample.label for sample in samples
            ),
        }
        for split in ("train", "validation", "test")
    }


def cross_split_subject_overlap(samples: list[Sample]) -> dict[str, int]:
    members = {
        split: {sample.subject_id for sample in samples if sample.split == split}
        for split in ("train", "validation", "test")
    }
    return {
        f"{left}_{right}": len(members[left] & members[right])
        for index, left in enumerate(members)
        for right in tuple(members)[index + 1 :]
    }


def hash_ints(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def read_source_build(source_root: Path) -> dict[str, object]:
    return json.loads(source_paths(source_root)["build"].read_text())
