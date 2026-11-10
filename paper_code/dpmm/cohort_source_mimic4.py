"""Operational reconstruction of DPMM's released MIMIC-IV source boundary.

This module deliberately calls the released MIMIC-IV cohort, item mapping,
value cleaning, discretization, normalization and dataset code from
`mimic4extract/` and `dataset_mimic4/`.  The only new work is bounded I/O
orchestration for the official compressed MIMIC-IV 2.2 tables and
serialization of the attributable intermediate views needed by the export.

It must never be imported in the same interpreter as `cohort_source`:
`mimic3extract/` and `mimic4extract/` both ship a package named
`mimic3benchmark`, and only one of them can win on `sys.path`.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


PAPER_ROOT = Path(__file__).resolve().parent
MIMIC4EXTRACT = PAPER_ROOT / "mimic4extract"
if str(PAPER_ROOT / "mimic3extract") in sys.path:  # pragma: no cover - guard
    raise ImportError("mimic3extract and mimic4extract cannot share a process")
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(MIMIC4EXTRACT))

from dataset_mimic4.ehr_dataset import EHRdataset, my_collate  # noqa: E402,F401
from ehr_utils.preprocessing import Discretizer, Normalizer  # noqa: E402
from mimic3benchmark import mimic3csv  # noqa: E402
from mimic3benchmark.preprocessing import (  # noqa: E402
    clean_events,
    clean_fns,
    map_itemids_to_variables,
    read_itemid_to_variable_map,
)

UPSTREAM_COMMIT = "17d0314ff0b21dde31f3f932e02d4683826029aa"
HOURS = 48.0
TIMESTEP = 2.0
SPLITS = ("train", "validation", "test")

IHM_TASK_ID = "mimic4_mortality_48h_partial_ehr"
READM_TASK_ID = "mimic4_readmission_30d_partial_ehr"
IHM_PUBLISHED_AUROC = 0.860
READM_PUBLISHED_AUROC = 0.771
IHM_PUBLISHED_CELL = (
    "Table 3, MIMIC-IV IHM, partial EHR-CXR DPMM, AUROC 0.860 (0.847, 0.877)"
)
READM_PUBLISHED_CELL = (
    "Table 3, MIMIC-IV READM, partial EHR-CXR DPMM, AUROC 0.771 (0.752, 0.788)"
)

# Supplement Table "Datasets summary" and the released `dataset_mimic4/fusion.py`
# docstring agree exactly on these MIMIC-IV numbers.
EXPECTED_COUNTS = {"train": 18064, "validation": 2035, "test": 4972}
EXPECTED_TOTAL = 25071
PUBLISHED_IHM_TOTAL_POSITIVES = 3153
PUBLISHED_READM_TOTAL_POSITIVES = 4603
PUBLISHED_LOADER_IHM_POSITIVES = {"train": 2268, "validation": 251, "test": 634}
PUBLISHED_LOADER_READM_POSITIVES = {"train": 3321, "validation": 382, "test": 900}
PUBLISHED_PAIRED_CXR = {"train": 4287, "validation": 465, "test": 1179}
PUBLISHED_PAIRED_CXR_TOTAL = 5931
PUBLISHED_PAIRED_IHM_POSITIVES = 890
PUBLISHED_PAIRED_READM_POSITIVES = 1262

# The released `mimic3csv.read_events_table_by_row` still carries MedFuse's
# MIMIC-IV v1.0 row counts. They are only tqdm totals and are never checked,
# and v1.0 cannot run this code at all (its admissions table has `ethnicity`,
# not the `race` column the released reader selects). The declared release is
# MIMIC-IV 2.2 (paper_code/dpmm/README.md links physionet .../mimiciv/2.2/).
RELEASED_STALE_ROW_TOTALS = {
    "chartevents": 329499788,
    "labevents": 122103667,
    "outputevents": 4457381,
}

RELEASE_TABLES = {
    "patients": "hosp/patients.csv.gz",
    "admissions": "hosp/admissions.csv.gz",
    "icustays": "icu/icustays.csv.gz",
    "chartevents": "icu/chartevents.csv.gz",
    "labevents": "hosp/labevents.csv.gz",
    "outputevents": "icu/outputevents.csv.gz",
}

ITEM_MAP = MIMIC4EXTRACT / "mimic3benchmark" / "resources" / "itemid_to_variable_map.csv"
VARIABLE_RANGES = (
    MIMIC4EXTRACT / "mimic3benchmark" / "resources" / "variable_ranges.csv"
)
TESTSET = MIMIC4EXTRACT / "mimic3benchmark" / "resources" / "testset_iv.csv"
VALSET = MIMIC4EXTRACT / "mimic3models" / "resources" / "valset_iv.csv"
DISCRETIZER_CONFIG = PAPER_ROOT / "ehr_utils" / "resources" / "discretizer_config.json"

# `clean_events` replaces the released `value` with a float for exactly these
# variables; every other retained variable keeps its raw source string, and the
# released per-subject de-duplication therefore orders it as text.
CLEANED_VARIABLES = tuple(sorted(clean_fns))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ordered_text_hash(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def assert_mimic4_release(root: Path) -> dict[str, str]:
    """Hash the exact MIMIC-IV 2.2 tables this reconstruction consumes.

    The local 2.2 tree ships no `SHA256SUMS.txt`, so the digests are computed
    and recorded rather than checked against a release manifest.
    """
    result = {}
    for name, relative in RELEASE_TABLES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing exact MIMIC-IV 2.2 table: {path}")
        result[relative] = sha256(path)
    return result


def released_all_stays(root: Path) -> pd.DataFrame:
    """The released `all_stays.csv` frame, before the 48-hour task filter."""
    patients = mimic3csv.read_patients_table(str(root / RELEASE_TABLES["patients"]))
    admits = mimic3csv.read_admissions_table(str(root / RELEASE_TABLES["admissions"]))
    stays = mimic3csv.read_icustays_table(str(root / RELEASE_TABLES["icustays"]))
    stays = mimic3csv.remove_icustays_with_transfers(stays)
    stays = mimic3csv.merge_on_subject_admission(stays, admits)
    stays = mimic3csv.merge_on_subject(stays, patients)
    stays = mimic3csv.filter_admissions_on_nb_icustays(stays)
    stays = mimic3csv.add_age_to_icustays(stays)
    stays = mimic3csv.add_inunit_mortality_to_icustays(stays)
    stays = mimic3csv.add_inhospital_mortality_to_icustays(stays)
    stays = mimic3csv.filter_icustays_on_age(stays)
    return stays


def released_candidates(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All released stays plus the 48-hour-eligible task candidates."""
    stays = released_all_stays(root)
    stays = stays.sort_values(
        ["subject_id", "intime", "outtime"], kind="stable"
    ).reset_index(drop=True)
    stays["episode_number"] = stays.groupby("subject_id").cumcount() + 1
    stays["source_name"] = (
        stays["subject_id"].astype(str)
        + "_episode"
        + stays["episode_number"].astype(str)
        + "_timeseries.csv"
    )
    stays["los_hours"] = 24.0 * stays["los"].astype(float)
    candidates = stays.loc[stays["los_hours"] >= HOURS - 1e-6].copy()
    if candidates["stay_id"].duplicated().any():
        raise AssertionError("released candidate ICU stay IDs are not unique")
    return stays, candidates


def released_diagnosis_stays(root: Path, stays: pd.DataFrame) -> set[int]:
    """Stays that survive the released episodic-label inner merge.

    `assemble_episodic_data` ends in
    `data.merge(extract_diagnosis_labels(diagnoses), left_index=True,
    right_index=True)` -- an inner join. A stay whose admission has no ICD
    diagnosis row (after the released inner merge with `d_icd_diagnoses`) is
    therefore absent from `episodic_data`, its `episode{i}.csv` is written with
    zero rows, and both `create_in_hospital_mortality.py` and
    `create_readmission_30d.py` skip it at their `label_df.shape[0] == 0`
    guard. This silently removes 12 otherwise eligible MIMIC-IV stays and is
    the difference between 25,083 candidates and the published 25,071.
    """
    original = pd.read_csv

    def shim(path, *args, **kwargs):
        candidate = Path(path)
        if not candidate.exists() and candidate.suffix == ".csv":
            candidate = candidate.with_suffix(".csv.gz")
        return original(candidate, *args, **kwargs)

    pd.read_csv = shim
    try:
        diagnoses = mimic3csv.read_icd_diagnoses_table(str(root / "hosp"))
        diagnoses = mimic3csv.filter_diagnoses_on_stays(diagnoses, stays)
    finally:
        pd.read_csv = original
    return set(int(value) for value in diagnoses["stay_id"])


def _load_flagged_subjects(path: Path) -> set[int]:
    result = set()
    with path.open() as handle:
        for line in handle:
            subject, flag = line.strip().split(",")
            if int(flag) == 1:
                result.add(int(subject))
    return result


def assign_split(frame: pd.DataFrame) -> pd.Series:
    test = _load_flagged_subjects(TESTSET)
    validation = _load_flagged_subjects(VALSET)
    if test & validation:
        raise AssertionError("released validation and test resources overlap")
    subject = frame["subject_id"].astype(int)
    return pd.Series(
        np.where(
            subject.isin(test),
            "test",
            np.where(subject.isin(validation), "validation", "train"),
        ),
        index=frame.index,
    )


def _write_part(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    for column, dtype in (
        ("stay_id", "int64"),
        ("subject_id", "int64"),
        ("raw_source_row", "int64"),
    ):
        if column in frame:
            frame[column] = frame[column].astype(dtype)
    if "charttime" in frame:
        frame["charttime"] = pd.to_datetime(frame["charttime"])
    for column in ("variable", "value_text", "sort_str", "categorical_value"):
        if column in frame:
            frame[column] = frame[column].astype("string")
    for column in ("numeric_value", "sort_num"):
        if column in frame:
            frame[column] = frame[column].astype("float64")
    frame.to_parquet(path, index=False, compression="zstd")


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": pd.Series(dtype="int64"),
            "subject_id": pd.Series(dtype="int64"),
            "charttime": pd.Series(dtype="datetime64[ns]"),
            "variable": pd.Series(dtype="string"),
            "numeric_value": pd.Series(dtype="float64"),
            "categorical_value": pd.Series(dtype="string"),
            "value_text": pd.Series(dtype="string"),
            "sort_num": pd.Series(dtype="float64"),
            "sort_str": pd.Series(dtype="string"),
            "raw_source_row": pd.Series(dtype="int64"),
        }
    )


TABLE_SPECS = {
    "outputevents": {
        "relative": "icu/outputevents.csv.gz",
        "usecols": [
            "subject_id",
            "charttime",
            "itemid",
            "value",
            "valueuom",
        ],
        # `read_events_table_and_break_up_by_subject` writes VALUEUOM into the
        # `valuenum` column for OUTPUTEVENTS only.
        "rename": {"valueuom": "valuenum"},
    },
    "chartevents": {
        "relative": "icu/chartevents.csv.gz",
        "usecols": [
            "subject_id",
            "charttime",
            "itemid",
            "value",
            "valuenum",
        ],
        "rename": {},
    },
    "labevents": {
        "relative": "hosp/labevents.csv.gz",
        "usecols": [
            "subject_id",
            "charttime",
            "itemid",
            "value",
            "valuenum",
        ],
        "rename": {},
    },
}


def _event_table_parts(
    mimic_root: Path,
    candidates: pd.DataFrame,
    cache_root: Path,
    *,
    chunksize: int = 1_000_000,
) -> dict[str, dict[str, int]]:
    """Bounded operational wrapper around the released mapping/cleaning."""
    part_root = cache_root / "event_parts"
    part_root.mkdir(parents=True, exist_ok=True)
    stats_path = cache_root / "event_part_stats.json"
    stats: dict[str, dict[str, int]] = (
        json.loads(stats_path.read_text()) if stats_path.exists() else {}
    )
    var_map = read_itemid_to_variable_map(str(ITEM_MAP))
    ready_itemids = set(int(value) for value in var_map["itemid"])
    categorical = json.loads(DISCRETIZER_CONFIG.read_text())["is_categorical_channel"]
    lookup = candidates[["subject_id", "stay_id", "intime"]].copy()
    subjects = set(lookup["subject_id"].astype(int))

    for table, spec in TABLE_SPECS.items():
        if stats.get(table, {}).get("complete"):
            print(f"{table}: using cached source checkpoints", flush=True)
            continue
        offset = 0
        retained = 0
        non_numeric = 0
        chunk_index = -1
        reader = pd.read_csv(
            mimic_root / spec["relative"],
            usecols=spec["usecols"],
            chunksize=chunksize,
            low_memory=False,
            dtype={"value": str, "valuenum": str, "valueuom": str},
        )
        for chunk_index, raw in enumerate(reader):
            path = part_root / f"{table}_{chunk_index:04d}.parquet"
            chunk_rows = len(raw)
            if path.exists():
                offset += chunk_rows
                continue
            raw = raw.rename(columns=spec["rename"])
            raw["raw_source_row"] = np.arange(
                offset, offset + chunk_rows, dtype=np.int64
            )
            offset += chunk_rows
            # Released `get_events_for_stay` keeps an event when its own
            # stay_id matches OR its charttime lies inside [intime, outtime].
            # Every retained stay has LOS >= 48h, so [intime, intime + 48h] is
            # contained in [intime, outtime] and the time branch alone decides
            # membership; the event's own stay_id is therefore not read.
            # Released `read_events` boundary: drop null values and treat the
            # released "valuenum" column (VALUEUOM for OUTPUTEVENTS) as text.
            selected = raw.loc[
                raw["value"].notna()
                & raw["itemid"].isin(ready_itemids)
                & raw["subject_id"].isin(subjects)
            ].copy()
            if selected.empty:
                _write_part(_empty_event_frame(), path)
                continue
            selected["valuenum"] = selected["valuenum"].fillna("").astype(str)
            selected["charttime"] = pd.to_datetime(
                selected["charttime"], errors="coerce"
            )
            selected = selected.loc[selected["charttime"].notna()]
            selected["itemid"] = selected["itemid"].astype(int)
            selected = map_itemids_to_variables(selected, var_map)
            if selected.empty:
                _write_part(_empty_event_frame(), path)
                continue
            if not selected["value"].map(lambda value: isinstance(value, str)).all():
                raise AssertionError(
                    "released cleaning assumes a text `value` column"
                )
            selected = clean_events(selected)
            if selected.empty:
                _write_part(_empty_event_frame(), path)
                continue
            selected = selected.merge(
                lookup, on="subject_id", how="inner", validate="m:m"
            )
            hours = (
                selected["charttime"] - selected["intime"]
            ).dt.total_seconds() / 3600.0
            selected = selected.loc[
                hours.ge(-1e-6) & hours.le(HOURS + 1e-6)
            ].copy()
            if selected.empty:
                _write_part(_empty_event_frame(), path)
                continue
            is_cleaned = selected["variable"].isin(CLEANED_VARIABLES).to_numpy()
            is_categorical = selected["variable"].map(categorical).astype(bool).to_numpy()
            value_text = np.array(
                [
                    str(value) if not isinstance(value, str) else value
                    for value in selected["value"].tolist()
                ],
                dtype=object,
            )
            text_series = pd.Series(value_text, index=selected.index)
            numeric_all = pd.to_numeric(text_series, errors="coerce")
            numeric = numeric_all.where(~is_categorical)
            non_numeric += int(
                (numeric.isna() & pd.Series(~is_categorical, index=selected.index)).sum()
            )
            narrowed = pd.DataFrame(
                {
                    "stay_id": selected["stay_id"].to_numpy(np.int64),
                    "subject_id": selected["subject_id"].to_numpy(np.int64),
                    "charttime": selected["charttime"].to_numpy("datetime64[ns]"),
                    "variable": selected["variable"].to_numpy(dtype=object),
                    "numeric_value": numeric.to_numpy(np.float64),
                    "categorical_value": np.where(is_categorical, value_text, None),
                    "value_text": value_text,
                    # Released de-duplication order: `clean_events` leaves a
                    # float for CLEANED_VARIABLES and the raw source string for
                    # every other variable, and pandas orders numbers before
                    # text, so only one of these two keys is ever set.
                    "sort_num": np.where(
                        is_cleaned, numeric_all.to_numpy(np.float64), np.nan
                    ),
                    "sort_str": np.where(is_cleaned, None, value_text),
                    "raw_source_row": selected["raw_source_row"].to_numpy(np.int64),
                }
            )
            retained += len(narrowed)
            _write_part(narrowed, path)
            print(
                f"{table}: chunk={chunk_index + 1} rows={offset:,} "
                f"retained={len(narrowed):,}",
                flush=True,
            )
        stats[table] = {
            "complete": 1,
            "source_rows": offset,
            "chunks": chunk_index + 1,
            "retained_rows": retained,
            "non_numeric_continuous_values": non_numeric,
        }
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return stats


def _collect_event_parts(cache_root: Path, candidates: pd.DataFrame) -> Path:
    """Apply the released per-timestamp de-duplication in bounded stay ranges."""
    output = cache_root / "structured_events.parquet"
    if output.exists():
        return output
    import pyarrow.parquet as pq

    stay_ids = np.sort(candidates["stay_id"].to_numpy(np.int64))
    buckets = 16
    edges = [int(stay_ids[0])]
    for index in range(1, buckets):
        edges.append(int(stay_ids[len(stay_ids) * index // buckets]))
    edges.append(int(stay_ids[-1]) + 1)
    pattern = str(cache_root / "event_parts" / "*.parquet")
    writer = None
    temporary = output.with_suffix(".partial")
    try:
        for low, high in zip(edges[:-1], edges[1:]):
            if low >= high:
                continue
            frame = (
                pl.scan_parquet(pattern)
                .filter(
                    pl.col("stay_id").is_not_null()
                    & (pl.col("stay_id") >= low)
                    & (pl.col("stay_id") < high)
                )
                # Released `convert_events_to_timeseries` sorts by
                # (charttime, variable, value) and keeps the last duplicate;
                # `clean_events` leaves a float for CLEANED_VARIABLES and the
                # raw source string for every other variable, and pandas orders
                # numbers before text, so exactly one key is set per row.
                .sort(
                    ["stay_id", "charttime", "variable", "sort_num", "sort_str"],
                    nulls_last=False,
                )
                .unique(
                    subset=["stay_id", "charttime", "variable"],
                    keep="last",
                    maintain_order=True,
                )
                .sort(["stay_id", "charttime", "variable"])
                .collect(engine="streaming")
            )
            if frame.is_empty():
                continue
            table = frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=250_000)
            print(f"dedup: stays [{low}, {high}) rows={frame.height:,}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    temporary.rename(output)
    return output


def _restrict_events_to_cohort(
    cache_root: Path, structured_path: Path, cohort: pd.DataFrame
) -> Path:
    """Drop events of candidate stays the released list-file builders skip.

    A stay with no released ICD diagnosis row still has chart/lab events, but
    its `episode{i}.csv` is empty so both released builders skip it. Restrict
    the persistent event stream to the exact released cohort.
    """
    import pyarrow.parquet as pq

    output = cache_root / "cohort_events.parquet"
    if output.exists():
        return output
    keep = set(int(value) for value in cohort["stay_id"])
    parquet = pq.ParquetFile(structured_path)
    writer = None
    temporary = output.with_suffix(".partial")
    dropped = 0
    try:
        for batch in parquet.iter_batches(batch_size=250_000):
            frame = pl.from_arrow(batch)
            filtered = frame.filter(
                pl.col("stay_id").is_in(pl.Series("keep", sorted(keep), dtype=pl.Int64))
            )
            dropped += frame.height - filtered.height
            if filtered.is_empty():
                continue
            table = filtered.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=250_000)
    finally:
        if writer is not None:
            writer.close()
    temporary.rename(output)
    print(f"restricted event stream: dropped {dropped:,} non-cohort rows", flush=True)
    return output


def _write_author_csvs(
    cache_root: Path,
    cohort: pd.DataFrame,
    structured_path: Path,
    task: str,
    label_column: str,
    episode_source: Path | None = None,
) -> Path:
    source_root = cache_root / "author_view" / task
    complete = source_root / ".complete"
    if complete.exists():
        return source_root
    source_root.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq

    # Both released list-file builders write byte-identical [0, 48]-truncated
    # episode files (identical `n_hours`, identical truncation, identical
    # names); only the list files differ. The second task therefore reuses the
    # first task's episode directories and writes only its own list files,
    # which `author_datasets` reads with the shared episode directory.
    if episode_source is None:
        (source_root / "train").mkdir(parents=True, exist_ok=True)
        (source_root / "test").mkdir(parents=True, exist_ok=True)

    variables = json.loads(DISCRETIZER_CONFIG.read_text())["id_to_channel"]
    csv_columns = ["Hours", *sorted(variables)]
    sorted_variables = sorted(variables)
    cohort_by_stay = cohort.set_index("stay_id")

    def flush(stay_id: int, rows: list) -> None:
        row = cohort_by_stay.loc[int(stay_id)]
        group = pd.DataFrame(rows, columns=["charttime", "variable", "value_text"])
        hours = (
            pd.to_datetime(group["charttime"]) - pd.Timestamp(row["intime"])
        ).dt.total_seconds() / 3600.0
        group = group.assign(Hours=hours.to_numpy())
        wide = group.pivot(index="Hours", columns="variable", values="value_text")
        wide = wide.reindex(columns=sorted_variables).reset_index()
        target = source_root / ("test" if row["split"] == "test" else "train")
        wide.to_csv(target / row["source_name"], index=False, columns=csv_columns)

    # `structured_events.parquet` is sorted by stay_id, so one episode is
    # buffered at a time and the full event table never enters memory.
    parquet = None if episode_source is not None else pq.ParquetFile(structured_path)
    current_stay = None
    buffer: list = []
    for batch in [] if parquet is None else parquet.iter_batches(
        batch_size=250_000, columns=["stay_id", "charttime", "variable", "value_text"]
    ):
        stay_ids = batch.column("stay_id").to_pylist()
        charttimes = batch.column("charttime").to_pylist()
        variables_column = batch.column("variable").to_pylist()
        value_texts = batch.column("value_text").to_pylist()
        for stay_id, charttime, variable, value_text in zip(
            stay_ids, charttimes, variables_column, value_texts
        ):
            if stay_id != current_stay:
                if current_stay is not None:
                    flush(current_stay, buffer)
                current_stay = stay_id
                buffer = []
            buffer.append((charttime, variable, value_text))
    if current_stay is not None:
        flush(current_stay, buffer)

    def list_frame(selected: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stay": selected["source_name"],
                "period_length": 0,
                "stay_id": selected["stay_id"].astype(np.int64),
                "y_true": selected[label_column].astype(np.int64),
            }
        )

    combined = cohort.loc[cohort["split"].isin(["train", "validation"])].copy()
    records = list_frame(combined).to_dict("records")
    random.Random(49297).shuffle(records)
    combined_path = (
        source_root / "train_listfile_combined.csv"
        if episode_source is not None
        else source_root / "train" / "listfile.csv"
    )
    pd.DataFrame(records).to_csv(combined_path, index=False)
    for split, filename in (
        ("train", "train_listfile.csv"),
        ("validation", "val_listfile.csv"),
        ("test", "test_listfile.csv"),
    ):
        list_frame(cohort.loc[cohort["split"] == split]).to_csv(
            source_root / filename, index=False
        )
    if episode_source is None:
        list_frame(cohort.loc[cohort["split"] == "test"]).to_csv(
            source_root / "test" / "listfile.csv", index=False
        )
    complete.write_text("exact released wide episode view\n")
    return source_root


def _fit_released_normalizer(
    cache_root: Path, source_root: Path, episodes_root: Path
) -> Path:
    """Released MIMIC-IV behavior: train-only moments from train_listfile.csv."""
    output = cache_root / "readm_ts.normalizer"
    if output.exists():
        return output
    listfile = source_root / "train_listfile.csv"
    names = pd.read_csv(listfile)["stay"].tolist()
    discretizer = Discretizer(
        timestep=TIMESTEP,
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
        config_path=str(DISCRETIZER_CONFIG),
    )
    first_path = episodes_root / "train" / names[0]
    first = pd.read_csv(first_path, dtype=str).fillna("").to_numpy(str)
    header = pd.read_csv(first_path, nrows=0).columns.tolist()
    new_header = discretizer.transform(first, header=header, end=HOURS)[1].split(",")
    continuous = [index for index, name in enumerate(new_header) if "->" not in name]
    normalizer = Normalizer(fields=continuous)
    for index, name in enumerate(names):
        path = episodes_root / "train" / name
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        data, _ = discretizer.transform(
            frame.to_numpy(str), header=frame.columns.tolist(), end=HOURS
        )
        normalizer._feed_data(data)
        if (index + 1) % 1000 == 0:
            print(f"normalizer: {index + 1:,}/{len(names):,}", flush=True)
    normalizer._save_params(str(output))
    return output


def released_readmission_labels(all_stays: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """`mimic4extract/.../create_readmission_30d.py`, verbatim plus controls."""
    records: list[tuple[int, int, int, int, float]] = []
    measurement = {
        "multi_stay_subjects": 0,
        "multi_stay_stays": 0,
        "subjects_reordered_by_admittime_sort": 0,
        "stays_changed_by_index_misuse": 0,
        "stays_changed_by_missing_lower_bound": 0,
        "negative_gap_stays": 0,
    }
    for _, subject_stays in all_stays.groupby("subject_id", sort=False):
        stay_df = subject_stays.sort_values(by="intime").reset_index(drop=True)
        before = stay_df["stay_id"].tolist()
        stay_df = stay_df.sort_values(by=["admittime", "dischtime"])
        count = len(stay_df)
        if count > 1:
            measurement["multi_stay_subjects"] += 1
            measurement["multi_stay_stays"] += count
            if stay_df["stay_id"].tolist() != before:
                measurement["subjects_reordered_by_admittime_sort"] += 1
        position_of_label = {label: index for index, label in enumerate(stay_df.index)}
        for idx, row in stay_df.iterrows():
            if idx == count - 1:
                released = 0
            else:
                next_row = stay_df.iloc[idx + 1]
                released_gap = pd.to_datetime(next_row.admittime) - pd.to_datetime(
                    row.dischtime
                )
                released = 1 if released_gap.days < 30 else 0
            position = position_of_label[idx]
            if position == count - 1:
                index_corrected = 0
                gap = None
            else:
                next_row = stay_df.iloc[position + 1]
                gap = pd.to_datetime(next_row.admittime) - pd.to_datetime(
                    row.dischtime
                )
                index_corrected = 1 if gap.days < 30 else 0
            bounded = (
                0
                if gap is None
                else int(pd.Timedelta(0) <= gap <= pd.Timedelta(days=30))
            )
            if released != index_corrected:
                measurement["stays_changed_by_index_misuse"] += 1
            if released != bounded:
                measurement["stays_changed_by_missing_lower_bound"] += 1
            if gap is not None and gap < pd.Timedelta(0):
                measurement["negative_gap_stays"] += 1
            records.append(
                (
                    int(row.stay_id),
                    int(released),
                    int(index_corrected),
                    int(bounded),
                    float("nan") if gap is None else gap.total_seconds() / 3600.0,
                )
            )
    frame = pd.DataFrame(
        records,
        columns=[
            "stay_id",
            "readmission_30d",
            "index_corrected",
            "bounded",
            "next_admission_gap_hours",
        ],
    )
    return frame, measurement


def ensure_source(mimic_root: Path, cache_root: Path) -> dict:
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        required = [
            cache_root / value
            for key, value in manifest["files"].items()
            if key != "author_task_root"
        ]
        if all(path.exists() for path in required):
            return manifest

    release = assert_mimic4_release(mimic_root)
    all_stays, candidates = released_candidates(mimic_root)
    stats = _event_table_parts(mimic_root, candidates, cache_root)
    events_path = _collect_event_parts(cache_root, candidates)
    observed = set(
        pl.scan_parquet(events_path)
        .select("stay_id")
        .unique()
        .collect()["stay_id"]
        .to_list()
    )
    diagnosis_stays = released_diagnosis_stays(mimic_root, all_stays)
    cohort = candidates.loc[
        candidates["stay_id"].isin(observed)
        & candidates["stay_id"].isin(diagnosis_stays)
    ].copy()
    cohort["split"] = assign_split(cohort)
    cohort["mortality_label"] = cohort["mortality"].astype(np.int8)

    labels, measurement = released_readmission_labels(all_stays)
    cohort = cohort.merge(
        labels[["stay_id", "readmission_30d", "index_corrected", "bounded"]],
        on="stay_id",
        how="left",
        validate="1:1",
    )
    if cohort["readmission_30d"].isna().any():
        raise AssertionError("a cohort stay has no released readmission label")
    cohort["readmission_label"] = np.where(
        cohort["mortality"].astype(int) == 1,
        1,
        cohort["readmission_30d"].astype(int),
    ).astype(np.int8)

    candidate_splits = candidates.assign(split=assign_split(candidates))
    observed_mask = candidates["stay_id"].isin(observed)
    diagnosis_mask = candidates["stay_id"].isin(diagnosis_stays)
    dropped_no_events = int((~observed_mask).sum())
    dropped_no_diagnosis = int((observed_mask & ~diagnosis_mask).sum())
    actual_counts = cohort["split"].value_counts().to_dict()
    if actual_counts != EXPECTED_COUNTS:
        raise AssertionError(f"DPMM MIMIC-IV cohort counts differ: {actual_counts}")
    cohort = cohort.sort_values(["split", "source_name"]).reset_index(drop=True)
    cohort_path = cache_root / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False, compression="zstd")
    labels_path = cache_root / "readmission_labels.parquet"
    labels.to_parquet(labels_path, index=False, compression="zstd")

    cohort_events_path = _restrict_events_to_cohort(cache_root, events_path, cohort)
    ihm_root = _write_author_csvs(
        cache_root,
        cohort,
        cohort_events_path,
        "in-hospital-mortality",
        "mortality_label",
    )
    readm_root = _write_author_csvs(
        cache_root,
        cohort,
        cohort_events_path,
        "readmission",
        "readmission_label",
        episode_source=ihm_root,
    )
    normalizer_path = _fit_released_normalizer(cache_root, readm_root, ihm_root)

    files = {
        "cohort": cohort_path.name,
        "structured_events": cohort_events_path.name,
        "candidate_events": events_path.name,
        "readmission_labels": labels_path.name,
        "normalizer": normalizer_path.name,
        "author_task_root": str(readm_root.parent.relative_to(cache_root)),
    }
    manifest = {
        "status": "exact_reconstruction",
        "upstream_commit": UPSTREAM_COMMIT,
        "mimic_iv_version": "2.2",
        "mimic_iv_version_evidence": (
            "paper_code/dpmm/README.md links https://physionet.org/content/mimiciv/2.2/; "
            "the released reader selects the `race` admissions column, which exists "
            "only from MIMIC-IV 2.0 onward; the stale nb_rows constants in "
            "mimic3csv.read_events_table_by_row are MedFuse v1.0 tqdm totals"
        ),
        "released_stale_row_totals": RELEASED_STALE_ROW_TOTALS,
        "observed_source_rows": {
            table: value["source_rows"] for table, value in stats.items()
        },
        "retained_source_rows": {
            table: value["retained_rows"] for table, value in stats.items()
        },
        "release_sha256": release,
        "source_resource_sha256": {
            "itemid_to_variable_map.csv": sha256(ITEM_MAP),
            "variable_ranges.csv": sha256(VARIABLE_RANGES),
            "testset_iv.csv": sha256(TESTSET),
            "valset_iv.csv": sha256(VALSET),
            "discretizer_config.json": sha256(DISCRETIZER_CONFIG),
        },
        "files": files,
        "file_sha256": {
            name: sha256(cache_root / relative)
            for name, relative in files.items()
            if (cache_root / relative).is_file()
        },
        "counts": actual_counts,
        "released_candidates_with_los_at_least_48h": len(candidates),
        "released_candidate_counts": candidate_splits["split"]
        .value_counts()
        .to_dict(),
        "dropped_for_no_retained_measurement": dropped_no_events,
        "dropped_for_no_released_diagnosis_row": dropped_no_diagnosis,
        "mortality_positives": cohort.groupby("split")["mortality_label"]
        .sum()
        .astype(int)
        .to_dict(),
        "readmission_positives": cohort.groupby("split")["readmission_label"]
        .sum()
        .astype(int)
        .to_dict(),
        "readmission_defect_measurement": measurement,
        "sample_id_sha256": ordered_text_hash(cohort["stay_id"]),
        "source_name_sha256": ordered_text_hash(cohort["source_name"]),
        "mortality_label_sha256": array_hash(
            cohort["mortality_label"].to_numpy(np.int8)
        ),
        "readmission_label_sha256": array_hash(
            cohort["readmission_label"].to_numpy(np.int8)
        ),
        "normalizer_reconstruction": (
            "released MIMIC-IV behavior: train_listfile.csv only, no validation "
            "leakage; the released paths.py points both MIMIC-IV tasks at the "
            "single data_mimic4/readm_ts.normalizer file"
        ),
        "author_view_tasks": ["in-hospital-mortality", "readmission"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_source(cache_root: Path) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((cache_root / "source_manifest.json").read_text())
    cohort = pd.read_parquet(cache_root / manifest["files"]["cohort"])
    return cohort, manifest


def make_discretizer_normalizer(cache_root: Path):
    discretizer = Discretizer(
        timestep=TIMESTEP,
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
        config_path=str(DISCRETIZER_CONFIG),
    )
    config = json.loads(DISCRETIZER_CONFIG.read_text())
    synthetic = np.asarray(
        [["0", *[config["normal_values"][x] for x in config["id_to_channel"]]]]
    )
    header = ["Hours", *config["id_to_channel"]]
    output_header = discretizer.transform(synthetic, header=header, end=HOURS)[1].split(
        ","
    )
    continuous = [index for index, name in enumerate(output_header) if "->" not in name]
    normalizer = Normalizer(fields=continuous)
    normalizer.load_params(str(cache_root / "readm_ts.normalizer"))
    return discretizer, normalizer, output_header


def author_datasets(cache_root: Path, task: str) -> dict[str, EHRdataset]:
    discretizer, normalizer, _ = make_discretizer_normalizer(cache_root)
    root = cache_root / "author_view" / task
    # Shared episode directories: the released readmission builder emits the
    # same [0, 48]-truncated episode files as the mortality builder.
    episodes = cache_root / "author_view" / "in-hospital-mortality"
    return {
        "train": EHRdataset(
            discretizer,
            normalizer,
            str(root / "train_listfile.csv"),
            str(episodes / "train"),
        ),
        "validation": EHRdataset(
            discretizer,
            normalizer,
            str(root / "val_listfile.csv"),
            str(episodes / "train"),
        ),
        "test": EHRdataset(
            discretizer,
            normalizer,
            str(root / "test_listfile.csv"),
            str(episodes / "test"),
        ),
    }


def source_codes() -> tuple[str, ...]:
    config = json.loads(DISCRETIZER_CONFIG.read_text())
    codes = []
    for variable in config["id_to_channel"]:
        if config["is_categorical_channel"][variable]:
            codes.extend(
                f"{variable}//{value}" for value in config["possible_values"][variable]
            )
        else:
            codes.append(variable)
    return tuple(codes)
