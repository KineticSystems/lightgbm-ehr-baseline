"""Operational reconstruction of DPMM's released MIMIC-III source boundary.

This module deliberately calls the released cohort, cleaning, discretization,
normalization, dataset, note-cleaning, and collate code.  The only new work is
bounded I/O orchestration for the official compressed MIMIC-III 1.4 tables and
serialization of the attributable intermediate views needed by the export.
"""

from __future__ import annotations

import csv
import ast
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import polars as pl


PAPER_ROOT = Path(__file__).resolve().parent
MIMIC3EXTRACT = PAPER_ROOT / "mimic3extract"
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(MIMIC3EXTRACT))

from dataset_mimic3.ehr_dataset import EHRdataset, my_collate  # noqa: E402
from ehr_utils.preprocessing import Discretizer, Normalizer  # noqa: E402
from mimic3benchmark import mimic3csv  # noqa: E402
from mimic3benchmark.preprocessing import (  # noqa: E402
    clean_events,
    map_itemids_to_variables,
    read_itemid_to_variable_map,
)
UPSTREAM_COMMIT = "17d0314ff0b21dde31f3f932e02d4683826029aa"
TASK_ID = "mortality_48h_partial_ehr_note"
PUBLISHED_AUROC = 0.860
HOURS = 48.0
TIMESTEP = 2.0
NOTE_CODE = "NOTE"
SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = {"train": 14681, "validation": 3222, "test": 3236}
EXPECTED_POSITIVES = {"train": 1987, "validation": 436, "test": 374}
PUBLISHED_TOTAL_POSITIVES = 2795
EXPECTED_PAIRED = {"train": 3652, "validation": 815, "test": 806}
EXPECTED_PAIRED_POSITIVES = {"train": 513, "validation": 125, "test": 99}
PUBLISHED_PAIRED_POSITIVES = 736
EXPECTED_RAW_EVENT_ROWS = {
    "CHARTEVENTS": 330_712_483,
    "LABEVENTS": 27_854_055,
    "OUTPUTEVENTS": 4_349_218,
}

RELEASE_SHA256 = {
    "PATIENTS.csv.gz": "0a13db69044d3b01ab9a397f3192792c2a1d2e98c9cb41c4f94e0403996dc7c5",
    "ADMISSIONS.csv.gz": "68f128f19e95434500982d95e623638f22eadf1c6a07f4bfa6c39c21751d8085",
    "ICUSTAYS.csv.gz": "65f8c8686e791bee80df0115fc18f9f3c1087b85889ba9ce2d9c441204e467ab",
    "CHARTEVENTS.csv.gz": "5961ac36d8bfbd399088811f9a1d30bd6f482b23a84628593d3122b5a6be8bc4",
    "LABEVENTS.csv.gz": "79d86315dde6e1121f7a88f62c2b91b7b835364ac1e777248a58e4f180224ae6",
    "OUTPUTEVENTS.csv.gz": "70f14321d228b12d243a10974b7e41e41690031d6dbc16354ef78e3d1dfb4445",
    "NOTEEVENTS.csv.gz": "5897590eb213c3adaab1a95b3a4bf7290821f2dfe02a977d47805b02e82e25ec",
}

ITEM_MAP = (
    MIMIC3EXTRACT
    / "mimic3benchmark"
    / "resources"
    / "itemid_to_variable_map.csv"
)
VARIABLE_RANGES = (
    MIMIC3EXTRACT / "mimic3benchmark" / "resources" / "variable_ranges.csv"
)
TESTSET = MIMIC3EXTRACT / "mimic3benchmark" / "resources" / "testset.csv"
VALSET = MIMIC3EXTRACT / "mimic3models" / "resources" / "valset.csv"
DISCRETIZER_CONFIG = PAPER_ROOT / "ehr_utils" / "resources" / "discretizer_config.json"
TOKENIZER = (
    PAPER_ROOT
    / ".export_assets"
    / "external_artifacts"
    / "huggingface"
    / "TinyBERT_General_4L_312D_34707a33cd59"
)


def _released_note_preprocessor():
    """Load only definitions from the released hardcoded script.

    The source file reads an author-local NOTEEVENTS path at import time.  AST
    selection avoids executing that broken command while compiling its exact
    regexes and cleaning/tokenization function bodies unchanged.
    """
    source_path = (
        MIMIC3EXTRACT / "ClinicalNotesICU" / "mimic3" / "extract_notes.py"
    )
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SECTION_TITLES"
            for target in node.targets
        ):
            keep.append(node)
    module = ast.Module(body=keep, type_ignores=[])
    namespace = {"__name__": "dpmm_released_note_definitions"}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["preprocess_mimic"]


preprocess_mimic = _released_note_preprocessor()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = str(path.relative_to(root)).encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
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


def assert_mimic_release(root: Path) -> dict[str, str]:
    sums = (root / "SHA256SUMS.txt").read_text()
    for filename, expected in RELEASE_SHA256.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing exact MIMIC-III 1.4 table: {path}")
        if f"{expected} {filename}" not in sums:
            raise AssertionError(f"MIMIC-III release checksum changed: {filename}")
    return dict(RELEASE_SHA256)


def _compressed_dataframe(path: str | Path, *args, **kwargs):
    path = Path(path)
    if not path.exists() and path.suffix == ".csv":
        path = path.with_suffix(".csv.gz")
    return pd.read_csv(path, *args, **kwargs)


def released_stays(root: Path) -> pd.DataFrame:
    """Invoke the released MIMIC-III cohort functions on gzip transparently."""
    old_reader = mimic3csv.dataframe_from_csv
    mimic3csv.dataframe_from_csv = _compressed_dataframe
    try:
        patients = mimic3csv.read_patients_table(root)
        admits = mimic3csv.read_admissions_table(root)
        stays = mimic3csv.read_icustays_table(root)
        stays = mimic3csv.remove_icustays_with_transfers(stays)
        stays = mimic3csv.merge_on_subject_admission(stays, admits)
        stays = mimic3csv.merge_on_subject(stays, patients)
        stays = mimic3csv.filter_admissions_on_nb_icustays(stays)
        stays = mimic3csv.add_age_to_icustays(stays)
        stays = mimic3csv.add_inunit_mortality_to_icustays(stays)
        stays = mimic3csv.add_inhospital_mortality_to_icustays(stays)
        stays = mimic3csv.filter_icustays_on_age(stays)
    finally:
        mimic3csv.dataframe_from_csv = old_reader

    stays = stays.sort_values(
        ["SUBJECT_ID", "INTIME", "OUTTIME"], kind="stable"
    ).reset_index(drop=True)
    stays["episode_number"] = stays.groupby("SUBJECT_ID").cumcount() + 1
    stays["source_name"] = (
        stays["SUBJECT_ID"].astype(str)
        + "_episode"
        + stays["episode_number"].astype(str)
        + "_timeseries.csv"
    )
    stays["los_hours"] = 24.0 * stays["LOS"].astype(float)
    candidates = stays.loc[stays["los_hours"] >= HOURS - 1e-6].copy()
    if candidates["ICUSTAY_ID"].duplicated().any():
        raise AssertionError("released candidate ICU stay IDs are not unique")
    return candidates


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
    subject = frame["SUBJECT_ID"].astype(int)
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
    if "stay_id" in frame:
        frame["stay_id"] = frame["stay_id"].astype("int64")
    if "raw_source_row" in frame:
        frame["raw_source_row"] = frame["raw_source_row"].astype("int64")
    if "CHARTTIME" in frame:
        frame["CHARTTIME"] = pd.to_datetime(frame["CHARTTIME"])
    if "VARIABLE" in frame:
        frame["VARIABLE"] = frame["VARIABLE"].astype("string")
    if "numeric_value" in frame:
        frame["numeric_value"] = frame["numeric_value"].astype("float64")
    if "categorical_value" in frame:
        frame["categorical_value"] = frame["categorical_value"].astype("string")
    for column in ("TEXT", "source_name", "split"):
        if column in frame:
            frame[column] = frame[column].astype("string")
    frame.to_parquet(path, index=False, compression="zstd")


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": pd.Series(dtype="int64"),
            "CHARTTIME": pd.Series(dtype="datetime64[ns]"),
            "VARIABLE": pd.Series(dtype="string"),
            "numeric_value": pd.Series(dtype="float64"),
            "categorical_value": pd.Series(dtype="string"),
            "raw_source_row": pd.Series(dtype="int64"),
        }
    )


def _empty_anchor_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": pd.Series(dtype="int64"),
            "CHARTTIME": pd.Series(dtype="datetime64[ns]"),
            "raw_source_row": pd.Series(dtype="int64"),
        }
    )


def _hour_floor(charttime: pd.Series, intime: pd.Series) -> np.ndarray:
    seconds = (charttime - intime).dt.total_seconds().to_numpy(np.float64)
    return np.floor(seconds / 3600.0).astype(np.int64)


def _event_table_parts(
    mimic_root: Path,
    candidates: pd.DataFrame,
    cache_root: Path,
    *,
    chunksize: int = 1_000_000,
) -> None:
    """Bounded operational wrapper around the released event mapping/cleaning."""
    part_root = cache_root / "event_parts"
    part_root.mkdir(parents=True, exist_ok=True)
    var_map = read_itemid_to_variable_map(str(ITEM_MAP))
    itemids = set(int(value) for value in var_map.index)
    categorical = json.loads(DISCRETIZER_CONFIG.read_text())["is_categorical_channel"]
    stay_lookup = candidates[
        ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]
    ].copy()
    stay_lookup.columns = [
        "SUBJECT_ID",
        "HADM_ID_STAY",
        "stay_id",
        "INTIME",
        "OUTTIME",
    ]
    anchor_lookup = stay_lookup[
        ["SUBJECT_ID", "HADM_ID_STAY", "stay_id", "INTIME"]
    ]
    table_specs = {
        "CHARTEVENTS": [
            "SUBJECT_ID",
            "HADM_ID",
            "ICUSTAY_ID",
            "CHARTTIME",
            "ITEMID",
            "VALUE",
            "VALUEUOM",
        ],
        "LABEVENTS": [
            "SUBJECT_ID",
            "HADM_ID",
            "CHARTTIME",
            "ITEMID",
            "VALUE",
            "VALUEUOM",
        ],
        "OUTPUTEVENTS": [
            "SUBJECT_ID",
            "HADM_ID",
            "ICUSTAY_ID",
            "CHARTTIME",
            "ITEMID",
            "VALUE",
            "VALUEUOM",
        ],
    }

    for table, usecols in table_specs.items():
        expected_chunks = math.ceil(EXPECTED_RAW_EVENT_ROWS[table] / chunksize)
        event_suffix = "parquet"
        expected_paths = [
            part_root / f"{table.lower()}_{index:04d}_events.{event_suffix}"
            for index in range(expected_chunks)
        ] + [
            part_root / f"{table.lower()}_{index:04d}_anchors.parquet"
            for index in range(expected_chunks)
        ]
        if all(path.exists() for path in expected_paths):
            print(
                f"{table}: using {expected_chunks} complete source checkpoints",
                flush=True,
            )
            continue
        offset = 0
        reader = pd.read_csv(
            mimic_root / f"{table}.csv.gz",
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk_index, raw in enumerate(reader):
            event_path = part_root / f"{table.lower()}_{chunk_index:04d}_events.parquet"
            anchor_path = part_root / f"{table.lower()}_{chunk_index:04d}_anchors.parquet"
            chunk_rows = len(raw)
            if event_path.exists() and anchor_path.exists():
                offset += chunk_rows
                continue
            raw["raw_source_row"] = np.arange(
                offset, offset + chunk_rows, dtype=np.int64
            )
            offset += chunk_rows
            raw["CHARTTIME"] = pd.to_datetime(raw["CHARTTIME"], errors="coerce")
            raw["ICUSTAY_ID"] = raw.get(
                "ICUSTAY_ID", pd.Series(index=raw.index, dtype=float)
            )
            raw["ICUSTAY_ID"] = raw["ICUSTAY_ID"].fillna(-1).astype(int)
            raw["HADM_ID"] = raw["HADM_ID"].fillna(-1).astype(int)

            # Preserve the released validate_events.py boundary: HADM_ID must
            # map to this released stay and a present ICUSTAY_ID must agree;
            # absent ICU IDs (notably labs) are recovered from that HADM_ID.
            validated = raw.loc[
                raw["CHARTTIME"].notna() & raw["HADM_ID"].notna(),
                [
                    "SUBJECT_ID",
                    "HADM_ID",
                    "ICUSTAY_ID",
                    "CHARTTIME",
                    "raw_source_row",
                ],
            ].copy()
            validated = validated.merge(
                anchor_lookup,
                left_on=["SUBJECT_ID", "HADM_ID"],
                right_on=["SUBJECT_ID", "HADM_ID_STAY"],
                how="inner",
            )
            validated = validated.loc[
                validated["ICUSTAY_ID"].eq(-1)
                | validated["ICUSTAY_ID"].eq(validated["stay_id"])
            ]
            anchors = validated
            if not anchors.empty:
                anchors = anchors.loc[
                    _hour_floor(anchors["CHARTTIME"], anchors["INTIME"]) > 0
                ]
                anchors = (
                    anchors.sort_values(["stay_id", "CHARTTIME", "raw_source_row"])
                    .groupby("stay_id", as_index=False, sort=False)
                    .first()[["stay_id", "CHARTTIME", "raw_source_row"]]
                )
            else:
                anchors = _empty_anchor_frame()
            _write_part(anchors, anchor_path)

            selected = raw.loc[
                raw["ITEMID"].isin(itemids) & raw["VALUE"].notna()
            ].copy()
            if selected.empty:
                narrowed = _empty_event_frame()
            else:
                selected["VALUEUOM"] = selected["VALUEUOM"].fillna("").astype(str)
                selected = map_itemids_to_variables(selected, var_map)
                selected = clean_events(selected)
                selected = selected.merge(
                    stay_lookup,
                    left_on=["SUBJECT_ID", "HADM_ID"],
                    right_on=["SUBJECT_ID", "HADM_ID_STAY"],
                    how="inner",
                )
                selected = selected.loc[
                    selected["ICUSTAY_ID"].eq(-1)
                    | selected["ICUSTAY_ID"].eq(selected["stay_id"])
                ]
                hours = (
                    selected["CHARTTIME"] - selected["INTIME"]
                ).dt.total_seconds() / 3600.0
                selected = selected.loc[
                    hours.ge(-1e-6) & hours.le(HOURS + 1e-6)
                ].copy()
                selected["numeric_value"] = np.nan
                selected["categorical_value"] = None
                is_cat = selected["VARIABLE"].map(categorical).astype(bool)
                selected.loc[~is_cat, "numeric_value"] = pd.to_numeric(
                    selected.loc[~is_cat, "VALUE"], errors="raise"
                )
                selected.loc[is_cat, "categorical_value"] = selected.loc[
                    is_cat, "VALUE"
                ].astype(str)
                narrowed = selected[
                    [
                        "stay_id",
                        "CHARTTIME",
                        "VARIABLE",
                        "numeric_value",
                        "categorical_value",
                        "raw_source_row",
                    ]
                ]
            _write_part(narrowed, event_path)
            print(
                f"{table}: chunk={chunk_index + 1} rows={offset:,} "
                f"retained={len(narrowed):,}",
                flush=True,
            )


def _collect_event_parts(cache_root: Path, candidates: pd.DataFrame) -> tuple[Path, Path]:
    event_output = cache_root / "structured_events.parquet"
    anchor_output = cache_root / "starttimes.parquet"
    if not event_output.exists():
        source = pl.scan_parquet(str(cache_root / "event_parts" / "*_events.parquet"))
        source = source.filter(pl.col("stay_id").is_not_null())
        events = (
            source.group_by(["stay_id", "CHARTTIME", "VARIABLE"])
            .agg(
                pl.col("numeric_value").max(),
                pl.col("categorical_value").max(),
                pl.col("raw_source_row").max(),
            )
            .sort(["stay_id", "CHARTTIME", "VARIABLE"])
            .collect(engine="streaming")
        )
        events.write_parquet(event_output, compression="zstd")
    if not anchor_output.exists():
        anchors = (
            pl.scan_parquet(str(cache_root / "event_parts" / "*_anchors.parquet"))
            .filter(pl.col("stay_id").is_not_null())
            .group_by("stay_id")
            .agg(
                pl.col("CHARTTIME").min(),
                pl.col("raw_source_row").min(),
            )
            .sort("stay_id")
            .collect(engine="streaming")
        )
        anchors.write_parquet(anchor_output, compression="zstd")
    return event_output, anchor_output


def _write_author_csvs(
    cache_root: Path, cohort: pd.DataFrame, structured_path: Path
) -> Path:
    source_root = cache_root / "author_view" / "in-hospital-mortality"
    complete = source_root / ".complete"
    if complete.exists():
        return source_root
    (source_root / "train").mkdir(parents=True, exist_ok=True)
    (source_root / "test").mkdir(parents=True, exist_ok=True)
    variables = json.loads(DISCRETIZER_CONFIG.read_text())["id_to_channel"]
    csv_columns = ["Hours", *sorted(variables)]
    cohort_by_stay = cohort.set_index("stay_id")
    events = pd.read_parquet(structured_path)
    events["CHARTTIME"] = pd.to_datetime(events["CHARTTIME"])
    for stay_id, group in events.groupby("stay_id", sort=True):
        row = cohort_by_stay.loc[int(stay_id)]
        hours = (
            group["CHARTTIME"] - pd.Timestamp(row["INTIME"])
        ).dt.total_seconds() / 3600.0
        group = group.assign(Hours=hours.to_numpy())
        group["value"] = np.where(
            group["categorical_value"].notna(),
            group["categorical_value"],
            group["numeric_value"],
        )
        wide = group.pivot(index="Hours", columns="VARIABLE", values="value")
        wide = wide.reindex(columns=sorted(variables)).reset_index()
        target = source_root / ("test" if row["split"] == "test" else "train")
        wide.to_csv(target / row["source_name"], index=False, columns=csv_columns)

    def list_frame(selected: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stay": selected["source_name"],
                "period_length": 0,
                "stay_id": selected["stay_id"].astype(np.int64),
                "y_true": selected["label"].astype(np.int64),
            }
        )

    combined = cohort.loc[cohort["split"].isin(["train", "validation"])].copy()
    records = list_frame(combined).to_dict("records")
    random.Random(49297).shuffle(records)
    pd.DataFrame(records).to_csv(source_root / "train" / "listfile.csv", index=False)
    for split, filename in (
        ("train", "train_listfile.csv"),
        ("validation", "val_listfile.csv"),
        ("test", "test_listfile.csv"),
    ):
        list_frame(cohort.loc[cohort["split"] == split]).to_csv(
            source_root / filename, index=False
        )
    list_frame(cohort.loc[cohort["split"] == "test"]).to_csv(
        source_root / "test" / "listfile.csv", index=False
    )
    complete.write_text("exact released wide episode view\n")
    return source_root


def _fit_released_normalizer(cache_root: Path, source_root: Path) -> Path:
    output = cache_root / "readm_ts.normalizer"
    if output.exists():
        return output
    # This is the released MIMIC-III behavior: the combined pre-val-split
    # train/listfile.csv is used, so validation rows enter the moments.
    listfile = source_root / "train" / "listfile.csv"
    names = pd.read_csv(listfile)["stay"].tolist()
    discretizer = Discretizer(
        timestep=TIMESTEP,
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
        config_path=str(DISCRETIZER_CONFIG),
    )
    first_path = source_root / "train" / names[0]
    first = pd.read_csv(first_path, dtype=str).fillna("").to_numpy(str)
    header = pd.read_csv(first_path, nrows=0).columns.tolist()
    new_header = discretizer.transform(first, header=header, end=HOURS)[1].split(",")
    continuous = [index for index, name in enumerate(new_header) if "->" not in name]
    normalizer = Normalizer(fields=continuous)
    for index, name in enumerate(names):
        path = source_root / "train" / name
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        data, _ = discretizer.transform(
            frame.to_numpy(str), header=frame.columns.tolist(), end=HOURS
        )
        normalizer._feed_data(data)
        if (index + 1) % 1000 == 0:
            print(f"normalizer: {index + 1:,}/{len(names):,}", flush=True)
    normalizer._save_params(str(output))
    return output


def _sample_note_lists(cohort: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for split in SPLITS:
        frame = cohort.loc[cohort["split"] == split].sort_values("source_name")
        sampled = frame.sample(frac=1 / 3, random_state=42)
        sampled = sampled.copy()
        sampled["note_list_position"] = np.arange(len(sampled), dtype=np.int64)
        selected.append(sampled)
    return pd.concat(selected, ignore_index=True)


def _note_parts(
    mimic_root: Path,
    sampled: pd.DataFrame,
    cache_root: Path,
    *,
    chunksize: int = 100_000,
) -> None:
    part_root = cache_root / "note_parts"
    part_root.mkdir(parents=True, exist_ok=True)
    lookup = sampled[
        ["stay_id", "SUBJECT_ID", "HADM_ID", "source_name", "split", "note_list_position"]
    ].copy()
    offset = 0
    reader = pd.read_csv(
        mimic_root / "NOTEEVENTS.csv.gz",
        usecols=["SUBJECT_ID", "HADM_ID", "CHARTTIME", "TEXT"],
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk_index, raw in enumerate(reader):
        path = part_root / f"notes_{chunk_index:04d}.parquet"
        chunk_rows = len(raw)
        if path.exists():
            offset += chunk_rows
            continue
        raw["raw_note_row"] = np.arange(offset, offset + chunk_rows, dtype=np.int64)
        offset += chunk_rows
        raw = raw.loc[
            raw["SUBJECT_ID"].notna()
            & raw["HADM_ID"].notna()
            & raw["CHARTTIME"].notna()
            & raw["TEXT"].notna()
        ].copy()
        raw["SUBJECT_ID"] = raw["SUBJECT_ID"].astype(np.int64)
        raw["HADM_ID"] = raw["HADM_ID"].astype(np.int64)
        raw = raw.merge(lookup, on=["SUBJECT_ID", "HADM_ID"], how="inner")
        raw["CHARTTIME"] = pd.to_datetime(raw["CHARTTIME"], errors="raise")
        _write_part(raw, path)
        if (chunk_index + 1) % 10 == 0:
            print(
                f"NOTEEVENTS: chunks={chunk_index + 1} rows={offset:,} retained={len(raw):,}",
                flush=True,
            )
    # Pandas/Arrow otherwise serializes empty object columns as Null, which
    # prevents Polars from scanning empty and nonempty source checkpoints as
    # one dataset.  Preserve the same empty rows with the explicit text schema.
    for path in sorted(part_root.glob("notes_*.parquet")):
        cached = pd.read_parquet(path)
        if cached.empty:
            _write_part(cached, path)


def _collect_notes(
    cache_root: Path, sampled: pd.DataFrame, anchors_path: Path
) -> tuple[Path, pd.DataFrame]:
    output = cache_root / "selected_notes.parquet"
    paired_output = cache_root / "paired_samples.parquet"
    if output.exists() and paired_output.exists():
        return output, pd.read_parquet(paired_output)
    notes = (
        pl.scan_parquet(str(cache_root / "note_parts" / "notes_*.parquet"))
        .sort(["stay_id", "CHARTTIME", "raw_note_row"])
        .unique(subset=["stay_id", "CHARTTIME"], keep="last", maintain_order=True)
        .sort(["stay_id", "CHARTTIME"])
        .collect(engine="streaming")
        .to_pandas()
    )
    anchors = pd.read_parquet(anchors_path).rename(columns={"CHARTTIME": "starttime"})
    start = anchors.set_index("stay_id")["starttime"]
    selected_rows = []
    paired_rows = []
    for split in SPLITS:
        split_sample = sampled.loc[sampled["split"] == split].sort_values(
            "note_list_position"
        )
        for row in split_sample.itertuples(index=False):
            if int(row.stay_id) not in start.index:
                continue
            source = notes.loc[notes["stay_id"] == int(row.stay_id)].sort_values(
                "CHARTTIME"
            )
            starttime = np.datetime64(start.loc[int(row.stay_id)])
            eligible = []
            for note in source.itertuples(index=False):
                delta = (
                    np.datetime64(note.CHARTTIME) - starttime
                ).astype("timedelta64[h]").astype(int)
                if delta <= HOURS + 1e-6:
                    eligible.append(note)
                else:
                    break
            # Preserve the released <= bug: exactly five notes are rejected.
            if len(eligible) <= 5:
                continue
            paired_rows.append(row._asdict())
            for slot, note in enumerate(eligible[-5:]):
                cleaned = " ".join(preprocess_mimic(str(note.TEXT)))
                selected_rows.append(
                    {
                        "stay_id": int(row.stay_id),
                        "split": split,
                        "slot": slot,
                        "CHARTTIME": pd.Timestamp(note.CHARTTIME),
                        "starttime": pd.Timestamp(starttime),
                        "raw_note_row": int(note.raw_note_row),
                        "text": cleaned,
                    }
                )
    selected_frame = pd.DataFrame(selected_rows)
    paired = pd.DataFrame(paired_rows)
    selected_frame.to_parquet(output, index=False, compression="zstd")
    paired.to_parquet(paired_output, index=False, compression="zstd")
    return output, paired


def _fusion_positions(cohort: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    result = []
    paired_names = {
        split: paired.loc[paired["split"] == split]
        .sort_values("note_list_position")["source_name"]
        .tolist()
        for split in SPLITS
    }
    for split in SPLITS:
        all_names = cohort.loc[cohort["split"] == split, "source_name"].tolist()
        ordered = paired_names[split] + sorted(set(all_names) - set(paired_names[split]))
        for position, name in enumerate(ordered):
            result.append((split, name, position, name in set(paired_names[split])))
    positions = pd.DataFrame(
        result, columns=["split", "source_name", "fusion_position", "note_present"]
    )
    return cohort.merge(positions, on=["split", "source_name"], validate="1:1")


def ensure_source(mimic_root: Path, cache_root: Path) -> dict:
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        required = [cache_root / value for value in manifest["files"].values()]
        if all(path.exists() for path in required):
            return manifest

    release = assert_mimic_release(mimic_root)
    nltk_root = Path(os.environ.get("NLTK_DATA", ""))
    punkt_root = nltk_root / "tokenizers" / "punkt_tab" / "english"
    if not punkt_root.is_dir():
        raise FileNotFoundError(
            f"missing pinned NLTK punkt_tab data in paper cache: {punkt_root}"
        )
    candidates = released_stays(mimic_root)
    _event_table_parts(mimic_root, candidates, cache_root)
    events_path, anchors_path = _collect_event_parts(cache_root, candidates)
    observed_stays = set(
        pl.scan_parquet(events_path).select("stay_id").unique().collect()["stay_id"].to_list()
    )
    cohort = candidates.loc[candidates["ICUSTAY_ID"].isin(observed_stays)].copy()
    cohort["split"] = assign_split(cohort)
    cohort["label"] = cohort["MORTALITY"].astype(np.int8)
    cohort = cohort.rename(columns={"ICUSTAY_ID": "stay_id"})

    actual_counts = cohort["split"].value_counts().to_dict()
    actual_positive = cohort.groupby("split")["label"].sum().astype(int).to_dict()
    if actual_counts != EXPECTED_COUNTS:
        raise AssertionError(f"DPMM cohort counts differ: {actual_counts}")
    if actual_positive != EXPECTED_POSITIVES:
        raise AssertionError(f"DPMM label counts differ: {actual_positive}")
    source_root = _write_author_csvs(cache_root, cohort, events_path)
    normalizer_path = _fit_released_normalizer(cache_root, source_root)
    sampled = _sample_note_lists(cohort)
    sampled_path = cache_root / "sampled_note_lists.parquet"
    sampled.to_parquet(sampled_path, index=False, compression="zstd")
    _note_parts(mimic_root, sampled, cache_root)
    notes_path, paired = _collect_notes(cache_root, sampled, anchors_path)
    actual_paired = paired["split"].value_counts().to_dict()
    if actual_paired != EXPECTED_PAIRED:
        raise AssertionError(f"DPMM paired-note counts differ: {actual_paired}")
    cohort = _fusion_positions(cohort, paired)
    cohort = cohort.sort_values(["split", "fusion_position"]).reset_index(drop=True)
    cohort_path = cache_root / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False, compression="zstd")

    files = {
        "cohort": cohort_path.name,
        "structured_events": events_path.name,
        "starttimes": anchors_path.name,
        "selected_notes": notes_path.name,
        "sampled_note_lists": sampled_path.name,
        "paired_samples": "paired_samples.parquet",
        "normalizer": normalizer_path.name,
        "author_task_root": str(source_root.relative_to(cache_root)),
    }
    hashes = {
        name: sha256(cache_root / relative)
        for name, relative in files.items()
        if (cache_root / relative).is_file()
    }
    manifest = {
        "status": "exact_reconstruction",
        "upstream_commit": UPSTREAM_COMMIT,
        "release_sha256": release,
        "source_resource_sha256": {
            "itemid_to_variable_map.csv": sha256(ITEM_MAP),
            "variable_ranges.csv": sha256(VARIABLE_RANGES),
            "testset.csv": sha256(TESTSET),
            "valset.csv": sha256(VALSET),
            "discretizer_config.json": sha256(DISCRETIZER_CONFIG),
            "nltk_punkt_tab_english_tree": tree_sha256(punkt_root),
            "tinybert_config.json": sha256(TOKENIZER / "config.json"),
            "tinybert_vocab.txt": sha256(TOKENIZER / "vocab.txt"),
        },
        "files": files,
        "file_sha256": hashes,
        "counts": actual_counts,
        "positives": actual_positive,
        "paired": actual_paired,
        "sample_id_sha256": ordered_text_hash(cohort["stay_id"]),
        "source_name_sha256": ordered_text_hash(cohort["source_name"]),
        "label_sha256": array_hash(cohort["label"].to_numpy(np.int8)),
        "notes": len(pd.read_parquet(notes_path)),
        "note_text_sha256": ordered_text_hash(pd.read_parquet(notes_path)["text"]),
        "normalizer_reconstruction": (
            "released full pre-validation train/listfile behavior; exact paper-run "
            "pickle was not released, so byte identity is unavailable"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_source(cache_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((cache_root / "source_manifest.json").read_text())
    cohort = pd.read_parquet(cache_root / manifest["files"]["cohort"])
    structured = pd.read_parquet(cache_root / manifest["files"]["structured_events"])
    notes = pd.read_parquet(cache_root / manifest["files"]["selected_notes"])
    return cohort, structured, notes, manifest


def make_discretizer_normalizer(cache_root: Path) -> tuple[Discretizer, Normalizer, list[str]]:
    discretizer = Discretizer(
        timestep=TIMESTEP,
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
        config_path=str(DISCRETIZER_CONFIG),
    )
    config = json.loads(DISCRETIZER_CONFIG.read_text())
    synthetic = np.asarray([["0", *[config["normal_values"][x] for x in config["id_to_channel"]]]])
    header = ["Hours", *config["id_to_channel"]]
    output_header = discretizer.transform(synthetic, header=header, end=HOURS)[1].split(",")
    continuous = [index for index, name in enumerate(output_header) if "->" not in name]
    normalizer = Normalizer(fields=continuous)
    normalizer.load_params(str(cache_root / "readm_ts.normalizer"))
    return discretizer, normalizer, output_header


def author_datasets(cache_root: Path) -> dict[str, EHRdataset]:
    discretizer, normalizer, _ = make_discretizer_normalizer(cache_root)
    root = cache_root / "author_view" / "in-hospital-mortality"
    return {
        "train": EHRdataset(
            discretizer,
            normalizer,
            str(root / "train_listfile.csv"),
            str(root / "train"),
        ),
        "validation": EHRdataset(
            discretizer,
            normalizer,
            str(root / "val_listfile.csv"),
            str(root / "train"),
        ),
        "test": EHRdataset(
            discretizer,
            normalizer,
            str(root / "test_listfile.csv"),
            str(root / "test"),
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
    return tuple(codes) + (NOTE_CODE,)
