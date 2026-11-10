"""Bounded export instrumentation of DiPro's released MIMIC pipeline.

The released shell pipeline eagerly expands and redistributes several hundred
million gzip-compressed rows.  This adapter calls the released cohort and split
functions directly, while keeping the exact release files compressed and the
large structured-event pass resumable in an ignored cache.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sqlite3
import sys
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
MIMIC_SOURCE = PAPER_ROOT / "mimic_data_extract"
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(MIMIC_SOURCE))

from mimic3benchmark import mimic3csv  # noqa: E402
from mimic3benchmark.constants import DEMOGRAPHIC, GCS, SELECTED_VAR  # noqa: E402
from mimic3benchmark.preprocessing import (  # noqa: E402
    clean_events,
    map_itemids_to_variables,
    read_hw_itemid_to_variable_map,
    read_itemid_to_variable_map,
)
from mimic3benchmark.subject import (  # noqa: E402
    add_hours_elpased_to_events,
    convert_events_to_timeseries,
    get_events_for_stay,
    get_first_valid_from_timeseries,
)


TASK_ID = "mortality_48h"
UPSTREAM_COMMIT = "64b12ec6fc9455ba873af0597804500f0d25afc2"
PAPER_SPLIT_COUNTS = {"train": 1889, "validation": 285, "test": 546}
PAPER_TOTAL_POSITIVE = 465
CHECKPOINT_TRAIN_POSITIVE = 316
SOURCE_TABLES = (
    (
        "chartevents",
        Path("icu/chartevents.csv.gz"),
        ("subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value", "valuenum"),
    ),
    (
        "outputevents",
        Path("icu/outputevents.csv.gz"),
        ("subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value", "valueuom"),
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _required(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def assert_exact_sources(
    mimic_root: Path,
    cxr_root: Path,
    imagenome_root: Path,
) -> dict[str, str]:
    """Require the paper's exact named releases before cohort construction."""
    required = {
        "mimic_manifest": mimic_root / "SHA256SUMS.txt",
        "patients": mimic_root / "hosp" / "patients.csv.gz",
        "admissions": mimic_root / "hosp" / "admissions.csv.gz",
        "icustays": mimic_root / "icu" / "icustays.csv.gz",
        "cxr_manifest": cxr_root / "SHA256SUMS.txt",
        "cxr_metadata": cxr_root / "mimic-cxr-2.0.0-metadata.csv.gz",
        "cxr_chexpert": cxr_root / "mimic-cxr-2.0.0-chexpert.csv.gz",
        "imagenome_manifest": imagenome_root / "SHA256SUMS.txt",
        "scene_graph_zip": imagenome_root
        / "silver_dataset"
        / "scene_graph.zip",
        "gold_comparison": imagenome_root
        / "gold_dataset"
        / "gold_object_comparison_with_coordinates.txt",
    }
    for path in required.values():
        _required(path)

    mimic_manifest = required["mimic_manifest"].read_text()
    expected_mimic = {
        "patients": "1557e9e412c69c32696a1b45d45e02b4b2ca99e4399aaa317ad2f97d8d510fb2",
        "admissions": "59f93820fda13ad3237e331b477b65ea7ce173108e8f470f0a2a4118b6ba9133",
        "icustays": "29fe37b0c7ad16c3d17d1cfc7b7b054553182ca98a52c6d21b45d40d4df33121",
    }
    for name, digest in expected_mimic.items():
        suffix = {
            "patients": "hosp/patients.csv.gz",
            "admissions": "hosp/admissions.csv.gz",
            "icustays": "icu/icustays.csv.gz",
        }[name]
        if f"{digest} {suffix}" not in mimic_manifest:
            raise AssertionError(f"MIMIC-IV 2.0 manifest mismatch for {suffix}")

    cxr_manifest = required["cxr_manifest"].read_text()
    expected_cxr = {
        "cxr_metadata": "6a3748ce77724c0dfe7d2def8f47643e989e3bbf0795bc13b89c1578e1649d6b",
        "cxr_chexpert": "8a827c9adfd17dc4c94b129b6a007e55d4f75af6e7686441eb386c5ff6d1ddde",
    }
    for name, digest in expected_cxr.items():
        suffix = required[name].name
        if f"{digest} {suffix}" not in cxr_manifest:
            raise AssertionError(f"MIMIC-CXR-JPG 2.0.0 manifest mismatch for {suffix}")

    return {name: sha256_file(path) for name, path in required.items()}


def build_all_stays(mimic_root: Path) -> pd.DataFrame:
    """Call the released cohort helpers in their executable order."""
    patients = mimic3csv.read_patients_table(
        mimic_root / "hosp" / "patients.csv.gz"
    )
    admissions = mimic3csv.read_admissions_table(
        mimic_root / "hosp" / "admissions.csv.gz"
    )
    stays = mimic3csv.read_icustays_table(mimic_root / "icu" / "icustays.csv.gz")
    stays = mimic3csv.remove_icustays_with_transfers(stays)
    stays = mimic3csv.merge_on_subject_admission(stays, admissions)
    stays = mimic3csv.merge_on_subject(stays, patients)
    stays = mimic3csv.filter_admissions_on_nb_icustays(stays)
    stays = mimic3csv.add_age_to_icustays(stays)
    stays = mimic3csv.add_inunit_mortality_to_icustays(stays)
    stays = mimic3csv.add_inhospital_mortality_to_icustays(stays)
    return mimic3csv.filter_icustays_on_age(stays)


@contextlib.contextmanager
def _import_args(arguments: list[str]):
    original = sys.argv
    sys.argv = arguments
    try:
        yield
    finally:
        sys.argv = original


def _scene_graph_names(scene_graph_zip: Path) -> list[str]:
    with zipfile.ZipFile(scene_graph_zip) as archive:
        return [
            Path(name).name
            for name in archive.namelist()
            if name.endswith("_SceneGraph.json")
        ]


def call_released_cxr_alignment(
    all_stays_path: Path,
    cxr_root: Path,
    scene_graph_zip: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Invoke the released alignment, virtualizing only the ZIP directory list."""
    module_name = "mimic3benchmark.scripts.get_cxr"
    with _import_args(
        [
            module_name,
            "--all_stay_csv_path",
            str(all_stays_path),
            "--mimic_cxr_meta",
            str(cxr_root / "mimic-cxr-2.0.0-metadata.csv.gz"),
            "--chexpert_label",
            str(cxr_root / "mimic-cxr-2.0.0-chexpert.csv.gz"),
            "--ImaGenome",
            str(scene_graph_zip),
            "--output_dir",
            str(output_dir),
        ]
    ):
        from mimic3benchmark.scripts import get_cxr

    names = _scene_graph_names(scene_graph_zip)
    real_listdir = os.listdir

    def listdir(path):
        if Path(path) == scene_graph_zip:
            return names
        return real_listdir(path)

    with mock.patch.object(get_cxr.os, "listdir", side_effect=listdir):
        get_cxr.align_cxr_with_stay(
            cxr_root / "mimic-cxr-2.0.0-metadata.csv.gz",
            cxr_root / "mimic-cxr-2.0.0-chexpert.csv.gz",
            scene_graph_zip,
            all_stays_path,
            output_dir,
        )
    return pd.read_csv(output_dir / "cxr_stay.csv")


def call_released_mortality_split(
    all_stays_path: Path,
    cxr_stay_path: Path,
    golden_path: Path,
    cache_root: Path,
) -> dict[str, pd.DataFrame]:
    """Invoke the released splitter; empty deferred-task files avoid new logic."""
    empty_progression = cache_root / "empty_progression"
    empty_progression.mkdir(parents=True, exist_ok=True)
    progression_columns = [
        "subject_id",
        "study_id",
        "cxr1_dicom_id",
        "cxr2_dicom_id",
        "disease",
        "progression",
        "corrected_progression",
        "from_golden",
        "stay_id",
        "cxr1_time",
        "cxr2_time",
    ]
    for disease in (
        "lung opacity",
        "pleural effusion",
        "atelectasis",
        "enlarged cardiac silhouette",
        "pulmonary edema",
        "consolidation",
        "pneumonia",
    ):
        pd.DataFrame(columns=progression_columns).to_csv(
            empty_progression / f"{disease}_image_pairs.csv", index=False
        )

    split_root = cache_root / "split"
    module_name = "mimic3benchmark.scripts.split_train_and_test"
    with _import_args(
        [
            module_name,
            "--all_stay_csv_path",
            str(all_stays_path),
            "--cxr_stay",
            str(cxr_stay_path),
            "--split_save_dir",
            str(split_root),
            "--golden_dir",
            str(golden_path),
            "--progression_task_label_dir",
            str(empty_progression),
            "--all_comparison_path",
            str(cache_root / "unused_all_comparison.csv"),
        ]
    ):
        from mimic3benchmark.scripts import split_train_and_test

    split_train_and_test.split_dataset_for_each_task(
        all_stays_path,
        cxr_stay_path,
        golden_path,
        split_root,
        empty_progression,
    )
    return {
        split: pd.read_csv(split_root / "mortality_los" / filename)
        for split, filename in (
            ("train", "train.csv"),
            ("validation", "val.csv"),
            ("test", "test.csv"),
        )
    }


def build_cohort(
    mimic_root: Path,
    cxr_root: Path,
    imagenome_root: Path,
    cache_root: Path,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    completed = (
        cache_root / "cohort_summary.json",
        cache_root / "all_stays.csv",
        cache_root / "cxr_stay.csv",
        cache_root / "split" / "mortality_los" / "train.csv",
        cache_root / "split" / "mortality_los" / "val.csv",
        cache_root / "split" / "mortality_los" / "test.csv",
    )
    if all(path.is_file() for path in completed):
        print((cache_root / "cohort_summary.json").read_text(), end="")
        print("cohort: using completed exact-source cache", flush=True)
        return
    source_hashes = assert_exact_sources(mimic_root, cxr_root, imagenome_root)
    all_stays = build_all_stays(mimic_root)
    all_stays_path = cache_root / "all_stays.csv"
    all_stays.to_csv(all_stays_path, index=False)
    cxr_stay = call_released_cxr_alignment(
        all_stays_path,
        cxr_root,
        imagenome_root / "silver_dataset" / "scene_graph.zip",
        cache_root,
    )
    splits = call_released_mortality_split(
        all_stays_path,
        cache_root / "cxr_stay.csv",
        imagenome_root
        / "gold_dataset"
        / "gold_object_comparison_with_coordinates.txt",
        cache_root,
    )
    summary = {
        "all_stays": len(all_stays),
        "cxr_rows": len(cxr_stay),
        "cxr_stays": int(cxr_stay["stay_id"].nunique()),
        "splits": {
            split: {
                "count": len(frame),
                "positive": int(frame["mortality_inhospital"].sum()),
                "subjects": int(frame["subject_id"].nunique()),
            }
            for split, frame in splits.items()
        },
        "source_hashes": source_hashes,
    }
    total = sum(item["count"] for item in summary["splits"].values())
    positive = sum(item["positive"] for item in summary["splits"].values())
    summary["total"] = total
    summary["positive"] = positive
    (cache_root / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _connect_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS cache_state "
        "(stage TEXT PRIMARY KEY, payload TEXT NOT NULL)"
    )
    return connection


def _state(connection: sqlite3.Connection, stage: str):
    row = connection.execute(
        "SELECT payload FROM cache_state WHERE stage = ?", (stage,)
    ).fetchone()
    return None if row is None else json.loads(row[0])


def _set_state(connection: sqlite3.Connection, stage: str, payload: dict) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO cache_state(stage, payload) VALUES (?, ?)",
        (stage, json.dumps(payload, sort_keys=True, separators=(",", ":"))),
    )
    connection.commit()


def selected_variable_map() -> pd.DataFrame:
    frame = read_itemid_to_variable_map(
        MIMIC_SOURCE
        / "mimic3benchmark"
        / "resources"
        / "itemid_to_variable_map.csv"
    )
    if len(frame) != 42 or not set(SELECTED_VAR).issubset(set(frame["variable"])):
        raise AssertionError("released DiPro ITEMID mapping changed")
    if frame["itemid"].duplicated().any() or frame["variable"].duplicated().any():
        raise AssertionError("released ITEMID mapping is no longer one-to-one")
    return frame


def scan_selected_events(
    mimic_root: Path,
    all_stays_path: Path,
    database_path: Path,
) -> None:
    """Scan exact gzip tables once and retain only released mapped ITEMIDs."""
    connection = _connect_cache(database_path)
    mapping = selected_variable_map()
    itemids = set(mapping["itemid"].astype(int))
    subjects = set(pd.read_csv(all_stays_path, usecols=["subject_id"])["subject_id"])
    connection.execute(
        "CREATE TABLE IF NOT EXISTS selected_events ("
        "source_table INTEGER NOT NULL, source_row INTEGER NOT NULL, "
        "subject_id INTEGER NOT NULL, hadm_id INTEGER, stay_id INTEGER, "
        "charttime TEXT NOT NULL, itemid INTEGER NOT NULL, value TEXT, "
        "valuenum TEXT, PRIMARY KEY(source_table, source_row))"
    )
    connection.commit()

    for table_number, (table_name, relative_path, columns) in enumerate(SOURCE_TABLES):
        stage = f"scan_{table_name}"
        if _state(connection, stage) is not None:
            print(f"{table_name}: using completed selected-event scan", flush=True)
            continue
        path = _required(mimic_root / relative_path)
        connection.execute(
            "DELETE FROM selected_events WHERE source_table = ?", (table_number,)
        )
        connection.commit()
        offset = 0
        retained = 0
        for chunk_number, chunk in enumerate(
            pd.read_csv(path, usecols=list(columns), chunksize=750_000), start=1
        ):
            chunk_rows = len(chunk)
            selected = chunk.loc[
                chunk["subject_id"].isin(subjects) & chunk["itemid"].isin(itemids)
            ].copy()
            if not selected.empty:
                selected.insert(
                    0,
                    "source_row",
                    np.arange(offset, offset + chunk_rows, dtype=np.int64)[
                        selected.index.to_numpy() - chunk.index[0]
                    ],
                )
                selected.insert(0, "source_table", table_number)
                if table_name == "outputevents":
                    selected = selected.rename(columns={"valueuom": "valuenum"})
                selected[
                    [
                        "source_table",
                        "source_row",
                        "subject_id",
                        "hadm_id",
                        "stay_id",
                        "charttime",
                        "itemid",
                        "value",
                        "valuenum",
                    ]
                ].to_sql(
                    "selected_events",
                    connection,
                    if_exists="append",
                    index=False,
                    chunksize=10_000,
                )
                retained += len(selected)
            offset += chunk_rows
            if chunk_number % 20 == 0:
                connection.commit()
                print(
                    f"{table_name}: chunks={chunk_number} rows={offset:,} "
                    f"selected={retained:,}",
                    flush=True,
                )
        connection.commit()
        _set_state(
            connection,
            stage,
            {"rows": offset, "selected": retained, "sha256": sha256_file(path)},
        )
        print(f"{table_name}: complete selected={retained:,}", flush=True)

    if _state(connection, "selected_index") is None:
        connection.execute(
            "CREATE INDEX IF NOT EXISTS selected_events_subject "
            "ON selected_events(subject_id, source_table, source_row)"
        )
        connection.commit()
        _set_state(connection, "selected_index", {"status": "complete"})

    hw_mapping = read_hw_itemid_to_variable_map(
        MIMIC_SOURCE
        / "mimic3benchmark"
        / "resources"
        / "itemid_to_variable_map.csv"
    )
    hw_mapping = hw_mapping.loc[hw_mapping["variable"].isin(["Height", "Weight"])]
    extra_hw_itemids = set(hw_mapping["itemid"].astype(int)) - itemids
    modern_extra_hw = {itemid for itemid in extra_hw_itemids if itemid > 220_000}
    if modern_extra_hw != {226_512, 226_531, 226_707}:
        raise AssertionError(
            f"released supplemental height/weight mapping changed: {modern_extra_hw}"
        )
    if _state(connection, "scan_height_weight_supplement") is None:
        path = _required(mimic_root / "icu" / "chartevents.csv.gz")
        offset = 0
        retained = 0
        columns = SOURCE_TABLES[0][2]
        for chunk_number, chunk in enumerate(
            pd.read_csv(path, usecols=list(columns), chunksize=750_000), start=1
        ):
            chunk_rows = len(chunk)
            selected = chunk.loc[
                chunk["subject_id"].isin(subjects)
                & chunk["itemid"].isin(modern_extra_hw)
            ].copy()
            if not selected.empty:
                selected.insert(
                    0,
                    "source_row",
                    np.arange(offset, offset + chunk_rows, dtype=np.int64)[
                        selected.index.to_numpy() - chunk.index[0]
                    ],
                )
                selected.insert(0, "source_table", 0)
                selected[
                    [
                        "source_table",
                        "source_row",
                        "subject_id",
                        "hadm_id",
                        "stay_id",
                        "charttime",
                        "itemid",
                        "value",
                        "valuenum",
                    ]
                ].to_sql(
                    "selected_events",
                    connection,
                    if_exists="append",
                    index=False,
                    chunksize=10_000,
                )
                retained += len(selected)
            offset += chunk_rows
            if chunk_number % 40 == 0:
                connection.commit()
                print(
                    f"height/weight supplement: rows={offset:,} selected={retained:,}",
                    flush=True,
                )
        connection.commit()
        _set_state(
            connection,
            "scan_height_weight_supplement",
            {
                "rows": offset,
                "selected": retained,
                "itemids": sorted(modern_extra_hw),
                "sha256": sha256_file(path),
            },
        )
        print(f"height/weight supplement: complete selected={retained:,}", flush=True)
    connection.close()


def _source_events_for_subject(
    connection: sqlite3.Connection,
    subject_id: int,
) -> pd.DataFrame:
    events = pd.read_sql_query(
        "SELECT subject_id, hadm_id, stay_id, charttime, itemid, value, valuenum "
        "FROM selected_events WHERE subject_id = ? "
        "ORDER BY source_table, source_row",
        connection,
        params=(int(subject_id),),
    )
    if events.empty:
        return events
    events["charttime"] = pd.to_datetime(events["charttime"])
    events["hadm_id"] = events["hadm_id"].fillna(-1).astype(int)
    events["stay_id"] = events["stay_id"].fillna(-1).astype(int)
    events["itemid"] = events["itemid"].astype(int)
    events["valuenum"] = events["valuenum"].fillna("").astype(str)
    return events


def _empty_moments() -> dict[str, dict[str, float]]:
    return {
        variable: {
            "total_sum": 0.0,
            "total_squared_sum": 0.0,
            "total_count": 0,
            "total_outlier_count": 0,
        }
        for variable in SELECTED_VAR
    }


def build_episode_cache(
    database_path: Path,
    all_stays_path: Path,
    split_root: Path,
) -> None:
    """Call released mapping/cleaning helpers and cache selected raw episodes."""
    connection = _connect_cache(database_path)
    if _state(connection, "episodes") is not None:
        print("episodes: using completed cache", flush=True)
        connection.close()
        return
    connection.execute("DROP TABLE IF EXISTS selected_episode_cells")
    connection.execute("DROP TABLE IF EXISTS episode_metadata")
    connection.execute("DROP TABLE IF EXISTS stay_height_weight")
    connection.execute(
        "CREATE TABLE selected_episode_cells ("
        "stay_id INTEGER NOT NULL, hours REAL NOT NULL, variable_index INTEGER NOT NULL, "
        "episode_row INTEGER NOT NULL, value TEXT NOT NULL, "
        "PRIMARY KEY(stay_id, variable_index, episode_row))"
    )
    connection.execute(
        "CREATE TABLE episode_metadata (stay_id INTEGER PRIMARY KEY, max_interval INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE stay_height_weight ("
        "stay_id INTEGER PRIMARY KEY, height REAL, weight REAL)"
    )
    connection.commit()

    all_stays = pd.read_csv(all_stays_path)
    all_stays["intime"] = pd.to_datetime(all_stays["intime"])
    all_stays["outtime"] = pd.to_datetime(all_stays["outtime"])
    selected_stays = set()
    for filename in ("train.csv", "val.csv", "test.csv"):
        selected_stays.update(
            pd.read_csv(split_root / "mortality_los" / filename)["stay_id"]
            .astype(int)
            .tolist()
        )
    mapping = selected_variable_map()
    hw_mapping = read_hw_itemid_to_variable_map(
        MIMIC_SOURCE
        / "mimic3benchmark"
        / "resources"
        / "itemid_to_variable_map.csv"
    )
    hw_mapping = hw_mapping.loc[hw_mapping["variable"].isin(["Height", "Weight"])]
    variables = list(mapping["variable"].unique())
    moments = _empty_moments()
    outliers = pd.read_csv(
        MIMIC_SOURCE / "mimic3benchmark" / "resources" / "variable_ranges.csv"
    )
    outliers["LEVEL2"] = outliers["LEVEL2"].str.lower()
    ranges = {
        row["LEVEL2"]: (float(row["OUTLIER LOW"]), float(row["OUTLIER HIGH"]))
        for _, row in outliers.drop_duplicates("LEVEL2", keep="first").iterrows()
    }
    missing_episodes = []
    inserted_cells = 0
    grouped = all_stays.groupby("subject_id", sort=False)
    for subject_number, (subject_id, stays) in enumerate(grouped, start=1):
        raw = _source_events_for_subject(connection, int(subject_id))
        if raw.empty:
            missing_episodes.extend(stays["stay_id"].astype(int).tolist())
            continue
        mapped = clean_events(map_itemids_to_variables(raw.copy(), mapping))
        if mapped.empty:
            missing_episodes.extend(stays["stay_id"].astype(int).tolist())
            continue
        timeseries = convert_events_to_timeseries(mapped, variables=variables)
        hw_mapped = clean_events(map_itemids_to_variables(raw.copy(), hw_mapping))
        hw_timeseries = (
            None
            if hw_mapped.empty
            else convert_events_to_timeseries(
                hw_mapped, variables=["Height", "Weight"]
            )
        )
        for _, stay in stays.iterrows():
            stay_id = int(stay["stay_id"])
            episode = get_events_for_stay(
                timeseries.copy(), stay_id, stay["intime"], stay["outtime"]
            )
            hw_episode = (
                None
                if hw_timeseries is None
                else get_events_for_stay(
                    hw_timeseries.copy(),
                    stay_id,
                    stay["intime"],
                    stay["outtime"],
                )
            )
            if episode.empty:
                missing_episodes.append(stay_id)
                continue
            episode = add_hours_elpased_to_events(episode, stay["intime"]).set_index(
                "HOURS"
            ).sort_index(axis=0)
            if hw_episode is None or hw_episode.empty:
                height = np.nan
                weight = np.nan
            else:
                hw_episode = add_hours_elpased_to_events(
                    hw_episode, stay["intime"]
                ).set_index("HOURS").sort_index(axis=0)
                height = get_first_valid_from_timeseries(hw_episode, "Height")
                weight = get_first_valid_from_timeseries(hw_episode, "Weight")
            connection.execute(
                "INSERT INTO stay_height_weight(stay_id, height, weight) VALUES (?, ?, ?)",
                (
                    stay_id,
                    None if pd.isna(height) else float(height),
                    None if pd.isna(weight) else float(weight),
                ),
            )

            for variable in SELECTED_VAR:
                values = episode[variable]
                if variable in GCS:
                    values = values.map(GCS[variable]["values"])
                values = pd.to_numeric(values, errors="coerce")
                low, high = ranges[variable.lower()]
                moments[variable]["total_outlier_count"] += int(
                    ((values < low) | (values > high)).sum()
                )
                values = values.clip(lower=low, upper=high)
                moments[variable]["total_sum"] += float(values.sum())
                moments[variable]["total_squared_sum"] += float((values**2).sum())
                moments[variable]["total_count"] += int(values.count())

            if stay_id in selected_stays:
                maximum = int(episode.index.max() // 1)
                connection.execute(
                    "INSERT INTO episode_metadata(stay_id, max_interval) VALUES (?, ?)",
                    (stay_id, maximum),
                )
                bounded = episode.loc[(episode.index >= -1.0) & (episode.index < 49.0)]
                rows = []
                for variable_index, variable in enumerate(SELECTED_VAR):
                    values = bounded[variable]
                    for episode_row, (hours, value) in enumerate(
                        values[values.notna()].items()
                    ):
                        rows.append(
                            (
                                stay_id,
                                float(hours),
                                variable_index,
                                episode_row,
                                str(value),
                            )
                        )
                connection.executemany(
                    "INSERT INTO selected_episode_cells("
                    "stay_id, hours, variable_index, episode_row, value) "
                    "VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
                inserted_cells += len(rows)
        if subject_number % 500 == 0:
            connection.commit()
            print(
                f"episodes: subjects={subject_number:,}/{len(grouped):,} "
                f"selected_cells={inserted_cells:,}",
                flush=True,
            )
    connection.commit()

    moment_rows = []
    for variable in SELECTED_VAR:
        item = moments[variable]
        count = item["total_count"]
        if count == 0:
            raise AssertionError(f"no source observations for {variable}")
        item["outlier_rate"] = item["total_outlier_count"] / count
        item["mean"] = item["total_sum"] / count
        item["std"] = (item["total_squared_sum"] / count - item["mean"] ** 2) ** 0.5
        moment_rows.append({"variables": variable, **item})
    moments_path = database_path.parent / "mean_std.csv"
    pd.DataFrame(moment_rows).to_csv(moments_path, index=False)
    missing_path = database_path.parent / "missing_episodes.json"
    missing_path.write_text(json.dumps(sorted(set(missing_episodes))) + "\n")
    _set_state(
        connection,
        "episodes",
        {
            "selected_cells": inserted_cells,
            "missing_stays": len(set(missing_episodes)),
            "mean_std_sha256": sha256_file(moments_path),
        },
    )
    _set_state(
        connection,
        "duplicate_range_moments_exact",
        {
            "policy": "released first matching variable_ranges.csv row",
            "mean_std_sha256": sha256_file(moments_path),
        },
    )
    connection.close()


def repair_cached_duplicate_range_moments(
    database_path: Path,
    all_stays_path: Path,
) -> None:
    """Repair a cache made before the released first-row range rule was explicit."""
    connection = _connect_cache(database_path)
    if _state(connection, "duplicate_range_moments_exact") is not None:
        connection.close()
        return
    moments_path = database_path.parent / "mean_std.csv"
    moments = pd.read_csv(moments_path).set_index("variables")
    # Bilirubin is the release's sole duplicated LEVEL2 row. Later duplicates
    # contain NaN limits; released preprocessing consistently selects row zero.
    outliers = pd.read_csv(
        MIMIC_SOURCE / "mimic3benchmark" / "resources" / "variable_ranges.csv"
    )
    outliers["LEVEL2"] = outliers["LEVEL2"].str.lower()
    duplicated = outliers.loc[
        outliers["LEVEL2"].duplicated(keep=False), "LEVEL2"
    ].unique().tolist()
    if duplicated != ["bilirubin"]:
        raise AssertionError(f"released duplicate range rows changed: {duplicated}")
    first = outliers.loc[outliers["LEVEL2"] == "bilirubin"].iloc[0]
    low = float(first["OUTLIER LOW"])
    high = float(first["OUTLIER HIGH"])
    mapping = selected_variable_map()
    itemids = mapping.loc[mapping["variable"] == "Bilirubin", "itemid"].astype(int)
    if itemids.tolist() != [225690]:
        raise AssertionError("released Bilirubin ITEMID changed")

    source = pd.read_sql_query(
        "SELECT subject_id, stay_id, charttime, value FROM selected_events "
        "WHERE itemid = ? AND (CAST(value AS REAL) < ? OR CAST(value AS REAL) > ?) "
        "ORDER BY subject_id, charttime, value",
        connection,
        params=(int(itemids.iloc[0]), low, high),
    )
    duplicate_times = source.groupby(["subject_id", "charttime"]).size()
    if len(duplicate_times) and int(duplicate_times.max()) != 1:
        raise AssertionError("outlying Bilirubin source values are no longer unique")
    stays = pd.read_csv(all_stays_path, parse_dates=["intime", "outtime"])
    correction_sum = 0.0
    correction_squared_sum = 0.0
    correction_count = 0
    for row in source.itertuples(index=False):
        timestamp = pd.Timestamp(row.charttime)
        metadata_stay = connection.execute(
            "SELECT MIN(COALESCE(stay_id, -1)) FROM selected_events "
            "WHERE subject_id = ? AND charttime = ?",
            (int(row.subject_id), str(row.charttime)),
        ).fetchone()[0]
        subject_stays = stays.loc[stays["subject_id"] == int(row.subject_id)]
        multiplicity = int(
            (
                (subject_stays["stay_id"] == int(metadata_stay))
                | (
                    (subject_stays["intime"] <= timestamp)
                    & (subject_stays["outtime"] >= timestamp)
                )
            ).sum()
        )
        value = float(row.value)
        clipped = min(max(value, low), high)
        correction_sum += multiplicity * (clipped - value)
        correction_squared_sum += multiplicity * (clipped**2 - value**2)
        correction_count += multiplicity
    moments.loc["Bilirubin", "total_sum"] += correction_sum
    moments.loc["Bilirubin", "total_squared_sum"] += correction_squared_sum
    moments.loc["Bilirubin", "total_outlier_count"] += correction_count
    count = float(moments.loc["Bilirubin", "total_count"])
    mean = float(moments.loc["Bilirubin", "total_sum"]) / count
    variance = (
        float(moments.loc["Bilirubin", "total_squared_sum"]) / count - mean**2
    )
    moments.loc["Bilirubin", "mean"] = mean
    moments.loc["Bilirubin", "std"] = variance**0.5
    moments.loc["Bilirubin", "outlier_rate"] = correction_count / count
    moments.reset_index().to_csv(moments_path, index=False)
    episodes_state = _state(connection, "episodes")
    episodes_state["mean_std_sha256"] = sha256_file(moments_path)
    _set_state(connection, "episodes", episodes_state)
    _set_state(
        connection,
        "duplicate_range_moments_exact",
        {
            "policy": "released first matching variable_ranges.csv row",
            "corrected_occurrences": correction_count,
            "mean_std_sha256": sha256_file(moments_path),
        },
    )
    connection.close()


def build_demographics(
    mimic_root: Path,
    database_path: Path,
    all_stays_path: Path,
    output_root: Path,
) -> None:
    connection = _connect_cache(database_path)
    if _state(connection, "demographics") is not None:
        print("demographics: using completed cache", flush=True)
        connection.close()
        return
    all_stays = pd.read_csv(all_stays_path)
    admissions = pd.read_csv(mimic_root / "hosp" / "admissions.csv.gz")
    merged = all_stays.merge(admissions, on=["hadm_id", "subject_id"], how="left")
    height_weight = pd.read_sql_query(
        "SELECT stay_id, height, weight FROM stay_height_weight", connection
    )
    merged = merged.merge(height_weight, on="stay_id", how="left")
    demographic = merged[
        [
            "subject_id",
            "stay_id",
            "age",
            "gender",
            "race",
            "language",
            "marital_status",
            "height",
            "weight",
        ]
    ]
    demographic_path = output_root / "demographic.csv"
    demographic.to_csv(demographic_path, index=False)

    # Invoke the released fitting/encoding function unchanged.
    module_name = "mimic3benchmark.scripts.ehr_preprocess"
    with _import_args(
        [
            module_name,
            "--all_stay_csv_path",
            str(all_stays_path),
            "--subject_episode_dir",
            str(output_root / "unused"),
            "--output_dir",
            str(output_root),
            "--meta_save_dir",
            str(output_root),
            "--admission_path",
            str(mimic_root / "hosp" / "admissions.csv.gz"),
            "--OUTLIER_FILE",
            str(
                MIMIC_SOURCE
                / "mimic3benchmark"
                / "resources"
                / "variable_ranges.csv"
            ),
        ]
    ):
        from mimic3benchmark.scripts import ehr_preprocess

    ehr_preprocess.process_demographic(demographic_path, output_root)
    processed = pd.read_csv(output_root / "demographic_processed.csv")
    missing = set(DEMOGRAPHIC) - set(processed.columns)
    if missing:
        raise AssertionError(f"released demographic vocabulary is missing {sorted(missing)}")
    _set_state(
        connection,
        "demographics",
        {
            "rows": len(processed),
            "columns": list(processed.columns),
            "sha256": sha256_file(output_root / "demographic_processed.csv"),
        },
    )
    connection.close()


def regularize_selected_episodes(database_path: Path, output_root: Path) -> None:
    connection = _connect_cache(database_path)
    if _state(connection, "regularized") is not None:
        print("regularized episodes: using completed cache", flush=True)
        connection.close()
        return
    output_dir = output_root / "ehr_preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    means = pd.read_csv(output_root / "mean_std.csv").set_index("variables")
    outliers = pd.read_csv(
        MIMIC_SOURCE / "mimic3benchmark" / "resources" / "variable_ranges.csv"
    )
    outliers["LEVEL2"] = outliers["LEVEL2"].str.lower()
    ranges = outliers.drop_duplicates("LEVEL2", keep="first").set_index("LEVEL2")
    metadata = connection.execute(
        "SELECT stay_id, max_interval FROM episode_metadata ORDER BY stay_id"
    ).fetchall()
    lengths = {}
    for number, (stay_id, max_interval) in enumerate(metadata, start=1):
        last_interval = min(int(max_interval), 48)
        interval_df = pd.DataFrame(
            {"regular_interval": range(-1, last_interval + 1)}
        )
        cells = pd.read_sql_query(
            "SELECT hours, variable_index, episode_row, value "
            "FROM selected_episode_cells WHERE stay_id = ? "
            "ORDER BY variable_index, episode_row",
            connection,
            params=(int(stay_id),),
        )
        for variable_index, variable in enumerate(SELECTED_VAR):
            values = cells.loc[cells["variable_index"] == variable_index, ["hours", "value"]].copy()
            if variable in GCS:
                values["value"] = values["value"].map(GCS[variable]["values"])
            values["value"] = pd.to_numeric(values["value"], errors="coerce")
            low = float(ranges.loc[variable.lower(), "OUTLIER LOW"])
            high = float(ranges.loc[variable.lower(), "OUTLIER HIGH"])
            values["value"] = values["value"].clip(lower=low, upper=high)
            values["regular_interval"] = values["hours"] // 1
            values = values.sort_values(["regular_interval", "hours"])
            values = values.dropna(subset=["value"])
            latest = values.groupby("regular_interval").nth(-1).reset_index()
            latest = latest[["regular_interval", "value"]].rename(
                columns={"value": variable}
            )
            interval_df = interval_df.merge(latest, on="regular_interval", how="left")
            interval_df[f"{variable}_mask"] = interval_df[variable].notna().astype(int)
            normal = float(ranges.loc[variable.lower(), "IMPUTE"])
            if pd.isna(interval_df[variable].iloc[0]):
                interval_df.loc[interval_df.index[0], variable] = normal
            interval_df[variable] = interval_df[variable].ffill()
        for variable in SELECTED_VAR:
            interval_df[variable] = (
                interval_df[variable] - float(means.loc[variable, "mean"])
            ) / float(means.loc[variable, "std"])
        interval_df = interval_df.iloc[1:].copy()
        interval_df.to_csv(
            output_dir / f"New_episode{int(stay_id)}_timeseries.csv", index=False
        )
        lengths[int(stay_id)] = len(interval_df)
        if number % 250 == 0:
            print(
                f"regularized episodes: {number:,}/{len(metadata):,}", flush=True
            )
    lengths_path = output_root / "episode_lengths.json"
    lengths_path.write_text(json.dumps(lengths, sort_keys=True) + "\n")
    _set_state(
        connection,
        "regularized",
        {
            "stays": len(metadata),
            "minimum_length": min(lengths.values()),
            "maximum_length": max(lengths.values()),
            "lengths_sha256": sha256_file(lengths_path),
        },
    )
    connection.close()


def build_structured_source(
    mimic_root: Path,
    cache_root: Path,
) -> None:
    cohort_summary = cache_root / "cohort_summary.json"
    if not cohort_summary.is_file():
        raise FileNotFoundError("build-cohort must complete first")
    database_path = cache_root / "source.sqlite"
    scan_selected_events(mimic_root, cache_root / "all_stays.csv", database_path)
    build_episode_cache(
        database_path, cache_root / "all_stays.csv", cache_root / "split"
    )
    repair_cached_duplicate_range_moments(
        database_path, cache_root / "all_stays.csv"
    )
    build_demographics(
        mimic_root, database_path, cache_root / "all_stays.csv", cache_root
    )
    regularize_selected_episodes(database_path, cache_root)
    print(json.dumps({"status": "complete", "cache": str(cache_root)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build-cohort", "build-structured"))
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cxr-root", type=Path, required=True)
    parser.add_argument("--imagenome-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build-cohort":
        build_cohort(
            args.mimic_root,
            args.cxr_root,
            args.imagenome_root,
            args.cache_root,
        )
    elif args.command == "build-structured":
        build_structured_source(args.mimic_root, args.cache_root)


if __name__ == "__main__":
    main()
