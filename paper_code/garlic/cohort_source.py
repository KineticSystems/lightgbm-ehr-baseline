"""Bounded the adapter around GARLIC's released MIMIC-III path.

The release builds one cohort-wide dense tensor and therefore cannot fit inside
the workspace resource envelope.  This adapter executes the pinned release's
feature-producing source blocks verbatim, then materializes the same last
deterministic boundary admission by admission in an ignored SQLite cache.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import math
import sqlite3
import sys
import textwrap
from datetime import datetime, timezone
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


REPOSITORY = Path(__file__).resolve().parent
WORKSPACE = REPOSITORY.parent.parent
AUTHOR_PREPROCESSOR = REPOSITORY / "data" / "preprocess_mimic_iii.py"
AUTHOR_DATA_UTILS = REPOSITORY / "data" / "data_utils.py"
UPSTREAM = json.loads((REPOSITORY / "UPSTREAM.json").read_text())

TASK_ID = "in_hospital_mortality"
PUBLISHED_COHORT_ROWS = 49_380
EXPECTED_COHORT_ROWS = 49_375
PUBLISHED_POSITIVE = 5_747
EXPECTED_POSITIVE = 5_742
EXPECTED_FEATURES = 103
OBSERVATION_HOURS = 48.0
MAX_SOURCE_TIMESTAMPS = 300
MAX_MODEL_TIMESTAMPS = 200
RAW_FILES = (
    "ADMISSIONS.csv.gz",
    "PATIENTS.csv.gz",
    "ICUSTAYS.csv.gz",
    "CHARTEVENTS.csv.gz",
    "LABEVENTS.csv.gz",
    "OUTPUTEVENTS.csv.gz",
    "INPUTEVENTS_CV.csv.gz",
    "INPUTEVENTS_MV.csv.gz",
    "D_ITEMS.csv.gz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def source_segment(start_marker: str, end_marker: str) -> tuple[str, str]:
    source = AUTHOR_PREPROCESSOR.read_text()
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    segment = textwrap.dedent(source[start:end])
    return segment, hashlib.sha256(segment.encode()).hexdigest()


PREFIX_SOURCE, PREFIX_SHA256 = source_segment(
    "    # Get all ICU stays.", "    # Observation"
)
OBSERVATION_SOURCE, OBSERVATION_SHA256 = source_segment(
    "    # blood pressure", "    # Free some memory."
)
OUTPUT_SOURCE, OUTPUT_SHA256 = source_segment(
    "    # Extract outputevents.", "    # Extract CV and MV inputevents."
)
INPUT_SOURCE, INPUT_SHA256 = source_segment(
    "    # Extract CV and MV inputevents.", "    # Save data."
)


class PandasProxy:
    """Delegate pandas calls while resolving the official compressed release."""

    def __getattr__(self, name: str) -> Any:
        return getattr(pd, name)

    @staticmethod
    def read_csv(path: Any, *args: Any, **kwargs: Any) -> Any:
        source = Path(path)
        if not source.exists() and source.suffix == ".csv":
            compressed = Path(f"{source}.gz")
            if compressed.exists():
                source = compressed
        return pd.read_csv(source, *args, **kwargs)


PANDAS_PROXY = PandasProxy()


def _exec_released(segment: str, namespace: dict[str, Any]) -> dict[str, Any]:
    globals_dict = {
        "pd": PANDAS_PROXY,
        "np": np,
        "tqdm": lambda value, **_: value,
    }
    globals_dict.update(namespace)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        exec(compile(segment, str(AUTHOR_PREPROCESSOR), "exec"), globals_dict)
    return globals_dict


def author_adult_icu(mimic_root: Path) -> pd.DataFrame:
    namespace = _exec_released(
        PREFIX_SOURCE,
        {"mimic_data_dir": f"{mimic_root.resolve()}/"},
    )
    return namespace["icu"].copy()


def _literal_integer_candidates() -> set[int]:
    """Return a conservative item-id superset parsed from the pinned source."""
    tree = ast.parse(AUTHOR_PREPROCESSOR.read_text())
    values: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        try:
            candidate = ast.literal_eval(node)
        except (ValueError, TypeError):
            continue

        def visit(value: Any) -> None:
            if isinstance(value, int) and value >= 0:
                values.add(value)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)

        visit(candidate)
    return values


ITEM_ID_CANDIDATES = _literal_integer_candidates()


def declared_feature_names() -> tuple[str, ...]:
    """Recover the 122 output names and declaration order from author source."""
    tree = ast.parse(AUTHOR_PREPROCESSOR.read_text())
    found: list[tuple[int, int, str]] = []
    serial = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "features"
            for target in node.targets
        ) and isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                found.append((node.lineno, serial, str(ast.literal_eval(key))))
                serial += 1
        elif isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, str
        ):
            if any("NAME" in ast.dump(target) for target in node.targets):
                found.append((node.lineno, serial, node.value.value))
                serial += 1
    ordered: list[str] = []
    for _, _, name in sorted(found):
        if name not in ordered:
            ordered.append(name)
    if len(ordered) != 122:
        raise AssertionError(f"released preprocessor declares {len(ordered)} features")
    return tuple(ordered)


DECLARED_FEATURES = declared_feature_names()


def removed_feature_indices() -> tuple[int, ...]:
    tree = ast.parse(AUTHOR_DATA_UTILS.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "remove_mod_idx"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if value:
                return tuple(int(index) for index in value)
    raise AssertionError("released 19-index feature removal was not found")


REMOVE_INDICES = removed_feature_indices()
if len(REMOVE_INDICES) != 19:
    raise AssertionError("GARLIC must remove exactly 19 of 122 source features")


def _empty_observation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "HADM_ID": pd.Series(dtype="float64"),
            "ICUSTAY_ID": pd.Series(dtype="float64"),
            "ITEMID": pd.Series(dtype="int64"),
            "CHARTTIME": pd.Series(dtype="object"),
            "VALUE": pd.Series(dtype="object"),
            "VALUENUM": pd.Series(dtype="float64"),
            "VALUEUOM": pd.Series(dtype="object"),
            "TABLE": pd.Series(dtype="object"),
            "SOURCE_ROW": pd.Series(dtype="int64"),
        }
    )


def _observation_sentinel() -> pd.DataFrame:
    """One discarded row keeps an unsafe released empty-.loc assignment valid."""
    return pd.DataFrame(
        {
            "HADM_ID": [-1.0],
            "ICUSTAY_ID": [np.nan],
            "ITEMID": [50885],
            "CHARTTIME": ["1900-01-01 00:00:00"],
            "VALUE": [None],
            "VALUENUM": [1.0],
            "VALUEUOM": [None],
            "TABLE": ["lab"],
            "SOURCE_ROW": [-1],
        }
    )


def _observation_features(ch: pd.DataFrame, la: pd.DataFrame) -> pd.DataFrame:
    result = _exec_released(OBSERVATION_SOURCE, {"ch": ch, "la": la})["events"]
    return result.loc[result.HADM_ID != -1]


def _feature_blocks() -> dict[str, int]:
    names = [name for name in DECLARED_FEATURES if name not in {"DBP", "SBP", "MBP"}]
    blocks = {name: position + 1 for position, name in enumerate(names)}
    blocks.update({"DBP": 0, "SBP": 0, "MBP": 0})
    return blocks


FEATURE_BLOCK = _feature_blocks()
TEMP_F_IDS = {223761, 678, 679, 3654}
VENT_IDS = {
    720, 223848, 223849, 445, 448, 449, 450, 1340, 1486, 1600, 224687,
    639, 654, 681, 682, 683, 684, 224685, 224684, 224686, 218, 436, 535,
    444, 459, 224697, 224695, 224696, 224746, 224747, 221, 1, 1211,
    1655, 2000, 226873, 224738, 224419, 224750, 227187, 543, 5865,
    5866, 224707, 224709, 224705, 224706, 60, 437, 505, 506, 686,
    220339, 224700, 3459, 501, 502, 503, 224702, 223, 667, 668, 669,
    670, 671, 672, 224701,
}


def _annotate_observation_order(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["AUTHOR_BLOCK"] = frame["NAME"].map(FEATURE_BLOCK).astype("int32")
    frame["AUTHOR_SUBBLOCK"] = np.where(frame["TABLE"].eq("lab"), 1, 0)
    temperature = frame["NAME"].eq("Temperature")
    frame.loc[temperature, "AUTHOR_SUBBLOCK"] = frame.loc[
        temperature, "ITEMID"
    ].isin(TEMP_F_IDS).astype("int8")
    ventilation = frame["NAME"].eq("Mechanical Ventilation")
    frame.loc[ventilation, "AUTHOR_SUBBLOCK"] = frame.loc[
        ventilation, "ITEMID"
    ].isin(VENT_IDS).astype("int8")
    frame["AUTHOR_ROW"] = frame["SOURCE_ROW"].astype("int64")
    return frame


def _chart_frames(
    mimic_root: Path,
    adult_stays: set[int],
    cohort_start: dict[int, int] | None = None,
) -> Iterator[pd.DataFrame]:
    path = mimic_root / "CHARTEVENTS.csv.gz"
    offset = 0
    empty_lab = _observation_sentinel()
    reader = pd.read_csv(
        path,
        chunksize=10_000_000,
        usecols=[
            "HADM_ID",
            "ICUSTAY_ID",
            "ITEMID",
            "CHARTTIME",
            "VALUE",
            "VALUENUM",
            "VALUEUOM",
            "ERROR",
        ],
    )
    for chunk in reader:
        chunk["SOURCE_ROW"] = np.arange(offset, offset + len(chunk), dtype=np.int64)
        offset += len(chunk)
        chunk = chunk.loc[chunk.ICUSTAY_ID.isin(adult_stays)]
        chunk = chunk.loc[chunk["ERROR"] != 1]
        chunk = chunk.loc[chunk.CHARTTIME.notna()]
        if cohort_start is not None:
            chart_time = pd.to_datetime(chunk.CHARTTIME, errors="raise")
            start = chunk.HADM_ID.map(cohort_start)
            chunk = chunk.loc[
                start.notna()
                & (
                    chart_time.astype("int64") // 1_000
                    <= start.fillna(0).astype("int64")
                    + int(OBSERVATION_HOURS * 3_600_000_000)
                )
            ]
        chunk = chunk.loc[
            chunk.ITEMID.isin(ITEM_ID_CANDIDATES) | chunk.VALUE.eq("Ventilator")
        ]
        chunk = chunk.drop(columns=["ERROR"])
        chunk = chunk.loc[~(chunk.VALUE.isna() & chunk.VALUENUM.isna())]
        chunk["TABLE"] = "chart"
        if chunk.empty:
            continue
        yield _annotate_observation_order(
            _observation_features(chunk, empty_lab.copy())
        )


def _lab_frame(
    mimic_root: Path,
    adult_hadm: set[int],
    cohort_start: dict[int, int] | None = None,
) -> pd.DataFrame:
    la = pd.read_csv(
        mimic_root / "LABEVENTS.csv.gz",
        usecols=["HADM_ID", "ITEMID", "CHARTTIME", "VALUE", "VALUENUM", "VALUEUOM"],
    )
    la["SOURCE_ROW"] = np.arange(len(la), dtype=np.int64)
    la = la.loc[la.HADM_ID.isin(adult_hadm)]
    la.HADM_ID = la.HADM_ID.astype(int)
    la = la.loc[la.CHARTTIME.notna()]
    if cohort_start is not None:
        chart_time = pd.to_datetime(la.CHARTTIME, errors="raise")
        start = la.HADM_ID.map(cohort_start)
        la = la.loc[
            start.notna()
            & (
                chart_time.astype("int64") // 1_000
                <= start.fillna(0).astype("int64")
                + int(OBSERVATION_HOURS * 3_600_000_000)
            )
        ]
    la = la.loc[~(la.VALUE.isna() & la.VALUENUM.isna())]
    la["ICUSTAY_ID"] = np.nan
    la["TABLE"] = "lab"
    la = la.loc[la.ITEMID.isin(ITEM_ID_CANDIDATES)]
    return _annotate_observation_order(
        _observation_features(_empty_observation_frame(), la)
    )


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "HADM_ID": pd.Series(dtype="float64"),
            "ICUSTAY_ID": pd.Series(dtype="float64"),
            "ITEMID": pd.Series(dtype="int64"),
            "CHARTTIME": pd.Series(dtype="object"),
            "VALUENUM": pd.Series(dtype="float64"),
            "TABLE": pd.Series(dtype="object"),
            "NAME": pd.Series(dtype="object"),
        }
    )


def _non_observation_frame(
    mimic_root: Path,
    adult_icu: pd.DataFrame,
    segment: str,
) -> pd.DataFrame:
    namespace = _exec_released(
        segment,
        {
            "mimic_data_dir": f"{mimic_root.resolve()}/",
            "icu": adult_icu.copy(),
            "events": _empty_events(),
        },
    )
    events = namespace["events"].copy()
    events["AUTHOR_BLOCK"] = events["NAME"].map(FEATURE_BLOCK).astype("int32")
    events["AUTHOR_SUBBLOCK"] = 0
    events["AUTHOR_ROW"] = np.arange(len(events), dtype=np.int64)
    return events


def released_frame_items(
    mimic_root: Path,
    adult_icu: pd.DataFrame,
    cohort_start: dict[int, int] | None = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    adult_stays = set(adult_icu.ICUSTAY_ID.astype("int64"))
    adult_hadm = set(adult_icu.HADM_ID.astype("int64"))
    for index, frame in enumerate(
        _chart_frames(mimic_root, adult_stays, cohort_start)
    ):
        yield f"chart:{index:03d}", frame
    lab = _lab_frame(mimic_root, adult_hadm, cohort_start)
    if not lab.empty:
        yield "lab", lab
    output = _non_observation_frame(mimic_root, adult_icu, OUTPUT_SOURCE)
    if not output.empty:
        yield "output", output
    input_events = _non_observation_frame(mimic_root, adult_icu, INPUT_SOURCE)
    if not input_events.empty:
        yield "input", input_events


def released_frames(
    mimic_root: Path,
    adult_icu: pd.DataFrame,
    cohort_start: dict[int, int] | None = None,
) -> Iterator[pd.DataFrame]:
    for _, frame in released_frame_items(mimic_root, adult_icu, cohort_start):
        yield frame


class CacheStateError(RuntimeError):
    """The cache cannot be resumed without an explicit stage restart."""


def _initialize_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-262144;
        CREATE TABLE IF NOT EXISTS metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
          name TEXT PRIMARY KEY,
          completed_at TEXT NOT NULL,
          payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cohort (
          source_position INTEGER NOT NULL,
          hadm_id INTEGER PRIMARY KEY,
          subject_id INTEGER NOT NULL,
          label INTEGER NOT NULL,
          split TEXT NOT NULL,
          observation_start_us INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS candidates (
          hadm_id INTEGER NOT NULL,
          event_time_us INTEGER NOT NULL,
          feature_idx INTEGER NOT NULL,
          raw_value REAL,
          author_block INTEGER NOT NULL,
          author_subblock INTEGER NOT NULL,
          author_row INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS raw_cells (
          hadm_id INTEGER NOT NULL,
          sequence_row INTEGER NOT NULL,
          feature_idx INTEGER NOT NULL,
          raw_value REAL NOT NULL,
          event_time_us INTEGER NOT NULL,
          source_time REAL NOT NULL,
          PRIMARY KEY (hadm_id, sequence_row, feature_idx)
        );
        CREATE TABLE IF NOT EXISTS normalization (
          feature_idx INTEGER PRIMARY KEY,
          mean_value REAL NOT NULL,
          std_value REAL NOT NULL,
          observed INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sample_boundary (
          hadm_id INTEGER PRIMARY KEY,
          retained_length INTEGER NOT NULL,
          observed_cells INTEGER NOT NULL
        );
        """
    )
    return connection


def _put_metadata(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        (key, json.dumps(value, sort_keys=True)),
    )


def metadata(connection: sqlite3.Connection, key: str) -> Any:
    row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    if row is None:
        raise KeyError(key)
    return json.loads(row[0])


def _checkpoint(
    connection: sqlite3.Connection,
    name: str,
    payload: Any,
) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO checkpoints(name, completed_at, payload) "
        "VALUES (?, ?, ?)",
        (
            name,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(payload, sort_keys=True),
        ),
    )


def checkpoint(connection: sqlite3.Connection, name: str) -> Any | None:
    row = connection.execute(
        "SELECT payload FROM checkpoints WHERE name=?", (name,)
    ).fetchone()
    return None if row is None else json.loads(row[0])


def _table_count(connection: sqlite3.Connection, name: str) -> int:
    return int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])


def restart_cache_from(connection: sqlite3.Connection, stage: str) -> None:
    """Explicitly discard one stage and every downstream checkpoint."""
    stages = ("pass1", "pass2", "raw_cells", "normalization")
    if stage not in stages:
        raise ValueError(f"restart stage must be one of {stages}")
    position = stages.index(stage)
    if position == 0:
        connection.execute("DELETE FROM cohort")
        connection.execute("DELETE FROM candidates")
        connection.execute("DELETE FROM raw_cells")
        connection.execute("DELETE FROM sample_boundary")
        connection.execute("DELETE FROM normalization")
        connection.execute("DELETE FROM checkpoints")
        connection.execute("DELETE FROM metadata")
    elif position == 1:
        connection.execute("DELETE FROM candidates")
        connection.execute("DELETE FROM raw_cells")
        connection.execute("DELETE FROM sample_boundary")
        connection.execute("DELETE FROM normalization")
        connection.execute(
            "DELETE FROM checkpoints WHERE name='cache_complete' "
            "OR name='pass2_complete' OR name='raw_cells_complete' "
            "OR name='raw_cells_progress' OR name='normalization_complete' "
            "OR name LIKE 'pass2_unit:%'"
        )
        connection.execute("DELETE FROM metadata WHERE key='cache_summary'")
    elif position == 2:
        connection.execute("DELETE FROM raw_cells")
        connection.execute("DELETE FROM sample_boundary")
        connection.execute("DELETE FROM normalization")
        connection.execute(
            "DELETE FROM checkpoints WHERE name IN "
            "('cache_complete','raw_cells_complete','raw_cells_progress',"
            "'normalization_complete')"
        )
        connection.execute("DELETE FROM metadata WHERE key='cache_summary'")
    else:
        connection.execute("DELETE FROM normalization")
        connection.execute(
            "DELETE FROM checkpoints WHERE name IN "
            "('cache_complete','normalization_complete')"
        )
        connection.execute("DELETE FROM metadata WHERE key='cache_summary'")
    connection.commit()


def _frame_times(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["CHARTTIME"], errors="raise")


def _first_pass(
    mimic_root: Path,
    adult_icu: pd.DataFrame,
) -> tuple[dict[int, int], set[int], set[int], tuple[str, ...]]:
    minimum: dict[int, int] = {}
    event_hadm: set[int] = set()
    event_icu: set[int] = set()
    feature_first: dict[str, tuple[int, int, int]] = {}
    for frame in released_frames(mimic_root, adult_icu):
        if frame.empty:
            continue
        times = _frame_times(frame)
        post_loader = ~(times.isna() & frame.VALUENUM.isna())
        identities = frame.loc[post_loader & frame.HADM_ID.notna()]
        event_hadm.update(identities.HADM_ID.astype("int64").tolist())
        event_icu.update(
            identities.ICUSTAY_ID.dropna().astype("int64").tolist()
        )
        valid = post_loader & frame.HADM_ID.notna() & times.notna()
        if valid.any():
            work = pd.DataFrame(
                {
                    "hadm": frame.loc[valid, "HADM_ID"].astype("int64"),
                    "time": times.loc[valid].astype("int64") // 1_000,
                }
            )
            for hadm, value in work.groupby("hadm", sort=False).time.min().items():
                hadm_int = int(hadm)
                value_int = int(value)
                previous = minimum.get(hadm_int)
                if previous is None or value_int < previous:
                    minimum[hadm_int] = value_int
        grouped = frame.loc[post_loader].groupby(
            ["NAME", "AUTHOR_BLOCK", "AUTHOR_SUBBLOCK"], sort=False
        ).AUTHOR_ROW.min()
        for (name, block, subblock), row in grouped.items():
            key = (int(block), int(subblock), int(row))
            if name not in feature_first or key < feature_first[name]:
                feature_first[str(name)] = key
    ordered = tuple(name for name, _ in sorted(feature_first.items(), key=lambda item: item[1]))
    if len(ordered) != 122 or set(ordered) != set(DECLARED_FEATURES):
        raise AssertionError(
            f"released event path yielded {len(ordered)} of 122 declared features"
        )
    return minimum, event_hadm, event_icu, ordered


def _author_cohort(
    mimic_root: Path,
    adult_icu: pd.DataFrame,
    minimum: dict[int, int],
    event_hadm: set[int],
    event_icu: set[int],
) -> list[dict[str, Any]]:
    icu = pd.read_csv(
        mimic_root / "ICUSTAYS.csv.gz",
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS"],
    )
    icu.drop_duplicates(inplace=True)
    icu = icu.loc[icu.HADM_ID.isin(event_hadm)]
    icu = icu.loc[icu.ICUSTAY_ID.isin(event_icu)]
    selected_subjects = set(icu.SUBJECT_ID.astype("int64"))
    admission_subject: dict[int, int] = {}
    for row in icu.itertuples(index=False):
        admission_subject.setdefault(int(row.HADM_ID), int(row.SUBJECT_ID))

    static_by_subject: dict[int, tuple[float, float]] = {}
    for row in adult_icu.itertuples(index=False):
        subject = int(row.SUBJECT_ID)
        if subject in static_by_subject:
            continue
        gender = 0.0 if row.GENDER == "F" else 1.0 if row.GENDER == "M" else np.nan
        age = float(row.AGE) if not pd.isna(row.AGE) else np.nan
        static_by_subject[subject] = (gender, age)

    admissions = pd.read_csv(
        mimic_root / "ADMISSIONS.csv.gz",
        usecols=[
            "SUBJECT_ID",
            "HADM_ID",
            "ADMITTIME",
            "DISCHTIME",
            "DEATHTIME",
            "HOSPITAL_EXPIRE_FLAG",
        ],
    )
    admissions.drop_duplicates(inplace=True)
    admissions = admissions.loc[admissions.SUBJECT_ID.isin(selected_subjects)]
    rows: list[dict[str, Any]] = []
    for source_position, row in enumerate(admissions.itertuples(index=False)):
        hadm = int(row.HADM_ID)
        if hadm not in minimum:
            continue
        subject = admission_subject.get(hadm)
        if subject is None:
            # This mirrors create_mor_large's tuple, but the selected-subject
            # filter should make the branch unreachable for retained rows.
            subject = int(row.SUBJECT_ID)
        gender, age = static_by_subject.get(subject, (np.nan, np.nan))
        rows.append(
            {
                "source_position": len(rows),
                "admission_row": source_position,
                "id": hadm,
                "hadm_id": hadm,
                "subject_id": subject,
                "label": int(row.HOSPITAL_EXPIRE_FLAG),
                "static": [gender, age],
                "observation_start_us": int(minimum[hadm]),
            }
        )

    sys.path.insert(0, str(REPOSITORY))
    from utils import get_data_split  # noqa: PLC0415

    train, validation, test = get_data_split(rows)
    split_by_hadm = {
        int(row["hadm_id"]): split
        for split, values in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
        for row in values
    }
    for row in rows:
        row["split"] = split_by_hadm[int(row["hadm_id"])]
    return rows


def _store_pass1_checkpoint(
    connection: sqlite3.Connection,
    cohort: list[dict[str, Any]],
    *,
    full_feature_order: tuple[str, ...],
    features: tuple[str, ...],
) -> None:
    """Atomically persist the cohort boundary before any second source scan."""
    if _table_count(connection, "cohort"):
        raise CacheStateError(
            "unmarked cohort rows already exist; use --restart-stage pass1"
        )
    connection.executemany(
        "INSERT INTO cohort VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                int(row["source_position"]),
                int(row["hadm_id"]),
                int(row["subject_id"]),
                int(row["label"]),
                str(row["split"]),
                int(row["observation_start_us"]),
            )
            for row in cohort
        ],
    )
    _checkpoint(
        connection,
        "pass1_complete",
        {
            "cohort_rows": len(cohort),
            "positive": sum(int(row["label"]) for row in cohort),
            "full_feature_order": list(full_feature_order),
            "features": list(features),
            "remove_indices": list(REMOVE_INDICES),
        },
    )
    connection.commit()


def _load_pass1_checkpoint(
    connection: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...]]:
    payload = checkpoint(connection, "pass1_complete")
    if payload is None:
        if _table_count(connection, "cohort"):
            raise CacheStateError(
                "cohort rows exist without pass1_complete; use --restart-stage pass1"
            )
        raise KeyError("pass1_complete")
    cohort = cohort_rows(connection)
    if len(cohort) != int(payload["cohort_rows"]):
        raise CacheStateError("pass1 cohort count disagrees with its checkpoint")
    return (
        cohort,
        tuple(payload["full_feature_order"]),
        tuple(payload["features"]),
    )


def _insert_candidates(
    connection: sqlite3.Connection,
    frame: pd.DataFrame,
    cohort_start: dict[int, int],
    feature_index: dict[str, int],
) -> int:
    if frame.empty:
        return 0
    times = _frame_times(frame)
    hadm = pd.to_numeric(frame.HADM_ID, errors="coerce")
    feature = frame.NAME.map(feature_index)
    work = pd.DataFrame(
        {
            "hadm": hadm,
            "time": times.astype("int64") // 1_000,
            "feature": feature,
            "value": pd.to_numeric(frame.VALUENUM, errors="coerce"),
            "block": frame.AUTHOR_BLOCK,
            "subblock": frame.AUTHOR_SUBBLOCK,
            "row": frame.AUTHOR_ROW,
        }
    )
    work = work.loc[work.hadm.notna() & times.notna() & work.feature.notna()]
    if work.empty:
        return 0
    work["hadm"] = work.hadm.astype("int64")
    work["start"] = work.hadm.map(cohort_start)
    work = work.loc[work.start.notna()]
    work = work.loc[
        work.time <= work.start.astype("int64") + int(OBSERVATION_HOURS * 3_600_000_000)
    ]
    if work.empty:
        return 0
    values = (
        (
            int(row.hadm),
            int(row.time),
            int(row.feature),
            None if pd.isna(row.value) else float(row.value),
            int(row.block),
            int(row.subblock),
            int(row.row),
        )
        for row in work.itertuples(index=False)
    )
    connection.executemany(
        "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?)", values
    )
    return len(work)


def _second_pass(
    connection: sqlite3.Connection,
    mimic_root: Path,
    adult_icu: pd.DataFrame,
    cohort: list[dict[str, Any]],
    feature_names: tuple[str, ...],
) -> int:
    completed = checkpoint(connection, "pass2_complete")
    if completed is not None:
        return int(completed["candidate_events"])
    starts = {int(row["hadm_id"]): int(row["observation_start_us"]) for row in cohort}
    feature_index = {name: index for index, name in enumerate(feature_names)}
    total = _table_count(connection, "candidates")
    if total and not connection.execute(
        "SELECT 1 FROM checkpoints WHERE name LIKE 'pass2_unit:%' LIMIT 1"
    ).fetchone():
        raise CacheStateError(
            "candidate rows exist without pass2 unit markers; use --restart-stage pass2"
        )
    units: list[str] = []
    for unit, frame in released_frame_items(mimic_root, adult_icu, starts):
        marker = f"pass2_unit:{unit}"
        units.append(unit)
        if checkpoint(connection, marker) is not None:
            continue
        inserted = _insert_candidates(connection, frame, starts, feature_index)
        total += inserted
        _checkpoint(connection, marker, {"rows": inserted})
        connection.commit()
    connection.execute(
        "CREATE INDEX IF NOT EXISTS candidates_order ON candidates "
        "(hadm_id, event_time_us, feature_idx, author_block DESC, "
        "author_subblock DESC, author_row DESC)"
    )
    _checkpoint(
        connection,
        "pass2_complete",
        {"candidate_events": total, "units": units},
    )
    connection.commit()
    return total


def _materialize_cells(
    connection: sqlite3.Connection,
    cohort: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    completed = checkpoint(connection, "raw_cells_complete")
    if completed is not None:
        return int(completed["retained_events"]), dict(completed["retained_length"])
    progress = checkpoint(connection, "raw_cells_progress") or {
        "last_source_position": -1
    }
    last_position = int(progress["last_source_position"])
    if _table_count(connection, "raw_cells") and last_position < 0:
        raise CacheStateError(
            "raw cells exist without a progress marker; use --restart-stage raw_cells"
        )
    for sample in cohort:
        source_position = int(sample["source_position"])
        if source_position <= last_position:
            continue
        hadm = int(sample["hadm_id"])
        cursor = connection.execute(
            "SELECT event_time_us, feature_idx, raw_value "
            "FROM candidates WHERE hadm_id=? "
            "ORDER BY event_time_us, feature_idx, author_block DESC, "
            "author_subblock DESC, author_row DESC",
            (hadm,),
        )
        current_time: int | None = None
        current_values = np.zeros(EXPECTED_FEATURES, dtype=np.float64)
        seen_features: set[int] = set()
        source_timestamps = 0
        kept_times: list[int] = []
        pending: list[tuple[int, int, int, float, int, float]] = []

        def finish_time() -> bool:
            nonlocal source_timestamps, current_values
            if current_time is None:
                return False
            source_timestamps += 1
            if source_timestamps > MAX_SOURCE_TIMESTAMPS:
                return True
            current_values[current_values < 0] = 0.0
            if np.sum(current_values) > 0:
                kept_times.append(current_time)
                sequence = len(kept_times) - 1
                first = kept_times[0]
                source_time = np.float32(
                    ((current_time - first) / 3_600_000_000.0) / 60.0
                ).item()
                for feature_idx in np.flatnonzero(current_values > 0):
                    pending.append(
                        (
                            hadm,
                            sequence,
                            int(feature_idx),
                            float(current_values[feature_idx]),
                            current_time,
                            float(source_time),
                        )
                    )
                if len(kept_times) >= MAX_MODEL_TIMESTAMPS:
                    return True
            return False

        exhausted = False
        for event_time, feature_idx, raw_value in cursor:
            event_time = int(event_time)
            feature_idx = int(feature_idx)
            if current_time is None:
                current_time = event_time
            elif event_time != current_time:
                if finish_time():
                    exhausted = True
                    break
                current_time = event_time
                current_values = np.zeros(EXPECTED_FEATURES, dtype=np.float64)
                seen_features.clear()
            if feature_idx not in seen_features:
                current_values[feature_idx] = (
                    np.nan if raw_value is None else float(raw_value)
                )
                seen_features.add(feature_idx)
        if not exhausted:
            finish_time()
        if pending:
            connection.executemany(
                "INSERT OR REPLACE INTO raw_cells VALUES (?, ?, ?, ?, ?, ?)", pending
            )
        connection.execute(
            "INSERT OR REPLACE INTO sample_boundary VALUES (?, ?, ?)",
            (hadm, len(kept_times), len(pending)),
        )
        if source_position % 250 == 0:
            _checkpoint(
                connection,
                "raw_cells_progress",
                {"last_source_position": source_position},
            )
            connection.commit()
    if cohort:
        _checkpoint(
            connection,
            "raw_cells_progress",
            {"last_source_position": int(cohort[-1]["source_position"])},
        )
    aggregates = connection.execute(
        "SELECT MIN(retained_length), MAX(retained_length), "
        "AVG(retained_length), SUM(observed_cells=0), COUNT(*) "
        "FROM sample_boundary"
    ).fetchone()
    if int(aggregates[4]) != len(cohort):
        raise CacheStateError("raw-cell checkpoint does not cover the full cohort")
    summary = {
        "minimum": int(aggregates[0]),
        "maximum": int(aggregates[1]),
        "mean": float(aggregates[2]),
        "samples_without_observed_cells": int(aggregates[3]),
    }
    inserted = _table_count(connection, "raw_cells")
    _checkpoint(
        connection,
        "raw_cells_complete",
        {"retained_events": inserted, "retained_length": summary},
    )
    connection.commit()
    return inserted, summary


def _normalization(
    connection: sqlite3.Connection,
    cohort: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    completed = checkpoint(connection, "normalization_complete")
    if completed is not None:
        return list(completed["normalization"])
    if _table_count(connection, "normalization"):
        raise CacheStateError(
            "normalization rows exist without a completion marker; "
            "use --restart-stage normalization"
        )
    position = {int(row["hadm_id"]): int(row["source_position"]) for row in cohort}
    connection.execute(
        "CREATE INDEX IF NOT EXISTS raw_cells_feature "
        "ON raw_cells(feature_idx, hadm_id, sequence_row)"
    )
    rows: list[dict[str, Any]] = []
    for feature_idx in range(EXPECTED_FEATURES):
        values = connection.execute(
            "SELECT hadm_id, sequence_row, raw_value FROM raw_cells WHERE feature_idx=?",
            (feature_idx,),
        ).fetchall()
        values.sort(key=lambda row: (position[int(row[0])], int(row[1])))
        array = np.asarray([float(row[2]) for row in values], dtype=np.float64)
        if not len(array):
            raise AssertionError(f"released feature {feature_idx} has no observations")
        mean_value = float(np.mean(array))
        std_value = float(np.std(array))
        connection.execute(
            "INSERT INTO normalization VALUES (?, ?, ?, ?)",
            (feature_idx, mean_value, std_value, len(array)),
        )
        rows.append(
            {
                "feature_idx": feature_idx,
                "mean": mean_value,
                "std": std_value,
                "observed": len(array),
            }
        )
    _checkpoint(connection, "normalization_complete", {"normalization": rows})
    connection.commit()
    return rows


def build_cache(
    mimic_root: Path,
    cache_path: Path,
    *,
    restart_stage: str | None = None,
) -> dict[str, Any]:
    mimic_root = mimic_root.resolve()
    connection = _initialize_database(cache_path)
    try:
        if (
            restart_stage is None
            and checkpoint(connection, "cache_complete") is not None
        ):
            return metadata(connection, "cache_summary")
        missing = [name for name in RAW_FILES if not (mimic_root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"MIMIC-III 1.4 is missing files: {missing}")
        if restart_stage is not None:
            restart_cache_from(connection, restart_stage)
        complete = checkpoint(connection, "cache_complete")
        if complete is not None:
            return metadata(connection, "cache_summary")
        adult_icu = author_adult_icu(mimic_root)
        if checkpoint(connection, "pass1_complete") is None:
            minimum, event_hadm, event_icu, full_feature_order = _first_pass(
                mimic_root, adult_icu
            )
            keep = tuple(
                name
                for index, name in enumerate(full_feature_order)
                if index not in set(REMOVE_INDICES)
            )
            if len(keep) != EXPECTED_FEATURES:
                raise AssertionError(
                    f"released feature removal yielded {len(keep)} channels"
                )
            cohort = _author_cohort(
                mimic_root, adult_icu, minimum, event_hadm, event_icu
            )
            if len(cohort) != EXPECTED_COHORT_ROWS:
                raise AssertionError(
                    f"released pipeline must reconstruct {EXPECTED_COHORT_ROWS} "
                    f"rows; found {len(cohort)}"
                )
            positives = sum(int(row["label"]) for row in cohort)
            if positives != EXPECTED_POSITIVE:
                raise AssertionError(
                    f"released pipeline must reconstruct {EXPECTED_POSITIVE} "
                    f"deaths; found {positives}"
                )
            _store_pass1_checkpoint(
                connection,
                cohort,
                full_feature_order=full_feature_order,
                features=keep,
            )
        cohort, full_feature_order, keep = _load_pass1_checkpoint(connection)
        positives = sum(int(row["label"]) for row in cohort)
        if len(cohort) != EXPECTED_COHORT_ROWS:
            raise CacheStateError("checkpointed pass-1 cohort has the wrong count")
        if positives != EXPECTED_POSITIVE:
            raise CacheStateError("checkpointed pass-1 labels have the wrong count")
        candidate_count = _second_pass(
            connection, mimic_root, adult_icu, cohort, keep
        )
        cell_count, length_summary = _materialize_cells(connection, cohort)
        normalization = _normalization(connection, cohort)

        raw_hashes = {}
        try:
            raw_hashes = metadata(connection, "raw_artifacts")
        except KeyError:
            for name in RAW_FILES:
                raw_hashes[name] = {
                    "sha256": sha256_file(mimic_root / name),
                    "bytes": (mimic_root / name).stat().st_size,
                }
                _put_metadata(connection, "raw_artifacts", raw_hashes)
                connection.commit()
        split_summary = {}
        for split in ("train", "validation", "test"):
            selected = [row for row in cohort if row["split"] == split]
            split_summary[split] = {
                "count": len(selected),
                "positive": sum(int(row["label"]) for row in selected),
                "subjects": len({int(row["subject_id"]) for row in selected}),
            }
        subjects = {
            split: {
                int(row["subject_id"])
                for row in cohort
                if row["split"] == split
            }
            for split in ("train", "validation", "test")
        }
        overlaps = {
            "train_validation": len(subjects["train"] & subjects["validation"]),
            "train_test": len(subjects["train"] & subjects["test"]),
            "validation_test": len(subjects["validation"] & subjects["test"]),
        }
        cache_summary = {
            "status": "complete",
            "mimic_version": "1.4",
            "cohort_rows": len(cohort),
            "published_cohort_rows": PUBLISHED_COHORT_ROWS,
            "paper_count_difference": len(cohort) - PUBLISHED_COHORT_ROWS,
            "paper_count_match": len(cohort) == PUBLISHED_COHORT_ROWS,
            "published_positive": PUBLISHED_POSITIVE,
            "positive": positives,
            "paper_positive_difference": positives - PUBLISHED_POSITIVE,
            "paper_positive_match": positives == PUBLISHED_POSITIVE,
            "features": list(keep),
            "full_feature_order": list(full_feature_order),
            "remove_indices": list(REMOVE_INDICES),
            "candidate_events": candidate_count,
            "retained_events": cell_count,
            "retained_length": length_summary,
            "splits": split_summary,
            "cross_split_subject_overlap": overlaps,
            "normalization": normalization,
            "raw_artifacts": raw_hashes,
            "released_source": {
                "preprocessor_sha256": sha256_file(AUTHOR_PREPROCESSOR),
                "data_utils_sha256": sha256_file(AUTHOR_DATA_UTILS),
                "prefix_segment_sha256": PREFIX_SHA256,
                "observation_segment_sha256": OBSERVATION_SHA256,
                "output_segment_sha256": OUTPUT_SHA256,
                "input_segment_sha256": INPUT_SHA256,
            },
        }
        _put_metadata(connection, "cache_summary", cache_summary)
        _checkpoint(
            connection,
            "cache_complete",
            {
                "cohort_rows": len(cohort),
                "candidate_events": candidate_count,
                "retained_events": cell_count,
            },
        )
        connection.commit()
        return cache_summary
    finally:
        connection.close()


def open_complete_cache(cache_path: Path) -> tuple[sqlite3.Connection, dict[str, Any]]:
    if not cache_path.is_file():
        raise FileNotFoundError(f"GARLIC source cache is missing: {cache_path}")
    connection = sqlite3.connect(cache_path)
    try:
        complete = checkpoint(connection, "cache_complete")
        summary = metadata(connection, "cache_summary")
    except (KeyError, sqlite3.OperationalError):
        connection.close()
        raise ValueError("GARLIC source cache is incomplete")
    if complete is None or summary.get("status") != "complete":
        connection.close()
        raise ValueError("GARLIC source cache is incomplete")
    return connection, summary


def cohort_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        {
            "source_position": int(row[0]),
            "hadm_id": int(row[1]),
            "subject_id": int(row[2]),
            "label": int(row[3]),
            "split": str(row[4]),
            "observation_start_us": int(row[5]),
        }
        for row in connection.execute(
            "SELECT source_position, hadm_id, subject_id, label, split, "
            "observation_start_us FROM cohort ORDER BY source_position"
        )
    ]


def sample_frame(connection: sqlite3.Connection) -> pl.DataFrame:
    rows = cohort_rows(connection)
    return pl.DataFrame(
        {
            "subject_id": [row["subject_id"] for row in rows],
            "sample_id": [row["hadm_id"] for row in rows],
            "observation_start": [row["observation_start_us"] for row in rows],
            "prediction_time": [
                row["observation_start_us"] + int(OBSERVATION_HOURS * 3_600_000_000)
                for row in rows
            ],
            "boolean_value": [bool(row["label"]) for row in rows],
            "split": [row["split"] for row in rows],
            "hadm_id": [row["hadm_id"] for row in rows],
            "source_position": [row["source_position"] for row in rows],
        },
        schema={
            "subject_id": pl.Int64,
            "sample_id": pl.Int64,
            "observation_start": pl.Datetime("us"),
            "prediction_time": pl.Datetime("us"),
            "boolean_value": pl.Boolean,
            "split": pl.String,
            "hadm_id": pl.Int64,
            "source_position": pl.Int64,
        },
    )


def event_batches(
    connection: sqlite3.Connection,
    feature_names: tuple[str, ...],
    batch_size: int = 250_000,
) -> Iterator[pl.DataFrame]:
    normal = {
        int(row[0]): (float(row[1]), float(row[2]))
        for row in connection.execute(
            "SELECT feature_idx, mean_value, std_value FROM normalization"
        )
    }
    cursor = connection.execute(
        "SELECT c.subject_id, r.hadm_id, r.sequence_row, r.feature_idx, "
        "r.raw_value, r.event_time_us, r.source_time, c.source_position "
        "FROM raw_cells r JOIN cohort c ON c.hadm_id=r.hadm_id "
        "ORDER BY c.source_position, r.sequence_row, r.feature_idx"
    )
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            return
        subject_id: list[int] = []
        sample_id: list[int] = []
        sequence_row: list[int] = []
        code: list[str] = []
        value: list[float] = []
        event_time: list[int] = []
        source_time: list[float] = []
        source_row: list[int] = []
        for subject, hadm, sequence, feature, raw, time_us, scaled_time, position in rows:
            mean_value, std_value = normal[int(feature)]
            normalized = float(raw) - mean_value
            if std_value != 0:
                normalized /= std_value
            subject_id.append(int(subject))
            sample_id.append(int(hadm))
            sequence_row.append(int(sequence))
            code.append(feature_names[int(feature)])
            value.append(np.float32(normalized).item())
            event_time.append(int(time_us))
            source_time.append(float(scaled_time))
            source_row.append(
                int(position) * MAX_MODEL_TIMESTAMPS * EXPECTED_FEATURES
                + int(sequence) * EXPECTED_FEATURES
                + int(feature)
            )
        yield pl.DataFrame(
            {
                "subject_id": subject_id,
                "time": event_time,
                "code": code,
                "numeric_value": value,
                "sample_id": sample_id,
                "source_row": source_row,
                "sequence_row": sequence_row,
                "source_time": source_time,
            },
            schema={
                "subject_id": pl.Int64,
                "time": pl.Datetime("us"),
                "code": pl.String,
                "numeric_value": pl.Float32,
                "sample_id": pl.Int64,
                "source_row": pl.Int64,
                "sequence_row": pl.Int32,
                "source_time": pl.Float64,
            },
        )


def selected_id_hash(connection: sqlite3.Connection) -> str:
    values = np.asarray(
        [row[0] for row in connection.execute("SELECT hadm_id FROM cohort ORDER BY hadm_id")],
        dtype=np.int64,
    )
    return hashlib.sha256(values.tobytes()).hexdigest()


def ordered_split_hash(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for row in connection.execute(
        "SELECT hadm_id, subject_id, label, split FROM cohort ORDER BY source_position"
    ):
        digest.update(json.dumps(row, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()
