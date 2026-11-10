"""Exact released TD-ICU source construction used by the exporter.

The released loader is called unchanged after an information-preserving CSV
projection: only item IDs that its inner join can retain and only columns that
the loader reads are written to the local author parquet cache.  This avoids
the released converter's unsafe temporary decompression beside credentialed
raw files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from utils.parquet_conversion.dtypes import get_dtypes
from utils.preprocessing.tools import (
    clean_combined_data,
    create_final_dataframe,
    create_labels_for_mimic,
    encode_combined_data_for_mimic,
    get_scaling_data_for_mimic,
    get_variable_names_for_mimic,
    load_mimic,
    save_patient_ids_for_mimic,
)


PAPER_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PAPER_ROOT / "data"
PARQUET_ROOT = DATA_ROOT / "mimic" / "parquet"
EXPORT_ROOT = DATA_ROOT / "export"
CURRENT_ROOT = EXPORT_ROOT / "current"
SAMPLES_PATH = EXPORT_ROOT / "samples.parquet"
SOURCE_CONVERSION_PATH = EXPORT_ROOT / "source_conversion.json"
SOURCE_BUILD_PATH = EXPORT_ROOT / "source_build.json"
CURRENT_BUILD_PATH = EXPORT_ROOT / "current_boundary.json"

UPSTREAM_COMMIT = "235ca921bb200dffe6c854ef16949a94435758f4"
MIMIC_RELEASE = "3.0"
INPUT_WINDOW_HOURS = 168
CONTEXT_LENGTH = 400
NEXT_STATE_DELAY_HOURS = 24
NEXT_STATE_WINDOW_HOURS = 24
MINIMUM_RUNTIME_FREE_BYTES = 30 * 1024**3
LABEL_COLUMNS = (
    "1-day-died",
    "3-day-died",
    "7-day-died",
    "14-day-died",
    "28-day-died",
)

RAW_SHA256 = {
    "hosp/admissions.csv.gz": "a9584ed88e9ed664a2f66f86a5cf9fd175c8bb0af50e3b6115598b19e978384e",
    "hosp/patients.csv.gz": "4c7507de7ecad8c5ba7647ea4d26018e88ff6672c5fc22ab59c2a5697d687cdd",
    "icu/chartevents.csv.gz": "fd0387653084e5b142756b98b74fdddc2e5e7eb0f496aa8bf5af3d4176e71098",
    "icu/d_items.csv.gz": "7f223a9570da9c02bd12f281b398cb4951d5a560a51836540f357bc247366b7f",
    "icu/inputevents.csv.gz": "09ffc40fade12d017debcb939225ab3924ac807625fdae27f09a260a8ee0ce48",
}

AUTHOR_COLUMNS = {
    "hosp/admissions": ("subject_id", "hadm_id", "deathtime"),
    "hosp/patients": (
        "subject_id",
        "gender",
        "anchor_age",
        "anchor_year",
        "dod",
    ),
    "icu/d_items": ("itemid", "label"),
    # hadm_id and stay_id are retained solely for cohort auditing.  The exact
    # released load_mimic() projection drops them before constructing inputs.
    "icu/chartevents": (
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "charttime",
        "valuenum",
        "valueuom",
    ),
    "icu/inputevents": (
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "starttime",
        "endtime",
        "amount",
        "amountuom",
        "originalrate",
        "rate",
        "rateuom",
        "patientweight",
        "ordercategoryname",
        "ordercategorydescription",
        "statusdescription",
        "orderid",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def assert_mimic_release(mimic_root: Path) -> dict[str, Any]:
    sums_path = mimic_root / "SHA256SUMS.txt"
    required = [sums_path, *(mimic_root / relative for relative in RAW_SHA256)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing exact MIMIC-IV 3.0 files: {missing}")
    sums = sums_path.read_text()
    for relative, expected in RAW_SHA256.items():
        if f"{expected} {relative}" not in sums:
            raise AssertionError(
                f"MIMIC-IV 3.0 release manifest changed for {relative}"
            )
    return {
        "release": MIMIC_RELEASE,
        "release_manifest_sha256": sha256_file(sums_path),
        "raw_sha256": RAW_SHA256,
    }


def _parquet_target(table: str) -> Path:
    return PARQUET_ROOT / f"{table.rsplit('/', 1)[-1]}.parquet"


def _assert_cache_record(record: dict[str, Any]) -> None:
    if record.get("upstream_commit") != UPSTREAM_COMMIT:
        raise AssertionError("cached TD-ICU source was built from another commit")
    if record.get("mimic_release") != MIMIC_RELEASE:
        raise AssertionError("cached TD-ICU source was built from another MIMIC release")
    for name, expected in record.get("parquet_sha256", {}).items():
        path = PARQUET_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise AssertionError(f"cached source parquet changed: {path}")


def _arrow_type(dtype: pl.DataType) -> pa.DataType:
    if dtype == pl.Int64:
        return pa.int64()
    if dtype == pl.Float64:
        return pa.float64()
    if dtype == pl.Datetime:
        return pa.timestamp("us")
    if dtype in {pl.String, pl.Utf8, pl.Categorical}:
        return pa.string()
    raise TypeError(f"no Arrow CSV type for {dtype}")


def _stream_csv_to_parquet(
    source_path: Path,
    target_path: Path,
    columns: tuple[str, ...],
    source_schema: dict[str, pl.DataType],
    selected_itemids: tuple[int, ...] | None = None,
) -> None:
    """Stream one gzip CSV with bounded memory and preserved source order."""
    compressed = pa.input_stream(str(source_path), compression="gzip")
    reader = pacsv.open_csv(
        compressed,
        read_options=pacsv.ReadOptions(block_size=16 * 1024 * 1024, use_threads=True),
        convert_options=pacsv.ConvertOptions(
            include_columns=list(columns),
            column_types={name: _arrow_type(source_schema[name]) for name in columns},
            null_values=[""],
            strings_can_be_null=True,
        ),
    )
    allowed = (
        pa.array(selected_itemids, type=pa.int64())
        if selected_itemids is not None
        else None
    )
    writer: pq.ParquetWriter | None = None
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch])
            if allowed is not None:
                table = table.filter(pc.is_in(table["itemid"], value_set=allowed))
            if not len(table):
                continue
            if writer is None:
                writer = pq.ParquetWriter(
                    target_path, table.schema, compression="zstd"
                )
            writer.write_table(table, row_group_size=250_000)
    finally:
        if writer is not None:
            writer.close()
        compressed.close()
    if writer is None:
        raise AssertionError(f"streaming projection retained no rows from {source_path}")


def prepare_author_parquets(mimic_root: Path) -> dict[str, Any]:
    """Safely create the exact input of the released ``load_mimic`` function."""
    release = assert_mimic_release(mimic_root)
    if SOURCE_CONVERSION_PATH.is_file():
        record = read_json(SOURCE_CONVERSION_PATH)
        _assert_cache_record(record)
        return record

    PARQUET_ROOT.mkdir(parents=True, exist_ok=True)
    dtypes = get_dtypes()
    variable_names = get_variable_names_for_mimic()
    d_items_source = pl.scan_csv(
        mimic_root / "icu" / "d_items.csv.gz",
        schema_overrides=dtypes["icu/d_items"],
        infer_schema_length=1_000_000,
        null_values=[""],
    )
    selected_items = (
        d_items_source
        .filter(pl.col("label").cast(pl.String).is_in(tuple(variable_names)))
        .select(AUTHOR_COLUMNS["icu/d_items"])
        .collect(streaming=True)
    )
    if selected_items.is_empty() or selected_items["itemid"].n_unique() != selected_items.height:
        raise AssertionError("released MIMIC item selection is empty or non-unique")
    item_ids = tuple(selected_items["itemid"].to_list())

    row_counts: dict[str, int] = {}
    identity_counts: dict[str, dict[str, int]] = {}
    for table in AUTHOR_COLUMNS:
        target = _parquet_target(table)
        temporary = target.with_suffix(".parquet.tmp")
        if table == "icu/d_items":
            selected_items.write_parquet(
                temporary, compression="zstd", row_group_size=250_000
            )
        else:
            _stream_csv_to_parquet(
                mimic_root / f"{table}.csv.gz",
                temporary,
                AUTHOR_COLUMNS[table],
                dtypes[table],
                item_ids if table in {"icu/chartevents", "icu/inputevents"} else None,
            )
        os.replace(temporary, target)
        available_ids = [
            column
            for column in ("subject_id", "hadm_id", "stay_id", "itemid")
            if column in pl.read_parquet_schema(target)
        ]
        summary = pl.scan_parquet(target).select(
            pl.len().alias("rows"),
            *(pl.col(column).n_unique().alias(column) for column in available_ids),
        ).collect()
        row_counts[target.name] = int(summary["rows"].item())
        identity_counts[target.name] = {
            column: int(summary[column].item()) for column in available_ids
        }
        print(f"prepared {target.name}: {row_counts[target.name]} rows", flush=True)

    parquet_hashes = {
        path.name: sha256_file(path) for path in sorted(PARQUET_ROOT.glob("*.parquet"))
    }
    record = {
        "status": "exact_information_preserving_projection",
        "upstream_commit": UPSTREAM_COMMIT,
        "mimic_release": MIMIC_RELEASE,
        "release": release,
        "selected_itemids": list(item_ids),
        "selected_itemid_count": len(item_ids),
        "selected_labels": sorted(variable_names),
        "row_counts": row_counts,
        "identity_counts": identity_counts,
        "parquet_sha256": parquet_hashes,
        "projection": {
            "row_filter": "itemid selected by released get_variable_names_for_mimic() and released d_items inner join",
            "columns": {key: list(value) for key, value in AUTHOR_COLUMNS.items()},
            "ordering": "raw CSV order preserved",
            "credentialed_raw_tree_modified": False,
        },
    }
    write_json(SOURCE_CONVERSION_PATH, record)
    return record


def _feature_files(features: list[str], feature_encoding: pl.DataFrame) -> None:
    feature_encoding.write_parquet(DATA_ROOT / "feature_encoding.parquet")
    (DATA_ROOT / "features.txt").write_text("\n".join(features))
    (DATA_ROOT / "label_features.txt").write_text("\n".join(LABEL_COLUMNS))


def build_author_source(mimic_root: Path) -> dict[str, Any]:
    """Call the exact released loader, cleaner, labeler, and encoder."""
    conversion = prepare_author_parquets(mimic_root)
    if SOURCE_BUILD_PATH.is_file():
        record = read_json(SOURCE_BUILD_PATH)
        if record.get("source_conversion_sha256") != sha256_file(SOURCE_CONVERSION_PATH):
            raise AssertionError("cached author source uses another conversion")
        for relative, expected in record.get("artifact_sha256", {}).items():
            path = DATA_ROOT / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise AssertionError(f"cached author source artifact changed: {path}")
        return record

    required_outputs = (
        PARQUET_ROOT / "combined_data.parquet",
        PARQUET_ROOT / "encoded_input_data.parquet",
        PARQUET_ROOT / "labels.parquet",
        DATA_ROOT / "feature_encoding.parquet",
        DATA_ROOT / "scaling_data.parquet",
    )
    existing = [path for path in required_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "unverified partial author source exists without source_build.json: "
            f"{existing}"
        )

    target_variables = get_variable_names_for_mimic()
    admissions, combined_data, patients, split_ids = load_mimic(target_variables)
    train_ids, validation_ids, test_ids = split_ids
    combined_data = clean_combined_data(
        combined_data, admissions, train_ids, "mimic"
    )
    labels = create_labels_for_mimic(
        combined_data,
        admissions,
        patients,
        INPUT_WINDOW_HOURS,
        NEXT_STATE_DELAY_HOURS,
        NEXT_STATE_WINDOW_HOURS,
    )
    save_patient_ids_for_mimic(train_ids, validation_ids, test_ids)
    encoded, features, feature_encoding, encodings = encode_combined_data_for_mimic(
        combined_data
    )
    scaling = get_scaling_data_for_mimic(
        encoded,
        labels,
        train_ids,
        INPUT_WINDOW_HOURS,
        encodings["age"],
        encodings["gender"],
        encodings["weight"],
    )

    combined_data.write_parquet(PARQUET_ROOT / "combined_data.parquet")
    encoded.write_parquet(PARQUET_ROOT / "encoded_input_data.parquet")
    labels.write_parquet(PARQUET_ROOT / "labels.parquet")
    scaling.write_parquet(DATA_ROOT / "scaling_data.parquet")
    _feature_files(features, feature_encoding)
    artifacts = {
        str(path.relative_to(DATA_ROOT)): sha256_file(path)
        for path in required_outputs
    }
    artifacts.update(
        {
            "features.txt": sha256_file(DATA_ROOT / "features.txt"),
            "label_features.txt": sha256_file(DATA_ROOT / "label_features.txt"),
            "mimic/patient_ids/train_patient_ids.npy": sha256_file(
                DATA_ROOT / "mimic" / "patient_ids" / "train_patient_ids.npy"
            ),
            "mimic/patient_ids/val_patient_ids.npy": sha256_file(
                DATA_ROOT / "mimic" / "patient_ids" / "val_patient_ids.npy"
            ),
            "mimic/patient_ids/test_patient_ids.npy": sha256_file(
                DATA_ROOT / "mimic" / "patient_ids" / "test_patient_ids.npy"
            ),
            "outlier_reference.pkl": sha256_file(DATA_ROOT / "outlier_reference.pkl"),
        }
    )
    split_summary = {
        name: {
            "source_label_rows": part.height,
            "subjects": part["subject_id"].n_unique(),
            "positive_28_day": int(part["28-day-died"].sum()),
        }
        for name in ("train", "validation", "test")
        for ids in (
            train_ids if name == "train" else validation_ids if name == "validation" else test_ids,
        )
        for part in (labels.filter(pl.col("subject_id").is_in(ids)),)
    }
    record = {
        "status": "exact_released_source",
        "upstream_commit": UPSTREAM_COMMIT,
        "source_conversion_sha256": sha256_file(SOURCE_CONVERSION_PATH),
        "parameters": {
            "input_window_hours": INPUT_WINDOW_HOURS,
            "context_length": CONTEXT_LENGTH,
            "next_state_delay_hours": NEXT_STATE_DELAY_HOURS,
            "next_state_window_hours": NEXT_STATE_WINDOW_HOURS,
            "numpy_split_seed": 42,
        },
        "source_conversion": conversion,
        "features": features,
        "feature_count": len(features),
        "source_label_rows": labels.height,
        "subjects": labels["subject_id"].n_unique(),
        "split_summary": split_summary,
        "artifact_sha256": artifacts,
    }
    write_json(SOURCE_BUILD_PATH, record)
    return record


def build_current_boundary() -> dict[str, Any]:
    """Capture every exact current-state author row before HDF5 drops IDs."""
    if not SOURCE_BUILD_PATH.is_file():
        raise FileNotFoundError("build_author_source must run first")
    if CURRENT_BUILD_PATH.is_file():
        record = read_json(CURRENT_BUILD_PATH)
        if record.get("source_build_sha256") != sha256_file(SOURCE_BUILD_PATH):
            raise AssertionError("cached current boundary uses another source build")
        if (
            not SAMPLES_PATH.is_file()
            or sha256_file(SAMPLES_PATH) != record.get("samples_sha256")
        ):
            raise AssertionError("cached current-boundary sample index changed")
        for relative, expected in record.get("parquet_sha256", {}).items():
            path = EXPORT_ROOT / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise AssertionError(f"cached current-boundary shard changed: {path}")
        return record
    if CURRENT_ROOT.exists() and any(CURRENT_ROOT.rglob("*.parquet")):
        raise FileExistsError(
            "unverified partial current-boundary cache exists without current_boundary.json"
        )

    encoded = pl.scan_parquet(PARQUET_ROOT / "encoded_input_data.parquet").drop(
        "str_feature"
    )
    labels = pl.scan_parquet(PARQUET_ROOT / "labels.parquet")
    feature_encoding = pl.read_parquet(DATA_ROOT / "feature_encoding.parquet")
    feature_names = (DATA_ROOT / "features.txt").read_text().splitlines()
    mapping = dict(
        zip(
            feature_encoding["str_feature"].cast(pl.String).to_list(),
            feature_encoding["encoded_feature"].to_list(),
            strict=True,
        )
    )
    drug_rate_names = [name for name in feature_names if "rate" in name]
    drug_names_encoded = [mapping[name] for name in drug_rate_names]
    lab_names_encoded = [
        mapping[name]
        for name in feature_names
        if name not in {"age", "gender", "patientweight"}
        and "rate" not in name
        and "bolus" not in name
    ]
    encodings = {
        "age": np.int16(mapping["age"]),
        "gender": np.int16(mapping["gender"]),
        "weight": np.int16(mapping["patientweight"]),
        "drug_names": drug_names_encoded,
        "lab_names": lab_names_encoded,
        "all_features": feature_encoding["encoded_feature"].to_list()[:-3],
    }
    train_ids = np.load(DATA_ROOT / "mimic" / "patient_ids" / "train_patient_ids.npy")
    validation_ids = np.load(DATA_ROOT / "mimic" / "patient_ids" / "val_patient_ids.npy")
    test_ids = np.load(DATA_ROOT / "mimic" / "patient_ids" / "test_patient_ids.npy")
    CURRENT_ROOT.mkdir(parents=True, exist_ok=True)
    sample_offset = 0
    shard_rows: dict[str, int] = {}

    def capture(group: str, index: int, frame: pl.DataFrame) -> None:
        nonlocal sample_offset
        if shutil.disk_usage(PAPER_ROOT).free < MINIMUM_RUNTIME_FREE_BYTES:
            raise RuntimeError(
                "TD-ICU current-boundary capture stopped before exhausting "
                "the workspace filesystem"
            )
        source_split = "validation" if group == "val" else group
        frame = frame.with_columns(
            pl.int_range(sample_offset, sample_offset + frame.height, dtype=pl.Int64)
            .alias("source_sample_row"),
            pl.int_range(sample_offset, sample_offset + frame.height, dtype=pl.Int64)
            .alias("sample_id"),
            pl.lit(source_split, dtype=pl.String).alias("split"),
        )
        sample_offset += frame.height
        target_dir = CURRENT_ROOT / source_split
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"dataframe{index:03}.parquet"
        frame.select(
            "subject_id",
            "sample_id",
            "source_sample_row",
            "split",
            "labeltime",
            "encoded_feature",
            "featuretime",
            "value",
            "delta_time",
            "delta_value",
            "next_labeltime",
            *LABEL_COLUMNS,
        ).write_parquet(target, compression="zstd", row_group_size=25_000)
        shard_rows[str(target.relative_to(EXPORT_ROOT))] = frame.height
        print(
            f"captured {source_split}/{target.name}: {frame.height} landmarks",
            flush=True,
        )

    create_final_dataframe(
        "mimic",
        encoded,
        labels,
        encodings,
        train_ids,
        validation_ids,
        test_ids,
        ["subject_id", "labeltime"],
        ["subject_id", "labeltime", "targets"],
        CONTEXT_LENGTH,
        export_chunk_callback=capture,
    )

    files = sorted(CURRENT_ROOT.rglob("*.parquet"))
    all_current = pl.scan_parquet(files)
    counts = all_current.select(
        pl.len().alias("rows"),
        pl.col("sample_id").n_unique().alias("unique_samples"),
        pl.col("subject_id").n_unique().alias("subjects"),
    ).collect()
    if counts["unique_samples"].item() != counts["rows"].item():
        raise AssertionError("author final boundary duplicated a source sample")
    samples = (
        all_current
        .select(
            pl.col("subject_id").cast(pl.Int64),
            pl.col("sample_id").cast(pl.Int64),
            (pl.col("labeltime") - pl.duration(hours=INPUT_WINDOW_HOURS))
            .cast(pl.Datetime("us"))
            .alias("observation_start"),
            pl.col("labeltime").cast(pl.Datetime("us")).alias("prediction_time"),
            pl.col("28-day-died").cast(pl.Boolean).alias("boolean_value"),
            pl.col("split").cast(pl.String),
            pl.col("value").list.get(2).cast(pl.Float64).alias("source_patientweight"),
            pl.col("next_labeltime").cast(pl.Datetime("us")).alias("source_labeltime_next"),
            *(
                pl.col(name).cast(pl.UInt8).alias(f"source_{name}")
                for name in LABEL_COLUMNS
            ),
            pl.col("value").list.get(0).cast(pl.UInt8).alias("source_age"),
            pl.col("value").list.get(1).cast(pl.UInt8).alias("source_gender"),
        )
        .sort("sample_id")
        .collect(streaming=True)
    )
    if not np.array_equal(
        samples["sample_id"].to_numpy(), np.arange(samples.height, dtype=np.int64)
    ):
        raise AssertionError("author final boundary row order is not a dense sample index")
    SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    samples.write_parquet(SAMPLES_PATH, compression="zstd", row_group_size=250_000)

    landmark_key_count = samples.select(
        "subject_id", "prediction_time", "source_patientweight"
    ).n_unique()

    hashes = {
        str(path.relative_to(EXPORT_ROOT)): sha256_file(path) for path in files
    }
    record = {
        "status": "exact_author_current_boundary",
        "source_build_sha256": sha256_file(SOURCE_BUILD_PATH),
        "rows": int(counts["rows"].item()),
        "subjects": int(counts["subjects"].item()),
        "source_label_rows": read_json(SOURCE_BUILD_PATH)["source_label_rows"],
        "duplicate_final_rows_beyond_unique_landmark_keys": int(
            samples.height - landmark_key_count
        ),
        "samples_sha256": sha256_file(SAMPLES_PATH),
        "shard_rows": shard_rows,
        "parquet_sha256": hashes,
        "excluded_from_persistent_boundary": [
            "aligned next-state tensors used only as privileged TD training targets",
            "padding tokens",
        ],
    }
    write_json(CURRENT_BUILD_PATH, record)
    return record


def current_files() -> list[Path]:
    order = {"train": 0, "validation": 1, "test": 2}
    return sorted(
        CURRENT_ROOT.rglob("*.parquet"),
        key=lambda path: (order[path.parent.name], path.name),
    )


def source_feature_names() -> tuple[str, ...]:
    return tuple((DATA_ROOT / "features.txt").read_text().splitlines())
