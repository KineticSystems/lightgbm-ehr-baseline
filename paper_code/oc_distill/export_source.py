"""Export the full observed OC-Distill vital stream for its exact prior cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch


REPOSITORY = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPOSITORY.parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from utils import utils as author_utils  # noqa: E402
from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402


VARIABLE_NAMES = (
    "Diastolic blood pressure",
    "Fraction inspired oxygen",
    "Glucose",
    "Heart Rate",
    "Height",
    "Mean blood pressure",
    "Oxygen saturation",
    "Respiratory rate",
    "Systolic blood pressure",
    "Temperature",
    "Weight",
    "pH",
)
HOURS = 48.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def raw_path(root: Path, stem: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        path = root / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing {stem}.csv[.gz] below {root}")


def item_mapping(repository: Path) -> pd.DataFrame:
    path = (
        repository
        / "mimic3_benchmark"
        / "mimic3benchmark"
        / "resources"
        / "itemid_to_variable_map.csv"
    )
    mapping = pd.read_csv(path).fillna("")
    mapping = mapping.loc[
        (mapping["LEVEL2"] != "")
        & (mapping["COUNT"].astype(int) > 0)
        & (mapping["STATUS"] == "ready")
        & mapping["LEVEL2"].isin(VARIABLE_NAMES),
        ["ITEMID", "LEVEL2", "MIMIC LABEL", "LINKSTO"],
    ].copy()
    mapping["ITEMID"] = mapping["ITEMID"].astype(np.int32)
    if set(mapping["LEVEL2"]) != set(VARIABLE_NAMES):
        raise ValueError("released item map is missing a selected variable")
    return mapping.rename(
        columns={"LEVEL2": "variable", "MIMIC LABEL": "mimic_label"}
    )


def clean_selected(frame: pd.DataFrame) -> np.ndarray:
    """Match the vendored MIMIC-III benchmark cleaning for retained channels."""
    variable = frame["variable"]
    raw = frame["VALUE"].astype("string")
    unit = frame["VALUEUOM"].fillna("").astype(str).str.lower()
    label = frame["mimic_label"].fillna("").astype(str).str.lower()
    values = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float64)
    for name, part in (("Systolic blood pressure", 0), ("Diastolic blood pressure", 1)):
        selected = variable.eq(name).to_numpy() & raw.str.contains(
            "/", regex=False, na=False
        ).to_numpy()
        if selected.any():
            components = raw.loc[selected].str.extract(r"^(\d+)/(\d+)$", expand=True)
            values[selected] = pd.to_numeric(
                components.iloc[:, part], errors="coerce"
            ).to_numpy()
    selected = variable.eq("Fraction inspired oxygen").to_numpy()
    values[selected & ~unit.str.contains("torr", regex=False).to_numpy()] /= 100.0
    selected = variable.eq("Oxygen saturation").to_numpy() & (values <= 1.0)
    values[selected] *= 100.0
    selected = variable.eq("Temperature").to_numpy()
    fahrenheit = selected & (
        unit.str.contains("F", regex=False).to_numpy()
        | label.str.contains("F", regex=False).to_numpy()
        | (values >= 79.0)
    )
    values[fahrenheit] = (values[fahrenheit] - 32.0) * 5.0 / 9.0
    selected = variable.eq("Weight").to_numpy()
    ounces = selected & (
        unit.str.contains("oz", regex=False).to_numpy()
        | label.str.contains("oz", regex=False).to_numpy()
    )
    values[ounces] /= 16.0
    pounds = selected & (
        ounces
        | unit.str.contains("lb", regex=False).to_numpy()
        | label.str.contains("lb", regex=False).to_numpy()
    )
    values[pounds] *= 0.453592
    selected = variable.eq("Height").to_numpy() & (
        unit.str.contains("in", regex=False).to_numpy()
        | label.str.contains("in", regex=False).to_numpy()
    )
    values[selected] = np.round(values[selected] * 2.54)
    return values


def load_cohort(
    cohort_root: Path,
    mimic_root: Path,
    auxiliary_root: Path,
    cache_root: Path,
    reuse_cache: bool = False,
):
    paths = {
        split: cohort_root / f"mortality_{split}_48h.npz"
        for split in ("train", "test")
    }
    rows = []
    source = {}
    for source_split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as archive:
            required = {"X", "y", "stay_id", "subject_id"}
            if set(archive.files) != required:
                raise ValueError(f"{path} has unexpected arrays {archive.files}")
            source[source_split] = {
                "X": archive["X"],
                "y": archive["y"].astype(np.uint8),
                "stay_id": archive["stay_id"].astype(np.int64),
                "subject_id": archive["subject_id"].astype(np.int64),
                "sha256": sha256(path),
            }
    icustays_path = raw_path(mimic_root, "ICUSTAYS")
    icustays = pd.read_csv(
        icustays_path,
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"],
    )
    icustays["INTIME"] = pd.to_datetime(icustays["INTIME"], errors="raise")
    lookup = {
        int(row.ICUSTAY_ID): (
            int(row.SUBJECT_ID),
            int(row.HADM_ID),
            int(row.INTIME.value // 1000),
        )
        for row in icustays.itertuples(index=False)
    }
    author_utils.BENCHMARK_DATA_DIR = cohort_root
    author_utils.DATA_DIR = auxiliary_root
    author_utils.DEMO_CSV = auxiliary_root / "demographic_table.csv"
    author_utils.NOTES_CSV = auxiliary_root / "notes_table.csv"
    author_utils.CACHE_DIR = cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    retained_frames = {}
    for source_split in ("train", "test"):
        cached = cache_root / "mortality" / "48h" / f"{source_split}_cohort.pkl"
        if reuse_cache and cached.is_file():
            retained = pd.read_pickle(cached).reset_index(drop=True)
        else:
            no_notes = author_utils.cohort_creation_no_notes(
                "mortality", source_split, "48h", str(cache_root)
            )
            retained = author_utils.attach_notes(
                no_notes, "mortality", source_split, "48h", str(cache_root)
            ).reset_index(drop=True)
        retained_frames[source_split] = retained
        block = source[source_split]
        position_for_stay = {
            int(stay_id): position for position, stay_id in enumerate(block["stay_id"])
        }
        for retained_position, record in enumerate(retained.itertuples(index=False)):
            stay_id = int(record.ICUSTAY_ID)
            position = position_for_stay[stay_id]
            subject_id = int(record.SUBJECT_ID)
            label = int(record.label)
            raw_subject, hadm_id, intime_us = lookup[int(stay_id)]
            if raw_subject != int(subject_id):
                raise AssertionError("cohort subject and raw ICUSTAYS disagree")
            if label != int(block["y"][position]):
                raise AssertionError("author loader changed the mortality label")
            if not np.array_equal(np.asarray(record.ts), block["X"][position]):
                raise AssertionError("author loader changed the benchmark tensor")
            rows.append(
                {
                    "source_split": source_split,
                    "source_position": position,
                    "retained_position": retained_position,
                    "stay_id": int(stay_id),
                    "subject_id": int(subject_id),
                    "hadm_id": hadm_id,
                    "intime_us": intime_us,
                    "label": int(label),
                }
            )
    return rows, source, retained_frames, icustays_path


def author_split(rows):
    train_rows = [row for row in rows if row["source_split"] == "train"]
    subjects = np.asarray(
        list(dict.fromkeys(row["subject_id"] for row in train_rows)), dtype=np.int64
    )
    random_state = np.random.RandomState(0)
    random_state.shuffle(subjects)
    fit_end = int(round(0.8 * len(subjects)))
    fit_subjects = set(subjects[:fit_end].tolist())
    for row in rows:
        if row["source_split"] == "test":
            row["split"] = "test"
        elif row["subject_id"] in fit_subjects:
            row["split"] = "train"
        else:
            row["split"] = "validation"


def stream_events(repository: Path, mimic_root: Path, rows, source, chunksize: int):
    mapping = item_mapping(repository)
    variable_index = {name: index for index, name in enumerate(VARIABLE_NAMES)}
    item_variable = mapping.set_index("ITEMID")["variable"].to_dict()
    item_label = mapping.set_index("ITEMID")["mimic_label"].to_dict()
    stay_frame = pd.DataFrame(rows)[
        ["subject_id", "stay_id", "intime_us", "source_split", "source_position"]
    ].rename(columns={"subject_id": "SUBJECT_ID"})
    stay_frame["INTIME"] = pd.to_datetime(stay_frame["intime_us"], unit="us")

    grids = {
        split: np.full((len(source[split]["y"]), 48, 12), np.nan, dtype=np.float32)
        for split in ("train", "test")
    }
    grid_times = {
        split: np.full(
            (len(source[split]["y"]), 48, 12), np.iinfo(np.int64).min, dtype=np.int64
        )
        for split in ("train", "test")
    }
    report = {"event_count": 0, "parity": {}}

    def generate():
        next_source_row = 0
        for stem, linksto in (
            ("CHARTEVENTS", "chartevents"),
            ("LABEVENTS", "labevents"),
        ):
            source_map = mapping.loc[mapping["LINKSTO"] == linksto]
            item_ids = set(source_map["ITEMID"].astype(int))
            reader = pd.read_csv(
                raw_path(mimic_root, stem),
                usecols=[
                    "SUBJECT_ID",
                    "ITEMID",
                    "CHARTTIME",
                    "VALUE",
                    "VALUEUOM",
                ],
                chunksize=chunksize,
                low_memory=False,
                dtype={
                    "SUBJECT_ID": np.int32,
                    "ITEMID": np.int32,
                    "VALUE": "string",
                    "VALUEUOM": "string",
                },
            )
            for chunk_number, chunk in enumerate(reader, start=1):
                chunk = chunk.loc[
                    chunk["ITEMID"].isin(item_ids)
                    & chunk["VALUE"].notna()
                    & chunk["CHARTTIME"].notna()
                ].copy()
                if chunk.empty:
                    continue
                chunk["CHARTTIME"] = pd.to_datetime(
                    chunk["CHARTTIME"], errors="coerce"
                )
                chunk = chunk.dropna(subset=["CHARTTIME"])
                chunk["variable"] = chunk["ITEMID"].map(item_variable)
                chunk["mimic_label"] = chunk["ITEMID"].map(item_label)
                chunk = chunk.merge(
                    stay_frame, on="SUBJECT_ID", how="inner", sort=False
                )
                chunk["event_hour"] = (
                    (chunk["CHARTTIME"] - chunk["INTIME"]).dt.total_seconds()
                    / 3600.0
                )
                chunk = chunk.loc[
                    (chunk["event_hour"] >= -1e-6)
                    & (chunk["event_hour"] <= HOURS + 1e-6)
                ].copy()
                if chunk.empty:
                    continue
                cleaned = clean_selected(chunk)
                valid = np.isfinite(cleaned)
                chunk = chunk.loc[valid].copy()
                cleaned = cleaned[valid].astype(np.float32)
                if chunk.empty:
                    continue
                count = len(chunk)
                code = chunk["variable"].map(variable_index).to_numpy(
                    dtype=np.int16
                )
                chart_us = (
                    chunk["CHARTTIME"].astype("int64") // 1000
                ).to_numpy(np.int64)
                source_rows = next_source_row + np.arange(count, dtype=np.int64)
                next_source_row += count

                bins = np.floor(
                    chunk["event_hour"].to_numpy(float) - 1e-6
                ).astype(int)
                bins = np.clip(bins, 0, 47)
                for source_split in ("train", "test"):
                    selected = chunk["source_split"].eq(source_split).to_numpy()
                    if not selected.any():
                        continue
                    positions = chunk.loc[
                        selected, "source_position"
                    ].to_numpy(np.int64)
                    flat = (
                        (positions * 48 + bins[selected]) * 12 + code[selected]
                    )
                    candidates = (
                        pd.DataFrame(
                            {
                                "flat": flat,
                                "time": chart_us[selected],
                                "value": cleaned[selected],
                            }
                        )
                        .sort_values(
                            ["flat", "time", "value"], kind="mergesort"
                        )
                        .drop_duplicates("flat", keep="last")
                    )
                    flat = candidates["flat"].to_numpy(np.int64)
                    times = candidates["time"].to_numpy(np.int64)
                    values = candidates["value"].to_numpy(np.float32)
                    old_times = grid_times[source_split].reshape(-1)[flat]
                    old_values = grids[source_split].reshape(-1)[flat]
                    replace = (times > old_times) | (
                        (times == old_times) & (values > old_values)
                    )
                    flat = flat[replace]
                    grid_times[source_split].reshape(-1)[flat] = times[replace]
                    grids[source_split].reshape(-1)[flat] = values[replace]

                report["event_count"] = next_source_row
                if chunk_number % 20 == 0:
                    print(
                        f"{stem}: processed {chunk_number} chunks; "
                        f"retained events={next_source_row}",
                        flush=True,
                    )
                yield pl.DataFrame(
                    {
                        "subject_id": pl.Series(
                            chunk["SUBJECT_ID"].to_numpy(np.int64),
                            dtype=pl.Int64,
                        ),
                        "time": pl.Series(chart_us, dtype=pl.Int64).cast(
                            pl.Datetime("us")
                        ),
                        "code": pl.Series(
                            np.asarray(VARIABLE_NAMES)[code], dtype=pl.String
                        ),
                        "numeric_value": pl.Series(cleaned, dtype=pl.Float32),
                        "sample_id": pl.Series(
                            chunk["stay_id"].to_numpy(np.int64), dtype=pl.Int64
                        ),
                        "source_row": pl.Series(source_rows, dtype=pl.Int64),
                    }
                )

        for source_split in ("train", "test"):
            positions = np.asarray(
                [
                    row["source_position"]
                    for row in rows
                    if row["source_split"] == source_split
                ],
                dtype=np.int64,
            )
            observed = np.isfinite(grids[source_split][positions])
            reference_x = source[source_split]["X"][positions]
            reference_mask = reference_x[:, :, 12:] > 0
            mask_equal = bool(np.array_equal(observed, reference_mask))
            if not mask_equal:
                raise AssertionError(
                    f"{source_split} raw-event mask does not reconstruct "
                    "reference tensor"
                )
            difference = np.abs(
                grids[source_split][observed]
                - reference_x[:, :, :12][observed]
            )
            max_difference = float(difference.max(initial=0.0))
            if max_difference > 1e-4:
                raise AssertionError(
                    f"{source_split} observed values differ from reference "
                    f"tensor: {max_difference}"
                )
            report["parity"][source_split] = {
                "observed_cells": int(observed.sum()),
                "mask_exact": mask_equal,
                "maximum_observed_value_difference": max_difference,
            }

    return generate(), report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--auxiliary-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()

    rows, source, retained_frames, icustays_path = load_cohort(
        args.cohort_root, args.mimic_root, args.auxiliary_root, args.cache_root
    )
    author_split(rows)
    event_batches, stream_report = stream_events(
        Path(__file__).resolve().parent, args.mimic_root, rows, source, args.chunksize
    )
    parity = stream_report["parity"]
    samples = pl.DataFrame(
        {
            "subject_id": pl.Series(
                [row["subject_id"] for row in rows], dtype=pl.Int64
            ),
            "sample_id": pl.Series([row["stay_id"] for row in rows], dtype=pl.Int64),
            "observation_start": pl.Series(
                [row["intime_us"] for row in rows], dtype=pl.Int64
            ).cast(pl.Datetime("us")),
            "prediction_time": pl.Series(
                [
                    row["intime_us"] + int(HOURS * 3600 * 1_000_000)
                    for row in rows
                ],
                dtype=pl.Int64,
            ).cast(pl.Datetime("us")),
            "boolean_value": pl.Series(
                [bool(row["label"]) for row in rows], dtype=pl.Boolean
            ),
            "split": pl.Series([row["split"] for row in rows], dtype=pl.String),
            "source_split": pl.Series(
                [row["source_split"] for row in rows], dtype=pl.String
            ),
            "source_position": pl.Series(
                [row["source_position"] for row in rows], dtype=pl.Int64
            ),
            "retained_position": pl.Series(
                [row["retained_position"] for row in rows], dtype=pl.Int64
            ),
            "hadm_id": pl.Series([row["hadm_id"] for row in rows], dtype=pl.Int64),
        }
    )
    split_counts = {
        split: {
            "rows": sum(row["split"] == split for row in rows),
            "subjects": len({row["subject_id"] for row in rows if row["split"] == split}),
            "positive": sum(row["label"] for row in rows if row["split"] == split),
        }
        for split in ("train", "validation", "test")
    }
    norm_stats_path = Path(__file__).resolve().parent / "vital_norm_stats_48h.pt"
    norm_stats = torch.load(norm_stats_path, map_location="cpu")
    metadata = {
        "paper": {
            "key": "oc_distill",
            "title": "OC-Distill: Ontology-aware Contrastive Learning with Cross-Modal Distillation for ICU Risk Prediction",
        },
        "task": {
            "id": "mortality_48h",
            "name": "In-hospital mortality from first 48 ICU hours",
            "primary_metric": "AUROC",
            "published_auroc": 0.793,
            "observation_window_hours": HOURS,
        },
        "dataset_name": "OC-Distill MIMIC-III mortality cohort",
        "dataset_version": "MIMIC-III 1.4 / prior validated exact-count reconstruction",
        "source": {
            "upstream_commit": "9599f6ec7dcf2d74cc1b7ac049b59053ece337b9",
            "loader_entrypoint": "utils.utils::cohort_creation_no_notes -> attach_notes -> build_vitals_tensors -> utils.data::PatientDataset/make_collate",
            "capture_boundary": "full cleaned numeric event stream proven against the note-restricted deployed-student batch",
            "reference_cohort_artifacts": {
                f"mortality_{split}_48h.npz": {
                    "sha256": source[split]["sha256"],
                    "rows": len(source[split]["y"]),
                }
                for split in ("train", "test")
            },
            "raw_event_to_reference_tensor_parity": parity,
            "note_restricted_loader_rows": {
                split: len(retained_frames[split]) for split in ("train", "test")
            },
            "batch_recreation": {
                "batch_size": 1024,
                "normal_values": [
                    59.0, 0.21, 128.0, 86.0, 170.0, 77.0, 98.0,
                    19.0, 118.0, 36.6, 81.0, 7.4,
                ],
                "normalization_artifact": "vital_norm_stats_48h.pt",
                "normalization_artifact_sha256": sha256(norm_stats_path),
                "normalization_mean": np.asarray(norm_stats["mean"]).tolist(),
                "normalization_std": np.asarray(norm_stats["std"]).tolist(),
            },
            "split_policy": "released seed-0 80/20 subject split on train side; fixed benchmark test",
            "split_counts": split_counts,
            "mimic_icustays_sha256": sha256(icustays_path),
        },
    }
    write_meds_task(
        args.output_dir,
        paper=metadata["paper"],
        task=metadata["task"],
        codes=VARIABLE_NAMES,
        events=event_batches,
        samples=samples,
        source=metadata["source"],
        dataset_name=metadata["dataset_name"],
        dataset_version=metadata["dataset_version"],
    )
    print(
        json.dumps(
            {
                "samples": len(rows),
                "events": stream_report["event_count"],
                "splits": split_counts,
                "parity": parity,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
