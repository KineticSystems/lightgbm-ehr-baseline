#!/usr/bin/env python3
"""Freeze task-equivalent note and CXR rows before MERGE embedding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HOURS = 48
EXPECTED_NOTES = 105_197
EXPECTED_CXRS = 19_940
EXPECTED_CXR_STAYS = 8_772
CXR_METADATA_NAME = "cxr_metadata_with_time_delta.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_text(digest: hashlib._Hash, value: object) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def note_identity(notes: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in notes[
        ["source_note_row", "subject_id", "stay_id", "note_id", "icu_time_delta"]
    ].itertuples(index=False):
        digest.update(
            np.asarray(
                [int(row.source_note_row), int(row.subject_id), int(row.stay_id)],
                dtype="<i8",
            ).tobytes()
        )
        update_text(digest, row.note_id)
        digest.update(np.asarray([float(row.icu_time_delta)], dtype="<f8").tobytes())
    return digest.hexdigest()


def cxr_identity(cxrs: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in cxrs[
        ["subject_id", "study_id", "dicom_id", "stay_id", "icu_time_delta"]
    ].itertuples(index=False):
        digest.update(
            np.asarray(
                [int(row.subject_id), int(row.study_id), int(row.stay_id)],
                dtype="<i8",
            ).tobytes()
        )
        update_text(digest, row.dicom_id)
        digest.update(np.asarray([float(row.icu_time_delta)], dtype="<f8").tobytes())
    return digest.hexdigest()


def image_path(root: Path, row: object) -> Path:
    subject_id = int(row.subject_id)
    return (
        root
        / "files"
        / f"p{str(subject_id)[:2]}"
        / f"p{subject_id}"
        / f"s{int(row.study_id)}"
        / f"{row.dicom_id}.jpg"
    )


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def safe_symlink(source: Path, target: Path) -> None:
    source = source.resolve()
    if target.is_symlink():
        if target.resolve() != source:
            raise RuntimeError(f"mismatched symlink: {target}")
        return
    if target.exists():
        if sha256(target) != sha256(source):
            raise RuntimeError(f"refusing to replace mismatched file: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--cxr-metadata", type=Path, required=True)
    parser.add_argument("--cxr-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark-rows", type=int, default=256)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    data_root = output_root / "data"
    bounded_root = output_root / "bounded_inputs"
    benchmark_root = output_root / "benchmarks"

    structured = pd.read_parquet(
        args.structured, columns=["stay_id", "icu_time_delta"]
    )
    candidate_ids = set(
        structured.loc[
            (structured["icu_time_delta"] >= 0)
            & (structured["icu_time_delta"] < HOURS),
            "stay_id",
        ]
        .astype(np.int64)
        .unique()
    )
    if len(candidate_ids) != 46_337:
        raise RuntimeError(f"candidate stay count changed: {len(candidate_ids)}")

    notes = pd.read_parquet(args.notes)
    if len(notes) != EXPECTED_NOTES:
        raise RuntimeError(f"note count changed: {len(notes)}")
    if notes["stay_id"].isna().any():
        raise RuntimeError("bounded note cache contains null stay_id")
    if not notes["stay_id"].astype(np.int64).isin(candidate_ids).all():
        raise RuntimeError("bounded note cache contains an out-of-cohort stay")
    if not ((notes["icu_time_delta"] >= 0) & (notes["icu_time_delta"] < HOURS)).all():
        raise RuntimeError("bounded note cache violates [0,48)")

    cxrs = pd.read_parquet(args.cxr_metadata)
    mapped = cxrs["stay_id"].notna()
    in_window = (cxrs["icu_time_delta"] >= 0) & (cxrs["icu_time_delta"] < HOURS)
    in_cohort = cxrs["stay_id"].astype("Int64").isin(candidate_ids)
    selected_cxrs = cxrs.loc[mapped & in_window & in_cohort].copy()
    if len(selected_cxrs) != EXPECTED_CXRS:
        raise RuntimeError(f"CXR count changed: {len(selected_cxrs)}")
    if selected_cxrs["stay_id"].nunique() != EXPECTED_CXR_STAYS:
        raise RuntimeError(
            f"CXR stay count changed: {selected_cxrs['stay_id'].nunique()}"
        )
    if selected_cxrs["dicom_id"].duplicated().any():
        raise RuntimeError("selected CXR DICOM identifiers are not unique")

    missing_images = [
        path
        for row in selected_cxrs.itertuples(index=False)
        if not (path := image_path(args.cxr_root, row)).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"missing selected CXR images: {len(missing_images)}")

    bounded_cxr_path = bounded_root / "cxr" / CXR_METADATA_NAME
    write_parquet(selected_cxrs, bounded_cxr_path)
    write_parquet(
        notes.iloc[: args.benchmark_rows].copy(),
        benchmark_root / "notes" / "rad_notes_text.parquet",
    )
    write_parquet(
        selected_cxrs.iloc[: args.benchmark_rows].copy(),
        benchmark_root / "cxr" / CXR_METADATA_NAME,
    )
    safe_symlink(args.structured, data_root / "ts_labs_vitals.parquet")

    report = {
        "status": "exact_bounded_inputs",
        "window": "[0,48) ICU hours",
        "candidate_stays": len(candidate_ids),
        "notes": {
            "rows": len(notes),
            "stays": int(notes["stay_id"].nunique()),
            "source_sha256": sha256(args.notes),
            "identity_sha256": note_identity(notes),
        },
        "cxr": {
            "source_metadata_rows": len(cxrs),
            "rows": len(selected_cxrs),
            "stays": int(selected_cxrs["stay_id"].nunique()),
            "images_present": len(selected_cxrs),
            "source_metadata_sha256": sha256(args.cxr_metadata),
            "bounded_metadata_sha256": sha256(bounded_cxr_path),
            "identity_sha256": cxr_identity(selected_cxrs),
        },
        "benchmark_rows": args.benchmark_rows,
    }
    report_path = output_root / "audit" / "embedding_inputs.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
