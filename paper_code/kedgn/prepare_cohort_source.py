"""Prepare KEDGN's pinned public MIMIC-III preprocessing chain."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import polars as pl

from cohort_source import (
    BENCHMARK_COMMIT,
    EXPECTED_BLACKLIST_REMOVALS,
    EXPECTED_POSITIVES,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_POSITIVES,
    EXPECTED_SPLIT_SUBJECTS,
    EXPECTED_TOTAL,
    MEDICAL_TS_COMMIT,
    SPLITS,
    assert_medical_ts_source,
    assert_mimic_release,
    blacklist,
    ordered_text_hash,
    sha256,
    source_task_root,
)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise AssertionError(f"existing cache file has changed size: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _validate_compatible_cache(root: Path) -> tuple[dict, pl.DataFrame, Path]:
    report_path = root / "source_build.json"
    cohort_path = root / "source_cohort.parquet"
    task_root = root / "in-hospital-mortality"
    if not report_path.is_file() or not cohort_path.is_file() or not task_root.is_dir():
        raise FileNotFoundError("compatible cache lacks the exact benchmark task and cohort map")
    report = json.loads(report_path.read_text())
    if (
        report.get("producer_commit") != BENCHMARK_COMMIT
        or report.get("mimic_release") != "MIMIC-III 1.4"
        or report.get("samples") != 21_139
        or report.get("positives") != 2_797
    ):
        raise AssertionError("compatible cache is not the pinned standard IHM extraction")
    cohort = pl.read_parquet(cohort_path)
    if cohort.height != 21_139 or cohort["source_name"].n_unique() != 21_139:
        raise AssertionError("compatible benchmark cohort inventory changed")
    return report, cohort, task_root


def _source_records(task_root: Path, base_cohort: pl.DataFrame) -> pl.DataFrame:
    excluded = set(blacklist())
    by_name = {
        row["source_name"]: row
        for row in base_cohort.iter_rows(named=True)
    }
    rows = []
    removal_counts = {}
    source_position = 0
    for split, mode, partition in (
        ("train", "train", "train"),
        ("validation", "val", "train"),
        ("test", "test", "test"),
    ):
        listfile = task_root / f"{mode}_listfile.csv"
        with listfile.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["stay", "y_true"]:
                raise AssertionError(f"unexpected standard listfile schema: {listfile}")
            records = list(reader)
        kept = [record for record in records if record["stay"] not in excluded]
        removal_counts[split] = len(records) - len(kept)
        for split_position, record in enumerate(kept):
            name = record["stay"]
            if name not in by_name:
                raise AssertionError(f"standard listfile lacks a cohort mapping: {name}")
            base = by_name[name]
            label = int(record["y_true"])
            if label != int(base["label"]):
                raise AssertionError(f"standard label disagrees with cohort map: {name}")
            rows.append(
                {
                    "subject_id": int(base["subject_id"]),
                    "hadm_id": int(base["hadm_id"]),
                    "sample_id": int(base["sample_id"]),
                    "intime": base["intime"],
                    "outtime": base["outtime"],
                    "source_name": name,
                    "source_partition": partition,
                    "split": split,
                    "split_position": split_position,
                    "source_position": source_position,
                    "label": bool(label),
                }
            )
            source_position += 1
    if removal_counts != EXPECTED_BLACKLIST_REMOVALS:
        raise AssertionError(f"medical_ts_datasets blacklist removals changed: {removal_counts}")
    cohort = pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "hadm_id": pl.Int64,
            "sample_id": pl.Int64,
            "intime": pl.Datetime("us"),
            "outtime": pl.Datetime("us"),
            "source_name": pl.String,
            "source_partition": pl.String,
            "split": pl.String,
            "split_position": pl.Int64,
            "source_position": pl.Int64,
            "label": pl.Boolean,
        },
    )
    if cohort.height != EXPECTED_TOTAL or cohort["sample_id"].n_unique() != EXPECTED_TOTAL:
        raise AssertionError("prepared KEDGN cohort count or stay uniqueness changed")
    counts = dict(cohort.group_by("split").len().iter_rows())
    positives = dict(cohort.group_by("split").agg(pl.col("label").sum()).iter_rows())
    subjects = dict(
        cohort.group_by("split").agg(pl.col("subject_id").n_unique()).iter_rows()
    )
    if counts != EXPECTED_SPLIT_COUNTS:
        raise AssertionError(f"prepared split counts changed: {counts}")
    if positives != EXPECTED_SPLIT_POSITIVES:
        raise AssertionError(f"prepared split labels changed: {positives}")
    if subjects != EXPECTED_SPLIT_SUBJECTS:
        raise AssertionError(f"prepared split subject counts changed: {subjects}")
    if int(cohort["label"].sum()) != EXPECTED_POSITIVES:
        raise AssertionError("prepared total positive count changed")
    return cohort


def prepare(mimic_root: Path, cache_root: Path, compatible_root: Path) -> dict:
    report_path = cache_root / "source_build.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if all((cache_root / value).exists() for value in report.get("files", {}).values()):
            return report

    release = assert_mimic_release(mimic_root)
    medical_hashes = assert_medical_ts_source()
    compatible_report, base_cohort, compatible_task = _validate_compatible_cache(
        compatible_root
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    task_root = source_task_root(cache_root)
    for mode, partition in (("train", "train"), ("val", "train"), ("test", "test")):
        _link_or_copy(
            compatible_task / f"{mode}_listfile.csv",
            task_root / f"{mode}_listfile.csv",
        )
        with (compatible_task / f"{mode}_listfile.csv").open(newline="") as handle:
            for record in csv.DictReader(handle):
                name = record["stay"]
                _link_or_copy(
                    compatible_task / partition / name,
                    task_root / partition / name,
                )

    cohort = _source_records(task_root, base_cohort)
    cohort_path = cache_root / "cohort.parquet"
    cohort.write_parquet(cohort_path, compression="zstd")
    listfiles = {
        mode: sha256(task_root / f"{mode}_listfile.csv")
        for mode in ("train", "val", "test")
    }
    report = {
        "status": "exact_structured_source_boundary",
        "fidelity_limitation": (
            "KEDGN did not release its final six object-array cache files or the "
            "few-line cache writer. The README-attributed public chain nevertheless "
            "fixes the exact cohort, irregular 16-variable values, finite-cell mask, "
            "timestamps, lengths, labels, and split. The only late adapter used for "
            "KEDGN tensor parity losslessly coalesces non-conflicting rows at identical "
            "Hours and replaces source NaN with zero while retaining the source mask, "
            "which is necessary for the released tensorizer to stay finite. A narrow "
            "pandas compatibility shim keeps the reader's explicitly mapped literal "
            "GCS category 'None' from being swallowed by the current default-NA set."
        ),
        "construction": (
            "pinned mimic3-benchmarks first-48-hour IHM extraction followed by the "
            "pinned medical_ts_datasets MIMICMortalityReader and 32-stay blacklist"
        ),
        "benchmark_commit": BENCHMARK_COMMIT,
        "medical_ts_commit": MEDICAL_TS_COMMIT,
        "mimic_release": "MIMIC-III 1.4",
        "release_sha256": release,
        "compatible_cache_report_sha256": sha256(compatible_root / "source_build.json"),
        "compatible_cache_producer_archive_sha256": compatible_report.get(
            "producer_archive_sha256"
        ),
        "medical_ts_source_sha256": medical_hashes,
        "listfile_sha256": listfiles,
        "files": {
            "cohort": cohort_path.name,
            "task_root": str(task_root.relative_to(cache_root)),
        },
        "samples": cohort.height,
        "subjects": cohort["subject_id"].n_unique(),
        "positives": int(cohort["label"].sum()),
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "split_positives": EXPECTED_SPLIT_POSITIVES,
        "blacklist_count": len(blacklist()),
        "blacklist_sha256": ordered_text_hash(blacklist()),
        "ordered_source_name_sha256": ordered_text_hash(cohort["source_name"]),
        "ordered_sample_id_sha256": ordered_text_hash(cohort["sample_id"]),
        "ordered_label_sha256": ordered_text_hash(cohort["label"].cast(pl.Int8)),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--compatible-cache", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(
        args.mimic_root.resolve(),
        args.cache_root.resolve(),
        args.compatible_cache.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
