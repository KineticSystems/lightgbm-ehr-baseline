"""Build SMART's omitted MIMIC-III decompensation source with its pinned producer.

Phase 2.  The paper's second binary MIMIC endpoint is decompensation (Appendix
Table 3, AUROC 94.20 +/- 0.12%).  SMART's README instructs users to replace the
producer's ``mimic3benchmark/scripts/create_decompensation.py`` with its own
``data/MIMIC-III/create_decompensation.py`` before running the producer's task
step, so this builder runs the pinned producer's cohort/episode chain and then
SMART's *own* released task builder verbatim.

Mirrors ``prepare_cohort_source.py`` (the sealed mortality builder): same
pinned producer, same gzip/Python-3 transport repairs, same released
seed-49297 ordering convention, same irreducible filesystem-order
qualification.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl

from prepare_cohort_source import (
    _apply_gzip_python3_repairs,
    _ensure_pinned_producer,
    _flagged_subjects,
)
from cohort_source import (
    COHORT_EVIDENCE,
    PAPER_ROOT,
    PRODUCER_COMMIT,
    PRODUCER_ROOT,
    SOURCE_ARCHIVE,
    assert_mimic_release,
    ordered_text_hash,
    sha256,
)
from cohort_decompensation import (
    BUILD_REPORT,
    BUILD_STATUS,
    EXPECTED_LISTFILE_COUNTS,
    EXPECTED_POSITIVES,
    EXPECTED_TOTAL,
    LISTFILE_HEADER,
    MAX_POSITIVE_DELTA,
    MAX_SAMPLE_DELTA,
    PAPER_LISTFILE_COUNTS,
    PAPER_POSITIVES,
    PAPER_TOTAL,
    SMART_TASK_BUILDER,
    TASK_DIRNAME,
    subject_of,
)


def _marker(markers: Path, name: str) -> Path:
    return markers / f"{name}.json"


def _run_step(markers: Path, name: str, command: list[str], cwd: Path, env: dict) -> None:
    marker = _marker(markers, name)
    if marker.is_file():
        return
    print(f"[decomp-source] {name}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)
    marker.write_text(
        json.dumps(
            {
                "command": command,
                "cwd": str(cwd),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def build_source(mimic_root: Path, cache_root: Path) -> tuple[Path, Path]:
    """Run the pinned producer chain, then SMART's replaced task builder."""
    cache_root.mkdir(parents=True, exist_ok=True)
    markers = cache_root / "markers"
    markers.mkdir(exist_ok=True)
    _ensure_pinned_producer()
    producer = cache_root / "producer"
    if not producer.exists():
        shutil.copytree(PRODUCER_ROOT, producer)
        _apply_gzip_python3_repairs(producer)
    benchmark_root = cache_root / "benchmark" / "root"
    task_root = cache_root / TASK_DIRNAME
    env = dict(os.environ)
    env["PYTHONPATH"] = str(producer)
    python = sys.executable
    itemids = producer / "mimic3benchmark" / "resources" / "itemid_to_variable_map.csv"
    _run_step(
        markers,
        "01_extract_subjects",
        [
            python,
            "-m",
            "mimic3benchmark.scripts.extract_subjects",
            str(mimic_root),
            str(benchmark_root),
            "--itemids_file",
            str(itemids),
        ],
        producer,
        env,
    )
    _run_step(
        markers,
        "02_validate_events",
        [python, "-m", "mimic3benchmark.scripts.validate_events", str(benchmark_root)],
        producer,
        env,
    )
    _run_step(
        markers,
        "03_extract_episodes",
        [
            python,
            "-m",
            "mimic3benchmark.scripts.extract_episodes_from_subjects",
            str(benchmark_root),
        ],
        producer,
        env,
    )
    _run_step(
        markers,
        "04_split_train_and_test",
        [
            python,
            "-m",
            "mimic3benchmark.scripts.split_train_and_test",
            str(benchmark_root),
        ],
        producer,
        env,
    )
    # SMART's own replaced task builder, run verbatim from paper_code/smart.
    _run_step(
        markers,
        "05_smart_create_decompensation",
        [python, str(SMART_TASK_BUILDER), str(benchmark_root), str(task_root)],
        PAPER_ROOT,
        env,
    )
    _run_step(
        markers,
        "06_split_train_val",
        [python, "-m", "mimic3models.split_train_val", str(task_root)],
        producer,
        env,
    )
    return task_root, benchmark_root


def _released_triples(task_root: Path) -> dict[str, dict[str, object]]:
    """Read SMART's released (stay, period_length, y_true) facts, order-free."""
    triples: dict[str, dict[str, object]] = {}
    for partition in ("train", "test"):
        path = task_root / partition / "listfile.csv"
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header != LISTFILE_HEADER:
                raise AssertionError(f"SMART decompensation listfile header changed: {header}")
            for name, period_length, label in reader:
                if name in triples:
                    raise AssertionError(f"duplicate SMART decompensation sample: {name}")
                triples[name] = {
                    "period_length": float(period_length),
                    "label": int(label),
                    "source_partition": partition,
                }
    return triples


def _freeze_order_and_listfiles(
    task_root: Path, triples: dict[str, dict[str, object]]
) -> list[str]:
    """Freeze the released seed-49297 order over lexicographic source names.

    SMART's builder shuffles an ``os.listdir``-ordered list, so the historical
    order is unrecoverable.  This reproduces the primary task's convention.
    """
    validation_subjects = _flagged_subjects(
        PRODUCER_ROOT / "mimic3models" / "resources" / "valset.csv"
    )
    train_names = sorted(
        name for name, fact in triples.items() if fact["source_partition"] == "train"
    )
    random.Random(49297).shuffle(train_names)
    train_order = [n for n in train_names if subject_of(n) not in validation_subjects]
    validation_order = [n for n in train_names if subject_of(n) in validation_subjects]
    test_order = sorted(
        name for name, fact in triples.items() if fact["source_partition"] == "test"
    )
    counts = {
        "train": len(train_order),
        "validation": len(validation_order),
        "test": len(test_order),
    }
    if counts != EXPECTED_LISTFILE_COUNTS:
        raise AssertionError(
            f"released decompensation listfile counts changed: {counts}"
        )
    for mode, names in (
        ("train", train_order),
        ("val", validation_order),
        ("test", test_order),
    ):
        with (task_root / f"{mode}_listfile.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(LISTFILE_HEADER)
            for name in names:
                fact = triples[name]
                writer.writerow(
                    [name, "{:.6f}".format(fact["period_length"]), int(fact["label"])]
                )
    return train_order + validation_order + test_order


def _cohort(benchmark_root: Path, triples: dict, order: list[str]) -> pd.DataFrame:
    stays = pd.read_csv(benchmark_root / "all_stays.csv")
    stays["INTIME"] = pd.to_datetime(stays["INTIME"])
    stays["OUTTIME"] = pd.to_datetime(stays["OUTTIME"])
    stays = stays.sort_values(["SUBJECT_ID", "INTIME", "OUTTIME"], kind="stable")
    stays["episode_number"] = stays.groupby("SUBJECT_ID").cumcount() + 1
    stays["source_name"] = (
        stays["SUBJECT_ID"].astype(str)
        + "_episode"
        + stays["episode_number"].astype(str)
        + "_timeseries.csv"
    )
    stays = stays.set_index("source_name", verify_integrity=True)
    rows = []
    for position, name in enumerate(order):
        fact = triples[name]
        stay = stays.loc[name]
        rows.append(
            {
                "subject_id": int(stay["SUBJECT_ID"]),
                "hadm_id": int(stay["HADM_ID"]),
                "sample_id": int(stay["ICUSTAY_ID"]),
                "intime": stay["INTIME"],
                "outtime": stay["OUTTIME"],
                "source_name": name,
                "source_partition": str(fact["source_partition"]),
                "label": bool(fact["label"]),
                "period_length": float(fact["period_length"]),
                "source_position": position,
            }
        )
    frame = pd.DataFrame(rows)
    prefix = frame["source_name"].str.split("_", n=1).str[0].astype("int64")
    if not (prefix == frame["subject_id"]).all():
        raise AssertionError("stays.csv episode numbering disagrees with the source name")
    return frame


def _write_cohort(cache_root: Path, cohort: pd.DataFrame) -> Path:
    path = cache_root / "decompensation_cohort.parquet"
    pl.DataFrame(
        {
            "subject_id": cohort["subject_id"].astype("int64"),
            "hadm_id": cohort["hadm_id"].astype("int64"),
            "sample_id": cohort["sample_id"].astype("int64"),
            "intime": pd.to_datetime(cohort["intime"]),
            "outtime": pd.to_datetime(cohort["outtime"]),
            "source_name": cohort["source_name"].astype(str),
            "source_partition": cohort["source_partition"].astype(str),
            "label": cohort["label"].astype(bool),
            "period_length": cohort["period_length"].astype("float64"),
            "source_position": cohort["source_position"].astype("int64"),
        },
        schema={
            "subject_id": pl.Int64,
            "hadm_id": pl.Int64,
            "sample_id": pl.Int64,
            "intime": pl.Datetime("us"),
            "outtime": pl.Datetime("us"),
            "source_name": pl.String,
            "source_partition": pl.String,
            "label": pl.Boolean,
            "period_length": pl.Float64,
            "source_position": pl.Int64,
        },
    ).write_parquet(path, compression="zstd")
    return path


def prepare(mimic_root: Path, cache_root: Path) -> dict:
    report_path = cache_root / BUILD_REPORT
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if all((cache_root / value).exists() for value in report["files"].values()):
            return report
    release = assert_mimic_release(mimic_root)
    task_root, benchmark_root = build_source(mimic_root, cache_root)
    triples = _released_triples(task_root)
    if len(triples) != EXPECTED_TOTAL:
        raise AssertionError(f"SMART decompensation sample count changed: {len(triples)}")
    positives = sum(1 for fact in triples.values() if fact["label"] == 1)
    if positives != EXPECTED_POSITIVES:
        raise AssertionError(f"SMART decompensation positives changed: {positives}")
    if abs(PAPER_TOTAL - len(triples)) > MAX_SAMPLE_DELTA:
        raise AssertionError(
            "rebuilt decompensation cohort is too far from the paper's 41,744 "
            f"examples: {len(triples)}"
        )
    if abs(PAPER_POSITIVES - positives) > MAX_POSITIVE_DELTA:
        raise AssertionError(
            "rebuilt decompensation positives are too far from the paper's 1,474: "
            f"{positives}"
        )
    order = _freeze_order_and_listfiles(task_root, triples)
    cohort = _cohort(benchmark_root, triples, order)
    cohort_path = _write_cohort(cache_root, cohort)
    report = {
        "status": BUILD_STATUS,
        "task": "decompensation",
        "fidelity_limitation": (
            "SMART did not release decompensation_normalized.pkl, its derived "
            "listfiles, its producer build environment, or the filesystem "
            "enumeration that fixed row order. (1) The released task builder "
            "shuffles an os.listdir-ordered list, so the historical example order "
            "is unrecoverable; this build freezes lexicographic source names "
            "before the released seed-49297 shuffle. (2) Rerunning the pinned "
            "producer chain on exact MIMIC-III 1.4 yields 41,726 eligible stays "
            "with 1,471 positives against the 41,744 / 1,474 the released "
            "notebook printed - an 0.043% shortfall caused by unreleased producer "
            "runtime details (pandas-version-dependent event cleaning), measured "
            "and reported rather than patched."
        ),
        "paper_targets": {
            "samples": PAPER_TOTAL,
            "positives": PAPER_POSITIVES,
            "listfile_counts": PAPER_LISTFILE_COUNTS,
            "source": "saved outputs of the released decompensation notebook cells",
        },
        "paper_delta": {
            "samples": PAPER_TOTAL - len(cohort),
            "positives": PAPER_POSITIVES - int(cohort["label"].sum()),
            "listfile_counts": {
                key: PAPER_LISTFILE_COUNTS[key] - EXPECTED_LISTFILE_COUNTS[key]
                for key in PAPER_LISTFILE_COUNTS
            },
            "sample_shortfall_fraction": (PAPER_TOTAL - len(cohort)) / PAPER_TOTAL,
        },
        "construction": (
            "pinned producer cohort/episode chain on exact compressed MIMIC-III 1.4, "
            "then SMART's own released data/MIMIC-III/create_decompensation.py and the "
            "producer's mimic3models.split_train_val"
        ),
        "producer_commit": PRODUCER_COMMIT,
        "producer_archive_sha256": sha256(SOURCE_ARCHIVE),
        "mimic_release": "MIMIC-III 1.4",
        "release_sha256": release,
        "files": {
            "cohort": cohort_path.name,
            "author_task_root": TASK_DIRNAME,
        },
        "samples": len(cohort),
        "subjects": int(cohort["subject_id"].nunique()),
        "positives": int(cohort["label"].sum()),
        "listfile_counts": EXPECTED_LISTFILE_COUNTS,
        "source_name_sha256": ordered_text_hash(cohort["source_name"]),
        "sample_id_sha256": ordered_text_hash(cohort["sample_id"]),
        "label_sha256": ordered_text_hash(cohort["label"].astype(int)),
        "period_length_sha256": ordered_text_hash(
            cohort["period_length"].map("{:.6f}".format)
        ),
        "ordering_convention": (
            "lexicographic source names -> released seed 49297 shuffle -> "
            "train/validation filter -> sorted test"
        ),
        "source_files": {
            "smart_task_builder": sha256(SMART_TASK_BUILDER),
            "smart_notebook": sha256(PAPER_ROOT / "data" / "MIMIC-III" / "preprocess_mimiciii.ipynb"),
            "smart_reader": sha256(PAPER_ROOT / "data" / "MIMIC-III" / "reader.py"),
            "smart_loader": sha256(PAPER_ROOT / "data" / "mimiciii.py"),
            "smart_collate": sha256(PAPER_ROOT / "data" / "dataloader.py"),
            "testset": sha256(COHORT_EVIDENCE / "testset.csv"),
            "valset": sha256(COHORT_EVIDENCE / "valset.csv"),
            "channel_info": sha256(COHORT_EVIDENCE / "channel_info.json"),
            "discretizer_config": sha256(COHORT_EVIDENCE / "discretizer_config.json"),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="run only the producer/task build, skipping cohort freezing",
    )
    args = parser.parse_args()
    mimic_root = args.mimic_root.resolve()
    cache_root = args.cache_root.resolve()
    if args.source_only:
        assert_mimic_release(mimic_root)
        task_root, benchmark_root = build_source(mimic_root, cache_root)
        print(json.dumps({"task_root": str(task_root), "root": str(benchmark_root)}, indent=2))
        return
    print(json.dumps(prepare(mimic_root, cache_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
