"""Build SMART's omitted MIMIC-III mortality source with its pinned producer."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from cohort_source import (
    COHORT_EVIDENCE,
    EXPECTED_IHM_INVENTORY_SHA256,
    EXPECTED_POSITIVES,
    EXPECTED_SOURCE_ARCHIVE_SHA256,
    EXPECTED_SUBJECTS,
    EXPECTED_TOTAL,
    PAPER_ROOT,
    PRODUCER_COMMIT,
    PRODUCER_ROOT,
    RELEASE_SHA256,
    SOURCE_ARCHIVE,
    assert_mimic_release,
    ordered_text_hash,
    sha256,
    source_root,
)


def _flagged_subjects(path: Path) -> set[int]:
    result = set()
    with path.open() as handle:
        for subject, flag in csv.reader(handle):
            if int(flag) == 1:
                result.add(int(subject))
    return result


def _ensure_pinned_producer() -> None:
    if sha256(SOURCE_ARCHIVE) != EXPECTED_SOURCE_ARCHIVE_SHA256:
        raise AssertionError("pinned producer archive hash changed")
    if PRODUCER_ROOT.is_dir():
        return
    with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (member.isfile() or member.isdir())
            ):
                raise AssertionError("unsafe member in pinned producer archive")
        archive.extractall(PAPER_ROOT / ".export_assets")
    if not PRODUCER_ROOT.is_dir():
        raise AssertionError("pinned producer archive extracted an unexpected root")


def _inventory_names() -> set[str]:
    path = COHORT_EVIDENCE / "ihm.csv"
    if sha256(path) != EXPECTED_IHM_INVENTORY_SHA256:
        raise AssertionError("released IHM inventory hash changed")
    names = set()
    with path.open() as handle:
        for relative, _ in csv.reader(handle):
            name = Path(relative).name
            if name != "listfile.csv":
                names.add(name)
    if len(names) != EXPECTED_TOTAL:
        raise AssertionError("released IHM inventory count changed")
    return names


def _apply_gzip_python3_repairs(producer: Path) -> None:
    """Make the exact pinned producer executable on compressed official files."""
    util = producer / "mimic3benchmark" / "util.py"
    util_text = util.read_text()
    old = "def dataframe_from_csv(path, header=0, index_col=0):\n    return pd.read_csv(path, header=header, index_col=index_col)"
    new = """def dataframe_from_csv(path, header=0, index_col=0):
    if not os.path.exists(path) and path.endswith('.csv') and os.path.exists(path + '.gz'):
        path = path + '.gz'
    return pd.read_csv(path, header=header, index_col=index_col)"""
    if old not in util_text:
        raise AssertionError("pinned producer util.py no longer matches")
    util.write_text("import os\n" + util_text.replace(old, new))

    mimic3csv = producer / "mimic3benchmark" / "mimic3csv.py"
    csv_text = mimic3csv.read_text()
    old = "reader = csv.DictReader(open(os.path.join(mimic3_path, table.upper() + '.csv'), 'r'))"
    new = """path = os.path.join(mimic3_path, table.upper() + '.csv')
    if os.path.exists(path):
        handle = open(path, 'r')
    else:
        import gzip
        handle = gzip.open(path + '.gz', 'rt')
    reader = csv.DictReader(handle)"""
    if old not in csv_text:
        raise AssertionError("pinned producer mimic3csv.py no longer matches")
    mimic3csv.write_text(csv_text.replace(old, new))

    preprocessing = producer / "mimic3benchmark" / "preprocessing.py"
    prep_text = preprocessing.read_text()
    old = "is_str = np.array(map(lambda x: type(x) == str, list(df.VALUE)), dtype=bool)"
    new = "is_str = np.array(list(map(lambda x: type(x) == str, list(df.VALUE))), dtype=bool)"
    if old not in prep_text:
        raise AssertionError("pinned producer preprocessing.py no longer matches")
    preprocessing.write_text(prep_text.replace(old, new))


def _run_exact_producer(mimic_root: Path, cache_root: Path) -> tuple[Path, Path]:
    producer = cache_root / "producer"
    if not producer.exists():
        shutil.copytree(PRODUCER_ROOT, producer)
        _apply_gzip_python3_repairs(producer)
    benchmark_root = cache_root / "benchmark"
    task_root = source_root(cache_root)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(producer)
    commands = (
        [
            sys.executable,
            "-m",
            "mimic3benchmark.scripts.extract_subjects",
            str(mimic_root),
            str(benchmark_root / "root"),
            "--itemids_file",
            str(producer / "mimic3benchmark" / "resources" / "itemid_to_variable_map.csv"),
        ],
        [sys.executable, "-m", "mimic3benchmark.scripts.validate_events", str(benchmark_root / "root")],
        [sys.executable, "-m", "mimic3benchmark.scripts.extract_episodes_from_subjects", str(benchmark_root / "root")],
        [sys.executable, "-m", "mimic3benchmark.scripts.split_train_and_test", str(benchmark_root / "root")],
        [sys.executable, "-m", "mimic3benchmark.scripts.create_in_hospital_mortality", str(benchmark_root / "root"), str(task_root)],
        [sys.executable, "-m", "mimic3models.split_train_val", str(task_root)],
    )
    for command in commands:
        subprocess.run(command, cwd=producer, env=env, check=True)
    return task_root, benchmark_root / "root" / "all_stays.csv"


def _cohort_from_stays(task_root: Path, stays_path: Path) -> pd.DataFrame:
    stays = pd.read_csv(stays_path)
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
    labels = []
    for partition in ("train", "test"):
        path = task_root / partition / "listfile.csv"
        frame = pd.read_csv(path)
        frame["source_partition"] = partition
        labels.append(frame.rename(columns={"stay": "source_name", "y_true": "label"}))
    labels = pd.concat(labels, ignore_index=True)
    result = labels.merge(stays, on="source_name", validate="1:1")
    return result.rename(
        columns={"SUBJECT_ID": "subject_id", "HADM_ID": "hadm_id", "ICUSTAY_ID": "sample_id", "INTIME": "intime", "OUTTIME": "outtime"}
    )


def _canonical_stays(mimic_root: Path) -> pd.DataFrame:
    sys.path.insert(0, str(PRODUCER_ROOT))
    from mimic3benchmark import mimic3csv

    def gzip_aware(path, *args, **kwargs):
        path = Path(path)
        if not path.is_file() and path.suffix == ".csv":
            path = path.with_suffix(".csv.gz")
        return pd.read_csv(path, *args, **kwargs)

    previous = mimic3csv.dataframe_from_csv
    mimic3csv.dataframe_from_csv = gzip_aware
    try:
        patients = mimic3csv.read_patients_table(mimic_root)
        admissions = mimic3csv.read_admissions_table(mimic_root)
        stays = mimic3csv.read_icustays_table(mimic_root)
    finally:
        mimic3csv.dataframe_from_csv = previous
    stays = mimic3csv.remove_icustays_with_transfers(stays)
    stays = mimic3csv.merge_on_subject_admission(stays, admissions)
    stays = mimic3csv.merge_on_subject(stays, patients)
    stays = mimic3csv.filter_admissions_on_nb_icustays(stays)
    stays = mimic3csv.add_age_to_icustays(stays)
    stays = mimic3csv.add_inunit_mortality_to_icustays(stays)
    stays = mimic3csv.add_inhospital_mortality_to_icustays(stays)
    stays = mimic3csv.filter_icustays_on_age(stays)
    stays = stays.sort_values(["SUBJECT_ID", "INTIME", "OUTTIME"], kind="stable")
    stays["episode_number"] = stays.groupby("SUBJECT_ID").cumcount() + 1
    stays["source_name"] = (
        stays["SUBJECT_ID"].astype(str)
        + "_episode"
        + stays["episode_number"].astype(str)
        + "_timeseries.csv"
    )
    return stays


MCP_NAME = re.compile(
    r"^(?P<subject>[0-9]+)_episode(?P<stay>[0-9]+)_timeseries_"
    r"intime_[0-9-]+_[0-9-]+_outtime_[0-9-]+_[0-9-]+\.csv$"
)


def _cohort_from_mcp_cache(
    task_root: Path, compatible_root: Path, mimic_root: Path
) -> tuple[pd.DataFrame, str, str]:
    source = compatible_root / "in-hospital-mortality"
    build_path = compatible_root / "source_build.json"
    if not source.is_dir() or not build_path.is_file():
        raise FileNotFoundError("compatible MCP cache lacks source task/build report")
    build = json.loads(build_path.read_text())
    if (
        build.get("producer_commit") != PRODUCER_COMMIT
        or build.get("mimic_release") != "MIMIC-III 1.4"
        or build.get("official_raw_sha256", {}).get("CHARTEVENTS.csv.gz")
        != RELEASE_SHA256["CHARTEVENTS.csv.gz"]
        or build.get("observed_source_cells") != 9_734_043
    ):
        raise AssertionError("compatible MCP cache is not the pinned exact-release source")
    rows = []
    for mode, partition in (("train", "train"), ("val", "train"), ("test", "test")):
        with (source / f"{mode}_listfile.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["stay", "y_true"]:
                raise AssertionError("compatible MCP listfile schema changed")
            for record in reader:
                match = MCP_NAME.fullmatch(str(record["stay"]))
                if match is None:
                    raise AssertionError("compatible MCP source filename changed")
                rows.append(
                    {
                        "subject_id": int(match.group("subject")),
                        "sample_id": int(match.group("stay")),
                        "compatible_name": str(record["stay"]),
                        "source_partition": partition,
                        "label": int(record["y_true"]),
                    }
                )
    frame = pd.DataFrame(rows)
    stays = _canonical_stays(mimic_root).set_index("ICUSTAY_ID", verify_integrity=True)
    selected = stays.loc[frame["sample_id"]]
    frame["hadm_id"] = selected["HADM_ID"].to_numpy(dtype="int64")
    frame["intime"] = pd.to_datetime(selected["INTIME"].to_numpy())
    frame["outtime"] = pd.to_datetime(selected["OUTTIME"].to_numpy())
    frame["source_name"] = selected["source_name"].to_numpy()
    if not np.array_equal(
        selected["SUBJECT_ID"].to_numpy(dtype="int64"),
        frame["subject_id"].to_numpy(dtype="int64"),
    ):
        raise AssertionError("compatible MCP filenames disagree with pinned producer stays")
    if not np.array_equal(
        selected["MORTALITY"].to_numpy(dtype="int64"),
        frame["label"].to_numpy(dtype="int64"),
    ):
        raise AssertionError("compatible MCP labels disagree with pinned producer stays")
    if not task_root.exists():
        (task_root / "train").mkdir(parents=True)
        (task_root / "test").mkdir(parents=True)
        for row in frame.itertuples(index=False):
            shutil.copy2(
                source / row.source_partition / row.compatible_name,
                task_root / row.source_partition / row.source_name,
            )
    return (
        frame[
            [
                "subject_id",
                "hadm_id",
                "sample_id",
                "intime",
                "outtime",
                "source_name",
                "label",
            ]
        ],
        sha256(build_path),
        (
            "bootstrap from a previously verified exact pinned-producer episode "
            "tree on MIMIC-III 1.4; filenames remapped to the released standard "
            "IHM inventory through pinned-producer ICU stay order"
        ),
    )


def _cohort_from_compatible_cache(
    task_root: Path, compatible_root: Path, mimic_root: Path
) -> tuple[pd.DataFrame, str | None, str]:
    if (compatible_root / "source_build.json").is_file():
        return _cohort_from_mcp_cache(task_root, compatible_root, mimic_root)
    source = compatible_root / "author_view" / "in-hospital-mortality"
    cohort_path = compatible_root / "cohort.parquet"
    manifest_path = compatible_root / "source_manifest.json"
    if not source.is_dir() or not cohort_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("compatible cache lacks author_view/cohort.parquet")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "exact_reconstruction":
        raise AssertionError("compatible cache is not an exact-release reconstruction")
    compatible_release = manifest.get("release_sha256", {})
    required_release = {
        "PATIENTS.csv.gz",
        "ADMISSIONS.csv.gz",
        "ICUSTAYS.csv.gz",
        "CHARTEVENTS.csv.gz",
        "LABEVENTS.csv.gz",
        "OUTPUTEVENTS.csv.gz",
    }
    if not required_release.issubset(compatible_release) or any(
        RELEASE_SHA256.get(filename) != value
        for filename, value in compatible_release.items()
        if filename in RELEASE_SHA256
    ):
        raise AssertionError("compatible cache used different MIMIC-III release bytes")
    resources = manifest.get("source_resource_sha256", {})
    for filename, path in (
        (
            "itemid_to_variable_map.csv",
            PRODUCER_ROOT / "mimic3benchmark" / "resources" / "itemid_to_variable_map.csv",
        ),
        ("testset.csv", COHORT_EVIDENCE / "testset.csv"),
        ("valset.csv", COHORT_EVIDENCE / "valset.csv"),
    ):
        if resources.get(filename) != sha256(path):
            raise AssertionError(f"compatible cache changed released resource: {filename}")
    if not task_root.exists():
        shutil.copytree(source, task_root, copy_function=shutil.copy2)
    cohort = pd.read_parquet(cohort_path)
    cohort = cohort.rename(
        columns={"SUBJECT_ID": "subject_id", "HADM_ID": "hadm_id", "stay_id": "sample_id", "INTIME": "intime", "OUTTIME": "outtime"}
    )[
        ["subject_id", "hadm_id", "sample_id", "intime", "outtime", "source_name", "label"]
    ]
    return (
        cohort,
        sha256(manifest_path),
        (
            "bootstrap from previously verified exact-release mimic3-benchmarks "
            "17-channel author view; cohort/resources/source semantics reverified here"
        ),
    )


def _freeze_order_and_listfiles(task_root: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    inventory = _inventory_names()
    if set(cohort["source_name"]) != inventory:
        raise AssertionError("reconstructed source-name set differs from released IHM inventory")
    test_subjects = _flagged_subjects(
        PRODUCER_ROOT / "mimic3benchmark" / "resources" / "testset.csv"
    )
    validation_subjects = _flagged_subjects(
        PRODUCER_ROOT / "mimic3models" / "resources" / "valset.csv"
    )
    cohort = cohort.copy()
    cohort["source_partition"] = "train"
    cohort.loc[cohort["subject_id"].isin(test_subjects), "source_partition"] = "test"
    missing_files = [
        str(task_root / row.source_partition / row.source_name)
        for row in cohort.itertuples(index=False)
        if not (task_root / row.source_partition / row.source_name).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"reconstructed SMART task is missing source files: {missing_files[:5]}")
    non_test = sorted(cohort.loc[cohort["source_partition"] == "train", "source_name"])
    random.Random(49297).shuffle(non_test)
    by_name = cohort.set_index("source_name")
    train_order = [name for name in non_test if int(by_name.loc[name, "subject_id"]) not in validation_subjects]
    validation_order = [name for name in non_test if int(by_name.loc[name, "subject_id"]) in validation_subjects]
    test_order = sorted(cohort.loc[cohort["source_partition"] == "test", "source_name"])
    notebook_order = train_order + validation_order + test_order
    if len(notebook_order) != EXPECTED_TOTAL or len(set(notebook_order)) != EXPECTED_TOTAL:
        raise AssertionError("frozen SMART notebook order is incomplete")

    for mode, names in (("train", train_order), ("val", validation_order), ("test", test_order)):
        with (task_root / f"{mode}_listfile.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["stay", "y_true"])
            for name in names:
                writer.writerow([name, int(by_name.loc[name, "label"])])

    position = {name: index for index, name in enumerate(notebook_order)}
    cohort["source_position"] = cohort["source_name"].map(position).astype("int64")
    cohort = cohort.sort_values("source_position", kind="stable").reset_index(drop=True)
    return cohort


def _write_cohort(cache_root: Path, cohort: pd.DataFrame) -> Path:
    path = cache_root / "source_cohort.parquet"
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
            "source_position": pl.Int64,
        },
    ).write_parquet(path, compression="zstd")
    return path


def prepare(mimic_root: Path, cache_root: Path, compatible_cache: Path | None) -> dict:
    report_path = cache_root / "source_build.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        required = [cache_root / value for value in report["files"].values()]
        if all(path.exists() for path in required):
            return report

    cache_root.mkdir(parents=True, exist_ok=True)
    release = assert_mimic_release(mimic_root)
    _ensure_pinned_producer()
    task_root = source_root(cache_root)
    if compatible_cache is None:
        task_root, stays_path = _run_exact_producer(mimic_root, cache_root)
        cohort = _cohort_from_stays(task_root, stays_path)
        construction = "pinned producer run on exact compressed MIMIC-III 1.4"
        compatible_hash = None
    else:
        cohort, compatible_hash, construction = _cohort_from_compatible_cache(
            task_root, compatible_cache, mimic_root
        )

    cohort = _freeze_order_and_listfiles(task_root, cohort)
    if len(cohort) != EXPECTED_TOTAL:
        raise AssertionError(f"SMART cohort count changed: {len(cohort)}")
    if cohort["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise AssertionError("SMART subject count changed")
    if int(cohort["label"].sum()) != EXPECTED_POSITIVES:
        raise AssertionError("SMART positive count changed")
    cohort_path = _write_cohort(cache_root, cohort)
    report = {
        "status": "exact_source_with_unreleased_pickle_and_filesystem_order",
        "fidelity_limitation": (
            "SMART did not release mortality_normalized.pkl, its historical producer/runtime "
            "artifacts, or the filesystem enumeration that fixed row order; the pinned-source "
            "reconstruction exactly matches the cohort and observation mask and freezes "
            "lexicographic source names before the released seed-49297 task shuffle"
        ),
        "construction": construction,
        "compatible_cache_manifest_sha256": compatible_hash,
        "producer_commit": PRODUCER_COMMIT,
        "producer_archive_sha256": sha256(SOURCE_ARCHIVE),
        "mimic_release": "MIMIC-III 1.4",
        "release_sha256": release,
        "files": {
            "cohort": cohort_path.name,
            "author_task_root": str(task_root.relative_to(cache_root)),
        },
        "samples": len(cohort),
        "subjects": int(cohort["subject_id"].nunique()),
        "positives": int(cohort["label"].sum()),
        "source_name_sha256": ordered_text_hash(cohort["source_name"]),
        "sample_id_sha256": ordered_text_hash(cohort["sample_id"]),
        "label_sha256": ordered_text_hash(cohort["label"].astype(int)),
        "ordering_convention": "lexicographic source names -> released seed 49297 shuffle -> train/validation filter -> sorted test",
        "source_files": {
            "smart_notebook": sha256(PAPER_ROOT / "data" / "MIMIC-III" / "preprocess_mimiciii.ipynb"),
            "smart_reader": sha256(PAPER_ROOT / "data" / "MIMIC-III" / "reader.py"),
            "smart_loader": sha256(PAPER_ROOT / "data" / "mimiciii.py"),
            "smart_collate": sha256(PAPER_ROOT / "data" / "dataloader.py"),
            "ihm_inventory": sha256(COHORT_EVIDENCE / "ihm.csv"),
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
    parser.add_argument("--compatible-cache", type=Path)
    args = parser.parse_args()
    report = prepare(args.mimic_root.resolve(), args.cache_root.resolve(), args.compatible_cache)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
