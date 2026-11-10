"""Build the pinned M3B and TLS source artifacts for the export.

The credentialed outputs live in disk-backed scratch outside the repository.
This wrapper changes only transport: PhysioNet distributes the exact v1.4 CSV
payload gzip-compressed, whereas the pinned M3B reader names plain CSV files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tables


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
ARCHIVE = (
    PAPER_ROOT
    / ".export_assets"
    / "external_artifacts"
    / "source_archives"
    / "mimic3-benchmarks-ea0314c7cbd369f62e2237ace6f683740f867e3a.tar.gz"
)
M3B_COMMIT = "ea0314c7cbd369f62e2237ace6f683740f867e3a"
M3B_REPOSITORY = "https://github.com/YerevaNN/mimic3-benchmarks.git"
TLS_COMMIT = "0582563440eee1b7998b68165a96cefe92d6077d"
CLINICAL_EMBEDDINGS_COMMIT = "e66cbdde0aff7dde7197d4520ec9df29b3edd58a"
ARCHIVE_SHA256 = "8215e4df1b1adcb7c885b0b92d5adf697968fa99a8c7413990986d8a5dfa500f"
MIMIC3CSV_SHA256 = "b09ca596162f97b0477ef6fa4a2c65dd365bf566bd1b25f0db0d5b2c4006e19c"
HORIZONS = (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)
RAW_FILES = (
    "ADMISSIONS.csv.gz",
    "PATIENTS.csv.gz",
    "ICUSTAYS.csv.gz",
    "D_ICD_DIAGNOSES.csv.gz",
    "DIAGNOSES_ICD.csv.gz",
    "CHARTEVENTS.csv.gz",
    "LABEVENTS.csv.gz",
    "OUTPUTEVENTS.csv.gz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _outside_repository(path: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError:
        return
    raise ValueError("credentialed source scratch must be outside the repository")


def _apply_gzip_transport_patch(producer: Path) -> str:
    path = producer / "mimic3benchmark" / "mimic3csv.py"
    original = path.read_text()
    original_hash = hashlib.sha256(original.encode()).hexdigest()
    if original_hash != MIMIC3CSV_SHA256:
        if "def _source_csv(mimic3_path, filename):" in original:
            return original_hash
        raise RuntimeError("pinned mimic3csv.py does not match its audited hash")

    replacements = (
        ("import csv\n", "import csv\nimport gzip\n"),
        (
            "from mimic3benchmark.util import dataframe_from_csv\n\n\n",
            "from mimic3benchmark.util import dataframe_from_csv\n\n\n"
            "def _source_csv(mimic3_path, filename):\n"
            '    """Resolve the byte-identical gzip transport used by PhysioNet v1.4."""\n'
            "    plain = os.path.join(mimic3_path, filename)\n"
            "    if os.path.isfile(plain):\n"
            "        return plain\n"
            "    compressed = plain + '.gz'\n"
            "    if os.path.isfile(compressed):\n"
            "        return compressed\n"
            "    raise FileNotFoundError(plain)\n\n\n",
        ),
        (
            "dataframe_from_csv(os.path.join(mimic3_path, 'PATIENTS.csv'))",
            "dataframe_from_csv(_source_csv(mimic3_path, 'PATIENTS.csv'))",
        ),
        (
            "dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'))",
            "dataframe_from_csv(_source_csv(mimic3_path, 'ADMISSIONS.csv'))",
        ),
        (
            "dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'))",
            "dataframe_from_csv(_source_csv(mimic3_path, 'ICUSTAYS.csv'))",
        ),
        (
            "dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_DIAGNOSES.csv'))",
            "dataframe_from_csv(_source_csv(mimic3_path, 'D_ICD_DIAGNOSES.csv'))",
        ),
        (
            "dataframe_from_csv(os.path.join(mimic3_path, 'DIAGNOSES_ICD.csv'))",
            "dataframe_from_csv(_source_csv(mimic3_path, 'DIAGNOSES_ICD.csv'))",
        ),
        (
            "    reader = csv.DictReader(open(os.path.join(mimic3_path, table.upper() + '.csv'), 'r'))\n"
            "    for i, row in enumerate(reader):\n"
            "        if 'ICUSTAY_ID' not in row:\n"
            "            row['ICUSTAY_ID'] = ''\n"
            "        yield row, i, nb_rows[table.lower()]\n",
            "    source = _source_csv(mimic3_path, table.upper() + '.csv')\n"
            "    opener = gzip.open if source.endswith('.gz') else open\n"
            "    with opener(source, 'rt', newline='') as handle:\n"
            "        reader = csv.DictReader(handle)\n"
            "        for i, row in enumerate(reader):\n"
            "            if 'ICUSTAY_ID' not in row:\n"
            "                row['ICUSTAY_ID'] = ''\n"
            "            yield row, i, nb_rows[table.lower()]\n",
        ),
    )
    patched = original
    for old, new in replacements:
        if patched.count(old) != 1:
            raise RuntimeError("gzip transport patch no longer applies uniquely")
        patched = patched.replace(old, new)
    path.write_text(patched)
    return sha256_file(path)


def _clone_producer(producer: Path) -> None:
    """Fetch the pinned MIMIC-III benchmark commit from its public repository.

    The archive under .export_assets is a local copy; it is not redistributed
    with the release because that directory holds credentialed derivatives.
    The commit is the integrity guarantee here, not the tarball bytes: GitHub
    does not promise byte-stable generated archives, so this checks out the
    pinned revision and verifies HEAD rather than hashing a download.
    """
    producer.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--quiet", str(producer)], check=True)
    subprocess.run(
        ["git", "-C", str(producer), "fetch", "--quiet", "--depth", "1",
         M3B_REPOSITORY, M3B_COMMIT],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(producer), "checkout", "--quiet", "FETCH_HEAD"],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(producer), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != M3B_COMMIT:
        raise RuntimeError(f"expected M3B commit {M3B_COMMIT}, checked out {head}")


def ensure_producer(source_root: Path) -> tuple[Path, str]:
    producer = source_root / "mimic3-benchmarks"
    if not producer.exists():
        if ARCHIVE.is_file():
            if sha256_file(ARCHIVE) != ARCHIVE_SHA256:
                raise RuntimeError("pinned M3B archive hash changed")
            producer.mkdir(parents=True)
            subprocess.run(
                ["tar", "-xzf", str(ARCHIVE), "--strip-components=1",
                 "-C", str(producer)],
                check=True,
            )
        else:
            _clone_producer(producer)
    return producer, _apply_gzip_transport_patch(producer)


def run_step(markers: Path, name: str, command: list[str], cwd: Path) -> None:
    complete = markers / f"{name}.json"
    running = markers / f"{name}.running.json"
    if complete.is_file():
        return
    if running.exists():
        raise RuntimeError(
            f"{name} was interrupted; inspect credentialed scratch before retrying"
        )
    write_json(
        running,
        {
            "command": command,
            "cwd": str(cwd),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    environment = os.environ.copy()
    subprocess.run(command, cwd=cwd, env=environment, check=True)
    write_json(
        complete,
        {
            "command": command,
            "cwd": str(cwd),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    running.unlink()


def _decode(array) -> list[str]:
    return [value.decode("utf-8") for value in array]


def summarize_hdf(path: Path) -> dict[str, object]:
    with tables.open_file(path, "r") as handle:
        root = handle.root
        columns = _decode(root.data.columns[:])
        tasks = _decode(root.labels.tasks[:])
        task_index = tasks.index("decomp_24Hours")
        split_summary = {}
        for split in ("train", "val", "test"):
            labels = root.labels._f_get_child(split)[:, task_index]
            selected = labels[np.isfinite(labels)]
            windows = root.patient_windows._f_get_child(split)[:]
            split_summary[split] = {
                "stays": int(len(windows)),
                "landmarks": int(len(selected)),
                "positive": int(np.sum(selected == 1)),
                "negative": int(np.sum(selected == 0)),
                "rows": int(root.data._f_get_child(split).shape[0]),
            }
        return {
            "sha256": sha256_file(path),
            "columns": columns,
            "column_hash": hashlib.sha256("\n".join(columns).encode()).hexdigest(),
            "tasks": tasks,
            "task_hash": hashlib.sha256("\n".join(tasks).encode()).hexdigest(),
            "splits": split_summary,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args()
    mimic_root = args.mimic_root.resolve()
    source_root = args.source_root.resolve()
    _outside_repository(source_root)
    for filename in RAW_FILES:
        if not (mimic_root / filename).is_file():
            raise FileNotFoundError(mimic_root / filename)
    source_root.mkdir(parents=True, exist_ok=True)
    markers = source_root / "markers"
    markers.mkdir(exist_ok=True)
    producer, patched_hash = ensure_producer(source_root)

    m3b_data = source_root / "m3b_data"
    root_data = m3b_data / "root"
    multitask = m3b_data / "multitask"
    tls_hdf = source_root / "tls_hdf"
    root_data.mkdir(parents=True, exist_ok=True)
    tls_hdf.mkdir(parents=True, exist_ok=True)
    itemids = producer / "mimic3benchmark" / "resources" / "itemid_to_variable_map.csv"
    # Do not resolve the venv launcher symlink: doing so bypasses its site-packages.
    python = sys.executable
    run_step(
        markers,
        "01_extract_subjects",
        [
            python,
            "-m",
            "mimic3benchmark.scripts.extract_subjects",
            str(mimic_root),
            str(root_data),
            "--itemids_file",
            str(itemids),
        ],
        producer,
    )
    run_step(
        markers,
        "02_validate_events",
        [python, "-m", "mimic3benchmark.scripts.validate_events", str(root_data)],
        producer,
    )
    run_step(
        markers,
        "03_extract_episodes",
        [
            python,
            "-m",
            "mimic3benchmark.scripts.extract_episodes_from_subjects",
            str(root_data),
        ],
        producer,
    )
    run_step(
        markers,
        "04_split_train_test",
        [
            python,
            "-m",
            "mimic3benchmark.scripts.split_train_and_test",
            str(root_data),
        ],
        producer,
    )
    run_step(
        markers,
        "05_create_multitask",
        [
            python,
            "-m",
            "mimic3benchmark.scripts.create_multitask",
            str(root_data),
            str(multitask),
        ],
        producer,
    )
    run_step(
        markers,
        "06_split_train_validation",
        [python, "-m", "mimic3models.split_train_val", str(multitask)],
        producer,
    )
    # The outer repository also has tls/preprocessing/resources; launch from
    # the pinned TLS package root so it cannot shadow tls.preprocessing.
    run_step(
        markers,
        "07_tls_hdf",
        [
            python,
            "-m",
            "tls.run",
            "preprocess",
            "--dataset",
            "mimic3",
            "--mimic3-data-root",
            str(multitask),
            "--work-dir",
            str(tls_hdf),
            "--resource-path",
            str(PAPER_ROOT / "tls" / "preprocessing" / "resources"),
            "--horizons",
            *[str(value) for value in HORIZONS],
            "--mimic3-static-columns",
            "Height",
        ],
        PAPER_ROOT / "tls",
    )

    non_scaled = tls_hdf / "non_scaled.h5"
    standard_scaled = tls_hdf / "standard_scaled.h5"
    summary = {
        "status": "exact_source",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "commits": {
            "clinical_embeddings": CLINICAL_EMBEDDINGS_COMMIT,
            "tls": TLS_COMMIT,
            "mimic3_benchmarks": M3B_COMMIT,
        },
        "gzip_transport_patch": {
            "source_sha256": MIMIC3CSV_SHA256,
            "patched_sha256": patched_hash,
            "patch_file": "m3b_gzip.patch",
        },
        "raw_files": {
            filename: {
                "bytes": (mimic_root / filename).stat().st_size,
                "sha256": sha256_file(mimic_root / filename),
            }
            for filename in RAW_FILES
        },
        "source_artifacts": {
            "m3b_archive": {"sha256": ARCHIVE_SHA256},
            "m3b_statistics": {
                "sha256": sha256_file(producer / "statistics.md")
            },
            "itemid_map": {"sha256": sha256_file(itemids)},
            "train_listfile": {
                "sha256": sha256_file(multitask / "train_listfile.csv")
            },
            "validation_listfile": {
                "sha256": sha256_file(multitask / "val_listfile.csv")
            },
            "test_listfile": {
                "sha256": sha256_file(multitask / "test_listfile.csv")
            },
            "non_scaled_hdf": summarize_hdf(non_scaled),
            "standard_scaled_hdf": summarize_hdf(standard_scaled),
        },
    }
    write_json(source_root / "preparation.json", summary)
    print(json.dumps(summary["source_artifacts"]["standard_scaled_hdf"]["splits"], indent=2))


if __name__ == "__main__":
    main()
