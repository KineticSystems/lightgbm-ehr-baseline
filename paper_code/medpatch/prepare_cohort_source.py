"""Run the released MedPatch MIMIC-IV 1.0 mortality construction path."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parent
EXTRACT_ROOT = REPOSITORY / "mimic4extract"
REQUIRED_RAW = (
    "core/patients.csv.gz",
    "core/admissions.csv.gz",
    "icu/icustays.csv.gz",
    "hosp/d_icd_diagnoses.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "icu/outputevents.csv.gz",
    "icu/chartevents.csv.gz",
    "hosp/labevents.csv.gz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def manifest_hashes(root: Path) -> dict[str, str]:
    result = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        result[name] = digest
    return result


def materialize_raw(source_root: Path, raw_root: Path) -> dict[str, dict[str, object]]:
    expected = manifest_hashes(source_root)
    inventory = {}
    for relative in REQUIRED_RAW:
        source = source_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        actual_hash = sha256(source)
        if actual_hash != expected.get(relative):
            raise AssertionError(f"MIMIC-IV 1.0 manifest mismatch: {relative}")
        target = raw_root / relative[:-3]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            temporary = target.with_suffix(target.suffix + ".partial")
            if temporary.exists():
                temporary.unlink()
            with gzip.open(source, "rb") as compressed, temporary.open("wb") as plain:
                shutil.copyfileobj(compressed, plain, length=8 * 1024 * 1024)
            os.replace(temporary, target)
        inventory[relative] = {
            "sha256": actual_hash,
            "compressed_bytes": source.stat().st_size,
            "decompressed_bytes": target.stat().st_size,
        }
    return inventory


def run_step(
    work_root: Path,
    name: str,
    command: list[str],
    *,
    expected: Path,
) -> None:
    marker = work_root / "markers" / f"{name}.json"
    if marker.is_file():
        if not expected.exists():
            raise RuntimeError(f"{name} marker exists but output is missing: {expected}")
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    log_path = work_root / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(EXTRACT_ROOT)
    environment["PYTHONHASHSEED"] = "0"
    with log_path.open("w") as log:
        process = subprocess.run(
            command,
            cwd=EXTRACT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    if not expected.exists():
        raise RuntimeError(f"{name} did not create expected output: {expected}")
    marker.write_text(
        json.dumps(
            {
                "command": command,
                "log": str(log_path.relative_to(work_root)),
                "log_sha256": sha256(log_path),
                "status": "complete",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def listfile_summary(path: Path) -> dict[str, object]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "rows": len(rows),
        "positive": sum(int(row["y_true"]) for row in rows),
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()

    work_root = args.work_root.resolve()
    raw_root = work_root / "raw_mimiciv_1_0"
    subjects_root = work_root / "subjects"
    task_root = work_root / "in-hospital-mortality"
    work_root.mkdir(parents=True, exist_ok=True)

    raw_inventory = materialize_raw(args.mimic_root.resolve(), raw_root)
    python = sys.executable
    if not (work_root / "markers" / "extract_subjects.json").is_file() and subjects_root.exists():
        shutil.rmtree(subjects_root)
    run_step(
        work_root,
        "extract_subjects",
        [
            python,
            "-m",
            "mimic3benchmark.scripts.extract_subjects_iv",
            str(raw_root),
            str(subjects_root),
        ],
        expected=subjects_root / "all_stays.csv",
    )
    run_step(
        work_root,
        "validate_events",
        [
            python,
            "-m",
            "mimic3benchmark.scripts.validate_events",
            str(subjects_root),
        ],
        expected=work_root / "markers" / "extract_subjects.json",
    )
    run_step(
        work_root,
        "extract_episodes",
        [
            python,
            "-m",
            "mimic3benchmark.scripts.extract_episodes_from_subjects",
            str(subjects_root),
        ],
        expected=subjects_root,
    )
    run_step(
        work_root,
        "split_train_test",
        [
            python,
            "-m",
            "mimic3benchmark.scripts.split_train_and_test",
            str(subjects_root),
        ],
        expected=subjects_root / "test",
    )
    run_step(
        work_root,
        "create_mortality",
        [
            python,
            "-m",
            "mimic3benchmark.scripts.create_in_hospital_mortality",
            str(subjects_root),
            str(task_root),
        ],
        expected=task_root / "train" / "listfile.csv",
    )
    run_step(
        work_root,
        "split_train_validation",
        [
            python,
            "-m",
            "mimic3models.split_train_val",
            str(task_root),
        ],
        expected=task_root / "val_listfile.csv",
    )

    report = {
        "status": "complete",
        "mimic_release": "MIMIC-IV 1.0",
        "raw_manifest_sha256": sha256(args.mimic_root / "SHA256SUMS.txt"),
        "raw_files": raw_inventory,
        "listfiles": {
            "train": listfile_summary(task_root / "train_listfile.csv"),
            "validation": listfile_summary(task_root / "val_listfile.csv"),
            "test": listfile_summary(task_root / "test_listfile.csv"),
        },
        "source_files": {
            "identifier_repaired_extractor": sha256(
                EXTRACT_ROOT / "mimic3benchmark" / "mimic3csv.py"
            ),
            "itemid_map": sha256(
                EXTRACT_ROOT
                / "mimic3benchmark"
                / "resources"
                / "itemid_to_variable_map.csv"
            ),
            "test_membership": sha256(
                EXTRACT_ROOT
                / "mimic3benchmark"
                / "resources"
                / "testset_iv.csv"
            ),
            "validation_membership": sha256(
                EXTRACT_ROOT / "mimic3models" / "resources" / "valset_iv.csv"
            ),
        },
        "steps": {
            name: json.loads((work_root / "markers" / f"{name}.json").read_text())
            for name in (
                "extract_subjects",
                "validate_events",
                "extract_episodes",
                "split_train_test",
                "create_mortality",
                "split_train_validation",
            )
        },
    }
    (work_root / "source_build.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["listfiles"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
