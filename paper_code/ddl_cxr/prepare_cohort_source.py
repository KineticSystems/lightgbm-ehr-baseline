"""Run DDL-CXR's released MIMIC-IV 2.0 prediction-cohort construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from cohort_source import (
    COMMITTED_COHORT_ROOT,
    CONFIG_PATH,
    ITEMID_MAP_PATH,
    NORMALIZER_PATH,
    PHENOTYPE_PATH,
    SOURCE_SPLITS,
    committed_subject_ids,
    sha256,
)


REPOSITORY = Path(__file__).resolve().parent
EXTRACT_ROOT = REPOSITORY / "mimic4extract"
REQUIRED_MIMIC = (
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "hosp/d_icd_diagnoses.csv.gz",
    "hosp/labevents.csv.gz",
    "icu/icustays.csv.gz",
    "icu/chartevents.csv.gz",
    "icu/outputevents.csv.gz",
    "icu/procedureevents.csv.gz",
)
REQUIRED_CXR = (
    "mimic-cxr-2.0.0-metadata.csv.gz",
    "mimic-cxr-2.0.0-chexpert.csv.gz",
)


def manifest(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        result[name] = digest
    return result


def verified_inventory(
    root: Path, relative_paths: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    expected = manifest(root / "SHA256SUMS.txt")
    result = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        if digest != expected.get(relative):
            raise AssertionError(f"release-manifest mismatch: {path}")
        result[relative] = {"sha256": digest, "bytes": path.stat().st_size}
    return result


def run_step(
    work_root: Path,
    name: str,
    command: list[str],
    *,
    cwd: Path,
    expected: Path,
    pythonpath: Path | None = None,
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
    environment["PYTHONHASHSEED"] = "0"
    if pythonpath is not None:
        environment["PYTHONPATH"] = str(pythonpath)
    with log_path.open("w") as log:
        process = subprocess.run(
            command,
            cwd=cwd,
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


def cohort_summary(path: Path) -> dict[str, object]:
    import pandas as pd

    frame = pd.read_csv(path)
    return {
        "rows": len(frame),
        "subjects": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "positive": int(frame["mortality_inhospital"].sum()),
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cxr-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()

    mimic_root = args.mimic_root.resolve()
    cxr_root = args.cxr_root.resolve()
    work_root = args.work_root.resolve()
    subjects_root = work_root / "subjects"
    cohort_root = work_root / "prediction_cohort"
    work_root.mkdir(parents=True, exist_ok=True)

    raw_inventory = verified_inventory(mimic_root, REQUIRED_MIMIC)
    cxr_inventory = verified_inventory(cxr_root, REQUIRED_CXR)
    subjects_file = work_root / "selected_subject_ids.txt"
    expected_subject_text = "".join(f"{value}\n" for value in committed_subject_ids())
    if not subjects_file.is_file() or subjects_file.read_text() != expected_subject_text:
        subjects_file.write_text(expected_subject_text)

    python = sys.executable
    if not (work_root / "markers" / "extract_subjects.json").is_file():
        if subjects_root.exists():
            shutil.rmtree(subjects_root)
    run_step(
        work_root,
        "extract_subjects",
        [
            python,
            "-m",
            "mimic3benchmark.scripts.extract_subjects_iv",
            str(mimic_root),
            str(subjects_root),
            "--subjects_file",
            str(subjects_file),
        ],
        cwd=EXTRACT_ROOT,
        expected=subjects_root / "all_stays.csv",
        pythonpath=EXTRACT_ROOT,
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
        cwd=EXTRACT_ROOT,
        expected=subjects_root,
        pythonpath=EXTRACT_ROOT,
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
        cwd=EXTRACT_ROOT,
        expected=subjects_root,
        pythonpath=EXTRACT_ROOT,
    )
    if not (work_root / "markers" / "prediction_cohort.json").is_file():
        if cohort_root.exists():
            shutil.rmtree(cohort_root)
        cohort_root.mkdir(parents=True)
    run_step(
        work_root,
        "prediction_cohort",
        [
            python,
            str(REPOSITORY / "pre_processing.py"),
            "--mimic_cxr_jpg_dir",
            str(cxr_root),
            "--mimic_iv_csv_dir",
            str(mimic_root),
            "--mimic_iv_subjects_dir",
            str(subjects_root),
            "--output_csv_dir",
            str(cohort_root),
            "--prediction_only",
        ],
        cwd=REPOSITORY,
        expected=cohort_root / "test_pred.csv",
        pythonpath=REPOSITORY,
    )
    run_step(
        work_root,
        "prediction_pickles",
        [
            python,
            str(REPOSITORY / "cohort_pickle_prediction.py"),
            "--metadata-path",
            str(cohort_root),
            "--subjects-root",
            str(subjects_root),
        ],
        cwd=REPOSITORY,
        expected=cohort_root / "test_ehr_pred.pkl",
        pythonpath=REPOSITORY,
    )

    report = {
        "status": "complete",
        "mimic_release": "MIMIC-IV 2.0",
        "cxr_release": "MIMIC-CXR-JPG 2.0.0 payload",
        "mimic_manifest_sha256": sha256(mimic_root / "SHA256SUMS.txt"),
        "cxr_manifest_sha256": sha256(cxr_root / "SHA256SUMS.txt"),
        "raw_files": raw_inventory,
        "cxr_files": cxr_inventory,
        "selected_subject_ids_sha256": hashlib.sha256(
            expected_subject_text.encode("utf-8")
        ).hexdigest(),
        "source_cohort": {
            split: cohort_summary(cohort_root / f"{split}_pred.csv")
            for split in SOURCE_SPLITS
        },
        "committed_cohort": {
            split: cohort_summary(COMMITTED_COHORT_ROOT / f"{split}_pred.csv")
            for split in SOURCE_SPLITS
        },
        "source_files": {
            "discretizer": sha256(CONFIG_PATH),
            "normalizer": sha256(NORMALIZER_PATH),
            "itemid_map": sha256(ITEMID_MAP_PATH),
            "phenotype_definitions": sha256(PHENOTYPE_PATH),
            "extractor": sha256(
                EXTRACT_ROOT / "mimic3benchmark" / "mimic3csv.py"
            ),
            "cohort_builder": sha256(REPOSITORY / "pre_processing.py"),
            "ehr_pickler": sha256(REPOSITORY / "ldm" / "data" / "dynamic_data.py"),
        },
        "steps": {
            name: json.loads((work_root / "markers" / f"{name}.json").read_text())
            for name in (
                "extract_subjects",
                "validate_events",
                "extract_episodes",
                "prediction_cohort",
                "prediction_pickles",
            )
        },
    }
    (work_root / "source_build.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["source_cohort"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
