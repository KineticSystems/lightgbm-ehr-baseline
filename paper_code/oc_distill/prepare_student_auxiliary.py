"""Prepare the exact note-availability inputs used by OC-Distill's student.

The released extractor loads all NOTEEVENTS at once. This streaming equivalent
preserves its two global de-duplication passes while retaining only metadata;
the student loader uses note text solely to decide cohort membership.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd


CATEGORIES = {"Nursing/other", "Nursing", "Radiology"}


def source_path(root: Path, stem: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        path = root / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing {stem}.csv[.gz] below {root}")


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def extract_notes(path: Path, output: Path):
    seen_text = set()
    retained_by_charttime = {}
    eligible = 0
    duplicate_text = 0
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        required = {"SUBJECT_ID", "HADM_ID", "CHARTTIME", "CATEGORY", "TEXT"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("NOTEEVENTS is missing a released extractor column")
        for row_number, row in enumerate(reader):
            if not row["CHARTTIME"] or not row["HADM_ID"]:
                continue
            if row["CATEGORY"] not in CATEGORIES:
                continue
            eligible += 1
            text = row["TEXT"] or ""
            text_digest = hashlib.sha256(text.encode("utf-8")).digest()
            if text_digest in seen_text:
                duplicate_text += 1
                continue
            seen_text.add(text_digest)
            retained_by_charttime[row["CHARTTIME"]] = (
                row_number,
                int(row["SUBJECT_ID"]),
                int(float(row["HADM_ID"])),
                row["CHARTTIME"],
                row["CATEGORY"],
                text_digest.hex(),
            )
    retained = sorted(retained_by_charttime.values())
    frame = pd.DataFrame(
        retained,
        columns=[
            "source_row", "SUBJECT_ID", "HADM_ID", "CHARTTIME", "CATEGORY", "TEXT"
        ],
    ).drop(columns="source_row")
    frame.to_csv(output, index=False)
    return {
        "eligible_rows": eligible,
        "duplicate_text_rows_removed": duplicate_text,
        "duplicate_charttime_rows_removed": eligible - duplicate_text - len(frame),
        "retained_rows": len(frame),
        "text_equality": "SHA-256 digest (no text leaves the temporary staging directory)",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    icustays_path = source_path(args.mimic_root, "ICUSTAYS")
    notes_path = source_path(args.mimic_root, "NOTEEVENTS")
    demographics = pd.read_csv(
        icustays_path,
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"],
    )
    demographics.to_csv(args.output_dir / "demographic_table.csv", index=False)
    note_summary = extract_notes(notes_path, args.output_dir / "notes_table.csv")
    report = {
        "demographic_rows": len(demographics),
        "icustays_sha256": sha256(icustays_path),
        "noteevents_sha256": sha256(notes_path),
        "released_bug_preserved": (
            "drop duplicate TEXT keep first, then duplicate CHARTTIME globally keep last"
        ),
        "notes": note_summary,
    }
    (args.output_dir / "preparation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
