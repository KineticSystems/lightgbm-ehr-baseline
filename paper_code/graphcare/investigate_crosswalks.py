#!/usr/bin/env python3
"""Measure exact ICD/NDC crosswalk retention on GraphCare's selected visits."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
DEFAULT_TASK_DIR = WORKSPACE_ROOT / "paper_tasks/graphcare/subsequent_visit_mortality"
DEFAULT_RESOURCE_ROOT = (
    PAPER_ROOT / ".export_assets/external_artifacts/pyhealth_resource_current"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_crosswalk(
    path: Path, source: str, target: str, selected_codes: set[str]
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row[source] in selected_codes:
                result[row[source]].append(row[target])
    return dict(result)


def standardize_icd9cm(code: str) -> str:
    if "." in code:
        return code
    boundary = 4 if code.startswith("E") else 3
    return code if len(code) <= boundary else code[:boundary] + "." + code[boundary:]


def standardize_icd9proc(code: str) -> str:
    if "." in code or len(code) <= 2:
        return code
    return code[:2] + "." + code[2:]


def current_visit_export_counts(events_path: Path) -> Counter[str]:
    result: Counter[str] = Counter()
    parquet = pq.ParquetFile(events_path)
    for batch in parquet.iter_batches(columns=["sequence_row", "code"]):
        rows = batch.to_pydict()
        for sequence_row, code in zip(rows["sequence_row"], rows["code"]):
            if int(sequence_row) == 27:
                result[str(code).split("//", 1)[0]] += 1
    return result


def audit_table(
    path: Path,
    selected_visits: set[str],
    code_column: str,
    namespace: str,
    crosswalk_path: Path,
    crosswalk_source: str,
    crosswalk_target: str,
    standardize: Callable[[str], str],
    convert_target: Callable[[str], str],
    expected_exported_occurrences: int,
) -> dict:
    raw_codes: Counter[str] = Counter()
    standardized_codes: Counter[str] = Counter()
    standardized_by_visit: dict[str, set[str]] = defaultdict(set)
    selected_rows = 0
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["HADM_ID"] not in selected_visits or not row[code_column]:
                continue
            selected_rows += 1
            raw = row[code_column]
            standardized = standardize(raw)
            raw_codes[raw] += 1
            standardized_codes[standardized] += 1
            standardized_by_visit[row["HADM_ID"]].add(standardized)

    mapping = read_crosswalk(
        crosswalk_path,
        crosswalk_source,
        crosswalk_target,
        set(standardized_codes),
    )
    mapping = {
        code: [convert_target(value) for value in targets]
        for code, targets in mapping.items()
    }
    unmapped_rows = sum(
        count for code, count in standardized_codes.items() if code not in mapping
    )
    multi_target_rows = sum(
        count
        for code, count in standardized_codes.items()
        if len(mapping.get(code, ())) > 1
    )
    mapped_occurrences_before_visit_deduplication = sum(
        count * len(mapping.get(code, ()))
        for code, count in standardized_codes.items()
    )
    targets_by_visit = {
        visit_id: {
            target
            for code in codes
            for target in mapping.get(code, ())
        }
        for visit_id, codes in standardized_by_visit.items()
    }

    target_codes = Counter(
        target for targets in targets_by_visit.values() for target in targets
    )
    mapped_occurrences = sum(target_codes.values())
    if mapped_occurrences != expected_exported_occurrences:
        raise AssertionError(
            f"{namespace} manual mapping produced {mapped_occurrences} occurrences; "
            f"the exact exported current visits contain {expected_exported_occurrences}"
        )
    return {
        "raw_selected_occurrences": selected_rows,
        "raw_distinct_codes": len(raw_codes),
        "standardized_distinct_codes": len(standardized_codes),
        "mapped_output_occurrences": mapped_occurrences,
        "mapped_occurrences_before_visit_deduplication": (
            mapped_occurrences_before_visit_deduplication
        ),
        "mapped_occurrences_removed_as_within_visit_duplicates": (
            mapped_occurrences_before_visit_deduplication - mapped_occurrences
        ),
        "mapped_distinct_codes": len(target_codes),
        "unmapped_occurrences": unmapped_rows,
        "unmapped_fraction": unmapped_rows / selected_rows,
        "multi_target_source_occurrences": multi_target_rows,
        "exported_current_visit_occurrences": expected_exported_occurrences,
        "most_frequent_unmapped_raw_codes": [
            {"code": code, "occurrences": count}
            for code, count in Counter(
                {
                    code: count
                    for code, count in standardized_codes.items()
                    if code not in mapping
                }
            ).most_common(20)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--resource-root", type=Path, default=DEFAULT_RESOURCE_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    mimic_root = args.mimic_root.resolve()
    task_dir = args.task_dir.resolve()
    resource_root = args.resource_root.resolve()
    selected_visits = {
        str(value)
        for value in pq.read_table(
            task_dir / "audit/task_index.parquet", columns=["sample_id"]
        ).column(0).to_pylist()
    }
    if len(selected_visits) != 9717:
        raise AssertionError("the selected GraphCare current-visit set is not exact")
    exported = current_visit_export_counts(task_dir / "data/events.parquet")

    specifications = {
        "conditions": {
            "raw_file": "DIAGNOSES_ICD.csv.gz",
            "code_column": "ICD9_CODE",
            "namespace": "CONDITION",
            "crosswalk": "ICD9CM_to_CCSCM.csv",
            "source": "ICD9CM",
            "target": "CCSCM",
            "standardize": standardize_icd9cm,
            "convert": lambda value: value,
        },
        "procedures": {
            "raw_file": "PROCEDURES_ICD.csv.gz",
            "code_column": "ICD9_CODE",
            "namespace": "PROCEDURE",
            "crosswalk": "ICD9PROC_to_CCSPROC.csv",
            "source": "ICD9PROC",
            "target": "CCSPROC",
            "standardize": standardize_icd9proc,
            "convert": lambda value: value,
        },
        "drugs": {
            "raw_file": "PRESCRIPTIONS.csv.gz",
            "code_column": "NDC",
            "namespace": "DRUG",
            "crosswalk": "NDC_to_ATC.csv",
            "source": "NDC",
            "target": "ATC",
            "standardize": lambda value: value,
            "convert": lambda value: value[:4],
        },
    }
    result = {
        "paper": "graphcare",
        "task": "MIMIC-III subsequent-visit mortality",
        "selected_current_visits": len(selected_visits),
        "crosswalks": {},
        "source_sha256": {},
    }
    for name, specification in specifications.items():
        raw_path = mimic_root / str(specification["raw_file"])
        crosswalk_path = resource_root / str(specification["crosswalk"])
        namespace = str(specification["namespace"])
        result["crosswalks"][name] = audit_table(
            raw_path,
            selected_visits,
            str(specification["code_column"]),
            namespace,
            crosswalk_path,
            str(specification["source"]),
            str(specification["target"]),
            specification["standardize"],
            specification["convert"],
            exported[namespace],
        )
        result["source_sha256"][raw_path.name] = sha256_file(raw_path)
        result["source_sha256"][crosswalk_path.name] = sha256_file(crosswalk_path)

    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
