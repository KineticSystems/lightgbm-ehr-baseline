#!/usr/bin/env python3
"""Directly round-trip raw MIMIC-III codes into GraphCare's exported cells."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
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
MAX_HISTORY = 28


def read_crosswalk(
    path: Path,
    source_column: str,
    target_column: str,
    selected_codes: set[str],
    convert_target: Callable[[str], str],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = row[source_column]
            if source in selected_codes:
                result[source].add(convert_target(row[target_column]))
    return dict(result)


def standardize_icd9cm(code: str) -> str:
    code = code.strip().upper()
    if "." in code:
        return code
    boundary = 4 if code.startswith("E") else 3
    return code if len(code) <= boundary else code[:boundary] + "." + code[boundary:]


def standardize_icd9proc(code: str) -> str:
    code = code.strip().upper()
    if "." in code or len(code) <= 2:
        return code
    return code[:2] + "." + code[2:]


def read_raw_visit_codes(
    path: Path,
    selected_visits: set[int],
    code_column: str,
    standardize: Callable[[str], str],
) -> tuple[dict[int, set[str]], Counter[str], int]:
    by_visit: dict[int, set[str]] = defaultdict(set)
    occurrences: Counter[str] = Counter()
    selected_rows = 0
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["HADM_ID"] or not row[code_column]:
                continue
            visit_id = int(row["HADM_ID"])
            if visit_id not in selected_visits:
                continue
            code = standardize(row[code_column])
            selected_rows += 1
            occurrences[code] += 1
            by_visit[visit_id].add(code)
    return dict(by_visit), occurrences, selected_rows


def reconstruct_cells(task_dir: Path) -> tuple[list[tuple[int, int, int]], set[int]]:
    index = pq.read_table(
        task_dir / "audit/task_index.parquet",
        columns=["subject_id", "sample_id", "current_visit_id", "history_length"],
    ).to_pydict()
    by_subject: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for subject_id, sample_id, current_visit_id, history_length in zip(
        index["subject_id"],
        index["sample_id"],
        index["current_visit_id"],
        index["history_length"],
        strict=True,
    ):
        if int(sample_id) != int(current_visit_id):
            raise AssertionError("sample_id differs from current_visit_id")
        by_subject[int(subject_id)].append((int(history_length), int(sample_id)))

    cells = []
    selected_visits = set()
    for subject_id, rows in by_subject.items():
        rows.sort()
        if [length for length, _ in rows] != list(range(1, len(rows) + 1)):
            raise AssertionError(f"non-contiguous history for subject {subject_id}")
        eligible_visits = [visit_id for _, visit_id in rows]
        selected_visits.update(eligible_visits)
        for history_length, sample_id in rows:
            history = eligible_visits[:history_length]
            first_sequence_row = MAX_HISTORY - history_length
            for offset, visit_id in enumerate(history):
                cells.append((sample_id, first_sequence_row + offset, visit_id))
    if len(selected_visits) != 9717:
        raise AssertionError("selected raw input visits differ from GraphCare")
    return cells, selected_visits


def exported_cells(
    task_dir: Path,
) -> dict[tuple[int, int, str], set[str]]:
    result: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    parquet = pq.ParquetFile(task_dir / "data/events.parquet")
    for batch in parquet.iter_batches(
        batch_size=65_536, columns=["sample_id", "sequence_row", "code"]
    ):
        values = batch.to_pydict()
        for sample_id, sequence_row, namespaced_code in zip(
            values["sample_id"],
            values["sequence_row"],
            values["code"],
            strict=True,
        ):
            namespace, code = str(namespaced_code).split("//", 1)
            result[(int(sample_id), int(sequence_row), namespace)].add(code)
    return dict(result)


def reverse_ambiguity(mapping: dict[str, set[str]]) -> dict:
    sources_by_target: dict[str, set[str]] = defaultdict(set)
    for source, targets in mapping.items():
        for target in targets:
            sources_by_target[target].add(source)
    counts = [len(sources) for sources in sources_by_target.values()]
    return {
        "mapped_targets": len(sources_by_target),
        "targets_with_multiple_raw_sources": sum(count > 1 for count in counts),
        "fraction_targets_with_multiple_raw_sources": (
            sum(count > 1 for count in counts) / len(counts)
        ),
        "median_raw_sources_per_target": statistics.median(counts),
        "maximum_raw_sources_per_target": max(counts),
        "most_ambiguous_targets": [
            {
                "target": target,
                "raw_source_count": len(sources),
                "raw_sources": sorted(sources)[:15],
            }
            for target, sources in sorted(
                sources_by_target.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )[:10]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--resource-root", type=Path, default=DEFAULT_RESOURCE_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "paper_results/graphcare/raw_roundtrip_audit.json",
    )
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args()
    if args.max_examples < 1:
        parser.error("max examples must be positive")
    mimic_root = args.mimic_root.resolve()
    task_dir = args.task_dir.resolve()
    resource_root = args.resource_root.resolve()

    cells, selected_visits = reconstruct_cells(task_dir)
    expected = exported_cells(task_dir)
    specifications = {
        "CONDITION": {
            "raw_path": mimic_root / "DIAGNOSES_ICD.csv.gz",
            "code_column": "ICD9_CODE",
            "standardize": standardize_icd9cm,
            "crosswalk_path": resource_root / "ICD9CM_to_CCSCM.csv",
            "crosswalk_source": "ICD9CM",
            "crosswalk_target": "CCSCM",
            "convert_target": lambda value: value,
        },
        "PROCEDURE": {
            "raw_path": mimic_root / "PROCEDURES_ICD.csv.gz",
            "code_column": "ICD9_CODE",
            "standardize": standardize_icd9proc,
            "crosswalk_path": resource_root / "ICD9PROC_to_CCSPROC.csv",
            "crosswalk_source": "ICD9PROC",
            "crosswalk_target": "CCSPROC",
            "convert_target": lambda value: value,
        },
        "DRUG": {
            "raw_path": mimic_root / "PRESCRIPTIONS.csv.gz",
            "code_column": "NDC",
            "standardize": lambda value: value.strip(),
            "crosswalk_path": resource_root / "NDC_to_ATC.csv",
            "crosswalk_source": "NDC",
            "crosswalk_target": "ATC",
            "convert_target": lambda value: value[:4],
        },
    }

    mapped_by_namespace = {}
    namespace_reports = {}
    for namespace, specification in specifications.items():
        raw_by_visit, occurrences, selected_rows = read_raw_visit_codes(
            specification["raw_path"],
            selected_visits,
            specification["code_column"],
            specification["standardize"],
        )
        mapping = read_crosswalk(
            specification["crosswalk_path"],
            specification["crosswalk_source"],
            specification["crosswalk_target"],
            set(occurrences),
            specification["convert_target"],
        )
        mapped_by_namespace[namespace] = {
            visit_id: {
                target
                for source in sources
                for target in mapping.get(source, ())
            }
            for visit_id, sources in raw_by_visit.items()
        }
        unmapped_occurrences = sum(
            count for code, count in occurrences.items() if code not in mapping
        )
        namespace_reports[namespace] = {
            "raw_selected_occurrences": selected_rows,
            "raw_distinct_codes": len(occurrences),
            "unmapped_occurrences": unmapped_occurrences,
            "unmapped_fraction": unmapped_occurrences / selected_rows,
            "reverse_mapping_ambiguity": reverse_ambiguity(mapping),
        }

    examples = []
    total_missing = 0
    total_unexpected = 0
    total_mismatched_cells = 0
    for namespace in specifications:
        report = namespace_reports[namespace]
        report.update(
            {
                "checked_sample_visit_cells": len(cells),
                "expected_exported_events": 0,
                "roundtrip_mapped_events": 0,
                "mismatched_cells": 0,
                "missing_exported_events": 0,
                "unexpected_roundtrip_events": 0,
            }
        )
        for sample_id, sequence_row, visit_id in cells:
            expected_codes = expected.get((sample_id, sequence_row, namespace), set())
            actual_codes = mapped_by_namespace[namespace].get(visit_id, set())
            missing = expected_codes - actual_codes
            unexpected = actual_codes - expected_codes
            report["expected_exported_events"] += len(expected_codes)
            report["roundtrip_mapped_events"] += len(actual_codes)
            if missing or unexpected:
                report["mismatched_cells"] += 1
                report["missing_exported_events"] += len(missing)
                report["unexpected_roundtrip_events"] += len(unexpected)
                if len(examples) < args.max_examples:
                    examples.append(
                        {
                            "namespace": namespace,
                            "sample_id": sample_id,
                            "sequence_row": sequence_row,
                            "raw_visit_id": visit_id,
                            "missing_from_roundtrip": sorted(missing),
                            "unexpected_from_roundtrip": sorted(unexpected),
                        }
                    )
        total_mismatched_cells += report["mismatched_cells"]
        total_missing += report["missing_exported_events"]
        total_unexpected += report["unexpected_roundtrip_events"]

    reconstructed_keys = {
        (sample_id, sequence_row, namespace)
        for sample_id, sequence_row, _ in cells
        for namespace in specifications
    }
    orphan_export_keys = sorted(set(expected) - reconstructed_keys)
    result = {
        "schema_version": 1,
        "paper": "graphcare",
        "task": "MIMIC-III subsequent-visit mortality",
        "method": (
            "raw gzip rows -> pinned crosswalk CSV -> per-admission code set -> "
            "right-aligned GraphCare sample cell"
        ),
        "selected_input_visits": len(selected_visits),
        "sample_visit_cells": len(cells),
        "checked_namespace_cells": len(cells) * len(specifications),
        "namespaces": namespace_reports,
        "total_mismatched_cells": total_mismatched_cells,
        "total_missing_exported_events": total_missing,
        "total_unexpected_roundtrip_events": total_unexpected,
        "orphan_export_cell_count": len(orphan_export_keys),
        "orphan_export_cell_examples": [list(key) for key in orphan_export_keys[:50]],
        "mismatch_examples": examples,
        "status": (
            "exact_roundtrip"
            if not total_mismatched_cells and not orphan_export_keys
            else "roundtrip_discrepancy"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_input_visits": result["selected_input_visits"],
                "sample_visit_cells": result["sample_visit_cells"],
                "checked_namespace_cells": result["checked_namespace_cells"],
                "total_mismatched_cells": result["total_mismatched_cells"],
                "total_missing_exported_events": result[
                    "total_missing_exported_events"
                ],
                "total_unexpected_roundtrip_events": result[
                    "total_unexpected_roundtrip_events"
                ],
                "orphan_export_cell_count": result["orphan_export_cell_count"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    if result["status"] != "exact_roundtrip":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
