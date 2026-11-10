"""Exact GraphCare/PyHealth source boundary used by the exporter.

This module deliberately stops before GraphCare's unreleased knowledge-graph
artifacts.  It invokes the released PyHealth parser, released mortality task
function, and released patient splitter without recreating their logic.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple


TABLES = ("DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS")
MODALITY_KEYS = ("conditions", "procedures", "drugs")
TABLE_BY_KEY = dict(zip(MODALITY_KEYS, TABLES))
PREFIX_BY_KEY = {
    "conditions": "CONDITION//",
    "procedures": "PROCEDURE//",
    "drugs": "DRUG//",
}
RESOURCE_HASHES = {
    "ATC.csv": "56edce0c5225ca340dce88565535f267984bcccf69028af44a43662f53b9f7ef",
    "CCSCM.csv": "5e61accf58a949a70ce11c777b95b6e20ab802c5c8922169c735d71cd7a0c5d2",
    "CCSPROC.csv": "90811b4b744659cfebf68442a9f6c444bf27baca17821060831030f53d29413b",
    "ICD9CM.csv": "7c620301cc56faeb603da1a84cdc0fb3b6ec5cbbff939016abc317a60210ae72",
    "ICD9CM_to_CCSCM.csv": "ac32ecd2a9ca4fef904584429412caa5652a5a5c8b8d6e0c10cb8125dc8f95ad",
    "ICD9PROC.csv": "b203d2af1a10849c7831167ebe91d947ac7451c6159604359c0a7dcae189de02",
    "ICD9PROC_to_CCSPROC.csv": "545ef94157dcfd02a2af6e4335a21029bfb30b2618a918a41b709ecc85aa9457",
    "NDC.csv": "5fd76c12d95386568ffcc06b660b9a8c0ca3e6a276ca29d92805eb10a4facfa1",
    "NDC_to_ATC.csv": "82af3bd9a983581af44754ba2b2b343cde46686a0c08d5aa18a4f46c94de6899",
}
VOCABULARY_FILES = {
    "conditions": "ccscm_id2clus.json",
    "procedures": "ccsproc_id2clus.json",
    "drugs": "atc3_id2clus.json",
}
EXPECTED_VOCABULARY_SIZES = {
    "conditions": 285,
    "procedures": 231,
    "drugs": 269,
}
EPOCH = datetime(1970, 1, 1)


@dataclass(frozen=True)
class SourceRecord:
    source_index: int
    split: str
    split_position: int
    sample: Dict[str, Any]
    target_visit_id: str


@dataclass(frozen=True)
class AuthorSource:
    base_dataset: Any
    sample_dataset: Any
    records: Tuple[SourceRecord, ...]
    max_history: int


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_values(values: Iterable[Any]) -> str:
    """Hash ordered JSON values without delimiter ambiguity."""
    digest = hashlib.sha256()
    for value in values:
        encoded = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _bootstrap_pyhealth_cache(paper_root: Path) -> Path:
    resources = (
        paper_root
        / ".export_assets"
        / "external_artifacts"
        / "pyhealth_resource_current"
    )
    for name, expected in RESOURCE_HASHES.items():
        path = resources / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(
                f"missing or modified audited PyHealth resource: {path}"
            )

    cache_root = paper_root / ".export_assets" / "pyhealth_runtime"
    medcode = cache_root / "medcode"
    medcode.mkdir(parents=True, exist_ok=True)
    for name in RESOURCE_HASHES:
        target = medcode / name
        source = (resources / name).resolve()
        if target.is_symlink():
            if target.resolve() != source:
                raise RuntimeError(f"unexpected PyHealth resource link: {target}")
        elif target.exists():
            if sha256_file(target) != RESOURCE_HASHES[name]:
                raise RuntimeError(f"unexpected PyHealth resource file: {target}")
        else:
            target.symlink_to(source)
    return cache_root


def _required_raw_files(mimic_root: Path) -> Dict[str, Path]:
    names = ("PATIENTS", "ADMISSIONS", *TABLES)
    result = {name: mimic_root / f"{name}.csv.gz" for name in names}
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "the exact MIMIC-III 1.4 gzip tables are missing: " + ", ".join(missing)
        )
    return result


def _install_gzip_read_shim(mimic_root: Path):
    """Make PyHealth 1.1.2 open the canonical .csv.gz payload in place."""
    import pandas as pd

    original = pd.read_csv
    resolved_root = mimic_root.resolve()

    def read_csv(path, *args, **kwargs):
        candidate = Path(os.fspath(path))
        try:
            under_root = candidate.resolve().parent == resolved_root
        except FileNotFoundError:
            under_root = candidate.parent.resolve() == resolved_root
        gzip_path = Path(str(candidate) + ".gz")
        if under_root and candidate.suffix == ".csv" and not candidate.exists():
            if not gzip_path.is_file():
                raise FileNotFoundError(candidate)
            path = gzip_path
        return original(path, *args, **kwargs)

    pd.read_csv = read_csv
    return pd, original


def _target_visit_id(base_dataset: Any, sample: Dict[str, Any]) -> str:
    patient = base_dataset.patients[sample["patient_id"]]
    current = sample["visit_id"]
    for position, visit in enumerate(patient):
        if visit.visit_id == current:
            if position + 1 >= len(patient):
                raise AssertionError("a mortality sample uses the final visit")
            target = patient[position + 1]
            expected = (
                0
                if target.discharge_status not in (0, 1)
                else int(target.discharge_status)
            )
            if expected != int(sample["label"]):
                raise AssertionError("target-visit mortality label disagrees")
            return target.visit_id
    raise AssertionError(f"current visit is absent from patient: {current}")


def load_author_source(paper_root: Path, mimic_root: Path) -> AuthorSource:
    """Run the authors' exact released parser, task function, and split."""
    paper_root = paper_root.resolve()
    mimic_root = mimic_root.resolve()
    _required_raw_files(mimic_root)
    cache_root = _bootstrap_pyhealth_cache(paper_root)
    if "pyhealth" in globals():
        raise RuntimeError("PyHealth must not be imported before cache setup")
    os.environ["PYHEALTH_CACHE_PATH"] = str(cache_root)

    pd, original_read_csv = _install_gzip_read_shim(mimic_root)
    try:
        from pyhealth.datasets import MIMIC3Dataset

        from graphcare_.splitter import split_by_patient
        from graphcare_.task_fn import mortality_prediction_mimic3_fn

        base_dataset = MIMIC3Dataset(
            root=str(mimic_root),
            tables=list(TABLES),
            code_mapping={
                "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
                "ICD9CM": "CCSCM",
                "ICD9PROC": "CCSPROC",
            },
            dev=False,
            refresh_cache=False,
        )
    finally:
        pd.read_csv = original_read_csv

    sample_dataset = base_dataset.set_task(mortality_prediction_mimic3_fn)
    subsets = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1], train_ratio=1.0, seed=528
    )
    split_by_index: Dict[int, Tuple[str, int]] = {}
    for split, subset in zip(("train", "validation", "test"), subsets):
        for position, raw_index in enumerate(subset.indices):
            index = int(raw_index)
            if index in split_by_index:
                raise AssertionError("an author sample appears in multiple splits")
            split_by_index[index] = (split, position)
    if len(split_by_index) != len(sample_dataset):
        raise AssertionError("the author split does not cover every sample")

    records: List[SourceRecord] = []
    for index, sample in enumerate(sample_dataset.samples):
        split, position = split_by_index[index]
        target_visit_id = _target_visit_id(base_dataset, sample)
        records.append(
            SourceRecord(index, split, position, sample, target_visit_id)
        )
    max_history = max(len(record.sample["conditions"]) for record in records)
    _assert_author_histories(base_dataset, records)
    return AuthorSource(
        base_dataset=base_dataset,
        sample_dataset=sample_dataset,
        records=tuple(records),
        max_history=max_history,
    )


def _assert_author_histories(
    base_dataset: Any, records: Sequence[SourceRecord]
) -> None:
    """Prove each nested history is the released eligible-visit prefix."""
    by_patient: Dict[str, List[SourceRecord]] = {}
    for record in records:
        by_patient.setdefault(record.sample["patient_id"], []).append(record)
    for patient_id, patient_records in by_patient.items():
        patient = base_dataset.patients[patient_id]
        expected = []
        for position in range(len(patient) - 1):
            visit = patient[position]
            code_lists = {
                key: visit.get_code_list(table=TABLE_BY_KEY[key])
                for key in MODALITY_KEYS
            }
            if all(code_lists.values()):
                expected.append((visit.visit_id, code_lists, patient[position + 1]))
        if len(expected) != len(patient_records):
            raise AssertionError("task samples differ from eligible visit prefixes")
        for offset, (record, expected_visit) in enumerate(
            zip(patient_records, expected)
        ):
            current_id, _, target = expected_visit
            if record.sample["visit_id"] != current_id:
                raise AssertionError("current visit order differs from task function")
            if record.target_visit_id != target.visit_id:
                raise AssertionError("target visit order differs from task function")
            for key in MODALITY_KEYS:
                prefix = [item[1][key] for item in expected[: offset + 1]]
                if record.sample[key] != prefix:
                    raise AssertionError(f"{key} history differs from task function")


def source_vocabularies(paper_root: Path) -> Dict[str, Tuple[str, ...]]:
    root = paper_root / "clustering" / "ccscm_ccsproc_atc3"
    result = {}
    for key, filename in VOCABULARY_FILES.items():
        values = tuple(json.loads((root / filename).read_text()).keys())
        if len(values) != EXPECTED_VOCABULARY_SIZES[key]:
            raise AssertionError(f"unexpected released {key} vocabulary size")
        result[key] = values
    return result


def meds_codes(paper_root: Path) -> Tuple[str, ...]:
    vocabularies = source_vocabularies(paper_root)
    return tuple(
        PREFIX_BY_KEY[key] + code
        for key in MODALITY_KEYS
        for code in vocabularies[key]
    )


def expected_event_rows(
    record: SourceRecord, max_history: int
) -> Iterator[Tuple[int, int, float, str]]:
    """Yield (source_row, sequence_row, source_time, MEDS code)."""
    sample = record.sample
    history_length = len(sample["conditions"])
    if not all(len(sample[key]) == history_length for key in MODALITY_KEYS):
        raise AssertionError("the author modalities are not visit-aligned")
    source_row = 0
    padding = max_history - history_length
    for visit_offset in range(history_length):
        sequence_row = padding + visit_offset
        for key in MODALITY_KEYS:
            prefix = PREFIX_BY_KEY[key]
            for raw_code in sample[key][visit_offset]:
                yield source_row, sequence_row, float(sequence_row), prefix + raw_code
                source_row += 1


def sample_time_bounds(max_history: int) -> Tuple[datetime, datetime]:
    return EPOCH, EPOCH + timedelta(hours=max_history)


def event_time(sequence_row: int) -> datetime:
    return EPOCH + timedelta(hours=sequence_row)


def paper_cohort_statistics(base_dataset: Any) -> Dict[str, Any]:
    """Recreate Table 1's all-eligible-visit MIMIC-III statistics."""
    patient_rows = []
    for patient in base_dataset.patients.values():
        counts = {key: 0 for key in MODALITY_KEYS}
        visits = 0
        unique = {key: set() for key in MODALITY_KEYS}
        for visit in patient:
            code_lists = {
                key: visit.get_code_list(table=TABLE_BY_KEY[key])
                for key in MODALITY_KEYS
            }
            if not all(code_lists.values()):
                continue
            visits += 1
            for key in MODALITY_KEYS:
                counts[key] += len(code_lists[key])
                unique[key].update(code_lists[key])
        if visits:
            patient_rows.append((visits, counts, unique))
    patient_count = len(patient_rows)
    visit_count = sum(row[0] for row in patient_rows)
    occurrence_totals = {
        key: sum(row[1][key] for row in patient_rows) for key in MODALITY_KEYS
    }
    unique_totals = {
        key: sum(len(row[2][key]) for row in patient_rows)
        for key in MODALITY_KEYS
    }
    return {
        "patients": patient_count,
        "visits": visit_count,
        "visits_per_patient": visit_count / patient_count,
        "code_occurrences_per_patient": {
            key: occurrence_totals[key] / patient_count for key in MODALITY_KEYS
        },
        "code_occurrences_per_eligible_visit": {
            key: occurrence_totals[key] / visit_count for key in MODALITY_KEYS
        },
        "unique_codes_per_patient": {
            key: unique_totals[key] / patient_count for key in MODALITY_KEYS
        },
    }


def source_artifact_hashes(paper_root: Path, mimic_root: Path) -> Dict[str, str]:
    result = {
        f"mimic_iii_1_4/{name}.csv.gz": sha256_file(path)
        for name, path in _required_raw_files(mimic_root).items()
    }
    resource_root = (
        paper_root
        / ".export_assets"
        / "external_artifacts"
        / "pyhealth_resource_current"
    )
    result.update(
        {
            f"pyhealth_resource_current/{name}": sha256_file(resource_root / name)
            for name in RESOURCE_HASHES
        }
    )
    for relative in (
        "UPSTREAM.json",
        "graphcare_/task_fn.py",
        "graphcare_/splitter.py",
        "data_prepare.py",
    ):
        result[f"paper_code/{relative}"] = sha256_file(paper_root / relative)
    return result


def source_summary(source: AuthorSource, paper_root: Path) -> Dict[str, Any]:
    records = source.records
    per_split = {}
    for split in ("train", "validation", "test"):
        selected = sorted(
            (record for record in records if record.split == split),
            key=lambda record: record.split_position,
        )
        per_split[split] = {
            "samples": len(selected),
            "patients": len({record.sample["patient_id"] for record in selected}),
            "positive": sum(int(record.sample["label"]) for record in selected),
            "ordered_source_index_sha256": hash_values(
                record.source_index for record in selected
            ),
            "ordered_current_visit_id_sha256": hash_values(
                record.sample["visit_id"] for record in selected
            ),
            "ordered_target_visit_id_sha256": hash_values(
                record.target_visit_id for record in selected
            ),
            "ordered_label_sha256": hash_values(
                int(record.sample["label"]) for record in selected
            ),
        }
    vocabulary = source_vocabularies(paper_root)
    return {
        "samples": len(records),
        "patients": len({record.sample["patient_id"] for record in records}),
        "max_history": source.max_history,
        "per_split": per_split,
        "vocabulary_sizes": {key: len(values) for key, values in vocabulary.items()},
        "ordered_meds_vocabulary_sha256": hash_values(meds_codes(paper_root)),
    }
