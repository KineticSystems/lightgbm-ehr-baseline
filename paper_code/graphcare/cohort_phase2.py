"""Phase-2 GraphCare exact-source boundaries.

Phase 1 exported the single prespecified task (MIMIC-III subsequent-visit
mortality) through ``cohort_source.py``.  This module adds the three remaining
published binary outcomes of Table 2 without touching that frozen module:

* ``readmission_15d``                    MIMIC-III 15-day readmission
* ``mimic4_subsequent_visit_mortality``  MIMIC-IV subsequent-visit mortality
* ``mimic4_readmission_15d``             MIMIC-IV 15-day readmission

Every task runs the released PyHealth parser methods, the released GraphCare
task function in ``graphcare_/task_fn.py`` and the released
``graphcare_.splitter.split_by_patient``.  No released label, cohort, order or
split rule is recreated here.

Memory.  ``cohort_source.load_author_source`` materialises the whole PyHealth
``patients`` object graph at once; that peaks above 6 GiB for MIMIC-III and far
higher for MIMIC-IV (~23M ``Event`` objects plus a 17.8M-row prescriptions
DataFrame), which does not fit the workstation's 7 GiB per-process budget.
Phase 2 therefore runs the *same released parser methods* over disjoint patient
shards and caches the released per-visit code lists.  ``build_parse_cache``
documents why that is equivalent; ``check_cohort_phase2.py`` proves it
empirically for MIMIC-III by reproducing the sealed phase-1 task exactly.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from cohort_source import (  # noqa: F401  (shared phase-1 helpers)
    EPOCH,
    EXPECTED_VOCABULARY_SIZES,
    MODALITY_KEYS,
    PREFIX_BY_KEY,
    RESOURCE_HASHES,
    TABLES,
    TABLE_BY_KEY,
    _bootstrap_pyhealth_cache,
    _required_raw_files,
    event_time,
    hash_values,
    meds_codes,
    sample_time_bounds,
    sha256_file,
    source_vocabularies,
)


PAPER_ROOT = Path(__file__).resolve().parent

# `data_prepare.py:55-77` fixes both released dataset configurations.
MIMIC3_CODE_MAPPING = {
    "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
    "ICD9CM": "CCSCM",
    "ICD9PROC": "CCSPROC",
}
MIMIC4_CODE_MAPPING = {
    "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
    "ICD9CM": "CCSCM",
    "ICD9PROC": "CCSPROC",
    "ICD10CM": "CCSCM",
    "ICD10PROC": "CCSPROC",
}

# The MIMIC-IV mapping adds two ICD-10 crosswalks that the code audit never acquired.
MIMIC4_EXTRA_RESOURCE_PATH = (
    PAPER_ROOT / ".export_assets" / "external_artifacts" / "pyhealth_resource_mimic4"
)
MIMIC4_EXTRA_HASH_FILE = MIMIC4_EXTRA_RESOURCE_PATH / "SHA256SUMS.json"

READMISSION_TIME_WINDOW = 15
PARSE_CACHE_VERSION = 1


# Only these raw columns are read by the released parser methods.  Restricting
# the read to them is output-preserving (nothing else is referenced) and is
# what makes a 7 GiB shard budget possible for `prescriptions`.
DATASETS: Dict[str, Dict[str, Any]] = {
    "mimic3": {
        "class": "MIMIC3Dataset",
        "module": "pyhealth.datasets.mimic3",
        "release": "MIMIC-III 1.4",
        "tables": ("DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"),
        "code_mapping": MIMIC3_CODE_MAPPING,
        "subject_column": "SUBJECT_ID",
        "patients_file": "PATIENTS",
        "admissions_file": "ADMISSIONS",
        "files": (
            "PATIENTS",
            "ADMISSIONS",
            "DIAGNOSES_ICD",
            "PROCEDURES_ICD",
            "PRESCRIPTIONS",
        ),
        "read_columns": {
            "PATIENTS.csv": None,
            "ADMISSIONS.csv": None,
            "DIAGNOSES_ICD.csv": ["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"],
            "PROCEDURES_ICD.csv": ["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"],
            "PRESCRIPTIONS.csv": [
                "SUBJECT_ID",
                "HADM_ID",
                "STARTDATE",
                "ENDDATE",
                "NDC",
            ],
        },
        "shards": 8,
        "cache": ".export_assets/mimic3_1_4_parse",
    },
    "mimic4": {
        "class": "MIMIC4Dataset",
        "module": "pyhealth.datasets.mimic4",
        "release": "MIMIC-IV 2.0",
        "tables": ("diagnoses_icd", "procedures_icd", "prescriptions"),
        "code_mapping": MIMIC4_CODE_MAPPING,
        "subject_column": "subject_id",
        "patients_file": "patients",
        "admissions_file": "admissions",
        "files": (
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "prescriptions",
        ),
        "read_columns": {
            "patients.csv": None,
            "admissions.csv": None,
            "diagnoses_icd.csv": [
                "subject_id",
                "hadm_id",
                "seq_num",
                "icd_code",
                "icd_version",
            ],
            "procedures_icd.csv": [
                "subject_id",
                "hadm_id",
                "seq_num",
                "icd_code",
                "icd_version",
            ],
            "prescriptions.csv": [
                "subject_id",
                "hadm_id",
                "starttime",
                "stoptime",
                "ndc",
            ],
        },
        "shards": 16,
        "cache": ".export_assets/mimic4_2_0_parse",
    },
}
for _name, _spec in DATASETS.items():
    _spec["table_by_key"] = dict(zip(MODALITY_KEYS, _spec["tables"]))
    _spec["vocabulary_by_key"] = {
        "conditions": "CCSCM",
        "procedures": "CCSPROC",
        "drugs": "ATC",
    }


def dataset_cache_dir(paper_root: Path, dataset: str) -> Path:
    return paper_root.resolve() / DATASETS[dataset]["cache"]


def required_raw_files(dataset: str, root: Path) -> Dict[str, Path]:
    spec = DATASETS[dataset]
    result = {name: root / f"{name}.csv.gz" for name in spec["files"]}
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"the exact {spec['release']} gzip tables are missing: "
            + ", ".join(missing)
        )
    return result


# --------------------------------------------------------------------------
# Pinned PyHealth crosswalk resources
# --------------------------------------------------------------------------


def mimic4_extra_resource_hashes() -> Dict[str, str]:
    if not MIMIC4_EXTRA_HASH_FILE.is_file():
        raise RuntimeError(
            "run acquire_mimic4_resources.py first: "
            f"{MIMIC4_EXTRA_HASH_FILE} is missing"
        )
    return json.loads(MIMIC4_EXTRA_HASH_FILE.read_text())["sha256"]


def bootstrap_pyhealth_cache(paper_root: Path) -> Path:
    """Pin the nine audited resources plus the four MIMIC-IV ICD-10 resources."""
    cache_root = _bootstrap_pyhealth_cache(paper_root)
    medcode = cache_root / "medcode"
    for name, digest in sorted(mimic4_extra_resource_hashes().items()):
        source = (MIMIC4_EXTRA_RESOURCE_PATH / name).resolve()
        if not source.is_file() or sha256_file(source) != digest:
            raise RuntimeError(f"missing or modified pinned resource: {source}")
        target = medcode / name
        if target.is_symlink():
            if target.resolve() != source:
                raise RuntimeError(f"unexpected PyHealth resource link: {target}")
        elif target.exists():
            if sha256_file(target) != digest:
                raise RuntimeError(f"unexpected PyHealth resource file: {target}")
        else:
            target.symlink_to(source)
    return cache_root


# --------------------------------------------------------------------------
# Sharded execution of the released parser
# --------------------------------------------------------------------------


def _install_read_shim(dataset: str, root: Path, subjects: Optional[set], chunk: int):
    """Redirect PyHealth's `.csv` reads to the canonical `.csv.gz` payload.

    When `subjects` is given the read is streamed in chunks and filtered to
    that patient shard.  Row order inside a shard is the raw file order, so the
    released `sort_values` (a stable multi-key lexsort) and the released
    `groupby(...)` reproduce exactly the per-visit event order that the
    unsharded read produces for those patients.
    """
    import pandas as pd

    spec = DATASETS[dataset]
    subject_column = spec["subject_column"]
    read_columns = spec["read_columns"]
    original = pd.read_csv
    resolved_root = root.resolve()

    def read_csv(path, *args, **kwargs):
        candidate = Path(os.fspath(path))
        try:
            under_root = candidate.parent.resolve() == resolved_root
        except OSError:  # pragma: no cover - defensive
            under_root = False
        if not (under_root and candidate.suffix == ".csv"):
            return original(path, *args, **kwargs)
        target = candidate
        if not candidate.exists():
            gzip_path = Path(str(candidate) + ".gz")
            if not gzip_path.is_file():
                raise FileNotFoundError(candidate)
            target = gzip_path
        columns = read_columns.get(candidate.name, "unknown")
        if columns == "unknown":
            raise AssertionError(f"unexpected released read: {candidate}")
        kwargs = dict(kwargs)
        if columns is not None:
            kwargs["usecols"] = columns
            if isinstance(kwargs.get("dtype"), dict):
                kwargs["dtype"] = {
                    key: value
                    for key, value in kwargs["dtype"].items()
                    if key in columns
                }
        if subjects is None:
            return original(target, *args, **kwargs)
        kwargs.pop("low_memory", None)
        kwargs["chunksize"] = chunk
        frames = []
        for block in original(target, *args, **kwargs):
            frames.append(block[block[subject_column].isin(subjects)])
        return pd.concat(frames, ignore_index=True)

    pd.read_csv = read_csv
    return pd, original


def _memoize_strptime(dataset: str):
    """Cache PyHealth's pure `strptime` helper (17.8M prescription rows).

    Memoising a pure string->datetime function cannot change any parsed value;
    it only avoids re-running `dateutil.parser.parse` on repeated timestamps.
    """
    import importlib

    module = importlib.import_module(DATASETS[dataset]["module"])
    original = module.strptime
    cache: Dict[Any, Any] = {}

    def cached(value):
        try:
            return cache[value]
        except KeyError:
            result = original(value)
            cache[value] = result
            return result
        except TypeError:
            return original(value)

    module.strptime = cached
    return module, original


def _sharded_class(dataset: str):
    """Subclass the released dataset class, removing only its pickle cache."""
    import importlib

    spec = DATASETS[dataset]
    released = getattr(importlib.import_module(spec["module"]), spec["class"])

    class Sharded(released):
        def __init__(self, root, tables, code_mapping):
            # Exactly the attributes the released __init__ sets before parsing.
            # Its dataset-level pickle cache is skipped because one shard is
            # not a complete dataset.
            self.dataset_name = spec["class"]
            self.root = root
            self.tables = list(tables)
            self.code_mapping = code_mapping
            self.dev = False
            self.code_mapping_tools = self._load_code_mapping_tools()

        def build(self):
            patients = self.parse_tables()
            return self._convert_code_in_patient_dict(patients)

    return Sharded


def ordered_subject_ids(dataset: str, root: Path) -> List[str]:
    """The released patient order: ascending string subject id from groupby."""
    import pandas as pd

    spec = DATASETS[dataset]
    column = spec["subject_column"]
    patients = pd.read_csv(
        root / f"{spec['patients_file']}.csv.gz", dtype={column: str}, usecols=[column]
    )
    admissions = pd.read_csv(
        root / f"{spec['admissions_file']}.csv.gz",
        dtype={column: str},
        usecols=[column],
    )
    return sorted(set(patients[column]) & set(admissions[column]))


def build_parse_cache(
    paper_root: Path,
    dataset: str,
    root: Path,
    cache_dir: Path,
    shards: int,
    chunk: int = 1_000_000,
    only_shard: Optional[int] = None,
) -> None:
    """Materialise the released parse as per-shard visit records.

    Equivalence to one unsharded `MIMIC{3,4}Dataset(root, tables, code_mapping)`
    call:

    * `parse_basic_info` merges patients/admissions on the subject id and
      groups by it; every group lies entirely inside one shard.
    * each clinical parser groups by (subject, admission) and only appends
      events to visits of that same patient, so no cross-patient state exists;
      events whose admission is absent are dropped identically.
    * `_convert_code_in_patient_dict` maps codes patient by patient.
    * the released dict order is ascending string subject id (pandas `groupby`
      sorts its keys), reproduced by walking shards in order and each shard's
      patients in sorted order.
    """
    paper_root = paper_root.resolve()
    root = root.resolve()
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    required_raw_files(dataset, root)
    os.environ["PYHEALTH_CACHE_PATH"] = str(bootstrap_pyhealth_cache(paper_root))

    subjects = ordered_subject_ids(dataset, root)
    (cache_dir / "subjects.json").write_text(
        json.dumps(
            {
                "version": PARSE_CACHE_VERSION,
                "dataset": dataset,
                "release": DATASETS[dataset]["release"],
                "shards": shards,
                "subject_count": len(subjects),
                "ordered_subject_sha256": hash_values(subjects),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    size = (len(subjects) + shards - 1) // shards
    for shard in range(shards):
        if only_shard is not None and shard != only_shard:
            continue
        target = cache_dir / f"shard_{shard:03d}.pkl"
        if target.is_file():
            print(f"shard {shard}: cached", flush=True)
            continue
        block = subjects[shard * size : (shard + 1) * size]
        rows = (
            _build_one_shard(dataset, root, set(block), block, chunk) if block else []
        )
        temporary = target.with_suffix(".partial")
        temporary.write_bytes(pickle.dumps(rows, protocol=4))
        temporary.rename(target)
        print(f"shard {shard}: {len(rows)} patients written", flush=True)
        del rows
        gc.collect()


def _build_one_shard(
    dataset: str,
    root: Path,
    subject_set: set,
    ordered_subjects: Sequence[str],
    chunk: int,
) -> List[Dict[str, Any]]:
    spec = DATASETS[dataset]
    pd, original_read_csv = _install_read_shim(dataset, root, subject_set, chunk)
    module, original_strptime = _memoize_strptime(dataset)
    try:
        parser = _sharded_class(dataset)(
            root=str(root),
            tables=spec["tables"],
            code_mapping=spec["code_mapping"],
        )
        patients = parser.build()
    finally:
        pd.read_csv = original_read_csv
        module.strptime = original_strptime

    table_by_key = spec["table_by_key"]
    rows: List[Dict[str, Any]] = []
    for subject_id in ordered_subjects:
        patient = patients.get(subject_id)
        if patient is None:
            continue
        visits = []
        for visit in patient:
            visits.append(
                {
                    "visit_id": visit.visit_id,
                    "encounter_time": visit.encounter_time,
                    "discharge_time": visit.discharge_time,
                    "discharge_status": int(visit.discharge_status),
                    "conditions": visit.get_code_list(table=table_by_key["conditions"]),
                    "procedures": visit.get_code_list(table=table_by_key["procedures"]),
                    "drugs": visit.get_code_list(table=table_by_key["drugs"]),
                }
            )
        rows.append({"patient_id": patient.patient_id, "visits": visits})
    del patients
    gc.collect()
    return rows


def _cached_visit_class(dataset: str):
    """A released `Visit` that serves the cached `get_code_list` output.

    The released task functions read only `patient[i]`, `visit.visit_id`,
    `visit.encounter_time`, `visit.discharge_status` and
    `visit.get_code_list(table=...)`.  The cache already stores exactly what
    the released `Visit.get_code_list` returned during the sharded parse (an
    order-preserving de-duplicated code list per table), so serving that list
    back is the identity for every released consumer.

    Rehydrating the underlying `Event` objects instead is behaviourally
    identical but costs roughly 10.6M Python objects for MIMIC-IV, which does
    not fit the 7 GiB per-process budget.  Each call returns a fresh list, as
    the released implementation does.
    """
    from pyhealth.data import Visit

    table_by_key = DATASETS[dataset]["table_by_key"]
    key_by_table = {table: key for key, table in table_by_key.items()}

    class CachedVisit(Visit):
        __slots__ = ("_code_lists",)

        def __init__(self, code_lists, **kwargs):
            super().__init__(**kwargs)
            self._code_lists = code_lists

        def get_code_list(self, table, remove_duplicate=True):
            key = key_by_table.get(table)
            if key is None:
                return []
            codes = self._code_lists[key]
            if not remove_duplicate:
                raise NotImplementedError(
                    "the released GraphCare task functions always de-duplicate"
                )
            return list(codes)

        def get_event_list(self, table):  # pragma: no cover - not used
            raise NotImplementedError(
                "the cached parse stores released code lists, not Event objects"
            )

    return CachedVisit


def iter_cached_patients(dataset: str, cache_dir: Path, shards: int) -> Iterator[Any]:
    """Rebuild released `Patient` objects from the cached parse, in order."""
    from pyhealth.data import Patient

    visit_class = _cached_visit_class(dataset)
    for shard in range(shards):
        path = cache_dir / f"shard_{shard:03d}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"missing parse shard: {path}")
        rows = pickle.loads(path.read_bytes())
        for row in rows:
            patient_id = sys.intern(row["patient_id"])
            patient = Patient(patient_id=patient_id)
            for record in row["visits"]:
                visit_id = sys.intern(record["visit_id"])
                patient.add_visit(
                    visit_class(
                        {key: [sys.intern(code) for code in record[key]]
                         for key in MODALITY_KEYS},
                        visit_id=visit_id,
                        patient_id=patient_id,
                        encounter_time=record["encounter_time"],
                        discharge_time=record["discharge_time"],
                        discharge_status=record["discharge_status"],
                    )
                )
            yield patient
        del rows
        gc.collect()


def parse_cache_summary(dataset: str, cache_dir: Path, shards: int) -> Dict[str, Any]:
    """Recreate Table 1's all-eligible-visit statistics from the cache."""
    parsed_patients = 0
    parsed_visits = 0
    occurrences = {key: 0 for key in MODALITY_KEYS}
    unique_totals = {key: 0 for key in MODALITY_KEYS}
    eligible_visits = 0
    eligible_patients = 0
    digest = hashlib.sha256()
    for shard in range(shards):
        rows = pickle.loads((cache_dir / f"shard_{shard:03d}.pkl").read_bytes())
        for row in rows:
            parsed_patients += 1
            parsed_visits += len(row["visits"])
            counts = {key: 0 for key in MODALITY_KEYS}
            unique = {key: set() for key in MODALITY_KEYS}
            local = 0
            for record in row["visits"]:
                digest.update(record["visit_id"].encode())
                if not all(record[key] for key in MODALITY_KEYS):
                    continue
                local += 1
                for key in MODALITY_KEYS:
                    counts[key] += len(record[key])
                    unique[key].update(record[key])
            if local:
                eligible_patients += 1
                eligible_visits += local
                for key in MODALITY_KEYS:
                    occurrences[key] += counts[key]
                    unique_totals[key] += len(unique[key])
        del rows
        gc.collect()
    return {
        "parsed_patients": parsed_patients,
        "parsed_visits": parsed_visits,
        "ordered_visit_id_sha256": digest.hexdigest(),
        "patients": eligible_patients,
        "visits": eligible_visits,
        "visits_per_patient": eligible_visits / eligible_patients,
        "code_occurrences_per_patient": {
            key: occurrences[key] / eligible_patients for key in MODALITY_KEYS
        },
        "unique_codes_per_patient": {
            key: unique_totals[key] / eligible_patients for key in MODALITY_KEYS
        },
    }


# --------------------------------------------------------------------------
# Released task functions, labels and split
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase2Record:
    """One released task sample plus the identifiers the export must audit."""

    source_index: int
    split: str
    split_position: int
    sample: Dict[str, Any]
    target_visit_id: str
    target_gap_days: Optional[int]
    history_visit_ids: Tuple[str, ...]


@dataclass(frozen=True)
class Phase2Source:
    dataset: str
    base_dataset: Any
    sample_dataset: Any
    records: Tuple[Phase2Record, ...]
    max_history: int


def _patient_position(patient: Any, visit_id: str) -> int:
    for position, visit in enumerate(patient):
        if visit.visit_id == visit_id:
            return position
    raise AssertionError(f"current visit is absent from patient: {visit_id}")


def _mortality_target(base_dataset: Any, sample: Dict[str, Any]):
    patient = base_dataset.patients[sample["patient_id"]]
    position = _patient_position(patient, sample["visit_id"])
    if position + 1 >= len(patient):
        raise AssertionError("a mortality sample uses the final visit")
    target = patient[position + 1]
    expected = (
        0 if target.discharge_status not in (0, 1) else int(target.discharge_status)
    )
    if expected != int(sample["label"]):
        raise AssertionError("target-visit mortality label disagrees")
    return target.visit_id, None


def _readmission_target(base_dataset: Any, sample: Dict[str, Any]):
    patient = base_dataset.patients[sample["patient_id"]]
    position = _patient_position(patient, sample["visit_id"])
    if position + 1 >= len(patient):
        raise AssertionError("a readmission sample uses the final visit")
    current = patient[position]
    target = patient[position + 1]
    gap_days = (target.encounter_time - current.encounter_time).days
    expected = 1 if gap_days < READMISSION_TIME_WINDOW else 0
    if expected != int(sample["label"]):
        raise AssertionError("target-visit readmission label disagrees")
    return target.visit_id, int(gap_days)


TARGET_RESOLVERS = {"mortality": _mortality_target, "readmission": _readmission_target}


class CachedBase:
    """Released `patients` mapping rebuilt from the sharded parse cache."""

    def __init__(self, dataset: str, cache_dir: Path, shards: int):
        self.dataset_name = DATASETS[dataset]["class"]
        self.patients: Dict[str, Any] = {}
        for patient in iter_cached_patients(dataset, cache_dir, shards):
            self.patients[patient.patient_id] = patient

    def set_task(self, task_fn):
        """The released `BaseDataset.set_task` body, verbatim."""
        from pyhealth.datasets.sample_dataset import SampleDataset

        samples = []
        for _, patient in self.patients.items():
            samples.extend(task_fn(patient))
        return SampleDataset(
            samples, dataset_name=self.dataset_name, task_name=task_fn.__name__
        )


def load_phase2_source(
    paper_root: Path, dataset: str, cache_dir: Path, shards: int, task: str
) -> Phase2Source:
    """Run the released task function and the released patient split."""
    if task not in TARGET_RESOLVERS:
        raise ValueError(f"unsupported task: {task}")
    os.environ["PYHEALTH_CACHE_PATH"] = str(
        bootstrap_pyhealth_cache(paper_root.resolve())
    )
    from graphcare_ import task_fn as released_task_fn

    base_dataset = CachedBase(dataset, cache_dir, shards)
    fn = getattr(released_task_fn, f"{task}_prediction_{dataset}_fn")
    sample_dataset = base_dataset.set_task(fn)
    records = _split_and_index(sample_dataset, base_dataset, task)
    max_history = max(len(record.sample["conditions"]) for record in records)
    _assert_histories(dataset, base_dataset, records)
    return Phase2Source(
        dataset=dataset,
        base_dataset=base_dataset,
        sample_dataset=sample_dataset,
        records=records,
        max_history=max_history,
    )


def _split_and_index(
    sample_dataset: Any, base_dataset: Any, task: str
) -> Tuple[Phase2Record, ...]:
    from graphcare_.splitter import split_by_patient

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

    resolve = TARGET_RESOLVERS[task]
    records: List[Phase2Record] = []
    seen: Dict[str, List[str]] = {}
    for index, sample in enumerate(sample_dataset.samples):
        split, position = split_by_index[index]
        target_visit_id, gap_days = resolve(base_dataset, sample)
        # The released task function emits one sample per eligible visit in
        # order, each carrying the cumulative prefix of the earlier eligible
        # visits, so the prefix visit ids are this patient's earlier sample ids.
        history = seen.setdefault(sample["patient_id"], [])
        history.append(sample["visit_id"])
        if len(history) != len(sample["conditions"]):
            raise AssertionError("history length differs from the eligible prefix")
        records.append(
            Phase2Record(
                index,
                split,
                position,
                sample,
                target_visit_id,
                gap_days,
                tuple(history),
            )
        )
    return tuple(records)


def _assert_histories(
    dataset: str, base_dataset: Any, records: Sequence[Phase2Record]
) -> None:
    """Prove each nested history is the released eligible-visit prefix."""
    table_by_key = DATASETS[dataset]["table_by_key"]
    by_patient: Dict[str, List[Phase2Record]] = {}
    for record in records:
        by_patient.setdefault(record.sample["patient_id"], []).append(record)
    for patient_id, patient_records in by_patient.items():
        patient = base_dataset.patients[patient_id]
        expected = []
        for position in range(len(patient) - 1):
            visit = patient[position]
            code_lists = {
                key: visit.get_code_list(table=table_by_key[key])
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
            if record.history_visit_ids != tuple(
                item[0] for item in expected[: offset + 1]
            ):
                raise AssertionError("history visit ids differ from task function")
            for key in MODALITY_KEYS:
                prefix = [item[1][key] for item in expected[: offset + 1]]
                if record.sample[key] != prefix:
                    raise AssertionError(f"{key} history differs from task function")


def phase2_event_rows(
    record: Phase2Record, max_history: int
) -> Iterator[Tuple[int, int, float, str, int]]:
    """Yield (source_row, sequence_row, source_time, MEDS code, visit id)."""
    sample = record.sample
    history_length = len(sample["conditions"])
    if not all(len(sample[key]) == history_length for key in MODALITY_KEYS):
        raise AssertionError("the author modalities are not visit-aligned")
    if len(record.history_visit_ids) != history_length:
        raise AssertionError("history visit ids are not visit-aligned")
    source_row = 0
    padding = max_history - history_length
    for visit_offset in range(history_length):
        sequence_row = padding + visit_offset
        visit_id = int(record.history_visit_ids[visit_offset])
        for key in MODALITY_KEYS:
            prefix = PREFIX_BY_KEY[key]
            for raw_code in sample[key][visit_offset]:
                yield (
                    source_row,
                    sequence_row,
                    float(sequence_row),
                    prefix + raw_code,
                    visit_id,
                )
                source_row += 1


# --------------------------------------------------------------------------
# Shared summaries and audits
# --------------------------------------------------------------------------


def phase2_source_summary(source: Phase2Source, paper_root: Path) -> Dict[str, Any]:
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


def readmission_defect_report(source: Phase2Source) -> Dict[str, Any]:
    """Quantify the released admission-to-admission readmission defects."""
    gaps = [record.target_gap_days for record in source.records]
    positives = sum(int(record.sample["label"]) for record in source.records)
    return {
        "released_label": (
            "1 if (next visit encounter_time - current visit encounter_time)"
            ".days < 15, measured admission-start to admission-start on the "
            "released visit order"
        ),
        "paper_statement": "Section 3.4: y = 1 if tau(x_t) - tau(x_{t-1}) <= 15 days",
        "code_paper_mismatch": (
            "the released code uses a strict `< 15` on whole truncated days, "
            "so a 15-day gap is negative in code and positive in the paper's "
            "stated definition"
        ),
        "prediction_point_defect": (
            "the endpoint is measured from the current admission's start, not "
            "its discharge, while the predictors are the current admission's "
            "complete discharge-time code set; for stays longer than the gap "
            "some predictor codes are assigned after the target admission has "
            "already begun (paper_audits/graphcare.md item 3)"
        ),
        "visit_order_defect": (
            "PyHealth orders a patient's visits by ascending string admission "
            "id (pandas groupby key order), not by encounter_time, so the "
            "'next' visit is not necessarily the chronologically next one"
        ),
        "samples": len(gaps),
        "positive": positives,
        "prevalence": positives / len(gaps),
        "negative_gap_samples": sum(1 for gap in gaps if gap < 0),
        "zero_gap_samples": sum(1 for gap in gaps if gap == 0),
        "gap_exactly_15_days": sum(1 for gap in gaps if gap == 15),
        "minimum_gap_days": min(gaps),
        "maximum_gap_days": max(gaps),
    }


def visit_order_report(source: Phase2Source) -> Dict[str, Any]:
    """Measure how often the released visit order is not chronological."""
    non_chronological_patients = 0
    non_chronological_pairs = 0
    total_pairs = 0
    for patient in source.base_dataset.patients.values():
        times = [visit.encounter_time for visit in patient]
        local = sum(
            1
            for left, right in zip(times, times[1:])
            if left is not None and right is not None and right < left
        )
        total_pairs += max(len(times) - 1, 0)
        non_chronological_pairs += local
        if local:
            non_chronological_patients += 1
    return {
        "consecutive_visit_pairs": total_pairs,
        "non_chronological_pairs": non_chronological_pairs,
        "patients_with_non_chronological_order": non_chronological_patients,
    }


def phase2_artifact_hashes(paper_root: Path, dataset: str, root: Path) -> Dict[str, str]:
    prefix = "mimic_iii_1_4" if dataset == "mimic3" else "mimic_iv_2_0_hosp"
    result = {
        f"{prefix}/{name}.csv.gz": sha256_file(path)
        for name, path in required_raw_files(dataset, root).items()
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
    if dataset == "mimic4":
        for name, digest in sorted(mimic4_extra_resource_hashes().items()):
            result[f"pyhealth_resource_mimic4/{name}"] = digest
    for relative in (
        "UPSTREAM.json",
        "graphcare_/task_fn.py",
        "graphcare_/splitter.py",
        "data_prepare.py",
    ):
        result[f"paper_code/{relative}"] = sha256_file(paper_root / relative)
    return result
