"""Validate and load one directly exported MEDS binary task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .common_format import (
    FORMAT_NAME,
    MEDS_VERSION,
    SCHEMA_VERSION,
    SPLITS,
    TASK_ID,
    sha256_file,
)
from .reference_features import (
    SEGMENTED_BY_PROFILE,
    TIMED_PROFILES,
)
from .types import FeatureMatrix
from .rust_features import build_rust_event_representations


EVENT_COLUMNS = {
    "subject_id": pa.int64(),
    "time": pa.timestamp("us"),
    "code": pa.string(),
    "numeric_value": pa.float32(),
    "sample_id": pa.int64(),
    "source_row": pa.int64(),
    "sequence_row": pa.int32(),
    "source_time": pa.float64(),
    "text_value": pa.string(),
    "split": pa.string(),
}
# Columns a paper's export carries for its own verification, which the shared
# featurizer never reads. GraphCare's batch check streams visit_id and asserts
# that every exported event still names the encounter it belongs to, so this is
# not a vestigial field: dropping it from the contract would delete that check
# rather than tidy an unused column. Permitted, type-checked, and never
# selected by any representation.
PAPER_VERIFIED_COLUMNS = {
    "visit_id": pa.int64(),
}

EVENT_REQUIRED_COLUMNS = {
    name: EVENT_COLUMNS[name]
    for name in (
        "subject_id",
        "time",
        "code",
        "numeric_value",
        "sample_id",
        "source_row",
        "split",
    )
}
INDEX_COLUMNS = {
    "subject_id": pa.int64(),
    "sample_id": pa.int64(),
    "observation_start": pa.timestamp("us"),
    "prediction_time": pa.timestamp("us"),
    "boolean_value": pa.bool_(),
    "split": pa.string(),
}
LABEL_COLUMNS = {
    "subject_id": pa.int64(),
    "sample_id": pa.int64(),
    "prediction_time": pa.timestamp("us"),
    "boolean_value": pa.bool_(),
}
SUBJECT_SPLIT_COLUMNS = {
    "subject_id": pa.int64(),
    "split": pa.string(),
}
CODE_COLUMNS = {
    "code": pa.string(),
}
CODE_OPTIONAL_COLUMNS = {
    "description": pa.string(),
}


def _compatible_type(actual: pa.DataType, expected: pa.DataType) -> bool:
    if pa.types.is_string(expected):
        return pa.types.is_string(actual) or pa.types.is_large_string(actual)
    return actual == expected


def _validate_parquet_schema(
    path: Path,
    expected: dict[str, pa.DataType],
) -> None:
    schema = pq.read_schema(path)
    if set(schema.names) != set(expected):
        raise ValueError(
            f"{path.name} columns differ: expected {sorted(expected)}, "
            f"found {sorted(schema.names)}"
        )
    for name, dtype in expected.items():
        if not _compatible_type(schema.field(name).type, dtype):
            raise ValueError(f"{path.name} column {name} has the wrong type")


def _validate_code_metadata_schema(path: Path) -> None:
    schema = pq.read_schema(path)
    missing = set(CODE_COLUMNS) - set(schema.names)
    unknown = set(schema.names) - set(CODE_COLUMNS) - set(CODE_OPTIONAL_COLUMNS)
    if missing or unknown:
        raise ValueError(
            f"{path.name} columns have missing={sorted(missing)} "
            f"and unsupported={sorted(unknown)}"
        )
    expected = {**CODE_COLUMNS, **CODE_OPTIONAL_COLUMNS}
    for name in schema.names:
        if not _compatible_type(schema.field(name).type, expected[name]):
            raise ValueError(f"{path.name} column {name} has the wrong type")


def _coerce_required(table: pa.Table, schema: dict[str, pa.DataType]) -> pa.Table:
    missing = set(schema) - set(table.column_names)
    if missing:
        raise ValueError(f"table is missing columns: {sorted(missing)}")
    columns = []
    for name in table.column_names:
        column = table[name].combine_chunks()
        target = schema.get(name)
        if target is not None and not _compatible_type(column.type, target):
            column = pc.cast(column, target)
        columns.append(column)
    return pa.table(columns, names=table.column_names)


def _validate_samples(samples: pa.Table, task_window_hours: float) -> np.ndarray:
    """Check the index and return each sample's true window width in hours."""
    sample_id = samples["sample_id"].to_numpy()
    if len(np.unique(sample_id)) != len(sample_id):
        raise ValueError("sample_id must be unique")
    split = samples["split"].to_pylist()
    if set(split) != set(SPLITS):
        raise ValueError(f"samples must use exactly the splits {SPLITS}")
    if np.any(
        samples["prediction_time"].to_numpy()
        <= samples["observation_start"].to_numpy()
    ):
        raise ValueError("every prediction_time must follow observation_start")
    starts = (
        samples["observation_start"]
        .to_numpy()
        .astype("datetime64[us]")
        .astype(np.int64)
    )
    stops = (
        samples["prediction_time"]
        .to_numpy()
        .astype("datetime64[us]")
        .astype(np.int64)
    )
    widths = (stops - starts) / 3_600_000_000.0
    if float(np.max(widths)) > task_window_hours + 1e-9:
        raise ValueError("a sample observation window exceeds the task window")
    for name in SPLITS:
        labels = samples.filter(pc.equal(samples["split"], name))["boolean_value"]
        if len(pc.unique(labels)) != 2:
            raise ValueError(f"{name} does not contain both label classes")
    return widths


def _validate_events(
    path: Path,
    samples: pa.Table,
    codes: tuple[str, ...],
    note_codes: tuple[str, ...],
    task_window_hours: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate every event; derive the code classification."""
    schema = pq.read_schema(path)
    missing = set(EVENT_REQUIRED_COLUMNS) - set(schema.names)
    unknown = set(schema.names) - set(EVENT_COLUMNS) - set(PAPER_VERIFIED_COLUMNS)
    if missing or unknown:
        raise ValueError(
            f"events columns have missing={sorted(missing)} "
            f"and unsupported={sorted(unknown)}"
        )
    permitted = {**EVENT_COLUMNS, **PAPER_VERIFIED_COLUMNS}
    for name in schema.names:
        if not _compatible_type(schema.field(name).type, permitted[name]):
            raise ValueError(f"events column {name} has the wrong type")
    if note_codes and "text_value" not in schema.names:
        raise ValueError("a note-using task must retain raw text_value")
    index_sample = samples["sample_id"].to_numpy()
    order = np.argsort(index_sample)
    sorted_sample = index_sample[order]
    sorted_subject = samples["subject_id"].to_numpy()[order]
    sorted_start = samples["observation_start"].to_numpy()[order]
    sorted_prediction = samples["prediction_time"].to_numpy()[order]
    sorted_split_values = samples["split"].to_pylist()
    # Split has a tiny vocabulary. Encoding it once lets the batch loop compare
    # with an Arrow kernel instead of building a Python string per event.
    split_vocab = sorted(set(sorted_split_values))
    split_ordinal = {value: rank for rank, value in enumerate(split_vocab)}
    sorted_split = np.asarray(
        [split_ordinal[value] for value in sorted_split_values], dtype=np.int64
    )[order]
    split_value_set = pa.array(split_vocab, type=samples["split"].type)
    # Arrow's membership kernel hashes the vocabulary.  NumPy's object-dtype
    # `isin` instead compares every event string with every vocabulary string,
    # which becomes effectively quadratic for large categorical vocabularies.
    code_type = schema.field("code").type
    allowed = pa.array(codes, type=code_type)
    notes = pa.array(note_codes, type=code_type)
    numeric_codes: set[str] = set()
    occurrence_codes: set[str] = set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=250_000):
        table = pa.Table.from_batches([batch])
        numeric_valid = pc.is_valid(table["numeric_value"]).to_numpy()
        numeric = table["numeric_value"].to_numpy(zero_copy_only=False)
        text_valid = (
            pc.is_valid(table["text_value"]).to_numpy()
            if "text_value" in table.column_names
            else np.zeros(len(table), dtype=bool)
        )
        sample_ids = table["sample_id"].to_numpy()
        subjects = table["subject_id"].to_numpy()
        times = table["time"].to_numpy()
        splits = pc.index_in(
            table["split"], value_set=split_value_set
        ).to_numpy(zero_copy_only=False)
        positions = np.searchsorted(sorted_sample, sample_ids)
        known = positions < len(sorted_sample)
        safe_positions = np.minimum(positions, len(sorted_sample) - 1)
        known &= sorted_sample[safe_positions] == sample_ids
        if not np.all(known):
            raise ValueError("an event references an unknown sample")
        if np.any(subjects != sorted_subject[safe_positions]) or np.any(
            splits != sorted_split[safe_positions]
        ):
            raise ValueError("an event disagrees with its sample identity or split")
        if np.any(times < sorted_start[safe_positions]) or np.any(
            times > sorted_prediction[safe_positions]
        ):
            raise ValueError("an event falls outside its sample window")
        declared = pc.is_in(table["code"], value_set=allowed).to_numpy(
            zero_copy_only=False
        )
        if not np.all(declared):
            raise ValueError("an event uses an undeclared code")
        is_note = pc.is_in(table["code"], value_set=notes).to_numpy(
            zero_copy_only=False
        )
        if np.any(is_note & (numeric_valid | ~text_valid)):
            raise ValueError("note events require text and no numeric value")
        if np.any(~is_note & (numeric_valid & ~np.isfinite(numeric) | text_valid)):
            raise ValueError(
                "structured events require either a finite numeric value or an "
                "occurrence-only null, and no text"
            )
        structured = ~is_note
        for target, mask in (
            (numeric_codes, structured & numeric_valid),
            (occurrence_codes, structured & ~numeric_valid),
        ):
            if mask.any():
                target.update(
                    pc.unique(pc.filter(table["code"], pa.array(mask))).to_pylist()
                )
    mixed = numeric_codes & occurrence_codes
    if mixed:
        raise ValueError(
            "structured codes cannot mix numeric and occurrence-only rows: "
            f"{sorted(mixed)[:10]}"
        )
    observed_structured = numeric_codes | occurrence_codes
    observed_codes = tuple(code for code in codes if code in observed_structured)
    return (
        tuple(code for code in codes if code in occurrence_codes),
        observed_codes,
    )


def _validate_label_and_split_tables(
    samples: pa.Table,
    labels: pa.Table,
    subject_splits: pa.Table,
) -> None:
    label_sample_ids = labels["sample_id"].to_numpy()
    if len(np.unique(label_sample_ids)) != len(label_sample_ids):
        raise ValueError("labels sample_id must be unique")
    expected_labels = {
        int(sample_id): (int(subject_id), prediction_time, bool(label))
        for subject_id, sample_id, prediction_time, label in zip(
            samples["subject_id"].to_numpy(),
            samples["sample_id"].to_numpy(),
            samples["prediction_time"].to_numpy(),
            samples["boolean_value"].to_numpy(),
            strict=True,
        )
    }
    actual_labels = {
        int(sample_id): (int(subject_id), prediction_time, bool(label))
        for subject_id, sample_id, prediction_time, label in zip(
            labels["subject_id"].to_numpy(),
            label_sample_ids,
            labels["prediction_time"].to_numpy(),
            labels["boolean_value"].to_numpy(),
            strict=True,
        )
    }
    if actual_labels != expected_labels:
        raise ValueError("labels.parquet disagrees with audit/task_index.parquet")

    actual_memberships = list(
        zip(
            subject_splits["subject_id"].to_numpy().tolist(),
            subject_splits["split"].to_pylist(),
            strict=True,
        )
    )
    if len(actual_memberships) != len(set(actual_memberships)):
        raise ValueError("subject_splits.parquet contains duplicate memberships")
    expected_memberships = set(
        zip(
            samples["subject_id"].to_numpy().tolist(),
            samples["split"].to_pylist(),
            strict=True,
        )
    )
    if set(actual_memberships) != expected_memberships:
        raise ValueError("subject_splits.parquet disagrees with task index membership")


class MedsTask:
    """One validated MEDS task with the authors' exact sample assignments."""

    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        occurrence_codes: tuple[str, ...],
        observed_structured_codes: tuple[str, ...],
        halves_split_hours: float,
        index: pa.Table,
        codes: tuple[str, ...],
    ):
        self.root = root
        self.manifest = manifest
        self.occurrence_codes = occurrence_codes
        self._observed_structured_codes = observed_structured_codes
        self.halves_split_hours = halves_split_hours
        self._index_table = index
        self._codes = codes

    @classmethod
    def open(
        cls, root: Path, *, allow_historical_snapshot: bool = False
    ) -> "MedsTask":
        root = root.resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"MEDS task manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != FORMAT_NAME:
            raise ValueError(f"unsupported MEDS task format: {manifest.get('format')!r}")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported MEDS task schema version")
        task = manifest.get("task", {})
        task_id = str(task.get("id", ""))
        matching_directory = root.name == task_id
        historical_snapshot = (
            allow_historical_snapshot and root.name.startswith(f"{task_id}.pre_")
        )
        if (
            not (matching_directory or historical_snapshot)
            or not TASK_ID.fullmatch(task_id)
        ):
            raise ValueError("MEDS task directory and task.id must match")
        if task.get("primary_metric") != "AUROC":
            raise ValueError("MEDS task does not use AUROC")
        note_codes = task.get("note_codes")
        if (
            not isinstance(note_codes, list)
            or len(note_codes) != len(set(note_codes))
            or not all(isinstance(code, str) and code for code in note_codes)
        ):
            raise ValueError("MEDS task must declare unique note_codes")
        expected_modalities = ["structured", "notes"] if note_codes else ["structured"]
        if task.get("modalities") != expected_modalities:
            raise ValueError("MEDS task has inconsistent modalities and note_codes")
        hours = task.get("observation_window_hours")
        if not isinstance(hours, (int, float)) or not np.isfinite(hours) or hours <= 0:
            raise ValueError("MEDS task must declare positive observation_window_hours")

        records = manifest.get("meds", {})
        required = {
            "events",
            "labels",
            "subject_splits",
            "codes",
            "dataset_metadata",
            "task_index",
        }
        if set(records) != required:
            raise ValueError(f"MEDS file inventory must contain {sorted(required)}")
        paths = {}
        for name, record in records.items():
            path = root / record["file"]
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise ValueError(f"MEDS {name} file is missing or modified")
            paths[name] = path

        _validate_parquet_schema(paths["labels"], LABEL_COLUMNS)
        _validate_parquet_schema(paths["subject_splits"], SUBJECT_SPLIT_COLUMNS)
        _validate_code_metadata_schema(paths["codes"])
        dataset_metadata = json.loads(paths["dataset_metadata"].read_text())
        if dataset_metadata.get("meds_version") != MEDS_VERSION:
            raise ValueError(f"dataset metadata must declare MEDS {MEDS_VERSION}")
        index = _coerce_required(pq.read_table(paths["task_index"]), INDEX_COLUMNS)
        widths = _validate_samples(index, float(hours))
        labels = pq.read_table(paths["labels"])
        subject_splits = pq.read_table(paths["subject_splits"])
        _validate_label_and_split_tables(index, labels, subject_splits)
        codes = tuple(
            pq.read_table(paths["codes"], columns=["code"])["code"].to_pylist()
        )
        if len(codes) != len(set(codes)) or not codes:
            raise ValueError("MEDS code vocabulary must be nonempty and unique")
        if not set(note_codes).issubset(codes) or set(note_codes) == set(codes):
            raise ValueError("MEDS note_codes do not define a valid structured task")
        occurrence_codes, observed_structured_codes = _validate_events(
            paths["events"],
            index,
            codes,
            tuple(note_codes),
            float(hours),
        )
        # The `visit_halves` boundary is one cohort-wide duration before
        # prediction time, fitted on training windows only so that validation
        # and test rows never influence the featurization they are scored with.
        training = np.asarray(index["split"].to_pylist(), dtype=object) == "train"
        halves_split_hours = 0.5 * float(np.median(widths[training]))
        pa.default_memory_pool().release_unused()
        return cls(
            root,
            manifest,
            occurrence_codes,
            observed_structured_codes,
            halves_split_hours,
            index,
            codes,
        )

    def _path(self, name: str) -> Path:
        return self.root / self.manifest["meds"][name]["file"]

    @property
    def task(self) -> dict[str, Any]:
        return self.manifest["task"]

    @property
    def task_id(self) -> str:
        return str(self.task["id"])

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(self.task["modalities"])

    @property
    def note_codes(self) -> tuple[str, ...]:
        return tuple(self.task["note_codes"])

    @property
    def observation_window_hours(self) -> float:
        return float(self.task["observation_window_hours"])

    @property
    def temporal_profile(self) -> str:
        """How the task carries time; see polars_meds.TEMPORAL_PROFILES."""
        return str(self.task.get("temporal_profile", "single_visit_timed"))

    @property
    def representations(self) -> tuple[str, ...]:
        """Representations the shared runner may search for this task.

        Visit bucketing is offered only where distinct event time is exactly
        the visit key.
        """
        return ("summary", *self.segment_kinds)

    @property
    def timed(self) -> bool:
        """Whether the source method is shown real elapsed time at all."""
        return self.temporal_profile in TIMED_PROFILES

    @property
    def segment_kinds(self) -> tuple[str, ...]:
        """Segmented representations this task may search."""
        return SEGMENTED_BY_PROFILE[self.temporal_profile]

    @property
    def segment_kind(self) -> str | None:
        """First segmented representation (legacy single-kind accessor)."""
        kinds = self.segment_kinds
        return kinds[0] if kinds else None

    @property
    def published_auroc(self) -> float | None:
        value = self.task.get("published_auroc")
        return None if value is None else float(value)

    def _index(self, split: str) -> pa.Table:
        if split not in SPLITS:
            raise ValueError(split)
        return self._index_table.filter(pc.equal(self._index_table["split"], split))

    def _structured_codes(self) -> tuple[str, ...]:
        # Retain the complete released vocabulary in MEDS provenance, but do
        # not materialize columns for codes with no event in any split. Such
        # columns are constant and their numeric/occurrence type is
        # unknowable from the event stream; treating them as numeric would
        # needlessly densify large categorical vocabularies.
        return self._observed_structured_codes

    def load(self, split: str) -> dict[str, np.ndarray]:
        index = self._index(split)
        return {
            "y": index["boolean_value"].to_numpy().astype(np.int8),
            "group_id": index["subject_id"].to_numpy(),
            "sample_id": index["sample_id"].to_numpy(),
        }

    def build_representations(self, split: str):
        """Materialize every representation this task allows, in memory."""
        index = self._index(split)
        structured_codes = self._structured_codes()
        built: dict[str, FeatureMatrix] = {}
        # A profile with no segmented representation still streams one; it
        # is discarded below rather than given a second executable path.
        for kind in self.segment_kinds or ("visit_halves",):
            built.update(
                build_rust_event_representations(
                    self._path("events"),
                    index,
                    structured_codes,
                    self.observation_window_hours,
                    occurrence_codes=self.occurrence_codes,
                    ignored_codes=self.note_codes,
                    segment_kind=kind,
                    halves_split_hours=self.halves_split_hours,
                    timed=self.timed,
                )
            )
        return {name: built[name] for name in self.representations}

    def load_documents(self, split: str) -> list[str]:
        """Join raw note rows in exporter order for every indexed sample."""
        if "notes" not in self.modalities:
            raise ValueError("MEDS task does not declare raw notes")
        index = self._index(split)
        # Apply both predicates and column projection in the lazy scan before
        # collecting.  Reading a split with PyArrow and filtering note codes
        # afterwards needlessly materializes millions of structured rows and
        # makes the subsequent Arrow sort allocate in proportion to the full
        # event table rather than the small note subset.
        events = (
            pl.scan_parquet(self._path("events"))
            .filter(
                (pl.col("split") == split)
                & pl.col("code").is_in(list(self.note_codes))
            )
            .select("sample_id", "source_row", "text_value")
            .sort("sample_id", "source_row")
            .group_by("sample_id", maintain_order=True)
            .agg(pl.col("text_value").str.join("\n\n").alias("document"))
            .collect(engine="streaming")
        )
        grouped = {
            int(sample_id): document
            for sample_id, document in events.iter_rows()
        }
        documents = []
        for sample_id in index["sample_id"].to_numpy():
            documents.append(grouped.get(int(sample_id), ""))
        return documents
