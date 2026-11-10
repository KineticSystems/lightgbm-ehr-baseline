"""Derive a sibling MEDS task from an existing task's events.

Phase 2 of the study runs every binary MIMIC outcome each paper reports.
Many of those share a base task's cohort and event stream and differ only
in the label (REMed's mortality horizons, td-icu's death horizons,
EHRMamba's condition targets). This tool builds a complete, standard task
directory for such an outcome by streaming the base task's already
validated events through the shared writer with a new index, so every
downstream contract (validator, runner audit gate, featurizers) is
unchanged. Prediction times and observation windows are inherited from
the base index unless the labels frame overrides them.

The derived task carries:
- `audit/full_batch_recreation.json`: copied from the base task, with a
  `derived_from` block (events are byte-identical up to the sample
  subset, so the base reconciliation still certifies them);
- `audit/cohort_reconstruction.json`: the label derivation record
  (status `exact_source_with_derived_labels`, accepted by the runner's
  audit gate) including per-split counts and the label-frame digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .common_format import sha256_file
from .polars_meds import EVENT_SCHEMA, write_meds_task

_EVENT_COLUMNS = tuple(name for name in EVENT_SCHEMA if name != "split")


def _base_manifest(base: Path) -> dict[str, Any]:
    return json.loads((base / "manifest.json").read_text())


def _labels_digest(labels: pl.DataFrame) -> str:
    frame = labels.sort("sample_id").select("sample_id", "boolean_value", "split")
    digest = hashlib.sha256()
    for column in frame.columns:
        digest.update(column.encode())
        digest.update(frame[column].cast(pl.String).str.join("\x1f").item().encode())
    return digest.hexdigest()


def _derived_index(base: Path, labels: pl.DataFrame) -> pl.DataFrame:
    """Join new labels onto the base index (anchors inherited unless given)."""
    index = pl.read_parquet(base / "audit" / "task_index.parquet")
    required = {"sample_id", "boolean_value", "split"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels frame is missing {sorted(missing)}")
    if labels["sample_id"].n_unique() != labels.height:
        raise ValueError("labels frame sample_id must be unique")
    overrides = [
        name
        for name in ("observation_start", "prediction_time")
        if name in labels.columns
    ]
    base_columns = [
        name
        for name in index.columns
        if name not in {"boolean_value", "split", *overrides}
    ]
    derived = index.select(base_columns).join(
        labels.select(["sample_id", "boolean_value", "split", *overrides]),
        on="sample_id",
        how="inner",
        maintain_order="left",
    )
    unknown = labels.height - derived.height
    if unknown:
        raise ValueError(f"{unknown} labeled samples are not in the base task index")
    return derived


def _event_batches(base: Path, sample_ids: set[int]) -> Iterator[pl.DataFrame]:
    parquet = pq.ParquetFile(base / "data" / "events.parquet")
    columns = [name for name in _EVENT_COLUMNS if name in parquet.schema_arrow.names]
    wanted = pa.array(sorted(sample_ids), type=pa.int64())
    for batch in parquet.iter_batches(batch_size=500_000, columns=columns):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.is_in(table["sample_id"], value_set=wanted))
        if table.num_rows:
            yield pl.from_arrow(table)


def derive_task(
    base: Path,
    output: Path,
    *,
    task: dict[str, Any],
    labels: pl.DataFrame,
    derivation: dict[str, Any],
) -> dict[str, Any]:
    """Write `output` as a sibling of `base` with the given labels.

    `task` supplies id/name/published_auroc/positive_class (window and
    profile are inherited from the base unless given); `labels` supplies
    sample_id/boolean_value/split and optional anchor overrides;
    `derivation` is free-form provenance recorded in both the manifest and
    the cohort audit (e.g. the paper's label definition and source column).
    """
    base = Path(base).resolve()
    output = Path(output).resolve()
    # Check what the base must already carry before writing anything. The
    # derived task inherits the base's batch reconciliation, which the base's
    # own batch check produces, so deriving before that check has run would
    # otherwise fail only after the output directory exists -- leaving a
    # half-written task the export wrapper then refuses to overwrite.
    required = base / "audit" / "full_batch_recreation.json"
    if not required.is_file():
        raise FileNotFoundError(
            f"{required} is missing; run the base task's batch check before "
            "deriving from it, because the derived task inherits that record"
        )
    manifest = _base_manifest(base)
    base_task = manifest["task"]
    codes = tuple(pq.read_table(base / "metadata" / "codes.parquet")["code"].to_pylist())
    note_codes = tuple(base_task.get("note_codes", ()))
    index = _derived_index(base, labels)
    derived_task = {
        "observation_window_hours": base_task["observation_window_hours"],
        "temporal_profile": base_task.get("temporal_profile", "single_visit_timed"),
        "primary_metric": "AUROC",
        **task,
    }
    source = {
        **manifest.get("source", {}),
        "derived_from": {
            "task_id": base_task["id"],
            "task_dir": str(base),
            "manifest_sha256": sha256_file(base / "manifest.json"),
            "events_sha256": manifest["meds"]["events"]["sha256"],
            "labels_sha256": _labels_digest(labels),
            **derivation,
        },
    }
    written = write_meds_task(
        output,
        paper=manifest["paper"],
        task=derived_task,
        codes=codes,
        note_codes=note_codes,
        events=_event_batches(base, set(index["sample_id"].to_list())),
        samples=index,
        source=source,
        dataset_name=manifest.get("dataset", {}).get("name", base_task["id"]),
        dataset_version=manifest.get("dataset", {}).get("version", "derived"),
    )
    _write_audits(base, output, written, labels, derivation)
    return written


def _write_audits(
    base: Path,
    output: Path,
    manifest: dict[str, Any],
    labels: pl.DataFrame,
    derivation: dict[str, Any],
) -> None:
    audit = output / "audit"
    audit.mkdir(exist_ok=True)
    batch = json.loads((base / "audit" / "full_batch_recreation.json").read_text())
    batch["derived_from"] = {
        "task_dir": str(base),
        "note": (
            "events are the base task's validated events restricted to this "
            "task's samples; the base reconciliation certifies them"
        ),
    }
    (audit / "full_batch_recreation.json").write_text(
        json.dumps(batch, indent=2, sort_keys=True) + "\n"
    )
    splits = {}
    for split in ("train", "validation", "test"):
        selected = labels.filter(pl.col("split") == split)
        splits[split] = {
            "count": selected.height,
            "positive": int(selected["boolean_value"].sum()),
        }
    cohort = {
        "status": "exact_source_with_derived_labels",
        "derived_from": str(base),
        "labels_sha256": _labels_digest(labels),
        "splits": splits,
        "derivation": derivation,
        "checks": [
            "every labeled sample exists in the base task index",
            "events restricted to labeled samples through the shared writer",
            "base batch reconciliation inherited",
        ],
    }
    (audit / "cohort_reconstruction.json").write_text(
        json.dumps(cohort, indent=2, sort_keys=True) + "\n"
    )
