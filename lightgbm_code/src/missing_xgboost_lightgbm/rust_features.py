"""Materialize a split's representations through the Rust featurizer.

This is the in-memory convenience entry (tests and `MedsTask.
build_representations`): it drives the same streaming featurizer the
executors use, via its `--dump-config` mode, so there is exactly one Rust
feature-construction path. Rows come back in the caller's index order.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .reference_features import SEGMENTS_BY_REPRESENTATION
from .split_contract import _binary, featurizer_threads
from .types import FeatureMatrix

_ALL_SPLIT = "__all__"


def build_rust_event_representations(
    path: Path,
    index: pa.Table,
    codes: tuple[str, ...],
    observation_window_hours: float,
    *,
    occurrence_codes: tuple[str, ...] = (),
    ignored_codes: tuple[str, ...] = (),
    segment_kind: str = "visit_halves",
    halves_split_hours: float = 0.0,
    timed: bool = True,
) -> dict[str, FeatureMatrix]:
    """Return {"summary": ..., segment_kind: ...} for the indexed samples.

    ``ignored_codes`` (e.g. a task's note codes) are skipped by the
    featurizer; any other undeclared code in an export with a ``split``
    column is a hard error, exactly as in the executors.
    """
    if segment_kind not in SEGMENTS_BY_REPRESENTATION:
        raise ValueError(
            f"segment_kind must be one of {sorted(SEGMENTS_BY_REPRESENTATION)}"
        )
    halves_split_hours = float(halves_split_hours)
    if not np.isfinite(halves_split_hours) or halves_split_hours < 0.0:
        raise ValueError("halves_split_hours must be finite and nonnegative")
    if segment_kind == "visit_halves" and halves_split_hours <= 0.0:
        raise ValueError("visit_halves requires a positive cohort-wide split")
    sample_ids = index["sample_id"].to_numpy().astype(np.int64, copy=False)
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("index sample_id must be unique")
    if index.num_rows == 0:
        raise ValueError("index must contain at least one sample")

    # The streaming featurizer selects by split; an index without one is
    # treated as a single split covering every row.
    columns = {
        "sample_id": index["sample_id"].cast(pa.int64()),
        "observation_start": index["observation_start"].cast(pa.timestamp("us")),
        "prediction_time": index["prediction_time"].cast(pa.timestamp("us")),
    }
    if "split" in index.column_names:
        splits = set(index["split"].to_pylist())
        if len(splits) != 1:
            raise ValueError("index must hold exactly one split")
        split = str(next(iter(splits)))
        columns["split"] = index["split"].cast(pa.string())
    else:
        split = _ALL_SPLIT
        columns["split"] = pa.array([split] * index.num_rows, type=pa.string())
    index_table = pa.table(columns)

    work_root = Path(__file__).resolve().parents[2] / ".feature_work"
    work_root.mkdir(exist_ok=True)
    result: dict[str, FeatureMatrix] = {}
    with tempfile.TemporaryDirectory(
        prefix="missing-xgboost-features-", dir=work_root
    ) as temporary:
        root = Path(temporary)
        index_path = root / "index.parquet"
        pq.write_table(index_table, index_path)
        # Dump rows arrive in ascending sample_id order; restore index order.
        order = np.argsort(sample_ids, kind="stable")
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        for representation in ("summary", segment_kind):
            stream = {
                "events_path": str(Path(path).resolve()),
                "index_path": str(index_path),
                "split": split,
                "representation": "summary" if representation == "summary" else "temporal",
                "codes": list(codes),
                "occurrence_codes": list(occurrence_codes),
                "ignored_codes": list(ignored_codes),
                "observation_window_hours": float(observation_window_hours),
                "timed": bool(timed),
                "max_component_rows": 1_000_000_000,
                "max_chunk_bytes": 64 << 20,
                "record_batch_rows": 65_536,
            }
            if representation != "summary":
                stream["segment_kind"] = segment_kind
                stream["halves_split_hours"] = halves_split_hours
            output_path = root / f"{representation}.f32"
            config_path = root / f"{representation}.json"
            config_path.write_text(
                json.dumps(
                    {
                        "stream": stream,
                        "output_path": str(output_path),
                        "featurizer_threads": featurizer_threads(),
                    }
                )
            )
            completed = subprocess.run(
                [str(_binary()), "--dump-config", str(config_path)],
                capture_output=True,
                env=os.environ.copy(),
                check=False,
            )
            if completed.returncode != 0:
                # Surface the featurizer's own message before raising the
                # conventional error callers already handle.
                print(
                    completed.stderr.decode("utf-8", errors="replace").strip(),
                    file=sys.stderr,
                )
                raise subprocess.CalledProcessError(
                    completed.returncode, completed.args, completed.stdout, completed.stderr
                )
            report = json.loads(completed.stdout)
            if report["rows"] != len(sample_ids):
                raise ValueError("Rust featurizer returned the wrong sample count")
            values = np.fromfile(output_path, dtype="<f4").reshape(
                report["rows"], report["feature_count"]
            )
            if np.any(np.isinf(values)):
                raise ValueError("Rust feature output contains infinity")
            result[representation] = FeatureMatrix(
                np.ascontiguousarray(values[inverse]), tuple(report["feature_names"])
            )
    return result
