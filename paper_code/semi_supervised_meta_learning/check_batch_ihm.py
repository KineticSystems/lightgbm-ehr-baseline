#!/usr/bin/env python3
"""Exhaustively recreate SSML's released multitask ihm loader boundary.

The common task stores the earliest attributable raw structured occurrence
stream, truncated at the released 48-hour in-hospital-mortality prediction
point.  This checker streams every exported sample, compares it with the raw
source CSV, and then drives the *released* ``reader.MultitaskReader`` over the
exact released listfile lines, applies the released ``Discretizer`` and
normalizer, and asserts the resulting ``[48, 76]`` model input.  Categorical
values, observation masks, labels, and every nonnumeric field remain bit-exact;
the twelve numeric channels use an exhaustively measured integer-ULP rule
solely for the required source-float64 to MEDS-float32 round trip.

The released study never wires an in-hospital-mortality trainer (see
``released_ihm_training_path`` in the manifest), so the last released
deterministic boundary for this Table 4 column is the reader/discretizer/
normalizer model input recreated here, not a trainer batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

import preprocessing
import reader
from check_batch import (
    completed_sample_frames,
    float32_ulp_distance,
    assert_raw_prefix,
    write_json,
)
from cohort_source import (
    COMMON_SPLIT,
    CONFIG_PATH,
    EPS,
    NORMALIZER_PATH,
    SOURCE_SPLIT,
    SPLIT_MAP,
    load_raw_stay,
    load_transform_spec,
    reconstruct_model_tensor,
    sha256_file,
)
from cohort_source_ihm import (
    IHM_POSITION,
    LISTFILE_HEADER,
    TASK_ID,
    WINDOW_HOURS,
    assert_source_ready,
    listfile_path,
    stay_path,
)


# Fixed after the exhaustive measurement written by measure_precision_ihm.py
# over all 21,139 released ihm stays.  This is an integer ULP bound on the
# final float32 numeric channels, never a relative/absolute allclose rule.
FIXED_NUMERIC_ULP_BOUND = 44_149
CHUNK = 128


class AuthorMultitaskLoader:
    """Drive the released MultitaskReader/Discretizer/Normalizer per chunk."""

    def __init__(self, paths: dict[str, Path], spec: dict[str, object], scratch: Path):
        self.paths = paths
        self.spec = spec
        self.scratch = scratch
        self.discretizer = preprocessing.Discretizer(
            timestep=1.0,
            store_masks=True,
            impute_strategy="previous",
            start_time="zero",
            config_path=str(CONFIG_PATH),
        )
        empty_row = np.asarray([["0"] + [""] * 17])
        self.header = self.discretizer.transform(empty_row, end=1.0)[1]
        if self.header != ",".join(spec["model_codes"]):
            raise AssertionError(
                "released discretizer header differs from the derived 76-channel boundary"
            )
        continuous = [
            index for index, name in enumerate(self.header.split(",")) if "->" not in name
        ]
        self.normalizer = preprocessing.Normalizer(fields=continuous)
        self.normalizer.load_params(str(NORMALIZER_PATH))

    def tensors(
        self, split: str, rows: list[dict[str, object]], lines: list[str]
    ) -> list[np.ndarray]:
        listfile = self.scratch / "released_chunk_listfile.csv"
        listfile.write_text(LISTFILE_HEADER + "\n" + "".join(lines))
        author_reader = reader.MultitaskReader(
            dataset_dir=str(self.paths[SOURCE_SPLIT[split]]), listfile=str(listfile)
        )
        if author_reader.get_number_of_examples() != len(rows):
            raise AssertionError("released reader lost a chunk row")
        results = []
        for position, row in enumerate(rows):
            example = author_reader.read_example(position)
            if example["name"] != row["source_filename"]:
                raise AssertionError("released list row no longer names the exported stay")
            if list(example["ihm"]) != [
                IHM_POSITION,
                1,
                int(bool(row["boolean_value"])),
            ]:
                raise AssertionError(
                    "released reader ihm triple differs from the exported landmark/label"
                )
            if float(example["t"]) < WINDOW_HOURS - EPS:
                raise AssertionError("released reader kept a stay shorter than the window")
            direct64 = self.discretizer.transform(
                example["X"], end=float(WINDOW_HOURS)
            )[0]
            direct64 = self.normalizer.transform(direct64)
            if direct64.shape != (WINDOW_HOURS, 76):
                raise AssertionError("released reader/discretizer tensor shape changed")
            results.append(direct64.astype(np.float32))
        return results


def released_lines(
    paths: dict[str, Path], samples: pl.DataFrame
) -> dict[int, str]:
    """The exact released listfile line behind every exported sample."""

    wanted: dict[int, str] = {}
    for split in SPLIT_MAP:
        selected = samples.filter(pl.col("split") == COMMON_SPLIT[split])
        rows = dict(
            zip(selected["source_list_row"].to_list(), selected["sample_id"].to_list())
        )
        with listfile_path(paths, split).open() as handle:
            handle.readline()
            for index, line in enumerate(handle):
                sample_id = rows.get(index)
                if sample_id is not None:
                    wanted[sample_id] = line
    if len(wanted) != samples.height:
        raise AssertionError("a released listfile line is missing for an exported sample")
    return wanted


def tensor_fingerprint(tensor: np.ndarray) -> np.ndarray:
    digest = hashlib.blake2b(
        np.ascontiguousarray(tensor).tobytes(), digest_size=16
    ).digest()
    return np.array(
        [
            int.from_bytes(digest[:8], "little"),
            int.from_bytes(digest[8:], "little"),
        ],
        dtype=np.uint64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    args = parser.parse_args()

    paths = assert_source_ready(args.source_root)
    manifest_path = args.task_dir / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    measurement_path = args.task_dir / "audit" / "source_precision_measurement.json"
    if not measurement_path.is_file():
        raise FileNotFoundError(
            "run measure_precision_ihm.py before check_batch_ihm.py"
        )
    measurement = json.loads(measurement_path.read_text())
    if (
        measurement.get("status") != "measurement"
        or measurement.get("smallest_covering_ulp_bound") != FIXED_NUMERIC_ULP_BOUND
    ):
        raise AssertionError(
            "source precision measurement does not fix the declared ULP bound"
        )

    task = manifest["task"]
    if task["id"] != TASK_ID:
        raise AssertionError("wrong SSML task")
    if task["modalities"] != ["structured"] or task["note_codes"] != []:
        raise AssertionError("SSML selected task retained a non-structured modality")
    if float(task["observation_window_hours"]) != float(WINDOW_HOURS):
        raise AssertionError("task window differs from the released ihm window")

    spec = load_transform_spec()
    events_path = args.task_dir / manifest["meds"]["events"]["file"]
    index = pl.read_parquet(args.task_dir / manifest["meds"]["task_index"]["file"])
    common_codes = tuple(
        pl.read_parquet(
            args.task_dir / manifest["meds"]["codes"]["file"], columns=["code"]
        )["code"].to_list()
    )
    if common_codes != spec["codes"] or len(common_codes) != 59:
        raise AssertionError(
            "common vocabulary is not the raw 59-code occurrence vocabulary"
        )
    index_by_id = index.sort("sample_id")
    if not np.array_equal(
        index_by_id["sample_id"].to_numpy(),
        np.arange(1, index.height + 1, dtype=np.int64),
    ):
        raise AssertionError("common sample IDs are not canonical consecutive integers")

    lines = released_lines(paths, index_by_id)
    numeric_positions = np.asarray(spec["numeric_indices"], dtype=np.int64)
    numeric_set = set(numeric_positions.tolist())
    categorical_positions = np.asarray(
        [position for position in range(59) if position not in numeric_set],
        dtype=np.int64,
    )

    split_reports = {
        split: {
            "examples": 0,
            "raw_occurrence_events": 0,
            "model_rows": 0,
            "model_cells": 0,
            "minimum_raw_occurrences": None,
            "maximum_raw_occurrences": 0,
        }
        for split in ("train", "validation", "test")
    }
    code_histogram = Counter()
    numeric_ulp_distribution = Counter()
    per_split_numeric_ulp = {
        split: Counter() for split in ("train", "validation", "test")
    }
    numeric_ulp_cells = numeric_ulp_nonzero = 0
    maximum_numeric_ulp = 0
    maximum_numeric_absolute_difference = 0.0
    fingerprints = np.zeros((index.height + 1, 2), dtype=np.uint64)
    seen = np.zeros(index.height + 1, dtype=bool)
    seen_count = 0

    scratch = Path(tempfile.mkdtemp(prefix="ssml-ihm-batch-"))
    loader = AuthorMultitaskLoader(paths, spec, scratch)
    pending: list[tuple[dict[str, object], pl.DataFrame]] = []
    pending_split: str | None = None

    def flush() -> None:
        nonlocal numeric_ulp_cells, numeric_ulp_nonzero, maximum_numeric_ulp
        nonlocal maximum_numeric_absolute_difference
        if not pending:
            return
        rows = [sample for sample, _ in pending]
        author = loader.tensors(
            pending_split, rows, [lines[int(row["sample_id"])] for row in rows]
        )
        for (sample, events), direct32 in zip(pending, author):
            raw = load_raw_stay(
                stay_path(paths, pending_split, str(sample["source_filename"])),
                spec,
                WINDOW_HOURS,
            )
            count = assert_raw_prefix(events, raw, sample)
            reconstructed = reconstruct_model_tensor(events, spec, WINDOW_HOURS)
            if reconstructed.shape != direct32.shape:
                raise AssertionError("raw MEDS stream changed the author tensor shape")
            if not np.array_equal(
                reconstructed[:, categorical_positions],
                direct32[:, categorical_positions],
            ):
                raise AssertionError("raw MEDS stream changed a categorical channel")
            if not np.array_equal(reconstructed[:, 59:], direct32[:, 59:]):
                raise AssertionError("raw MEDS stream changed an observation-mask channel")
            ulp = float32_ulp_distance(
                reconstructed[:, numeric_positions], direct32[:, numeric_positions]
            )
            unique, counts = np.unique(ulp, return_counts=True)
            observed = {int(a): int(b) for a, b in zip(unique, counts)}
            numeric_ulp_distribution.update(observed)
            per_split_numeric_ulp[pending_split].update(observed)
            numeric_ulp_cells += int(ulp.size)
            numeric_ulp_nonzero += int(np.count_nonzero(ulp))
            observed_maximum = int(ulp.max(initial=0))
            if observed_maximum > FIXED_NUMERIC_ULP_BOUND:
                raise AssertionError(
                    "numeric tensor exceeds the fixed ULP bound: "
                    f"observed={observed_maximum} bound={FIXED_NUMERIC_ULP_BOUND}"
                )
            maximum_numeric_ulp = max(maximum_numeric_ulp, observed_maximum)
            maximum_numeric_absolute_difference = max(
                maximum_numeric_absolute_difference,
                float(
                    np.max(
                        np.abs(
                            reconstructed[:, numeric_positions]
                            - direct32[:, numeric_positions]
                        ),
                        initial=0.0,
                    )
                ),
            )
            report = split_reports[pending_split]
            report["examples"] += 1
            report["raw_occurrence_events"] += count
            report["model_rows"] += WINDOW_HOURS
            report["model_cells"] += WINDOW_HOURS * 76
            report["maximum_raw_occurrences"] = max(
                report["maximum_raw_occurrences"], count
            )
            report["minimum_raw_occurrences"] = (
                count
                if report["minimum_raw_occurrences"] is None
                else min(report["minimum_raw_occurrences"], count)
            )
            code_histogram.update(events["code"].to_list())
            fingerprints[int(sample["sample_id"])] = tensor_fingerprint(direct32)
        pending.clear()

    for events in completed_sample_frames(events_path):
        sample_id = int(events["sample_id"][0])
        if not 1 <= sample_id <= index.height:
            raise AssertionError("common events reference an unknown task sample")
        if seen[sample_id]:
            raise AssertionError("common events are not contiguous by sample")
        seen[sample_id] = True
        seen_count += 1
        sample = index_by_id.row(sample_id - 1, named=True)
        split = str(sample["split"])
        if pending_split is not None and split != pending_split:
            flush()
        pending_split = split
        pending.append((sample, events))
        if len(pending) >= CHUNK:
            flush()
        if seen_count % 2_000 == 0:
            print(f"checked {seen_count:,} / {index.height:,} released ihm stays")
    flush()

    if seen_count != index.height or not seen[1:].all():
        missing = (np.flatnonzero(~seen[1:]) + 1)[:20].tolist()
        raise AssertionError(f"not every task sample had a raw event block: {missing}")
    total_events = sum(
        value["raw_occurrence_events"] for value in split_reports.values()
    )
    if total_events != manifest["event_count"]:
        raise AssertionError("exhaustive common event total differs from manifest")
    if sum(code_histogram.values()) != total_events:
        raise AssertionError("code histogram does not cover every exported event")
    unexpected = set(code_histogram) - set(spec["codes"])
    if unexpected:
        raise AssertionError(f"exported events use undeclared codes: {sorted(unexpected)}")
    if sum(numeric_ulp_distribution.values()) != numeric_ulp_cells:
        raise AssertionError("numeric ULP histogram does not cover every measured cell")
    expected_distribution = {
        int(distance): int(count)
        for distance, count in measurement["ulp_distribution"].items()
    }
    if dict(numeric_ulp_distribution) != expected_distribution:
        raise AssertionError(
            "MEDS-derived numeric ULP distribution differs from the source measurement"
        )
    if numeric_ulp_cells != int(measurement["cells"]) or numeric_ulp_nonzero != int(
        measurement["nonzero_cells"]
    ):
        raise AssertionError("MEDS-derived precision census has different coverage")
    for split, distribution in per_split_numeric_ulp.items():
        expected = {
            int(distance): int(count)
            for distance, count in measurement["per_split"][split][
                "ulp_distribution"
            ].items()
        }
        if dict(distribution) != expected:
            raise AssertionError(f"{split} MEDS/source ULP distributions differ")
    if maximum_numeric_ulp != FIXED_NUMERIC_ULP_BOUND:
        raise AssertionError(
            "exhaustive numeric maximum no longer reproduces the smallest fixed ULP "
            f"bound: observed={maximum_numeric_ulp} bound={FIXED_NUMERIC_ULP_BOUND}"
        )

    serializable = {}
    for split, values in split_reports.items():
        if values["examples"] != manifest["splits"][split]["count"]:
            raise AssertionError(f"{split} exhaustive sample total differs")
        ordered = index.filter(pl.col("split") == split).sort("source_order")
        ordered_ids = ordered["sample_id"].to_numpy().astype("<i8", copy=False)
        ordered_labels = ordered["boolean_value"].to_numpy().astype(np.uint8, copy=False)
        ordered_rows = ordered["ihm_position"].to_numpy().astype("<i8", copy=False)
        digest = hashlib.sha256()
        digest.update(ordered_ids.tobytes())
        digest.update(ordered_rows.tobytes())
        digest.update(ordered_labels.tobytes())
        digest.update(np.ascontiguousarray(fingerprints[ordered_ids]).tobytes())
        serializable[split] = {
            **values,
            "ordered_tensor_fingerprint_sha256": digest.hexdigest(),
        }

    report = {
        "status": "exact_at_released_multitask_reader_boundary",
        "paper": "semi_supervised_meta_learning",
        "task": TASK_ID,
        "protocol_status_meaning": (
            "every exported sample was reconciled against the raw source CSV and "
            "against the released reader.MultitaskReader + Discretizer + normalizer "
            "model input; the released study wires no in-hospital-mortality trainer, "
            "so this loader output is the last released deterministic boundary for "
            "the selected Table 4 column"
        ),
        "common_boundary": {
            "kind": "raw pre-discretization structured occurrence stream",
            "codes": 59,
            "fractional_source_times": True,
            "within_bin_repeated_values": True,
            "source_csv_order": True,
            "derived_masks_or_imputed_values_retained": False,
            "note_codes": [],
            "observed_codes": len(code_histogram),
            "unused_declared_codes": sorted(set(spec["codes"]) - set(code_histogram)),
            "code_histogram": {
                code: int(code_histogram[code]) for code in spec["codes"] if code in code_histogram
            },
        },
        "author_boundary": {
            "loader": "reader.MultitaskReader.read_example (released)",
            "shape": f"[{WINDOW_HOURS},76]",
            "numpy_dtype": "float64",
            "model_dtype": "float32",
            "value_channels": 59,
            "mask_channels": 17,
            "prediction_row": IHM_POSITION,
            "hourly_overwrite": "last raw value in CSV order",
            "imputation": "previous, initialized with released normal values",
            "normalization": "released SSML normalizer state (decomp_ts1.0)",
            "released_ihm_trainer": False,
        },
        "numeric_precision": {
            "scope": (
                "12 numeric channels of the released float32 model input for every "
                "one of the released ihm stays"
            ),
            "rule": "integer distance between adjacent IEEE-754 float32 values",
            "authorized_reason": (
                "required attributable source-float64 to MEDS-float32 round-trip only; "
                "no rtol/atol allclose"
            ),
            "fixed_ulp_bound": FIXED_NUMERIC_ULP_BOUND,
            "maximum_observed_ulp": maximum_numeric_ulp,
            "maximum_absolute_difference": maximum_numeric_absolute_difference,
            "cells": numeric_ulp_cells,
            "nonzero_cells": numeric_ulp_nonzero,
            "nonzero_fraction": numeric_ulp_nonzero / numeric_ulp_cells,
            "ulp_distribution": {
                str(distance): int(count)
                for distance, count in sorted(numeric_ulp_distribution.items())
            },
            "per_split": {
                split: {
                    "maximum_observed_ulp": max(distribution, default=0),
                    "cells": sum(distribution.values()),
                    "nonzero_cells": sum(
                        count for distance, count in distribution.items() if distance != 0
                    ),
                    "ulp_distribution": {
                        str(distance): int(count)
                        for distance, count in sorted(distribution.items())
                    },
                }
                for split, distribution in per_split_numeric_ulp.items()
            },
        },
        "exhaustive_recreation": serializable,
        "checks": [
            "every common occurrence against exact raw CSV time, row, channel, and value",
            "every exported stay's released (pos;mask;label) ihm triple through reader.MultitaskReader",
            "released Discretizer(timestep=1, previous, zero) + normalizer [48,76] model input for every stay",
            "exact categorical and mask rematerialization for every stay",
            "numeric float32 channels within the fixed exhaustive integer-ULP bound",
            "structured-only 59-code common vocabulary, empty note_codes, and no retained derived masks",
            "no exported occurrence postdates the released 48-hour prediction point",
        ],
        "source_artifacts": {
            "discretizer_config_sha256": sha256_file(CONFIG_PATH),
            "normalizer_sha256": sha256_file(NORMALIZER_PATH),
            "source_precision_measurement_sha256": sha256_file(measurement_path),
            "common_events_sha256": sha256_file(events_path),
        },
        "manifest_sha256": manifest_sha256,
    }
    write_json(args.task_dir / "audit" / "full_batch_recreation.json", report)
    if sha256_file(manifest_path) != manifest_sha256:
        raise AssertionError("batch checker modified immutable manifest.json")
    for path in scratch.iterdir():
        path.unlink()
    scratch.rmdir()
    print(json.dumps({k: v for k, v in report.items() if k != "common_boundary"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
