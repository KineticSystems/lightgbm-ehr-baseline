#!/usr/bin/env python3
"""Measure the sole MEDS-float32 numeric precision boundary for the ihm task.

This pass never reads the exported MEDS directory: it recreates the released
model input twice from the source alone -- once through the released
MultitaskReader/Discretizer/normalizer in float64, and once through the exact
float32 occurrence values the export stores -- and records the integer-ULP
distribution that check_batch_ihm.py must then reproduce from MEDS.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from check_batch import float32_ulp_distance
from check_batch_ihm import CHUNK, AuthorMultitaskLoader, released_lines
from measure_precision import raw_event_frame, write_json
from cohort_source import (
    load_raw_stay,
    load_transform_spec,
    reconstruct_model_tensor,
)
from cohort_source_ihm import (
    WINDOW_HOURS,
    assert_source_ready,
    build_samples,
    stay_path,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = assert_source_ready(args.source_root)
    samples, sample_audit = build_samples(args.source_root)
    spec = load_transform_spec()
    numeric_positions = np.asarray(spec["numeric_indices"], dtype=np.int64)
    numeric_set = set(numeric_positions.tolist())
    categorical_positions = np.asarray(
        [position for position in range(59) if position not in numeric_set],
        dtype=np.int64,
    )
    scratch = Path(tempfile.mkdtemp(prefix="ssml-ihm-precision-"))
    loader = AuthorMultitaskLoader(paths, spec, scratch)
    lines = released_lines(paths, samples)

    distribution = Counter()
    per_split = {split: Counter() for split in ("train", "validation", "test")}
    cells = nonzero = stays = 0
    maximum_ulp = 0
    maximum_absolute_difference = 0.0

    ordered = samples.sort("sample_id").iter_rows(named=True)
    pending: list[dict[str, object]] = []
    pending_split: str | None = None

    def flush() -> None:
        nonlocal cells, nonzero, stays, maximum_ulp, maximum_absolute_difference
        if not pending:
            return
        author = loader.tensors(
            pending_split, pending, [lines[int(row["sample_id"])] for row in pending]
        )
        for row, direct32 in zip(pending, author):
            raw = load_raw_stay(
                stay_path(paths, pending_split, str(row["source_filename"])),
                spec,
                WINDOW_HOURS,
            )
            recreated = reconstruct_model_tensor(
                raw_event_frame(raw), spec, WINDOW_HOURS
            )
            if not np.array_equal(
                recreated[:, categorical_positions], direct32[:, categorical_positions]
            ):
                raise AssertionError("source-float32 changed a categorical channel")
            if not np.array_equal(recreated[:, 59:], direct32[:, 59:]):
                raise AssertionError("source-float32 changed an observation-mask channel")
            ulp = float32_ulp_distance(
                recreated[:, numeric_positions], direct32[:, numeric_positions]
            )
            unique, counts = np.unique(ulp, return_counts=True)
            observed = {int(a): int(b) for a, b in zip(unique, counts)}
            distribution.update(observed)
            per_split[pending_split].update(observed)
            cells += int(ulp.size)
            nonzero += int(np.count_nonzero(ulp))
            maximum_ulp = max(maximum_ulp, int(ulp.max(initial=0)))
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                float(
                    np.max(
                        np.abs(
                            recreated[:, numeric_positions]
                            - direct32[:, numeric_positions]
                        ),
                        initial=0.0,
                    )
                ),
            )
            stays += 1
        pending.clear()
        if stays % 2_000 < CHUNK:
            print(f"measured {stays:,} released ihm stays")

    for row in ordered:
        split = str(row["split"])
        if pending_split is not None and split != pending_split:
            flush()
        pending_split = split
        pending.append(row)
        if len(pending) >= CHUNK:
            flush()
    flush()

    if stays != samples.height:
        raise AssertionError("precision measurement did not cover every released stay")
    if sum(distribution.values()) != cells:
        raise AssertionError("ULP distribution does not cover every numeric cell")
    report = {
        "status": "measurement",
        "scope": (
            "every released ihm stay's complete [48,76] model input; the exported "
            "task contains exactly one sample per measured stay"
        ),
        "stays": stays,
        "samples_covered": samples.height,
        "window_hours": WINDOW_HOURS,
        "numeric_channels": 12,
        "categorical_channels_exact": 47,
        "mask_channels_exact": 17,
        "rule": "integer distance between adjacent IEEE-754 float32 values",
        "reason": (
            "required attributable source-float64 to MEDS-float32 round-trip only; "
            "no rtol/atol allclose"
        ),
        "smallest_covering_ulp_bound": maximum_ulp,
        "maximum_absolute_difference": maximum_absolute_difference,
        "cells": cells,
        "nonzero_cells": nonzero,
        "nonzero_fraction": nonzero / cells,
        "ulp_distribution": {
            str(distance): int(count) for distance, count in sorted(distribution.items())
        },
        "per_split": {
            split: {
                "maximum_observed_ulp": max(values, default=0),
                "cells": sum(values.values()),
                "nonzero_cells": sum(
                    count for distance, count in values.items() if distance != 0
                ),
                "ulp_distribution": {
                    str(distance): int(count)
                    for distance, count in sorted(values.items())
                },
            }
            for split, values in per_split.items()
        },
        "source_sample_audit": sample_audit["splits"],
    }
    write_json(args.output, report)
    for path in scratch.iterdir():
        path.unlink()
    scratch.rmdir()
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "ulp_distribution"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
