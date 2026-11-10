#!/usr/bin/env python3
"""Measure the sole MEDS-float32 numeric precision boundary exhaustively."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa

from check_batch import (
    build_author_components,
    direct_author_maximum,
    float32_ulp_distance,
)
from cohort_source import (
    _group_samples,
    assert_source_ready,
    build_samples,
    load_raw_stay,
    load_transform_spec,
    reconstruct_model_tensor,
    stay_path,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def raw_event_frame(raw: dict[str, np.ndarray]) -> pl.DataFrame:
    numeric = pa.array(
        raw["numeric_values"], mask=~raw["numeric_valid"], type=pa.float32()
    )
    return pl.DataFrame(
        {
            "source_time": pl.Series(raw["times"], dtype=pl.Float64),
            "source_row": pl.Series(raw["source_rows"], dtype=pl.Int64),
            "code": pl.Series(raw["codes"], dtype=pl.String),
            "numeric_value": pl.from_arrow(numeric),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = assert_source_ready(args.source_root)
    samples, sample_audit = build_samples(args.source_root)
    spec = load_transform_spec()
    dataset_dirs, discretizer, normalizer, header = build_author_components(paths)
    if header != ",".join(spec["model_codes"]):
        raise AssertionError("released loader header differs from the 76-channel spec")
    numeric_positions = np.asarray(spec["numeric_indices"], dtype=np.int64)
    numeric_set = set(numeric_positions.tolist())
    categorical_positions = np.asarray(
        [position for position in range(59) if position not in numeric_set],
        dtype=np.int64,
    )

    distribution = Counter()
    per_split = {split: Counter() for split in ("train", "validation", "test")}
    cells = nonzero = groups = 0
    maximum_ulp = 0
    maximum_absolute_difference = 0.0
    for split, filename, group in _group_samples(samples):
        maximum = group.row(-1, named=True)
        cutoff = int(maximum["source_cutoff_hour"])
        raw = load_raw_stay(stay_path(paths, split, filename), spec, cutoff)
        recreated = reconstruct_model_tensor(raw_event_frame(raw), spec, cutoff)
        _, author = direct_author_maximum(
            dataset_dirs[split], discretizer, normalizer, maximum
        )
        if not np.array_equal(
            recreated[:, categorical_positions], author[:, categorical_positions]
        ):
            raise AssertionError("source-float32 changed a categorical trainer channel")
        if not np.array_equal(recreated[:, 59:], author[:, 59:]):
            raise AssertionError("source-float32 changed an observation-mask channel")
        ulp = float32_ulp_distance(
            recreated[:, numeric_positions], author[:, numeric_positions]
        )
        unique, counts = np.unique(ulp, return_counts=True)
        observed = {
            int(distance): int(count) for distance, count in zip(unique, counts)
        }
        distribution.update(observed)
        per_split[split].update(observed)
        cells += int(ulp.size)
        nonzero += int(np.count_nonzero(ulp))
        maximum_ulp = max(maximum_ulp, int(ulp.max(initial=0)))
        maximum_absolute_difference = max(
            maximum_absolute_difference,
            float(
                np.max(
                    np.abs(
                        recreated[:, numeric_positions] - author[:, numeric_positions]
                    ),
                    initial=0.0,
                )
            ),
        )
        groups += 1
        if groups % 1_000 == 0:
            print(f"measured {groups:,} stay maxima")

    if groups != sum(
        values["stays"] for values in sample_audit["splits"].values()
    ):
        raise AssertionError("precision measurement did not cover every stay")
    if sum(distribution.values()) != cells:
        raise AssertionError("ULP distribution does not cover every numeric cell")
    report = {
        "status": "measurement",
        "scope": (
            "every attributable stay maximum; every eligible landmark is an exact "
            "prefix of one measured tensor"
        ),
        "groups": groups,
        "landmarks_covered_by_prefix": samples.height,
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
            str(distance): int(count)
            for distance, count in sorted(distribution.items())
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
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
