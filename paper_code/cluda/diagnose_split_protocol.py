"""Build a compact per-stay view for CLUDA split-provenance diagnosis."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cohort_source import source_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = source_files(args.source_root)
    admissions = pd.read_csv(paths["admissions"])
    static = pd.read_csv(paths["static"])
    scaling = pd.read_csv(paths["static_scaling"]).set_index("feature")
    age_mean = float(scaling.loc["age", "mean"])
    age_sd = float(scaling.loc["age", "sd"])

    lengths: Counter[int] = Counter()
    for chunk in pd.read_csv(
        paths["timeseries"], usecols=["stay_id"], chunksize=1_000_000
    ):
        counts = chunk["stay_id"].value_counts(sort=False)
        lengths.update({int(stay_id): int(count) for stay_id, count in counts.items()})

    static = static.set_index("stay_id", verify_integrity=True)
    stay_ids = admissions["stay_id"].to_numpy(np.int64)
    frame = pd.DataFrame(
        {
            "stay_id": stay_ids,
            "hospital_expire_flag": admissions["hospital_expire_flag"].to_numpy(
                np.int8
            ),
            "released_split": admissions["splitType"].astype(str),
            "length": np.asarray([lengths[int(value)] for value in stay_ids], np.int64),
            "age": static.loc[stay_ids, "age"].to_numpy(np.float64) * age_sd
            + age_mean,
        }
    )
    if (frame["length"] <= 0).any() or not np.isfinite(frame["age"]).all():
        raise AssertionError("compact CLUDA split view is incomplete")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(
        {
            "rows": len(frame),
            "eligible": int((frame["length"] >= 4).sum()),
            "samples": int((frame["length"] - 3).clip(lower=0).sum()),
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
