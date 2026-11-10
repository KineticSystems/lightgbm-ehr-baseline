"""Export the prespecified rolling decompensation task to MEDS Parquet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from missing_xgboost_lightgbm.landmark_sampling import (  # noqa: E402
    LANDMARK_SAMPLE_CAPS,
    LANDMARK_SAMPLE_SEED,
    select_uniform_landmarks,
)
from cohort_source import (  # noqa: E402
    TASK_ID,
    build_samples,
    count_events,
    expected_source,
    inspect_hdf,
    iter_events,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    metadata = inspect_hdf(args.source_root)
    full_samples, sample_audit = build_samples(args.source_root)
    samples, sampling_audit = select_uniform_landmarks(full_samples)
    selected_sample_ids = np.sort(
        samples["sample_id"].to_numpy().astype(np.int64, copy=False)
    )
    event_count, event_counts_by_split = count_events(
        args.source_root, selected_sample_ids
    )
    print(
        json.dumps(
            {
                "samples": sample_audit["splits"],
                "sampling": sampling_audit,
                "event_count": event_count,
                "event_counts_by_split": event_counts_by_split,
            },
            indent=2,
        )
    )
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "stepwise_embeddings",
            "title": "On the Importance of Step-wise Embeddings for Heterogeneous Clinical Time-Series",
        },
        task={
            "id": TASK_ID,
            "name": "Hourly ICU decompensation within the next 24 hours",
            "primary_metric": "AUROC",
            "published_auroc": 0.916,
            "observation_window_hours": float(sample_audit["max_author_target_row"]),
            "evaluation_unit": "eligible ICU-stay landmark",
            "positive_class": "in-unit death within 24 hours at the M3B source cutoff",
            "source_task": "decomp_24Hours",
        },
        codes=tuple(metadata["columns"]),
        note_codes=(),
        events=iter_events(
            args.source_root, selected_sample_ids=selected_sample_ids
        ),
        samples=samples,
        source={
            **expected_source(args.source_root, event_count),
            "sample_audit": sample_audit,
            "sampling": sampling_audit,
            "event_counts_by_split": event_counts_by_split,
            "resource_limited_reproduction": True,
            "sampling_contract": {
                "seed": LANDMARK_SAMPLE_SEED,
                "caps": LANDMARK_SAMPLE_CAPS,
            },
        },
        dataset_name="MIMIC-III",
        dataset_version="1.4",
    )
    if int(manifest["event_count"]) != event_count:
        raise AssertionError("written MEDS event count differs from source pre-count")
    print(f"wrote {args.output_dir} with {event_count:,} events")


if __name__ == "__main__":
    main()
