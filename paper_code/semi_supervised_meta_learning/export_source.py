"""Export the complete deterministic SSML decompensation benchmark to MEDS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from cohort_source import (  # noqa: E402
    TASK_ID,
    build_samples,
    count_events,
    expected_source,
    iter_events,
    load_transform_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    spec = load_transform_spec()
    samples, sample_audit = build_samples(args.source_root)
    event_count, event_counts_by_split = count_events(args.source_root, samples)
    print(
        json.dumps(
            {
                "samples": sample_audit["splits"],
                "event_count": event_count,
                "event_counts_by_split": event_counts_by_split,
            },
            indent=2,
        )
    )
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "semi_supervised_meta_learning",
            "title": "Semi-Supervised Meta-Learning for Cross-Domain Few-Shot Prediction in Time-Series",
        },
        task={
            "id": TASK_ID,
            "name": "Hourly ICU decompensation within the next 24 hours",
            "primary_metric": "AUROC",
            "published_auroc": 0.906,
            "observation_window_hours": float(sample_audit["maximum_cutoff_hour"]),
            "evaluation_unit": "eligible ICU-stay landmark",
            "positive_class": "in-unit death within 24 hours at the benchmark cutoff",
            "source_task": "Table 4 SSML-TDV full-sequence decompensation",
        },
        codes=spec["codes"],
        note_codes=(),
        events=iter_events(args.source_root, samples),
        samples=samples,
        source={
            **expected_source(args.source_root, event_count),
            "sample_audit": sample_audit,
            "event_counts_by_split": event_counts_by_split,
            "complete_deterministic_eligible_benchmark": True,
        },
        dataset_name="MIMIC-III",
        dataset_version="1.4",
    )
    if int(manifest["event_count"]) != event_count:
        raise AssertionError("written MEDS event count differs from source pre-count")
    print(f"wrote {args.output_dir} with {event_count:,} events")


if __name__ == "__main__":
    main()
