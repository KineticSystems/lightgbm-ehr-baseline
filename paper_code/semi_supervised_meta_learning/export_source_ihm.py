"""Export the released SSML multitask in-hospital-mortality task to MEDS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from cohort_source import load_transform_spec  # noqa: E402
from cohort_source_ihm import (  # noqa: E402
    IHM_POSITION,
    RELEASED_LABEL_DEFINITION,
    TASK_ID,
    WINDOW_HOURS,
    build_samples,
    count_events,
    expected_source,
    iter_events,
)

PUBLISHED_AUROC = 0.869
PUBLISHED_AUROC_SD = 0.007
PUBLISHED_BASELINE = {
    "name": "MAML",
    "auroc": 0.858,
    "rule": (
        "canonical best baseline: the non-proposed method in the paper's own "
        "results table that wins most often across the outcomes this campaign "
        "fits, ties broken by mean AUROC over those outcomes (user rule, "
        "2026-09-02)"
    ),
}
PUBLISHED_CELL = (
    "Table 4 (average performance on MIMIC-III full sequences with time domain "
    "variation), In-hospital Mortality AUCROC, row SSML-TDV: 0.869 (0.007); "
    "best non-proposed row MAML 0.858 (0.009)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-rows", type=int, default=2_000_000)
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
            "name": "In-hospital mortality after the first 48 ICU hours",
            "primary_metric": "AUROC",
            "published_auroc": PUBLISHED_AUROC,
            "published_auroc_sd": PUBLISHED_AUROC_SD,
            "published_baseline": PUBLISHED_BASELINE,
            "published_cell": PUBLISHED_CELL,
            "observation_window_hours": float(WINDOW_HOURS),
            "evaluation_unit": "eligible ICU stay",
            "positive_class": "in-hospital death (mimic3-benchmarks episode Mortality flag)",
            "label_definition": RELEASED_LABEL_DEFINITION,
            "author_target_row": IHM_POSITION,
            "source_task": "Table 4 SSML-TDV full-sequence in-hospital mortality",
            "temporal_profile": "single_visit_timed",
        },
        codes=spec["codes"],
        note_codes=(),
        events=iter_events(args.source_root, samples, batch_rows=args.batch_rows),
        samples=samples,
        source={
            **expected_source(args.source_root, event_count),
            "sample_audit": sample_audit,
            "event_counts_by_split": event_counts_by_split,
            "complete_released_eligible_cohort": True,
        },
        dataset_name="MIMIC-III",
        dataset_version="1.4",
    )
    if int(manifest["event_count"]) != event_count:
        raise AssertionError("written MEDS event count differs from source pre-count")
    print(f"wrote {args.output_dir} with {event_count:,} events")


if __name__ == "__main__":
    main()
