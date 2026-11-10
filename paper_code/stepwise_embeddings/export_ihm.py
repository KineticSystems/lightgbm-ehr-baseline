"""Export the second published task (in-hospital mortality at 48h) to MEDS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from cohort_ihm import (  # noqa: E402
    AUTHOR_MAXLEN,
    AUTHOR_TARGET_ROW,
    AUTHOR_TASK,
    TASK_ID,
    build_samples,
    count_events,
    expected_ihm_source,
    iter_events,
)
from cohort_source import inspect_hdf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    metadata = inspect_hdf(args.source_root)
    samples, sample_audit = build_samples(args.source_root)
    event_count, event_counts_by_split = count_events(args.source_root)
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
            "key": "stepwise_embeddings",
            "title": "On the Importance of Step-wise Embeddings for Heterogeneous Clinical Time-Series",
        },
        task={
            "id": TASK_ID,
            "name": "In-hospital mortality after the first 48 ICU hours",
            "primary_metric": "AUROC",
            "published_auroc": 0.858,
            "published_auroc_sd": 0.002,
            "published_source": "Appendix Table 9, MIMIC-III Mortality, '+ FTT embedding (Gorishniy et al., 2021)' = 85.8 +/- 0.2",
            "observation_window_hours": float(AUTHOR_MAXLEN),
            "evaluation_unit": "eligible ICU stay",
            "clinical_endpoint": "in-hospital mortality predicted from the first 48 ICU hours",
            "positive_class": "in-hospital death (mimic3-benchmarks episode Mortality flag)",
            "label_definition": (
                "mimic3-benchmarks create_multitask.py: ihm_label = episode Mortality; "
                "ihm_mask = 1 only when the ICU length of stay is at least 48 hours and "
                "the first in-ICU event occurs within 48 hours; ihm_position = 47. TLS "
                "stores that label at author grid row 47 and the released launcher sets "
                "MAXLEN=48, so the model consumes author rows 0..47 only."
            ),
            "source_task": AUTHOR_TASK,
            "author_maxlen": AUTHOR_MAXLEN,
            "author_target_row": AUTHOR_TARGET_ROW,
        },
        codes=tuple(metadata["columns"]),
        note_codes=(),
        events=iter_events(args.source_root),
        samples=samples,
        source={
            **expected_ihm_source(args.source_root, event_count),
            "sample_audit": sample_audit,
            "event_counts_by_split": event_counts_by_split,
            "resource_limited_reproduction": False,
        },
        dataset_name="MIMIC-III",
        dataset_version="1.4",
    )
    if int(manifest["event_count"]) != event_count:
        raise AssertionError("written MEDS event count differs from source pre-count")
    print(f"wrote {args.output_dir} with {event_count:,} events")


if __name__ == "__main__":
    main()
