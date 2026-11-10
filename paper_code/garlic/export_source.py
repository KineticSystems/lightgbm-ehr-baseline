"""Export GARLIC's released MIMIC-III mortality boundary directly as MEDS."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parent
WORKSPACE = REPOSITORY.parent.parent
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(WORKSPACE / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from cohort_source import (  # noqa: E402
    EXPECTED_COHORT_ROWS,
    EXPECTED_FEATURES,
    EXPECTED_POSITIVE,
    OBSERVATION_HOURS,
    PUBLISHED_COHORT_ROWS,
    PUBLISHED_POSITIVE,
    TASK_ID,
    UPSTREAM,
    build_cache,
    event_batches,
    open_complete_cache,
    ordered_split_hash,
    sample_frame,
    selected_id_hash,
    sha256_file,
    sha256_json,
)


def load_cache(
    mimic_root: Path,
    cache_path: Path,
    *,
    resume: bool,
    restart_stage: str | None,
):
    """Open a final cache directly or explicitly continue its build.

    A completed reconstruction never consults the raw source tree.  This keeps
    exporter/report metadata changes independent of the expensive source
    boundary and makes writing MEDS from a completed cache deterministic.
    """
    if restart_stage is None and cache_path.is_file():
        try:
            return open_complete_cache(cache_path)
        except (KeyError, ValueError, OSError):
            pass
    if not resume and restart_stage is None:
        raise ValueError(
            "GARLIC source cache is absent or incomplete; pass --resume, or "
            "explicitly discard a stage with --restart-stage"
        )
    build_cache(mimic_root, cache_path, restart_stage=restart_stage)
    return open_complete_cache(cache_path)


def write_task_from_cache(
    connection,
    summary: dict,
    output_dir: Path,
) -> None:
    """Write MEDS directly from a completed reconstruction cache."""
    features = tuple(summary["features"])
    if len(features) != EXPECTED_FEATURES:
        raise AssertionError("GARLIC source cache has the wrong vocabulary")
    normalization_hash = sha256_json(summary["normalization"])
    source = {
        "upstream_repository": UPSTREAM["repositories"][0]["url"],
        "upstream_commit": UPSTREAM["repositories"][0]["commit"],
        "loader_entrypoints": [
            "data.preprocess_mimic_iii.prepare_mimic_data",
            "data.data_utils.process_mimiciii",
            "utils.z_score_normalize",
            "utils.get_data_split",
            "main.prepare_batch",
        ],
        "capture_boundary": (
            "positive-mask cells after released first-300/first-200 exact-"
            "timestamp construction and full-cohort z-score; source_time "
            "equals main.prepare_batch time/60"
        ),
        "bounded_instrumentation": {
            "adapter": "cohort_source.py",
            "adapter_sha256": sha256_file(REPOSITORY / "cohort_source.py"),
            "cache": "ignored SQLite reconstruction boundary; never a model input",
            "cache_lifecycle": (
                "pass1/pass2/raw-cell/normalization states use separate atomic "
                "checkpoints; completed caches write MEDS without source rescans"
            ),
            "released_segments": summary["released_source"],
        },
        "selected_paper_cell": (
            "MainResults.tex MIMIC-III mortality, GARLIC, AUROC 90.09 +/- 0.45"
        ),
        "selected_model_inputs": ["structured"],
        "selection_tier": 1,
        "omitted_released_fields": ["static", "extended_static"],
        "omitted_modalities": [],
        "cohort_selection": {
            "published_count": PUBLISHED_COHORT_ROWS,
            "reconstructed_count": EXPECTED_COHORT_ROWS,
            "difference": EXPECTED_COHORT_ROWS - PUBLISHED_COHORT_ROWS,
            "paper_count_match": False,
            "published_positive": PUBLISHED_POSITIVE,
            "reconstructed_positive": EXPECTED_POSITIVE,
            "positive_difference": EXPECTED_POSITIVE - PUBLISHED_POSITIVE,
            "paper_positive_match": False,
            "sorted_hadm_id_sha256": selected_id_hash(connection),
            "protocol_exception": (
                "user explicitly authorized the reproducible released 49,375-"
                "admission cohort as the best attainable source-faithful "
                "reconstruction; no rows were invented or substituted"
            ),
        },
        "split_policy": (
            "released admission-row 80/10/10 stratified split, both calls "
            "random_state=0; patient overlap preserved"
        ),
        "ordered_identifier_label_split_sha256": ordered_split_hash(connection),
        "split_counts": summary["splits"],
        "cross_split_subject_overlap": summary["cross_split_subject_overlap"],
        "raw_source_artifacts": summary["raw_artifacts"],
        "feature_order_sha256": hashlib.sha256(
            json.dumps(features, separators=(",", ":")).encode()
        ).hexdigest(),
        "normalization_sha256": normalization_hash,
        "source_tensor": {
            "values": [EXPECTED_COHORT_ROWS, 200, 103],
            "values_dtype_after_prepare_batch": "float32",
            "mask": [EXPECTED_COHORT_ROWS, 200, 103],
            "mask_dtype_after_prepare_batch": "float32",
            "time": [200, "batch"],
            "time_dtype_after_prepare_batch": "float32",
            "label_dtype_after_prepare_batch": "float32",
            "mask_rule": "released x < 0 clamp followed by x > 0",
            "padding_value": 0.0,
            "timestamp_scale": "elapsed hours divided by 60",
        },
        "candidate_events_within_released_cutoff": summary["candidate_events"],
        "retained_observed_cells": summary["retained_events"],
        "retained_length": summary["retained_length"],
        "known_source_defects_preserved": [
            "normalization and raw outlier medians fitted on the full cohort",
            "admission-row split permits patient overlap",
            "nonpositive values are discarded by the released mask",
            "window begins at earliest retained event rather than ICU INTIME",
            "elapsed MIMIC hours are divided by 60 in prepare_batch",
        ],
    }
    write_meds_task(
        output_dir,
        paper={
            "key": "garlic",
            "title": (
                "GARLIC: Graph Attention-based Relational Learning of "
                "Multivariate Time Series in Intensive Care"
            ),
        },
        task={
            "id": TASK_ID,
            "name": "MIMIC-III in-hospital mortality",
            "primary_metric": "AUROC",
            "published_auroc": 0.9009,
            "observation_window_hours": OBSERVATION_HOURS,
        },
        codes=features,
        note_codes=(),
        events=event_batches(connection, features),
        samples=sample_frame(connection),
        source=source,
        dataset_name="GARLIC released MIMIC-III mortality cohort",
        dataset_version="MIMIC-III 1.4 / GARLIC code-faithful boundary",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--resume",
        action="store_true",
        help="resume an incomplete cache, or write directly from a complete cache",
    )
    mode.add_argument(
        "--restart-stage",
        choices=("pass1", "pass2", "raw_cells", "normalization"),
        help="explicitly discard this cache stage and every downstream stage",
    )
    args = parser.parse_args()

    connection, summary = load_cache(
        args.mimic_root,
        args.cache_path,
        resume=args.resume,
        restart_stage=args.restart_stage,
    )

    try:
        write_task_from_cache(connection, summary, args.output_dir)
    finally:
        connection.close()

    print(
        json.dumps(
            {
                "task": str(args.output_dir),
                "samples": summary["cohort_rows"],
                "positive": summary["positive"],
                "events": summary["retained_events"],
                "splits": summary["splits"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
