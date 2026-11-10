"""Export REMed's prespecified 48-hour LOS-7 task to shared MEDS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from plain_events import (  # noqa: E402
    DISCLOSED_DROPPED_COLUMNS,
    MAXIMUM_PLAIN_CODES,
    MINIMUM_VALUE_SUPPORT_STAYS,
    PlainEventDecoder,
    clamp_duration,
    load_vocabulary,
)


from cohort_source import (  # noqa: E402
    LABEL_COLUMN,
    PREDICTION_HOURS,
    PUBLISHED_AUROC,
    REMED_COMMIT,
    SNUB_COMMIT,
    TASK_ID,
    TASK_NAME,
    cross_split_subject_overlap,
    hash_ints,
    load_samples,
    read_source_build,
    sample_frame,
    sha256_file,
    source_paths,
    split_summary,
)

# Interval-completion clamp (user-directed 2026-09-01): procedure durations
# denominated in minutes are truncated to the observable remainder of the
# prediction window. The model keeps its full leaked durations upstream, so
# the baseline is strictly conservative on this channel.
CLAMP_INTERVAL_DURATIONS = True
PREDICTION_MINUTES = float(PREDICTION_HOURS * 60)

BERT_VOCAB_PATH = (
    PAPER_ROOT
    / ".export_assets"
    / "external_artifacts"
    / "Bio_ClinicalBERT"
    / "vocab.txt"
)


def build_decoder(source_root: Path, samples) -> PlainEventDecoder:
    """Survey categorical support over TRAIN stays, then freeze the policy."""
    decoder = PlainEventDecoder(load_vocabulary(BERT_VOCAB_PATH))
    paths = source_paths(source_root)
    with h5py.File(paths["h5"], "r") as source:
        ehr = source["ehr"]
        for sample in samples:
            if sample.split != "train":
                continue
            decoder.survey_sample(
                np.asarray(ehr[str(sample.stay_id)]["hi"][:], dtype=np.int16)
            )
    decoder.finalize_survey()
    return decoder


def collect_plain_codes(
    source_root: Path, samples, decoder: PlainEventDecoder
) -> tuple[str, ...]:
    """One full emission pass: the deterministic plain vocabulary."""
    codes: set[str] = set()
    paths = source_paths(source_root)
    with h5py.File(paths["h5"], "r") as source:
        ehr = source["ehr"]
        for sample in samples:
            for event in np.asarray(
                ehr[str(sample.stay_id)]["hi"][:], dtype=np.int16
            ):
                for code, _numeric, _clamp in decoder.emissions(event):
                    codes.add(code)
    if len(codes) > MAXIMUM_PLAIN_CODES:
        raise AssertionError(
            f"plain vocabulary {len(codes):,} breached the "
            f"{MAXIMUM_PLAIN_CODES:,} guard"
        )
    return tuple(sorted(codes))


def plain_event_batches(
    source_root: Path,
    samples,
    decoder: PlainEventDecoder,
    *,
    batch_size: int = 100_000,
):
    """Stream decoded emissions in sample order (writer-ready batches)."""
    paths = source_paths(source_root)
    rows: dict[str, list[object]] = {
        "subject_id": [],
        "time": [],
        "code": [],
        "numeric_value": [],
        "sample_id": [],
        "source_row": [],
        "sequence_row": [],
        "source_time": [],
    }
    source_row = 0
    with h5py.File(paths["h5"], "r") as source:
        ehr = source["ehr"]
        for sample in samples:
            group = ehr[str(sample.stay_id)]
            times = np.asarray(group["time"][:], dtype=np.int64)
            icu_us = sample.prediction_time_us - PREDICTION_HOURS * 3_600_000_000
            events = np.asarray(group["hi"][:], dtype=np.int16)
            for sequence_row, (event, minute) in enumerate(zip(events, times)):
                for code, numeric, clamp in decoder.emissions(event):
                    if (
                        clamp
                        and CLAMP_INTERVAL_DURATIONS
                        and numeric is not None
                    ):
                        numeric = clamp_duration(
                            numeric, float(minute), PREDICTION_MINUTES
                        )
                    rows["subject_id"].append(sample.subject_id)
                    rows["time"].append(icu_us + int(minute) * 60_000_000)
                    rows["code"].append(code)
                    rows["numeric_value"].append(numeric)
                    rows["sample_id"].append(sample.stay_id)
                    rows["source_row"].append(source_row)
                    rows["sequence_row"].append(sequence_row)
                    rows["source_time"].append(float(minute))
                    source_row += 1
            if len(rows["sample_id"]) >= batch_size:
                yield pl.DataFrame(rows)
                rows = {name: [] for name in rows}
    if rows["sample_id"]:
        yield pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = source_paths(args.source_root)
    samples, observation_hours = load_samples(args.source_root)
    decoder = build_decoder(args.source_root, samples)
    codes = collect_plain_codes(args.source_root, samples, decoder)
    decode_statistics = decoder.statistics.as_dict()
    summaries = split_summary(samples)
    source = {
        "upstream_commit": REMED_COMMIT,
        "producer_repository": "https://github.com/Jwoo5/integrated-ehr-pipeline",
        "producer_commit": SNUB_COMMIT,
        "loader_entrypoint": "src.dataset.EHRDataset over snub mimiciv.h5",
        "capture_boundary": (
            "the exact ordered snub hi[event,channel,token] int16 tensor and "
            "integer-minute time vector before the released EHRDataset creates "
            "Torch tensors"
        ),
        "selected_paper_cell": (
            "Appendix Table 10, MIMIC-IV, 48-hour prediction, infinite history, "
            "LOS 7, REMed AUROC"
        ),
        "selected_model_inputs": ["structured"],
        "selection_tier": 1,
        "published_auroc": PUBLISHED_AUROC,
        "selected_label": LABEL_COLUMN,
        "positive_class": (
            "complete ICU length of stay is strictly greater than seven days "
            "(released snub los_7 label)"
        ),
        "source_prediction_hours": PREDICTION_HOURS,
        "source_history": "infinite (all snub-retained events before 48 hours)",
        "split_policy": "released patient-grouped split_2020, mapped valid->validation",
        "split_summary": summaries,
        "cross_split_subject_overlap": cross_split_subject_overlap(samples),
        "ordered_sample_id_sha256": hash_ints(sample.stay_id for sample in samples),
        "ordered_subject_id_sha256": hash_ints(
            sample.subject_id for sample in samples
        ),
        "ordered_label_sha256": hash_ints(int(sample.label) for sample in samples),
        "source_artifacts": {
            "mimiciv.h5": {
                "sha256": sha256_file(paths["h5"]),
                "bytes": paths["h5"].stat().st_size,
            },
            "mimiciv_cohort.csv": {
                "sha256": sha256_file(paths["cohort"]),
                "bytes": paths["cohort"].stat().st_size,
            },
            "source_build.json": {
                "sha256": sha256_file(paths["build"]),
                "bytes": paths["build"].stat().st_size,
            },
            "remed_src_dataset.py": {
                "sha256": sha256_file(PAPER_ROOT / "src" / "dataset.py"),
                "bytes": (PAPER_ROOT / "src" / "dataset.py").stat().st_size,
            },
        },
        "source_build": read_source_build(args.source_root),
        "event_encoding": {
            "design": (
                "plain decoded timeline (docs/remed_plain_timeline.md): each "
                "snub [3,128] event is segmented by its GenHPF type-id "
                "channel, detokenized through the pinned Bio_ClinicalBERT "
                "vocabulary, and emitted as MIMICIV//<table>//<identity> "
                "codes with real numeric values; event membership and times "
                "are the model's own h5 stream, so both sides see exactly "
                "the same events (user parity directive 2026-09-01)"
            ),
            "interval_duration_clamp": {
                "enabled": CLAMP_INTERVAL_DURATIONS,
                "rule": (
                    "procedureevents values with valueuom=min are truncated "
                    "to min(duration, minutes remaining to prediction); the "
                    "model's own input keeps the full start-anchored "
                    "durations (known_source_defects_preserved), so the "
                    "baseline is conservative on this channel"
                ),
            },
            "categorical_value_policy": (
                "a value embeds into its item's code only when that exact "
                "(item, value) pair appears in at least "
                f"{MINIMUM_VALUE_SUPPORT_STAYS} TRAIN stays; decided per "
                "pair, never per item; masked markers normalized to absent"
            ),
            "decode_statistics": decode_statistics,
            "disclosed_dropped_columns": {
                table: list(columns)
                for table, columns in DISCLOSED_DROPPED_COLUMNS.items()
            },
            "decoder_module_sha256": sha256_file(
                PAPER_ROOT / "plain_events.py"
            ),
            "bert_vocabulary_sha256": sha256_file(BERT_VOCAB_PATH),
        },
        "same_minute_order": (
            "preserved exactly from this source HDF5 build; the upstream Spark/pandas "
            "producer does not define a reproducible equal-minute tie breaker"
        ),
        "multitask_asymmetry": (
            "the paper trained 27 endpoints jointly; the export covers only the "
            "prespecified los_7 label while preserving its exact input/cohort"
        ),
        "event_encoder_limitation": (
            "the fitted GenHPF event-encoder checkpoint used to make REMed's cached "
            "512-dimensional representations was not released; the reversible "
            "released EHRDataset input is therefore the last attributable boundary"
        ),
        "known_source_defects_preserved": [
            "result-bearing lab/microbiology fields are assigned specimen/chart time",
            "interval-completion fields are assigned start time",
            "diagnosis construction inner-joins and filters every endpoint cohort",
            "same-minute order is unspecified",
        ],
    }
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "remed",
            "title": (
                "General-Purpose Retrieval-Enhanced Medical Prediction Model "
                "Using Near-Infinite History"
            ),
        },
        task={
            "id": TASK_ID,
            "name": TASK_NAME,
            "primary_metric": "AUROC",
            "published_auroc": PUBLISHED_AUROC,
            "observation_window_hours": observation_hours,
        },
        codes=codes,
        events=plain_event_batches(args.source_root, samples, decoder),
        samples=sample_frame(samples, observation_hours),
        source=source,
        dataset_name="REMed/snub MIMIC-IV 48-hour infinite-history cohort",
        dataset_version="MIMIC-IV 2.0 + MIMIC-IV-ED 2.0 / snub bab6c07",
    )
    print(
        json.dumps(
            {
                "task": TASK_ID,
                "samples": len(samples),
                "events": manifest["event_count"],
                "codes": len(codes),
                "observation_window_hours": observation_hours,
                "splits": summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
