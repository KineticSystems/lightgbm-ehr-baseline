"""Export KEDGN's selected MIMIC-III mortality task directly to MEDS."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import polars as pl


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from cohort_source import (  # noqa: E402
    BENCHMARK_COMMIT,
    EXPECTED_TOTAL,
    HOURS,
    MEDICAL_TS_COMMIT,
    SPLITS,
    TASK_ID,
    code_vocabulary,
    export_code_vocabulary,
    iter_source_examples,
    kedgn_visible_fields,
    load_build_report,
    load_cohort,
    sha256,
    variable_names,
)


def event_batches(cache_root: Path, batch_size: int = 240_000):
    codes = code_vocabulary()
    exported_codes = export_code_vocabulary()
    source_names = variable_names()
    if len(codes) != len(source_names):
        raise AssertionError("KEDGN semantic/source channel orders differ in width")
    columns = {
        "subject_id": [],
        "time": [],
        "code": [],
        "numeric_value": [],
        "sample_id": [],
        "source_row": [],
        "sequence_row": [],
        "source_time": [],
    }
    next_source_row = 0

    def emit():
        nonlocal columns
        frame = pl.DataFrame(
            columns,
            schema={
                "subject_id": pl.Int64,
                "time": pl.Datetime("us"),
                "code": pl.String,
                "numeric_value": pl.Float32,
                "sample_id": pl.Int64,
                "source_row": pl.Int64,
                "sequence_row": pl.Int32,
                "source_time": pl.Float64,
            },
        )
        columns = {name: [] for name in columns}
        return frame

    for split in SPLITS:
        for example in iter_source_examples(cache_root, split):
            values, model_mask, _, average, length = kedgn_visible_fields(example)
            for sequence_row, source_time in enumerate(example.time):
                numeric_time = float(source_time)
                if not (-1e-6 <= numeric_time <= HOURS + 1e-6):
                    raise AssertionError("KEDGN source event lies outside first 48 hours")
                event_time = example.intime + timedelta(hours=numeric_time)
                for channel, code in enumerate(codes):
                    if not bool(model_mask[sequence_row, channel]):
                        continue
                    for event_code, event_value in (
                        (code, float(values[sequence_row, channel])),
                        (
                            f"{code}//KEDGN_AVERAGE_INTERVAL",
                            float(average[sequence_row, channel]),
                        ),
                    ):
                        columns["subject_id"].append(example.subject_id)
                        columns["time"].append(event_time)
                        columns["code"].append(event_code)
                        columns["numeric_value"].append(event_value)
                        columns["sample_id"].append(example.sample_id)
                        columns["source_row"].append(next_source_row)
                        columns["sequence_row"].append(sequence_row)
                        columns["source_time"].append(numeric_time)
                        next_source_row += 1
            columns["subject_id"].append(example.subject_id)
            columns["time"].append(example.intime)
            columns["code"].append(exported_codes[-1])
            columns["numeric_value"].append(float(length))
            columns["sample_id"].append(example.sample_id)
            columns["source_row"].append(next_source_row)
            columns["sequence_row"].append(-1)
            columns["source_time"].append(0.0)
            next_source_row += 1
            # Keep every source example in one event batch so exhaustive parity
            # can compare writer row groups without joining a split sample.
            if len(columns["sample_id"]) >= batch_size:
                yield emit()
    if columns["sample_id"]:
        yield emit()


def task_samples(cohort: pl.DataFrame) -> pl.DataFrame:
    return cohort.select(
        "subject_id",
        "sample_id",
        pl.col("intime").alias("observation_start"),
        (pl.col("intime") + pl.duration(hours=HOURS)).alias("prediction_time"),
        pl.col("label").alias("boolean_value"),
        "split",
        "hadm_id",
        "source_name",
        "source_partition",
        "split_position",
        "source_position",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    output = args.output_root.resolve()
    build = load_build_report(cache_root)
    cohort = load_cohort(cache_root)
    if cohort.height != EXPECTED_TOTAL:
        raise AssertionError("KEDGN selected cohort count changed")
    upstream = json.loads((PAPER_ROOT / "UPSTREAM.json").read_text())["repositories"][0]

    source = {
        "upstream": upstream,
        "producer_chain": [
            {
                "url": "https://github.com/YerevaNN/mimic3-benchmarks.git",
                "commit": BENCHMARK_COMMIT,
            },
            {
                "url": "https://github.com/ExpectationMax/medical_ts_datasets.git",
                "commit": MEDICAL_TS_COMMIT,
            },
        ],
        "mimic_release": "MIMIC-III 1.4",
        "source_protocol": "attributed benchmark split with 32-stay blacklist",
        "selected_experiment": "Table 1 KEDGN-ChatGPT MIMIC-III in-hospital mortality",
        "selected_launcher": "README MIMIC-III command; train.py dataset=mimic3",
        "selected_model_inputs": ["structured"],
        "selection_tier": 3,
        "retained_projection": ["structured"],
        "unused_modality_restriction": False,
        "common_profile": "masked_timeseries",
        "source_boundary": (
            "exact KEDGN-visible boundary after lossless equal-Hours coalescing: "
            "positive clinical values/model-mask occurrences, source-mask-derived "
            "average intervals, float32 Hours, sequence length, label, and split"
        ),
        "author_loader_boundary": (
            "KEDGN float32 values/positivity mask [B,292,32], per-variable "
            "normalized times and average intervals [B,292,16], length [B,1], "
            "and int64 mortality label"
        ),
        "fidelity_status": build["status"],
        "fidelity_limitation": build["fidelity_limitation"],
        "missing_value_behavior": {
            "source": "NaN plus explicit finite-cell Boolean mask",
            "cache_adapter": (
                "non-conflicting equal-Hours rows coalesced; NaN-to-zero with source "
                "mask retained"
            ),
            "reader_runtime_compatibility": (
                "literal GCS eye-opening category 'None' retained for the released "
                "explicit None-to-1 mapping despite current pandas default-NA behavior"
            ),
            "released_kedgn_model_mask": "raw zero-filled value > 0",
            "persistent_projection": (
                "positive value events plus their derived average-interval events; "
                "raw source-observed zero/negative values withheld"
            ),
        },
        "source_channel_order": list(variable_names()),
        "kedgn_channel_order": list(code_vocabulary()),
        "source_artifacts": {
            "source_build": sha256(cache_root / "source_build.json"),
            "source_cohort": sha256(cache_root / build["files"]["cohort"]),
            "kedgn_utils": sha256(PAPER_ROOT / "utils.py"),
            "kedgn_semantic_names": sha256(
                PAPER_ROOT / "data" / "mimic3" / "mimic3_variables_name.json"
            ),
            "kedgn_default_semantic_tensor": sha256(
                PAPER_ROOT / "data" / "mimic3" / "mimic3_bert_var_rep_gpt_source.pt"
            ),
            **{
                f"medical_ts_{name}": value
                for name, value in build["medical_ts_source_sha256"].items()
            },
        },
    }
    manifest = write_meds_task(
        output,
        paper={
            "key": "kedgn",
            "title": "Knowledge-Empowered Dynamic Graph Network for Irregularly Sampled Medical Time Series",
        },
        task={
            "id": TASK_ID,
            "name": "In-hospital mortality from the first 48 ICU hours",
            "primary_metric": "AUROC",
            "published_auroc": 0.851,
            "published_auroc_sd": 0.003,
            "observation_window_hours": HOURS,
            "positive_class": "death during the hospital admission",
            "clinical_endpoint": "in-hospital mortality from first 48 ICU hours",
            "temporal_profile": "single_visit_timed",
        },
        codes=export_code_vocabulary(),
        events=event_batches(cache_root),
        samples=task_samples(cohort),
        source=source,
        dataset_name="KEDGN README-attributed MIMIC-III mortality reconstruction",
        dataset_version="mimiciii-1.4-mimic3benchmarks-medicalts-1.0.1",
    )
    print(
        json.dumps(
            {"events": manifest["event_count"], "splits": manifest["splits"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
