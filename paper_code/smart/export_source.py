"""Export SMART's three released-seed mortality replicas directly to MEDS."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

import polars as pl


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from cohort_source import (  # noqa: E402
    CHANNEL_INFO_PATH,
    COLLATE_PATH,
    CONFIG_PATH,
    HOURS,
    LOADER_PATH,
    NOTEBOOK_PATH,
    PRODUCER_COMMIT,
    READER_PATH,
    SEEDS,
    code_vocabulary,
    configuration,
    load_build_report,
    load_cohort,
    replica_task_id,
    sha256,
    source_root,
    split_assignment,
    variable_names,
)


def event_batches(cache_root: Path, cohort: pl.DataFrame, batch_size: int = 250_000):
    config = configuration()
    variables = variable_names()
    categorical = config["is_categorical_channel"]
    possible = config["possible_values"]
    task_root = source_root(cache_root)
    accumulator = {
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
        nonlocal accumulator
        frame = pl.DataFrame(
            accumulator,
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
        accumulator = {name: [] for name in accumulator}
        return frame

    for sample in cohort.iter_rows(named=True):
        partition = str(sample["source_partition"])
        path = task_root / partition / str(sample["source_name"])
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header != ["Hours", *sorted(variables)]:
                raise AssertionError(f"unexpected released SMART header: {path}")
            header_variables = header[1:]
            for sequence_row, row in enumerate(reader):
                source_time = float(row[0])
                if not (-1e-6 < source_time < HOURS + 1e-6):
                    raise AssertionError("SMART source event lies outside first 48 hours")
                event_time = sample["intime"] + timedelta(hours=source_time)
                for variable, value in zip(header_variables, row[1:]):
                    if value == "":
                        continue
                    if categorical[variable]:
                        if value not in possible[variable]:
                            raise AssertionError(f"unexpected {variable} category: {value!r}")
                        code = f"{variable}//{value}"
                        numeric_value = None
                    else:
                        numeric_value = float(value)
                        if not math.isfinite(numeric_value):
                            raise AssertionError("SMART source contains non-finite numeric data")
                        code = variable
                    accumulator["subject_id"].append(int(sample["subject_id"]))
                    accumulator["time"].append(event_time)
                    accumulator["code"].append(code)
                    accumulator["numeric_value"].append(numeric_value)
                    accumulator["sample_id"].append(int(sample["sample_id"]))
                    accumulator["source_row"].append(next_source_row)
                    accumulator["sequence_row"].append(sequence_row)
                    accumulator["source_time"].append(source_time)
                    next_source_row += 1
                    if len(accumulator["sample_id"]) >= batch_size:
                        yield emit()
    if accumulator["sample_id"]:
        yield emit()


def samples_for_replica(cohort: pl.DataFrame, seed: int) -> pl.DataFrame:
    assigned = split_assignment(cohort, seed)
    return assigned.select(
        "subject_id",
        "sample_id",
        pl.col("intime").alias("observation_start"),
        (pl.col("intime") + pl.duration(hours=HOURS)).alias("prediction_time"),
        pl.col("label").alias("boolean_value"),
        "split",
        "hadm_id",
        "source_name",
        "source_partition",
        "source_position",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    report = load_build_report(cache_root)
    cohort = load_cohort(cache_root)
    upstream = json.loads((PAPER_ROOT / "UPSTREAM.json").read_text())["repositories"][0]

    for replica, seed in enumerate(SEEDS, start=1):
        task_id = replica_task_id(replica)
        output = args.output_root.resolve() / task_id
        source = {
            "upstream": upstream,
            "producer": {
                "url": "https://github.com/YerevaNN/mimic3-benchmarks.git",
                "commit": PRODUCER_COMMIT,
            },
            "mimic_release": "MIMIC-III 1.4",
            "source_protocol": "three source seed-specific split replicas",
            "replica": replica,
            "paper_seed": seed,
            "released_split_assignment_unit": "example/sample_id",
            "selected_experiment": "Appendix Table 3 SMART in-hospital mortality AUROC",
            "selected_launcher": "run.sh mortality branch with seeds 1, 42, and 3407",
            "selected_model_inputs": ["structured"],
            "selection_tier": 1,
            "retained_projection": ["structured"],
            "unused_modality_restriction": False,
            "common_profile": "repeated_split",
            "source_boundary": "released 17-variable IHM episode CSVs before SMART hourly alignment and pooled normalization",
            "author_loader_boundary": "collate_fn float32 value/mask [B,48,17], int64 labels/lens",
            "fidelity_status": report["status"],
            "fidelity_limitation": report["fidelity_limitation"],
            "ordering_convention": report["ordering_convention"],
            "author_preprocessing": {
                "hours": 48,
                "channels": 17,
                "categorical_encoding": "released scalar channel_info mapping",
                "normalization_fit": "all 21,139 pooled examples and only observed cells",
                "mask_zero_normalized_value": 0,
            },
            "source_artifacts": {
                "smart_notebook": sha256(NOTEBOOK_PATH),
                "smart_reader": sha256(READER_PATH),
                "smart_loader": sha256(LOADER_PATH),
                "smart_collate": sha256(COLLATE_PATH),
                "channel_info": sha256(CHANNEL_INFO_PATH),
                "discretizer_config": sha256(CONFIG_PATH),
                "source_build": sha256(cache_root / "source_build.json"),
                "source_cohort": sha256(cache_root / report["files"]["cohort"]),
            },
        }
        manifest = write_meds_task(
            output,
            paper={
                "key": "smart",
                "title": "SMART: Towards Pre-trained Missing-Aware Model for Patient Health Status Prediction",
            },
            task={
                "id": task_id,
                "name": f"In-hospital mortality from first 48 ICU hours (SMART seed {seed})",
                "primary_metric": "AUROC",
                "published_auroc": 0.8561,
                "published_auroc_sd": 0.0062,
                "observation_window_hours": HOURS,
                "positive_class": "death during the hospital admission",
                "clinical_endpoint": "in-hospital mortality from first 48 ICU hours",
                "replica_seed": seed,
            },
            codes=code_vocabulary(),
            events=event_batches(cache_root, cohort),
            samples=samples_for_replica(cohort, seed),
            source=source,
            dataset_name="SMART exact-source MIMIC-III mortality reconstruction",
            dataset_version=f"mimiciii-1.4-smart-{seed}",
        )
        print(
            json.dumps(
                {"task": task_id, "events": manifest["event_count"], "splits": manifest["splits"]},
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
