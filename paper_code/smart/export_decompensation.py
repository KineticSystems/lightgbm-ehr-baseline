"""Export SMART's three released-seed decompensation replicas directly to MEDS."""

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
    LOADER_PATH,
    NOTEBOOK_PATH,
    PRODUCER_COMMIT,
    READER_PATH,
    SEEDS,
    code_vocabulary,
    configuration,
    sha256,
    variable_names,
)
from cohort_decompensation import (  # noqa: E402
    NOTEBOOK_PERIOD_LENGTH,
    PUBLISHED_AUROC,
    PUBLISHED_AUROC_SD,
    PUBLISHED_CELL,
    SMART_TASK_BUILDER,
    load_build_report,
    load_cohort,
    replica_task_id,
    source_root,
    split_assignment,
)


READER_EPS = 1e-6


def event_batches(cache_root: Path, cohort: pl.DataFrame, batch_size: int = 250_000):
    """One MEDS event per observed cell of the rows SMART's reader returns.

    ``DecompensationReader._read_timeseries`` stops at the first row whose hour
    exceeds ``period_length + 1e-6``; this reproduces that bound exactly.
    """
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
        period_length = float(sample["period_length"])
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header != ["Hours", *sorted(variables)]:
                raise AssertionError(f"unexpected released SMART header: {path}")
            header_variables = header[1:]
            for sequence_row, row in enumerate(reader):
                source_time = float(row[0])
                if source_time > period_length + READER_EPS:
                    break
                if source_time < 0.0:
                    raise AssertionError("SMART source event precedes ICU admission")
                if source_time > period_length:
                    raise AssertionError(
                        "a retained source event lies after the released landmark"
                    )
                event_time = sample["intime"] + timedelta(hours=source_time)
                for variable, value in zip(header_variables, row[1:]):
                    if value == "":
                        continue
                    if categorical[variable]:
                        if value not in possible[variable]:
                            raise AssertionError(
                                f"unexpected {variable} category: {value!r}"
                            )
                        code = f"{variable}//{value}"
                        numeric_value = None
                    else:
                        numeric_value = float(value)
                        if not math.isfinite(numeric_value):
                            raise AssertionError(
                                "SMART source contains non-finite numeric data"
                            )
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
    if (assigned["period_length"] % 1 != 0).any():
        raise AssertionError("released decompensation landmark is not an integer hour")
    if (assigned["period_length"] <= 0).any():
        raise AssertionError(
            "a released decompensation landmark is at or before ICU admission"
        )
    return assigned.select(
        "subject_id",
        "sample_id",
        pl.col("intime").alias("observation_start"),
        (
            pl.col("intime")
            + pl.duration(hours=pl.col("period_length").cast(pl.Int64))
        ).alias("prediction_time"),
        pl.col("label").alias("boolean_value"),
        "split",
        "hadm_id",
        "source_name",
        "source_partition",
        "source_position",
        pl.col("period_length").alias("released_period_length_hours"),
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
            "selected_experiment": "Appendix Table 3 SMART decompensation AUROC",
            "selected_launcher": "run.sh mimic_decompensation branch with seeds 1, 42, and 3407",
            "selected_model_inputs": ["structured"],
            "selection_tier": 1,
            "retained_projection": ["structured"],
            "unused_modality_restriction": False,
            "common_profile": "repeated_split",
            "phase": "phase 2 second published binary outcome",
            "locally_replaced_task_builder": {
                "file": "data/MIMIC-III/create_decompensation.py",
                "sha256": sha256(SMART_TASK_BUILDER),
                "readme_instruction": (
                    "SMART README: 'we adopt different settings when generating "
                    "decompensation and length-of-stay data. Please replace the "
                    "original mimic3benchmark/scripts/create_decompensation.py and "
                    "mimic3benchmark/scripts/create_length_of_stay.py with the "
                    "scripts in the data/MIMIC-III folder.'"
                ),
                "difference_from_producer": (
                    "the producer emits one hourly landmark per eligible stay-hour; "
                    "SMART emits exactly one example per eligible stay at "
                    "t = min(24, int(los), int(lived_time)) and labels it "
                    "mortality and (lived_time - t < 24)"
                ),
            },
            "source_boundary": (
                "released 17-variable decompensation episode CSV rows up to the "
                "released landmark, before SMART hourly binning and pooled normalization"
            ),
            "author_loader_boundary": (
                "collate_fn float32 value/mask [B,<=24,17] with pad_sequence padding, "
                "int64 labels/lens"
            ),
            "fidelity_status": report["status"],
            "fidelity_limitation": report["fidelity_limitation"],
            "ordering_convention": report["ordering_convention"],
            "author_preprocessing": {
                "notebook_period_length": NOTEBOOK_PERIOD_LENGTH,
                "bins": "N_bins = min(int(t + 1 - 1e-6), 24)",
                "channels": 17,
                "within_bin_aggregation": "mean of the observed values in the bin",
                "unobserved_fill": "released discretizer normal_values (no carry-forward)",
                "categorical_encoding": "released scalar channel_info mapping",
                "normalization_fit": "all 41,744 pooled examples, every cell",
                "normalization_applied_to": "every cell, observed or not",
            },
            "released_defects_preserved": [
                "outcome-derived prediction landmark and sequence length: "
                "t = min(24, int(los), int(lived_time)) is set by the realized "
                "death or discharge time",
                "example-level 80/10/10 split redrawn per seed over the pooled "
                "benchmark partitions, so patients cross split boundaries",
                "normalization moments fitted on the full pooled cohort "
                "including validation and test",
                "eligibility requires a second event by the landmark "
                "(len(event_times) >= 2 and event_times[1] <= t)",
            ],
            "source_artifacts": {
                "smart_task_builder": sha256(SMART_TASK_BUILDER),
                "smart_notebook": sha256(NOTEBOOK_PATH),
                "smart_reader": sha256(READER_PATH),
                "smart_loader": sha256(LOADER_PATH),
                "smart_collate": sha256(COLLATE_PATH),
                "channel_info": sha256(CHANNEL_INFO_PATH),
                "discretizer_config": sha256(CONFIG_PATH),
                "source_build": sha256(cache_root / "decompensation_build.json"),
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
                "name": f"Decompensation at the released landmark (SMART seed {seed})",
                "primary_metric": "AUROC",
                "published_auroc": PUBLISHED_AUROC,
                "published_auroc_sd": PUBLISHED_AUROC_SD,
                "published_source": PUBLISHED_CELL,
                "observation_window_hours": float(NOTEBOOK_PERIOD_LENGTH),
                "positive_class": (
                    "in-hospital death occurring within 24 hours after the released "
                    "prediction landmark"
                ),
                "clinical_endpoint": "decompensation (death within 24h) at the released landmark",
                "label_definition": (
                    "SMART data/MIMIC-III/create_decompensation.py: for each eligible "
                    "stay t = min(24, int(los), int(lived_time)); y = 0 when the episode "
                    "Mortality flag is 0, else int(lived_time - t < 24). Eligible means "
                    "a non-empty label file, a known length of stay, at least one in-ICU "
                    "event, and at least two events with the second at or before t. The "
                    "model is shown the episode rows up to t only."
                ),
                "source_task": "decompensation",
                "released_landmark_rule": "t = min(24, int(los), int(lived_time))",
                "replica_seed": seed,
            },
            codes=code_vocabulary(),
            events=event_batches(cache_root, cohort),
            samples=samples_for_replica(cohort, seed),
            source=source,
            dataset_name="SMART exact-source MIMIC-III decompensation reconstruction",
            dataset_version=f"mimiciii-1.4-smart-decomp-{seed}",
        )
        print(
            json.dumps(
                {
                    "task": task_id,
                    "events": manifest["event_count"],
                    "splits": manifest["splits"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
