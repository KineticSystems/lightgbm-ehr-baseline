"""Export the selected released IHM source stream directly as MEDS Parquet."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl


REPOSITORY = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPOSITORY.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from cohort_source import (  # noqa: E402
    DISCRETIZER_SHA256,
    HOURS,
    NORMALIZER_SHA256,
    PRODUCER_COMMIT,
    STUDY_COMMIT,
    TASK_ID,
    TORCHMIMIC_COMMIT,
    code_vocabulary,
    configuration,
    load_source_rows,
    sha256,
    torchmimic_artifacts,
    variable_names,
)


def event_batches(rows: list[dict[str, object]], batch_size: int = 250_000):
    config = configuration()
    variables = variable_names()
    categorical = config["is_categorical_channel"]
    possible = config["possible_values"]
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

    for sample in rows:
        with Path(sample["timeseries_path"]).open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header[0] != "Hours" or set(header[1:]) != set(variables):
                raise AssertionError("unexpected released IHM timeseries header")
            for sequence_row, values in enumerate(reader):
                source_time = float(values[0])
                if not (-1e-6 < source_time < HOURS + 1e-6):
                    raise AssertionError("source task contains an out-of-window event")
                event_time = sample["intime"] + timedelta(
                    hours=min(HOURS, max(0.0, source_time))
                )
                for variable, value in zip(header[1:], values[1:]):
                    if value == "":
                        continue
                    if categorical[variable]:
                        if value not in possible[variable]:
                            raise AssertionError(
                                f"unexpected released category {variable}={value!r}"
                            )
                        code = f"{variable}//{value}"
                        numeric_value = None
                    else:
                        code = variable
                        numeric_value = float(np.float32(float(value)))
                    accumulator["subject_id"].append(int(sample["subject_id"]))
                    accumulator["time"].append(event_time)
                    accumulator["code"].append(code)
                    accumulator["numeric_value"].append(numeric_value)
                    accumulator["sample_id"].append(int(sample["stay_id"]))
                    accumulator["source_row"].append(next_source_row)
                    accumulator["sequence_row"].append(sequence_row)
                    accumulator["source_time"].append(source_time)
                    next_source_row += 1
                    if len(accumulator["sample_id"]) >= batch_size:
                        yield emit()
    if accumulator["sample_id"]:
        yield emit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    build_path = source_root / "source_build.json"
    if not build_path.is_file():
        raise FileNotFoundError(build_path)
    build = json.loads(build_path.read_text())
    if build.get("mimic_release") != "MIMIC-III 1.4":
        raise AssertionError("selected experiment requires exact MIMIC-III v1.4")
    rows = load_source_rows(source_root)
    samples = pl.DataFrame(
        {
            "subject_id": [int(row["subject_id"]) for row in rows],
            "sample_id": [int(row["stay_id"]) for row in rows],
            "observation_start": [row["intime"] for row in rows],
            "prediction_time": [
                row["intime"] + timedelta(hours=HOURS) for row in rows
            ],
            "boolean_value": [bool(row["label"]) for row in rows],
            "split": [str(row["split"]) for row in rows],
            "source_file": [str(row["source_file"]) for row in rows],
            "source_position": [int(row["source_position"]) for row in rows],
            "source_outtime": [row["outtime"] for row in rows],
        },
        schema={
            "subject_id": pl.Int64,
            "sample_id": pl.Int64,
            "observation_start": pl.Datetime("us"),
            "prediction_time": pl.Datetime("us"),
            "boolean_value": pl.Boolean,
            "split": pl.String,
            "source_file": pl.String,
            "source_position": pl.Int64,
            "source_outtime": pl.Datetime("us"),
        },
    )
    upstream = json.loads((REPOSITORY / "UPSTREAM.json").read_text())
    artifacts = torchmimic_artifacts()
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "multimodal_clinical_pretraining",
            "title": "Multimodal Pretraining of Medical Time Series and Notes",
        },
        task={
            "id": TASK_ID,
            "name": "Full-label in-hospital mortality from first 48 ICU hours",
            "primary_metric": "AUROC",
            "published_auroc": 0.856,
            "observation_window_hours": HOURS,
            "positive_class": "death during the hospitalization",
        },
        codes=code_vocabulary(),
        events=event_batches(rows),
        samples=samples,
        source={
            "upstream": upstream["repositories"][0],
            "study_commit": STUDY_COMMIT,
            "producer_commit": PRODUCER_COMMIT,
            "torchmimic_commit": TORCHMIMIC_COMMIT,
            "mimic_release": "MIMIC-III 1.4",
            "source_protocol": "single released split",
            "released_split_assignment_unit": "ICU stay with patient-disjoint lists",
            "selected_experiment": "Table 1 Ours, IHM, 100% labels",
            "selected_command": (
                "experiments/measurement_notes/measurement_notes_downstream.py "
                "--experiment full_eval --task IHM"
            ),
            "selected_model_inputs": ["structured"],
            "common_profile": "fixed_numeric",
            "selection_tier": 1,
            "retained_projection": ["structured"],
            "unused_modality_restriction": False,
            "source_boundary": (
                "released cleaned first-48-hour episode timeseries before the "
                "torchmimic one-hour discretizer"
            ),
            "author_loader_boundary": "IHMDataset.__getitem__ torch.float32 [48,76]",
            "author_preprocessing": {
                "timestep_hours": 1.0,
                "observation_hours": 48.0,
                "imputation": "previous then released normal value",
                "start_time": "zero",
                "store_masks": True,
                "value_channels": 59,
                "mask_channels": 17,
                "normalized_continuous_channels": 12,
                "batch_padding": "none for fixed 48-bin IHM",
            },
            "filename_correction": build["filename_correction"],
            "source_artifacts": {
                "author_listfiles": build["study_listfile_sha256"],
                "producer_source": build["producer_source_sha256"],
                "raw_release_manifest": build["official_manifest"],
                "raw_release_files": build["official_raw_sha256"],
                "torchmimic_discretizer": {
                    "sha256": sha256(artifacts["discretizer"]),
                    "expected_sha256": DISCRETIZER_SHA256,
                },
                "torchmimic_normalizer": {
                    "sha256": sha256(artifacts["normalizer"]),
                    "expected_sha256": NORMALIZER_SHA256,
                    "fit_scope": "producer pre-validation train pool",
                },
            },
            "source_build": build,
            "known_source_defects": [
                "dependency normalizer includes the later author validation partition",
                "study normalizer reassignment occurs after eager loading and is inert",
                "released pretraining selects checkpoints using IHM test labels",
                "released pretraining train/validation membership tests pandas indices",
                "released producer datetime rendering is incompatible with acquired names",
            ],
        },
        dataset_name="Exact released multimodal-clinical-pretraining IHM cohort",
        dataset_version="mimiciii-1.4-study-655c26a",
    )
    if sum(value["count"] for value in manifest["splits"].values()) != len(rows):
        raise AssertionError("MEDS manifest lost a released IHM row")
    print(json.dumps(manifest["splits"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
