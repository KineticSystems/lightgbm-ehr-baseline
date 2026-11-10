"""Export DDL-CXR's structured first-48-hour mortality projection to MEDS."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

import polars as pl


REPOSITORY = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPOSITORY.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from cohort_source import (  # noqa: E402
    CONFIG_PATH,
    EPS,
    HOURS,
    ITEMID_MAP_PATH,
    NORMALIZER_PATH,
    PHENOTYPE_PATH,
    code_vocabulary,
    configuration,
    iter_raw_rows,
    load_source_rows,
    sha256,
    variable_names,
)


def event_batches(rows: list[dict[str, object]], batch_size: int = 250_000):
    config = configuration()
    categorical = config["is_categorical_channel"]
    possible = config["possible_values"]
    variables = variable_names()
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
        for sequence_row, source_time, source_values in iter_raw_rows(sample):
            # The released Discretizer drops rows before 1e-6 hours. Do not make
            # those unused measurements available to the shared model.
            if source_time < EPS:
                continue
            event_time = sample["intime"] + timedelta(hours=source_time)
            for variable, value in zip(variables, source_values[1:]):
                if value == "":
                    continue
                if categorical[variable]:
                    rendered = str(float(value)) if not isinstance(value, str) else value
                    if rendered not in possible[variable]:
                        raise AssertionError(
                            f"unexpected released category {variable}={rendered!r}"
                        )
                    code = f"{variable}//{rendered}"
                    numeric_value = None
                else:
                    code = variable
                    numeric_value = float(value)
                    if not math.isfinite(numeric_value):
                        raise AssertionError("released structured value is non-finite")
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
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    work_root = args.work_root.resolve()
    build_report_path = work_root / "source_build.json"
    if not build_report_path.is_file():
        raise FileNotFoundError(build_report_path)
    build_report = json.loads(build_report_path.read_text())
    if build_report.get("mimic_release") != "MIMIC-IV 2.0":
        raise AssertionError("DDL-CXR requires exact MIMIC-IV 2.0")
    if build_report.get("cxr_release") != "MIMIC-CXR-JPG 2.0.0 payload":
        raise AssertionError("DDL-CXR requires the exact CXR-JPG 2.0.0 payload")
    rows = load_source_rows(work_root)
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
            "hadm_id": [int(row["hadm_id"]) for row in rows],
            "source_position": [int(row["source_position"]) for row in rows],
            "source_index": [int(row["source_index"]) for row in rows],
        },
        schema={
            "subject_id": pl.Int64,
            "sample_id": pl.Int64,
            "observation_start": pl.Datetime("us"),
            "prediction_time": pl.Datetime("us"),
            "boolean_value": pl.Boolean,
            "split": pl.String,
            "hadm_id": pl.Int64,
            "source_position": pl.Int64,
            "source_index": pl.Int64,
        },
    )
    upstream = json.loads((REPOSITORY / "UPSTREAM.json").read_text())
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "ddl_cxr",
            "title": (
                "Addressing Asynchronicity in Clinical Multimodal Fusion via "
                "Individualized Chest X-ray Generation"
            ),
        },
        task={
            "id": "mortality_48h",
            "name": "In-hospital mortality from first 48 ICU hours",
            "primary_metric": "AUROC",
            "published_auroc": 0.822,
            "observation_window_hours": HOURS,
            "positive_class": "death during the hospital admission",
        },
        codes=code_vocabulary(),
        events=event_batches(rows),
        samples=samples,
        source={
            "upstream": upstream["repositories"][0],
            "mimic_release": "MIMIC-IV 2.0",
            "cxr_release": "MIMIC-CXR-JPG 2.0.0 payload",
            "source_protocol": "single released subject split",
            "released_split_assignment_unit": "subject_id",
            "selected_experiment": "Table 1 DDL-CXR mortality AUROC",
            "selected_launcher": "configs/Prediction/Pred_fusion.yaml with task=mortality",
            "selected_model_inputs": [
                "structured",
                "last_cxr_image",
                "generated_cxr_latent",
            ],
            "common_profile": "fixed_numeric",
            "selection_tier": 4,
            "retained_projection": ["structured"],
            "unused_modality_restriction": True,
            "source_boundary": (
                "released cleaned episode timeseries before hourly discretization, "
                "restricted to rows consumed by the prediction Discretizer"
            ),
            "author_loader_boundary": "PredictDataset EHR float64 [48,79]",
            "author_preprocessing": {
                "timestep_hours": 1.0,
                "observation_hours": HOURS,
                "imputation": "previous",
                "start_time": "zero",
                "store_masks": True,
                "expanded_value_channels": 62,
                "mask_channels": 17,
                "normalized_continuous_positions": [
                    2,
                    3,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    61,
                ],
            },
            "excluded_modalities": {
                "last_cxr_image": "imaging prohibited by the common task contract",
                "generated_cxr_latent": "imaging embedding prohibited by the common task contract",
            },
            "source_artifacts": {
                "discretizer": sha256(CONFIG_PATH),
                "normalizer": sha256(NORMALIZER_PATH),
                "itemid_map": sha256(ITEMID_MAP_PATH),
                "phenotype_definitions": sha256(PHENOTYPE_PATH),
                "prediction_loader": sha256(
                    REPOSITORY / "ldm" / "data" / "dynamic_data.py"
                ),
                "prediction_config": sha256(
                    REPOSITORY / "configs" / "Prediction" / "Pred_fusion.yaml"
                ),
            },
            "normalizer_provenance": (
                "released bytes preserved; fit partition and clipping-bound "
                "provenance are not released"
            ),
            "source_build": build_report,
        },
        dataset_name="DDL-CXR exact image-matched structured mortality projection",
        dataset_version="mimiciv-2.0-ddl-cxr-0ddd5485",
    )
    print(
        json.dumps(
            {"event_count": manifest["event_count"], "splits": manifest["splits"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
