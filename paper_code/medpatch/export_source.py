"""Export MedPatch's released first-48-hour structured mortality task."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import timedelta
from pathlib import Path

import polars as pl


NOTE_CODE = "MEDPATCH//RADIOLOGY_REPORT"

REPOSITORY = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPOSITORY.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from missing_xgboost_lightgbm.polars_meds import write_meds_task  # noqa: E402
from radiology_notes import radiology_text_by_stay, radiology_source_path  # noqa: E402
from cohort_source import (  # noqa: E402
    CONFIG_PATH,
    HOURS,
    ITEMID_MAP_PATH,
    NORMALIZER_PATH,
    TEST_MEMBERSHIP_PATH,
    VALIDATION_MEMBERSHIP_PATH,
    code_vocabulary,
    configuration,
    load_source_rows,
    sha256,
    variable_names,
)


def event_batches(
    rows: list[dict[str, object]],
    radiology_text: dict[int, str],
    batch_size: int = 250_000,
):
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
        "text_value": [],
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
                "text_value": pl.String,
            },
        )
        accumulator = {name: [] for name in accumulator}
        return frame

    for sample in rows:
        with Path(sample["timeseries_path"]).open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header[0] != "Hours" or set(header[1:]) != set(variables):
                raise AssertionError("unexpected released mortality timeseries header")
            for sequence_row, values in enumerate(reader):
                source_time = float(values[0])
                if not (-1e-6 < source_time < HOURS + 1e-6):
                    raise AssertionError("source task contains an out-of-window event")
                event_time = sample["intime"] + timedelta(
                    hours=min(HOURS, max(0.0, source_time))
                )
                for column, value in zip(header[1:], values[1:]):
                    if value == "":
                        continue
                    if categorical[column]:
                        if value not in possible[column]:
                            raise AssertionError(
                                f"unexpected category {column}={value!r}"
                            )
                        code = f"{column}//{value}"
                        numeric_value = None
                    else:
                        code = column
                        numeric_value = float(value)
                    accumulator["subject_id"].append(int(sample["subject_id"]))
                    accumulator["time"].append(event_time)
                    accumulator["code"].append(code)
                    accumulator["numeric_value"].append(numeric_value)
                    accumulator["sample_id"].append(int(sample["stay_id"]))
                    accumulator["source_row"].append(next_source_row)
                    accumulator["sequence_row"].append(sequence_row)
                    accumulator["source_time"].append(source_time)
                    accumulator["text_value"].append(None)
                    next_source_row += 1
                    if len(accumulator["sample_id"]) >= batch_size:
                        yield emit()
        # The released BioBERT branch receives one space-joined string per
        # stay, so the note channel is one MEDS event rather than one event
        # per report. A stay with no in-window report emits nothing and
        # becomes an empty document, which is the released left-join
        # behaviour and not a cohort filter.
        note_text = radiology_text.get(int(sample["stay_id"]))
        if note_text:
            accumulator["subject_id"].append(int(sample["subject_id"]))
            accumulator["time"].append(
                sample["intime"] + timedelta(hours=HOURS) - timedelta(microseconds=1)
            )
            accumulator["code"].append(NOTE_CODE)
            accumulator["numeric_value"].append(None)
            accumulator["sample_id"].append(int(sample["stay_id"]))
            accumulator["source_row"].append(next_source_row)
            accumulator["sequence_row"].append(0)
            accumulator["source_time"].append(float(HOURS))
            accumulator["text_value"].append(note_text)
            next_source_row += 1
            if len(accumulator["sample_id"]) >= batch_size:
                yield emit()
    if accumulator["sample_id"]:
        yield emit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--note-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    build_report_path = source_root.parent / "source_build.json"
    if not build_report_path.is_file():
        raise FileNotFoundError(build_report_path)
    build_report = json.loads(build_report_path.read_text())
    if build_report.get("mimic_release") != "MIMIC-IV 1.0":
        raise AssertionError("MedPatch requires exact MIMIC-IV 1.0")
    rows = load_source_rows(source_root)
    radiology_text = radiology_text_by_stay(
        source_root.parent / "subjects" / "all_stays.csv", args.note_root
    )
    covered = sum(
        1 for row in rows if radiology_text.get(int(row["stay_id"]))
    )
    print(
        f"radiology reports attached to {covered} of {len(rows)} samples "
        f"({100.0 * covered / len(rows):.1f}%)"
    )
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
            "source_file": [str(row["source_file"]) for row in rows],
            "source_position": [int(row["source_position"]) for row in rows],
        },
        schema={
            "subject_id": pl.Int64,
            "sample_id": pl.Int64,
            "observation_start": pl.Datetime("us"),
            "prediction_time": pl.Datetime("us"),
            "boolean_value": pl.Boolean,
            "split": pl.String,
            "hadm_id": pl.Int64,
            "source_file": pl.String,
            "source_position": pl.Int64,
        },
    )
    upstream = json.loads((REPOSITORY / "UPSTREAM.json").read_text())
    manifest = write_meds_task(
        args.output_dir,
        paper={
            "key": "medpatch",
            "title": "MedPatch: Towards Improving Clinical Predictive Modeling under Missingness",
        },
        task={
            "id": "mortality_48h",
            "name": "In-hospital mortality from first 48 ICU hours",
            "primary_metric": "AUROC",
            "published_auroc": 0.876,
            "observation_window_hours": HOURS,
            "positive_class": "death during the hospital admission",
        },
        codes=code_vocabulary() + (NOTE_CODE,),
        note_codes=(NOTE_CODE,),
        events=event_batches(rows, radiology_text),
        samples=samples,
        source={
            "upstream": upstream["repositories"][0],
            "mimic_release": "MIMIC-IV 1.0",
            "source_protocol": "single released patient split",
            "released_split_assignment_unit": "subject_id",
            "selected_experiment": (
                "Table 3 trimodal EHR+CXR+RR mortality (MedPatch)"
            ),
            "selected_launcher": (
                "scripts/mortality/MedPatch/Confidence-Patching.sh"
            ),
            "selected_model_inputs": ["structured", "notes", "imaging"],
            "common_profile": "fixed_numeric_notes",
            "selection_tier": 2,
            "retained_projection": ["structured", "notes"],
            "omitted_model_inputs": ["chest radiograph pixels"],
            "unused_modality_restriction": False,
            "note_channel": {
                "source": "mimic-iv-note 2.2 radiology.csv",
                "source_sha256": sha256(
                    radiology_source_path(args.note_root)
                ),
                "released_construction": (
                    "datasets/DataFusion.py loadmetadata, data_pairs=partial: "
                    "inner merge on (subject_id, hadm_id), charttime within "
                    "[intime, intime+48h], groupby then ' '.join(text), then "
                    "left join onto every EHR stay"
                ),
                "samples_with_report": covered,
                "join_separator": " ",
            },
            "source_boundary": "released cleaned episode timeseries before one-hour discretization",
            "author_loader_boundary": "EHRdataset.__getitem__ float64 [48,76]",
            "author_preprocessing": {
                "timestep_hours": 1.0,
                "observation_hours": 48.0,
                "imputation": "previous",
                "start_time": "zero",
                "store_masks": True,
                "value_channels": 59,
                "mask_channels": 17,
                "normalized_continuous_channels": 12,
            },
            "extractor_instrumentation": {
                "file": "mimic4extract/mimic3benchmark/mimic3csv.py",
                "operation": (
                    "retained-row counter and one summary print consumed by "
                    "check_cohort.py; the released identifier filtering is "
                    "unmodified. An earlier export 'identifier type repair' "
                    "was reverted after audit: upstream already casts "
                    "items_to_keep and subjects_to_keep to strings before the "
                    "string-valued DictReader comparison, so the repair was a "
                    "verified no-op and is no longer applied."
                ),
                "sha256": build_report["source_files"][
                    "identifier_repaired_extractor"
                ],
            },
            "source_artifacts": {
                "discretizer": sha256(CONFIG_PATH),
                "normalizer": sha256(NORMALIZER_PATH),
                "itemid_map": sha256(ITEMID_MAP_PATH),
                "test_membership": sha256(TEST_MEMBERSHIP_PATH),
                "validation_membership": sha256(VALIDATION_MEMBERSHIP_PATH),
                "selected_launcher": sha256(
                    REPOSITORY / "scripts" / "mortality" / "Unimodal" / "EHR.sh"
                ),
                "metadata_loader": sha256(REPOSITORY / "datasets" / "DataFusion.py"),
                "ehr_loader": sha256(REPOSITORY / "datasets" / "ehr_dataset.py"),
                "ehr_preprocessing": sha256(
                    REPOSITORY / "ehr_utils" / "preprocessing.py"
                ),
                "raw_manifest": build_report["raw_manifest_sha256"],
            },
            "source_build": build_report,
        },
        dataset_name="MedPatch exact structured mortality cohort",
        dataset_version="mimiciv-1.0-medpatch-f33c5ed6",
    )
    if sum(item["count"] for item in manifest["splits"].values()) != len(rows):
        raise AssertionError("manifest lost a source sample")
    print(json.dumps(manifest["splits"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
