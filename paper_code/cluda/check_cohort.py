"""Assert CLUDA's paper-specific MIMIC-IV age-domain cohort reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from cohort_source import (  # noqa: E402
    AGE_BOUNDS,
    AGE_MEAN,
    AGE_SD,
    CODES,
    MIMIC_VERSION,
    MIN_HOURS,
    PAPER_LOADER_COUNTS,
    PAPER_MORTALITY_PERCENT,
    PAPER_PREFILTER_COUNTS,
    PAPER_SAMPLE_COUNTS,
    PAPER_TOTAL_PREFILTER,
    PUBLISHED_AUROC,
    SOURCE_AGE_GROUP,
    STATIC_FEATURES,
    TARGET_AGE_GROUP,
    TASK_ID,
    TS_FEATURES,
    iter_sample_rows,
    load_author_source,
    require_source,
    selected_splits,
    sha256,
    source_files,
    source_summary,
    validate_age_scaling,
    validate_upstream_files,
    write_json,
)


def text_hash(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def compare_index(index: pd.DataFrame, selected) -> pd.DataFrame:
    expected = pd.DataFrame(list(iter_sample_rows(selected)))
    left = index.sort_values("sample_id").reset_index(drop=True)
    right = expected.sort_values("sample_id").reset_index(drop=True)
    fields = (
        "subject_id",
        "sample_id",
        "observation_start",
        "prediction_time",
        "boolean_value",
        "split",
        "stay_id",
        "stay_hour",
        "loader_patient_id",
        "source_original_split",
        "loader_age_group",
        "paper_age_group",
    )
    if len(left) != len(right):
        raise AssertionError("MEDS task changed the selected CLUDA sample count")
    for field in fields:
        actual = left[field].to_numpy()
        wanted = right[field].to_numpy()
        if field in {"observation_start", "prediction_time"}:
            actual = pd.to_datetime(actual).to_numpy(dtype="datetime64[us]")
            wanted = pd.to_datetime(wanted).to_numpy(dtype="datetime64[us]")
        if not np.array_equal(actual, wanted):
            raise AssertionError(f"MEDS changed exact CLUDA cohort field: {field}")
    return left


def check_event_projection(task_dir: Path, manifest: dict, index: pd.DataFrame) -> dict:
    declared = pq.read_table(task_dir / manifest["meds"]["codes"]["file"])["code"].to_pylist()
    if declared != list(CODES):
        raise AssertionError("MEDS code vocabulary/order differs from CLUDA")
    if manifest["task"]["modalities"] != ["structured"] or manifest["task"]["note_codes"]:
        raise AssertionError("CLUDA is a structured-only selected experiment")
    if any("image" in code.lower() or "note" in code.lower() for code in declared):
        raise AssertionError("an undeclared modality entered CLUDA's task")

    ordered_ids = np.sort(index["sample_id"].to_numpy(np.int64))
    static_counts = np.zeros(len(ordered_ids), dtype=np.int16)
    observed_codes: set[str] = set()
    event_count = 0
    events_path = task_dir / manifest["meds"]["events"]["file"]
    for batch in pq.ParquetFile(events_path).iter_batches(
        columns=["sample_id", "code"], batch_size=250_000
    ):
        ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        codes = batch.column(1).to_pylist()
        positions = np.searchsorted(ordered_ids, ids)
        if (positions == len(ordered_ids)).any() or not np.array_equal(ordered_ids[positions], ids):
            raise AssertionError("an event references an unselected CLUDA sample")
        is_static = np.fromiter((value.startswith("STATIC/") for value in codes), bool, len(codes))
        np.add.at(static_counts, positions[is_static], 1)
        observed_codes.update(codes)
        event_count += len(codes)
    if not np.all(static_counts == len(STATIC_FEATURES)):
        raise AssertionError("every CLUDA landmark must retain its exact eight static inputs")
    if event_count != manifest["event_count"]:
        raise AssertionError("MEDS event count differs from its immutable manifest")
    if not observed_codes.issubset(CODES):
        raise AssertionError("MEDS contains an undeclared CLUDA code")
    return {
        "declared_codes": len(declared),
        "observed_codes": len(observed_codes),
        "event_count": event_count,
        "static_events_per_sample": len(STATIC_FEATURES),
        "all_samples_have_exact_static_projection": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--allow-synthetic", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    source_manifest = require_source(args.source_root, allow_synthetic=args.allow_synthetic)
    upstream_files = validate_upstream_files()
    age_scaling = validate_age_scaling(
        args.source_root, allow_synthetic=args.allow_synthetic
    )
    source = load_author_source(args.source_root, allow_synthetic=args.allow_synthetic)
    summary = source_summary(source)
    if not args.allow_synthetic:
        if source_manifest["dataset_version"] != MIMIC_VERSION:
            raise AssertionError("the selected source is not exact MIMIC-IV v1.0")
        if summary["prefilter_counts"] != PAPER_PREFILTER_COUNTS:
            raise AssertionError(f"executed notebook prefilter counts differ: {summary['prefilter_counts']}")
        if summary["loader_counts"] != PAPER_LOADER_COUNTS:
            raise AssertionError(f"paper loader patient counts differ: {summary['loader_counts']}")
        if summary["sample_counts"] != PAPER_SAMPLE_COUNTS:
            raise AssertionError(f"paper rolling-sample counts differ: {summary['sample_counts']}")
        if summary["total_prefilter"] != PAPER_TOTAL_PREFILTER:
            raise AssertionError("paper's 49,351-stay source cohort differs")
        admissions = pd.read_csv(source_files(args.source_root)["admissions"])
        mortality_percent = 100.0 * admissions["hospital_expire_flag"].mean()
        if round(mortality_percent, 2) != PAPER_MORTALITY_PERCENT:
            raise AssertionError(f"paper mortality marginal differs: {mortality_percent}")
    else:
        mortality_percent = float("nan")

    manifest_path = args.task_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_manifest_sha256 = sha256(args.source_root / "source_manifest.json")
    if manifest["source"].get("source_manifest_sha256") != source_manifest_sha256:
        raise AssertionError("MEDS task is not sealed to the exact CLUDA source manifest")
    if manifest["source"].get("source_reconstruction_status") != source_manifest["status"]:
        raise AssertionError("MEDS task changed the CLUDA source reconstruction status")
    if manifest["task"]["id"] != TASK_ID or manifest["task"]["published_auroc"] != PUBLISHED_AUROC:
        raise AssertionError("wrong CLUDA task or published Table 1 cell")
    if manifest["task"].get("positive_class") != (
        "hospital_expire_flag == 1 (death during the hospital admission)"
    ):
        raise AssertionError("CLUDA mortality positive-class definition changed")
    if not args.allow_synthetic and manifest["source"].get("upstream_file_sha256") != upstream_files:
        raise AssertionError("MEDS task is not sealed to the executed CLUDA upstream files")
    configuration = manifest["source"]["selected_source_configuration"]
    expected_configuration = {
        "task": "mortality",
        "path_src": "miiv_fullstays",
        "path_trg": "miiv_fullstays",
        "age_src": SOURCE_AGE_GROUP,
        "age_trg": TARGET_AGE_GROUP,
        "use_mask": True,
        "min_hours": 4,
        "history_hours": 48,
    }
    if configuration != expected_configuration:
        raise AssertionError("selected CLUDA source configuration changed")

    selected = selected_splits(source)
    source_clock_offsets = {}
    for split_name, arrays in selected.items():
        starts = np.asarray(
            [int(np.asarray(value)[0]) for value in arrays.source_charttime],
            dtype=np.int64,
        )
        affected = starts != 0
        source_clock_offsets[split_name] = {
            "affected_stays": int(affected.sum()),
            "affected_samples": int(
                np.maximum(arrays.lengths[affected] - MIN_HOURS + 1, 0).sum()
            ),
            "minimum_start_hour": int(starts.min()),
            "maximum_start_hour": int(starts.max()),
        }
    index = pq.read_table(
        args.task_dir / manifest["meds"]["task_index"]["file"]
    ).to_pandas()
    index = compare_index(index, selected)
    counts = index["split"].value_counts().to_dict()
    positives = index.groupby("split")["boolean_value"].sum().astype(int).to_dict()
    subjects = index.groupby("split")["subject_id"].nunique().to_dict()
    loader_age_groups = index.groupby("split")["loader_age_group"].unique().to_dict()
    paper_age_groups = index.groupby("split")["paper_age_group"].unique().to_dict()
    if loader_age_groups["train"].tolist() != [SOURCE_AGE_GROUP] or loader_age_groups["validation"].tolist() != [SOURCE_AGE_GROUP]:
        raise AssertionError("source age-domain membership changed")
    if loader_age_groups["test"].tolist() != [TARGET_AGE_GROUP]:
        raise AssertionError("target age-domain membership changed")
    if paper_age_groups["train"].tolist() != [1] or paper_age_groups["validation"].tolist() != [1]:
        raise AssertionError("source paper-domain label changed")
    if paper_age_groups["test"].tolist() != [2]:
        raise AssertionError("target paper-domain label changed")

    for split_name, arrays in selected.items():
        ages = arrays.static[:, 0] * AGE_SD + AGE_MEAN
        group = SOURCE_AGE_GROUP if split_name != "test" else TARGET_AGE_GROUP
        lower, upper = AGE_BOUNDS[group]
        if not np.logical_and(ages > lower, ages <= upper).all():
            raise AssertionError(f"{split_name} contains a patient outside age group {group}")

    subject_sets = {
        split: set(index.loc[index["split"] == split, "subject_id"].astype(int))
        for split in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": len(subject_sets["train"] & subject_sets["validation"]),
        "train_test": len(subject_sets["train"] & subject_sets["test"]),
        "validation_test": len(subject_sets["validation"] & subject_sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"released first-stay subject splits overlap: {overlaps}")
    event_projection = check_event_projection(args.task_dir, manifest, index)

    report = {
        "status": (
            "synthetic_exact_loader_projection" if args.allow_synthetic else
            "exact_source_with_paper_artifact_triangulated_split"
        ),
        "selected_experiment": {
            "paper_cell": "Main Table 1, Group 1 to Group 2",
            "task": "MIMIC-IV age-domain-shift mortality",
            "selection_tier": 3,
            "published_auroc": PUBLISHED_AUROC,
            "configuration": expected_configuration,
            "supervised_projection": manifest["source"]["supervised_projection"],
        },
        "paper_full_cohort": {
            "prefilter_stays": summary["prefilter_counts"],
            "loader_patients": summary["loader_counts"],
            "rolling_samples": summary["sample_counts"],
            "mortality_percent": mortality_percent,
        },
        "selected_splits": {
            split: {
                "samples": counts[split],
                "subjects": subjects[split],
                "positive": positives[split],
                "prevalence": positives[split] / counts[split],
                "paper_age_group": 1 if split != "test" else 2,
                "loader_age_group": SOURCE_AGE_GROUP if split != "test" else TARGET_AGE_GROUP,
            }
            for split in ("train", "validation", "test")
        },
        "cross_split_subject_overlap": overlaps,
        "absolute_source_clock_offsets": source_clock_offsets,
        "retained_projection": {
            "modalities": ["structured"],
            "time_series_features": list(TS_FEATURES),
            "static_features": list(STATIC_FEATURES),
            "event_projection": event_projection,
        },
        "ordered_sha256": {
            "sample_id": hashlib.sha256(index["sample_id"].to_numpy(np.int64).tobytes()).hexdigest(),
            "subject_id": hashlib.sha256(index["subject_id"].to_numpy(np.int64).tobytes()).hexdigest(),
            "label": hashlib.sha256(index["boolean_value"].to_numpy(bool).tobytes()).hexdigest(),
            "split": text_hash(index["split"]),
        },
        "source_manifest_sha256": source_manifest_sha256,
        "upstream_file_sha256": upstream_files,
        "age_scaling_reconstruction": age_scaling,
        "fidelity_qualifications": {
            "ricu": "released code did not pin RICU; the export uses time-compatible RICU 0.5.2",
            "split_rng": (
                "the released script reaches a 37-name-draw pre-split RNG state, which does not "
                "match the paper's downstream counts or released age-scaling constants; the "
                "paper-evaluated signatures jointly identify the 72-draw state, exactly one "
                "additional 35-draw released time-series loader invocation. The export applies "
                "that fail-closed transition immediately before createDataPartition"
            ),
            "static_tail": (
                "identifier-renamed released AUMC static tail, accepted only because "
                "the executed MIMIC training age scaling matches the released loader constants"
            ),
            "dataset_version": (
                "v1.0 selected from RICU 0.5.2 executable configuration and the "
                "paper's 49,351-stay signature; README v0.4 link is stale"
            ),
        },
        "manifest_sha256": sha256(manifest_path),
        "checks": [
            "exact paper-artifact-triangulated mortality-stratified split and notebook shuffle",
            "exact source Group 1 train/validation and target Group 2 test membership",
            "all endpoints from hour 4 through each released sequence end",
            "paper full-cohort patient/sample counts and mortality marginal",
            "exact 41-channel dynamic and eight-field static declarations",
            "zero cross-split subject overlap from the first-stay cohort",
            "executed producer/loader and feature/static-tail evidence match the pinned upstream commit files",
            "training age mean/SD match the released loader constants within printed-precision tolerance",
            "absolute RICU charttime retained while loader slicing remains sequence-relative",
        ],
    }
    output = args.task_dir / "audit" / "cohort_reconstruction.json"
    write_json(output, report)
    print(json.dumps({"status": report["status"], "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
