"""Verify the exact TD-ICU cohort, labels, and released source split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "lightgbm_code" / "src"))

from export_source import meds_feature_names  # noqa: E402
from missing_xgboost_lightgbm.common_format import write_json  # noqa: E402
from missing_xgboost_lightgbm.landmark_sampling import (  # noqa: E402
    select_uniform_landmarks,
)
from cohort_source import (  # noqa: E402
    CURRENT_BUILD_PATH,
    INPUT_WINDOW_HOURS,
    LABEL_COLUMNS,
    MIMIC_RELEASE,
    PARQUET_ROOT,
    SAMPLES_PATH,
    SOURCE_BUILD_PATH,
    SOURCE_CONVERSION_PATH,
    UPSTREAM_COMMIT,
    assert_mimic_release,
    build_author_source,
    read_json,
    sha256_file,
    source_feature_names,
)


TASK_ID = "mortality_28d"
PAPER_PATIENTS = 65_050
PAPER_HOSPITAL_ADMISSIONS = 84_659
PAPER_ICU_ADMISSIONS = 93_508
PAPER_FEMALE = 28_496
HOSPITAL_DURATION_SECONDS_EXCLUSIVE = 6_660
ICU_DURATION_SECONDS_EXCLUSIVE = 7_227
ICUSTAYS_SHA256 = "fd87afb6fc6cf28a0d806b80d4377a11bbe3afd8a57a420ffeba1b6dd5ace237"


def array_hash(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def overlap(samples: pl.DataFrame) -> dict[str, int]:
    members = {
        name: set(
            samples.filter(pl.col("split") == name)["subject_id"].unique().to_list()
        )
        for name in ("train", "validation", "test")
    }
    return {
        "train_validation": len(members["train"] & members["validation"]),
        "train_test": len(members["train"] & members["test"]),
        "validation_test": len(members["validation"] & members["test"]),
    }


def selected_encounters(mimic_root: Path) -> dict:
    chart = pl.scan_parquet(PARQUET_ROOT / "chartevents.parquet").select(
        "subject_id", "hadm_id", "stay_id"
    )
    input_events = pl.scan_parquet(PARQUET_ROOT / "inputevents.parquet").select(
        "subject_id", "hadm_id", "stay_id"
    )
    events = pl.concat([chart, input_events])
    encounters = events.select(
        pl.col("subject_id").n_unique().alias("patients"),
        pl.col("hadm_id").drop_nulls().n_unique().alias("hospital_admissions"),
        pl.col("stay_id").drop_nulls().n_unique().alias("icu_admissions"),
    ).collect()
    selected_hadm = events.select("hadm_id").drop_nulls().unique().collect()
    selected_stay = events.select("stay_id").drop_nulls().unique().collect()

    admissions = (
        pl.scan_csv(
            mimic_root / "hosp" / "admissions.csv.gz",
            schema_overrides={
                "subject_id": pl.Int64,
                "hadm_id": pl.Int64,
                "admittime": pl.Datetime,
                "dischtime": pl.Datetime,
                "deathtime": pl.Datetime,
            },
            infer_schema_length=100_000,
            null_values=[""],
        )
        .select("subject_id", "hadm_id", "admittime", "dischtime", "deathtime")
        # Polars 1.7 can under-read this semi-join when both inputs remain lazy.
        .join(selected_hadm.lazy(), on="hadm_id", how="semi")
        .with_columns(
            (pl.col("dischtime") - pl.col("admittime"))
            .dt.total_seconds()
            .alias("duration_seconds")
        )
        .collect(streaming=True)
    )
    icustays_path = mimic_root / "icu" / "icustays.csv.gz"
    if not icustays_path.is_file() or sha256_file(icustays_path) != ICUSTAYS_SHA256:
        raise AssertionError("exact MIMIC-IV 3.0 icustays.csv.gz is missing or changed")
    stays = (
        pl.scan_csv(
            icustays_path,
            schema_overrides={
                "subject_id": pl.Int64,
                "hadm_id": pl.Int64,
                "stay_id": pl.Int64,
                "intime": pl.Datetime,
                "outtime": pl.Datetime,
            },
            infer_schema_length=100_000,
            null_values=[""],
        )
        .select("subject_id", "hadm_id", "stay_id", "intime", "outtime")
        .join(selected_stay.lazy(), on="stay_id", how="semi")
        .with_columns(
            (pl.col("outtime") - pl.col("intime"))
            .dt.total_seconds()
            .alias("duration_seconds")
        )
        .collect(streaming=True)
    )

    duration_hadm = admissions.filter(
        pl.col("duration_seconds").is_not_null()
        & (pl.col("duration_seconds") > HOSPITAL_DURATION_SECONDS_EXCLUSIVE)
    )
    duration_stay = stays.filter(
        pl.col("duration_seconds").is_not_null()
        & (pl.col("duration_seconds") > ICU_DURATION_SECONDS_EXCLUSIVE)
    )
    if duration_hadm["hadm_id"].n_unique() != PAPER_HOSPITAL_ADMISSIONS:
        raise AssertionError("inferred hospital-duration boundary no longer matches Table 1")
    if duration_stay["stay_id"].n_unique() != PAPER_ICU_ADMISSIONS:
        raise AssertionError("inferred ICU-duration boundary no longer matches Table 1")

    hospital_excluded = admissions.filter(
        pl.col("duration_seconds").is_null()
        | (pl.col("duration_seconds") <= HOSPITAL_DURATION_SECONDS_EXCLUSIVE)
    )
    stay_excluded = stays.filter(
        pl.col("duration_seconds").is_null()
        | (pl.col("duration_seconds") <= ICU_DURATION_SECONDS_EXCLUSIVE)
    )
    if hospital_excluded.height != 105 or stay_excluded.height != 124:
        raise AssertionError("inferred Table 1 duration exclusion cardinality changed")

    joint_event_filter = (
        events.join(duration_hadm.select("hadm_id").lazy(), on="hadm_id", how="semi")
        .join(duration_stay.select("stay_id").lazy(), on="stay_id", how="semi")
        .select(
            pl.col("subject_id").n_unique().alias("patients"),
            pl.col("hadm_id").n_unique().alias("hospital_admissions"),
            pl.col("stay_id").n_unique().alias("icu_stays"),
        )
        .collect()
        .to_dicts()[0]
    )
    if joint_event_filter != {
        "patients": 64_940,
        "hospital_admissions": 84_627,
        "icu_stays": 93_402,
    }:
        raise AssertionError("joint application of inferred encounter filters changed")
    deaths = (
        admissions
        .select(pl.col("deathtime").is_not_null().sum())
        .item()
    )
    return {
        "paper_hospital_admissions": PAPER_HOSPITAL_ADMISSIONS,
        "paper_icu_admissions": PAPER_ICU_ADMISSIONS,
        "raw_selected_patients": int(encounters["patients"].item()),
        "raw_selected_hospital_admissions": int(
            encounters["hospital_admissions"].item()
        ),
        "raw_selected_icu_stays": int(encounters["icu_admissions"].item()),
        "raw_selected_admissions_with_deathtime": int(deaths),
        "duration_reconciliation": {
            "status": "inferred_exact_duration_marginals",
            "scope": "Table 1 descriptive encounter counting; not model-cohort membership",
            "hospital": {
                "duration_seconds_exclusive": HOSPITAL_DURATION_SECONDS_EXCLUSIVE,
                "excluded": hospital_excluded.height,
                "retained": duration_hadm.height,
                "max_excluded_positive_seconds": int(
                    hospital_excluded.filter(pl.col("duration_seconds") > 0)[
                        "duration_seconds"
                    ].max()
                ),
                "min_retained_seconds": int(duration_hadm["duration_seconds"].min()),
                "retained_subjects": duration_hadm["subject_id"].n_unique(),
            },
            "icu": {
                "duration_seconds_exclusive": ICU_DURATION_SECONDS_EXCLUSIVE,
                "excluded": stay_excluded.height,
                "retained": duration_stay.height,
                "null_duration_excluded": stay_excluded["duration_seconds"].null_count(),
                "max_excluded_positive_seconds": int(
                    stay_excluded.filter(pl.col("duration_seconds") > 0)[
                        "duration_seconds"
                    ].max()
                ),
                "min_retained_seconds": int(duration_stay["duration_seconds"].min()),
                "retained_subjects": duration_stay["subject_id"].n_unique(),
                "retained_los_mean_days": float(
                    (duration_stay["duration_seconds"] / 86_400).mean()
                ),
                "retained_los_std_days": float(
                    (duration_stay["duration_seconds"] / 86_400).std()
                ),
            },
            "joint_event_filter": joint_event_filter,
            "paper_model_patients": PAPER_PATIENTS,
            "model_membership_changed": False,
            "interpretation": "the independently reconstructed encounter marginals cannot be applied jointly to events: their subject intersection is smaller than the exact 65,050-patient released model cohort",
        },
        "icustays_sha256": ICUSTAYS_SHA256,
    }


def check(mimic_root: Path, task_dir: Path) -> dict:
    release = assert_mimic_release(mimic_root)
    source_build = build_author_source(mimic_root)
    if not CURRENT_BUILD_PATH.is_file():
        raise FileNotFoundError("exact author current-boundary record is missing")
    current_build = read_json(CURRENT_BUILD_PATH)
    if sha256_file(SAMPLES_PATH) != current_build["samples_sha256"]:
        raise AssertionError("exact author final-row sample index changed")
    manifest_path = task_dir / "manifest.json"
    manifest_hash_before = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["paper"]["key"] != "td_icu_mortality":
        raise AssertionError("task belongs to another paper")
    task = manifest["task"]
    if task_dir.name != TASK_ID or task["id"] != TASK_ID:
        raise AssertionError("selected task identity changed")
    if task["primary_metric"] != "AUROC" or task["published_auroc"] != 0.839:
        raise AssertionError("selected paper AUROC cell changed")
    if task["modalities"] != ["structured"] or task["note_codes"] != []:
        raise AssertionError("TD-ICU selected task must be structured-only")
    if task["observation_window_hours"] != float(INPUT_WINDOW_HOURS):
        raise AssertionError("released seven-day input window changed")
    source = manifest["source"]
    if source["commit"] != UPSTREAM_COMMIT or source["mimic_release"] != MIMIC_RELEASE:
        raise AssertionError("source commit or exact MIMIC release changed")
    if source["configuration"] != {
        "context_length": 400,
        "input_window_hours": 168,
        "next_state_delay_hours": 24,
        "next_state_window_hours": 24,
        "split_seed": 42,
    }:
        raise AssertionError("selected released source configuration changed")

    source_samples = pl.read_parquet(SAMPLES_PATH).sort("sample_id")
    expected, sampling_audit = select_uniform_landmarks(source_samples)
    expected = expected.sort("sample_id")
    actual = pl.read_parquet(task_dir / "audit" / "task_index.parquet").sort(
        "sample_id"
    )
    if source.get("sampling") != sampling_audit:
        raise AssertionError("manifest sampling record differs from deterministic contract")
    if actual.schema != expected.schema or not actual.equals(expected):
        raise AssertionError("MEDS task index differs from deterministic landmark subsample")
    if source_samples.height != current_build["rows"]:
        raise AssertionError("full source cohort differs from exact author final rows")
    if actual["sample_id"].n_unique() != actual.height:
        raise AssertionError("rolling landmark sample_id is not unique")
    if source_samples["subject_id"].n_unique() != PAPER_PATIENTS:
        raise AssertionError("paper's 65,050-patient cohort was not reconstructed")
    unique_landmark_keys = source_samples.select(
        ["subject_id", "prediction_time", "source_patientweight"]
    ).n_unique()
    duplicate_final_rows = source_samples.height - unique_landmark_keys
    if duplicate_final_rows != current_build[
        "duplicate_final_rows_beyond_unique_landmark_keys"
    ]:
        raise AssertionError("released duplicate-landmark multiplicity changed")
    if source_samples.filter(
        pl.col("prediction_time") - pl.col("observation_start")
        != pl.duration(hours=INPUT_WINDOW_HOURS)
    ).height:
        raise AssertionError("a sample does not use the released seven-day window")

    subject_gender = source_samples.group_by("subject_id").agg(
        pl.col("source_gender").n_unique().alias("values"),
        pl.col("source_gender").first().alias("gender"),
        pl.col("source_age").first().alias("first_age"),
    )
    if subject_gender.filter(pl.col("values") != 1).height:
        raise AssertionError("source gender changes within a patient")
    female = subject_gender.filter(pl.col("gender") == 1).height
    if female != PAPER_FEMALE:
        raise AssertionError(
            f"paper female count changed: expected {PAPER_FEMALE}, found {female}"
        )

    source_overlap = overlap(source_samples)
    if any(source_overlap.values()):
        raise AssertionError("released subject-level split is no longer disjoint")
    expected_split_subjects = {"train": 52_040, "validation": 6_505, "test": 6_505}
    source_split_summary: dict[str, dict[str, int | float | str]] = {}
    exported_split_summary: dict[str, dict[str, int | float | str]] = {}
    for name in ("train", "validation", "test"):
        source_selected = source_samples.filter(pl.col("split") == name)
        if source_selected["subject_id"].n_unique() != expected_split_subjects[name]:
            raise AssertionError(f"released {name} subject split changed")
        source_split_summary[name] = {
            "examples": source_selected.height,
            "subjects": source_selected["subject_id"].n_unique(),
            "positive_28_day": int(source_selected["boolean_value"].sum()),
            "prevalence_28_day": float(source_selected["boolean_value"].mean()),
        }
        selected = actual.filter(pl.col("split") == name)
        if selected["boolean_value"].n_unique() != 2:
            raise AssertionError(f"sampled {name} split lost a binary label class")
        ids = selected["sample_id"].to_numpy().astype(np.int64)
        labels = selected["boolean_value"].to_numpy().astype(np.uint8)
        exported_split_summary[name] = {
            "examples": selected.height,
            "subjects": selected["subject_id"].n_unique(),
            "positive_28_day": int(labels.sum()),
            "prevalence_28_day": float(labels.mean()),
            "ordered_sample_id_sha256": array_hash(ids),
            "ordered_label_sha256": array_hash(labels),
        }
        if manifest["splits"][name] != {
            "count": selected.height,
            "subjects": selected["subject_id"].n_unique(),
            "positive": int(labels.sum()),
        }:
            raise AssertionError(f"manifest {name} summary changed")

    labels = pl.read_parquet(task_dir / "tasks" / TASK_ID / "labels.parquet").sort(
        "sample_id"
    )
    expected_labels = expected.select(
        "subject_id", "sample_id", "prediction_time", "boolean_value"
    )
    if labels.schema != expected_labels.schema or not labels.equals(expected_labels):
        raise AssertionError("MEDS labels differ from exact released 28-day labels")
    codes = tuple(
        pl.read_parquet(task_dir / "metadata" / "codes.parquet")["code"].to_list()
    )
    if codes != meds_feature_names(source_feature_names()):
        raise AssertionError("MEDS vocabulary differs from exact source channel order")

    encounter_comparison = selected_encounters(mimic_root)
    declared_reconciliation = source.get("table1_cohort_reconciliation")
    if declared_reconciliation is not None:
        reconstructed = encounter_comparison["duration_reconciliation"]
        expected_declaration = {
            "status": "inferred_exact_duration_marginals",
            "scope": "descriptive encounter counting only; not a filter on the released model cohort",
            "hospital_duration_seconds_exclusive": reconstructed["hospital"][
                "duration_seconds_exclusive"
            ],
            "icu_duration_seconds_exclusive": reconstructed["icu"][
                "duration_seconds_exclusive"
            ],
            "reconstructed_hospital_admissions": reconstructed["hospital"][
                "retained"
            ],
            "reconstructed_icu_admissions": reconstructed["icu"]["retained"],
            "model_task_membership_changed": False,
            "reason_membership_is_unchanged": "applying both encounter filters to model events would reduce the exact paper-matching 65,050-patient cohort and would not preserve either published encounter marginal jointly",
        }
        if declared_reconciliation != expected_declaration:
            raise AssertionError("manifest Table 1 reconciliation declaration changed")

    ages = subject_gender["first_age"]
    report = {
        "status": "exact_source_with_inferred_table1_duration_boundary",
        "paper": "td_icu_mortality",
        "task": TASK_ID,
        "selected_experiment": {
            "cell": "Table 2, TD model evaluated for 28-day mortality",
            "published_auroc": 0.839,
            "modalities": ["structured"],
            "source_release": "MIMIC-IV 3.0",
            "external_sicdb_required": False,
        },
        "source_release": release,
        "source_records": {
            "source_conversion_sha256": sha256_file(SOURCE_CONVERSION_PATH),
            "source_build_sha256": sha256_file(SOURCE_BUILD_PATH),
            "current_boundary_sha256": sha256_file(CURRENT_BUILD_PATH),
        },
        "full_source_cohort": {
            "rolling_landmarks": source_samples.height,
            "unique_patients": source_samples["subject_id"].n_unique(),
            "paper_unique_patients": PAPER_PATIENTS,
            "female_patients": female,
            "paper_female_patients": PAPER_FEMALE,
            "first_landmark_age_median": float(ages.median()),
            "first_landmark_age_q25": float(ages.quantile(0.25)),
            "first_landmark_age_q75": float(ages.quantile(0.75)),
            "landmark_key": ["subject_id", "prediction_time", "source_patientweight"],
            "source_label_rows_before_final_next_state_join": current_build[
                "source_label_rows"
            ],
            "unique_landmark_keys": unique_landmark_keys,
            "duplicate_final_rows_beyond_unique_landmark_keys": duplicate_final_rows,
            "duplicates_preserved_as_distinct_sample_ids": True,
        },
        "resource_limited_export": {
            "sampling": sampling_audit,
            "splits": exported_split_summary,
            "interpretation": "uniform unweighted landmark subsample; the paper AUROC used all released test landmarks",
        },
        "paper_admission_count_comparison": encounter_comparison,
        "full_source_splits": source_split_summary,
        "cross_split_subject_overlap": source_overlap,
        "all_five_source_labels_retained_for_audit_only": list(LABEL_COLUMNS),
        "selected_model_label": "28-day-died",
        "manifest_sha256": manifest_hash_before,
    }
    report_path = task_dir / "audit" / "cohort_reconstruction.json"
    write_json(report_path, report)
    if sha256_file(manifest_path) != manifest_hash_before:
        raise AssertionError("cohort checker changed immutable manifest.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    args = parser.parse_args()
    check(args.mimic_root.resolve(), args.task_dir.resolve())


if __name__ == "__main__":
    main()
