"""Released MIMIC-III 30-day readmission labels for the DPMM partial cohort.

The DPMM readmission experiment reuses the exact benchmark extraction of the
selected in-hospital-mortality task: `create_readmission_30d.py` applies the
same 48-hour `n_hours` window, the same `los < 48h` exclusion, the same
"no events in ICU" exclusion, and the same `<patient>_<episode>_timeseries.csv`
naming as `create_in_hospital_mortality.py`.  Only the label differs, so this
module reconstructs the released label producer and reuses the already
validated `cohort_source` event/note cache untouched.

The released labeler carries an index-misuse defect (audited as "readmission
labels are scrambled").  It is reproduced verbatim here, together with the two
control variants that measure it on the real cohort.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_ROOT = Path(__file__).resolve().parent
MIMIC3EXTRACT = PAPER_ROOT / "mimic3extract"
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(MIMIC3EXTRACT))

from mimic3benchmark import mimic3csv  # noqa: E402

from cohort_source import (  # noqa: E402
    HOURS,
    UPSTREAM_COMMIT,
    _compressed_dataframe,
    array_hash,
    assert_mimic_release,
    make_discretizer_normalizer,
    sha256,
)
from dataset_mimic3.ehr_dataset import EHRdataset  # noqa: E402


TASK_ID = "readmission_30d_partial_ehr_note"
PUBLISHED_AUROC = 0.756
PUBLISHED_CELL = (
    "Table 3, MIMIC-III READM, partial EHR-note DPMM, "
    "AUROC 0.756 (0.734, 0.780)"
)
# Released `dataset_mimic3/fusion.py` docstring, readmission / paired_ehr_note.
# It is produced by summing a `shuffle=True, drop_last=True` training loader,
# so the training number omits a random remainder of four paired examples.
PUBLISHED_PAIRED_POSITIVES = {"train": 692, "validation": 167, "test": 139}
PUBLISHED_PAIRED_TOTAL = 998
# Supplement Table "Datasets summary": MIMIC-III / READM, 14681 / 3222 / 3236
# stays and 3,987 positives overall, 998 of them matched to clinical notes.
PUBLISHED_COUNTS = {"train": 14681, "validation": 3222, "test": 3236}
PUBLISHED_TOTAL_POSITIVES = 3987
PUBLISHED_TOTAL_SAMPLES = 21139

EXPECTED_COUNTS = {"train": 14681, "validation": 3222, "test": 3236}
EXPECTED_POSITIVES = {"train": 2794, "validation": 640, "test": 556}
EXPECTED_PAIRED = {"train": 3652, "validation": 815, "test": 806}
EXPECTED_PAIRED_POSITIVES = {"train": 693, "validation": 167, "test": 139}

# Measured properties of the audited index-misuse defect on this exact cohort.
EXPECTED_DEFECT_MEASUREMENT = {
    "multi_stay_subjects": 5240,
    "multi_stay_stays": 13718,
    "subjects_reordered_by_admittime_sort": 0,
    "stays_changed_by_index_misuse": 0,
    "stays_changed_by_missing_lower_bound": 1,
    "negative_gap_stays": 1,
}


def released_all_stays(mimic_root: Path) -> pd.DataFrame:
    """The exact released `all_stays` frame that `stays.csv` is broken out of.

    This is the same released call chain used by `cohort_source.released_stays`
    but *before* its `los >= 48h` task filter, because the readmission labeler
    reads every stay of a subject to find the next admission.
    """
    old_reader = mimic3csv.dataframe_from_csv
    mimic3csv.dataframe_from_csv = _compressed_dataframe
    try:
        patients = mimic3csv.read_patients_table(mimic_root)
        admits = mimic3csv.read_admissions_table(mimic_root)
        stays = mimic3csv.read_icustays_table(mimic_root)
        stays = mimic3csv.remove_icustays_with_transfers(stays)
        stays = mimic3csv.merge_on_subject_admission(stays, admits)
        stays = mimic3csv.merge_on_subject(stays, patients)
        stays = mimic3csv.filter_admissions_on_nb_icustays(stays)
        stays = mimic3csv.add_age_to_icustays(stays)
        stays = mimic3csv.add_inunit_mortality_to_icustays(stays)
        stays = mimic3csv.add_inhospital_mortality_to_icustays(stays)
        stays = mimic3csv.filter_icustays_on_age(stays)
    finally:
        mimic3csv.dataframe_from_csv = old_reader
    return stays


def released_readmission_labels(mimic_root: Path) -> tuple[pd.DataFrame, dict]:
    """Run `create_readmission_30d.py`'s label rule exactly, plus two controls.

    `readmission_30d` is the released value.  `index_corrected` fixes only the
    `iloc[idx + 1]` / `idx == len - 1` label-versus-position confusion.
    `bounded` additionally requires a non-negative gap of at most 30 days.
    Comparing the three measures the audited defect on real data.
    """
    stays = released_all_stays(mimic_root)
    records: list[tuple[int, int, int, int, float]] = []
    measurement = {
        "multi_stay_subjects": 0,
        "multi_stay_stays": 0,
        "subjects_reordered_by_admittime_sort": 0,
        "stays_changed_by_index_misuse": 0,
        "stays_changed_by_missing_lower_bound": 0,
        "negative_gap_stays": 0,
    }
    for _, subject_stays in stays.groupby("SUBJECT_ID", sort=False):
        # `break_up_stays_by_subject` writes stays.csv sorted by INTIME; the
        # labeler reads it back with a fresh RangeIndex.
        stay_df = subject_stays.sort_values(by="INTIME").reset_index(drop=True)
        before = stay_df["ICUSTAY_ID"].tolist()
        # Released line: sort without resetting the index.
        stay_df = stay_df.sort_values(by=["ADMITTIME", "DISCHTIME"])
        count = len(stay_df)
        if count > 1:
            measurement["multi_stay_subjects"] += 1
            measurement["multi_stay_stays"] += count
            if stay_df["ICUSTAY_ID"].tolist() != before:
                measurement["subjects_reordered_by_admittime_sort"] += 1
        position_of_label = {label: index for index, label in enumerate(stay_df.index)}
        for idx, row in stay_df.iterrows():
            # ---- released behavior (label used as a positional index) ----
            if idx == count - 1:
                released = 0
                released_gap = None
            else:
                next_row = stay_df.iloc[idx + 1]
                released_gap = pd.to_datetime(next_row.ADMITTIME) - pd.to_datetime(
                    row.DISCHTIME
                )
                released = 1 if released_gap.days < 30 else 0
            # ---- control A: identical rule at the correct position ----
            position = position_of_label[idx]
            if position == count - 1:
                index_corrected = 0
                gap = None
            else:
                next_row = stay_df.iloc[position + 1]
                gap = pd.to_datetime(next_row.ADMITTIME) - pd.to_datetime(
                    row.DISCHTIME
                )
                index_corrected = 1 if gap.days < 30 else 0
            # ---- control B: also require 0 <= gap <= 30 days ----
            if gap is None:
                bounded = 0
            else:
                bounded = int(
                    pd.Timedelta(0) <= gap <= pd.Timedelta(days=30)
                )
            if released != index_corrected:
                measurement["stays_changed_by_index_misuse"] += 1
            if released != bounded:
                measurement["stays_changed_by_missing_lower_bound"] += 1
            if gap is not None and gap < pd.Timedelta(0):
                measurement["negative_gap_stays"] += 1
            records.append(
                (
                    int(row.ICUSTAY_ID),
                    int(released),
                    int(index_corrected),
                    int(bounded),
                    float("nan") if gap is None else gap.total_seconds() / 3600.0,
                )
            )
    frame = pd.DataFrame(
        records,
        columns=[
            "stay_id",
            "readmission_30d",
            "index_corrected",
            "bounded",
            "next_admission_gap_hours",
        ],
    )
    if frame["stay_id"].duplicated().any():
        raise AssertionError("released stays produced duplicate ICU stay IDs")
    return frame, measurement


def ensure_readmission_labels(mimic_root: Path, cache_root: Path) -> dict:
    """Cache the released readmission labels beside the shared source cache."""
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "readmission_manifest.json"
    labels_path = cache_root / "readmission_labels.parquet"
    if manifest_path.exists() and labels_path.exists():
        return json.loads(manifest_path.read_text())
    release = assert_mimic_release(mimic_root)
    frame, measurement = released_readmission_labels(mimic_root)
    frame.to_parquet(labels_path, index=False, compression="zstd")
    manifest = {
        "status": "exact_reconstruction",
        "upstream_commit": UPSTREAM_COMMIT,
        "producer": "mimic3extract/mimic3benchmark/scripts/create_readmission_30d.py",
        "producer_sha256": sha256(
            MIMIC3EXTRACT
            / "mimic3benchmark"
            / "scripts"
            / "create_readmission_30d.py"
        ),
        "release_sha256": {
            name: value
            for name, value in release.items()
            if name in ("PATIENTS.csv.gz", "ADMISSIONS.csv.gz", "ICUSTAYS.csv.gz")
        },
        "observation_window_hours": HOURS,
        "label_definition": (
            "next hospital admission starts strictly fewer than 30 whole days "
            "after this admission's DISCHTIME (pandas Timedelta.days floor, no "
            "lower bound); the last admission of a subject is negative; "
            "in-hospital death overrides the outcome to positive"
        ),
        "labeled_stays": len(frame),
        "defect_measurement": measurement,
        "released_label_sha256": array_hash(
            frame.sort_values("stay_id")["readmission_30d"].to_numpy(np.int8)
        ),
        "files": {"readmission_labels": labels_path.name},
        "file_sha256": {"readmission_labels": sha256(labels_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def readmission_cohort(cohort: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach the released readmission label to the exact partial cohort."""
    merged = cohort.merge(
        labels[["stay_id", "readmission_30d", "index_corrected", "bounded"]],
        on="stay_id",
        how="left",
        validate="1:1",
    )
    if merged["readmission_30d"].isna().any():
        raise AssertionError("a cohort stay has no released readmission label")
    mortality = merged["MORTALITY"].astype(int).to_numpy()
    # Released lines 47-49: in-hospital death is a positive READM outcome.
    merged["readm_label"] = np.where(
        mortality == 1, 1, merged["readmission_30d"].astype(int)
    ).astype(np.int8)
    merged["ihm_label"] = merged["label"].astype(np.int8)
    merged["label"] = merged["readm_label"]
    return merged


def write_readmission_author_view(
    cache_root: Path, readm_cache: Path, cohort: pd.DataFrame
) -> Path:
    """Mirror the released `data_mimic3/readmission` task directory.

    `create_readmission_30d.py` writes the same `[0, 48]`-truncated episode
    files as `create_in_hospital_mortality.py`; only the list files differ, so
    the episode directories are linked to the already validated in-hospital
    mortality view instead of duplicating 167 MiB of identical CSV.
    """
    import random

    source_root = readm_cache / "author_view" / "readmission"
    complete = source_root / ".complete"
    ihm_root = cache_root / "author_view" / "in-hospital-mortality"
    if complete.exists():
        return source_root
    source_root.mkdir(parents=True, exist_ok=True)
    for partition in ("train", "test"):
        link = source_root / partition
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(ihm_root / partition, target_is_directory=True)

    def list_frame(selected: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stay": selected["source_name"],
                "period_length": 0,
                "stay_id": selected["stay_id"].astype(np.int64),
                "y_true": selected["label"].astype(np.int64),
            }
        )

    combined = cohort.loc[cohort["split"].isin(["train", "validation"])].copy()
    records = list_frame(combined).to_dict("records")
    random.Random(49297).shuffle(records)
    pd.DataFrame(records).to_csv(
        source_root / "train_listfile_combined.csv", index=False
    )
    for split, filename in (
        ("train", "train_listfile.csv"),
        ("validation", "val_listfile.csv"),
        ("test", "test_listfile.csv"),
    ):
        list_frame(cohort.loc[cohort["split"] == split]).to_csv(
            source_root / filename, index=False
        )
    complete.write_text("released readmission list files over the shared episodes\n")
    return source_root


def readmission_author_datasets(
    cache_root: Path, readm_cache: Path
) -> dict[str, EHRdataset]:
    discretizer, normalizer, _ = make_discretizer_normalizer(cache_root)
    root = readm_cache / "author_view" / "readmission"
    return {
        "train": EHRdataset(
            discretizer, normalizer, str(root / "train_listfile.csv"), str(root / "train")
        ),
        "validation": EHRdataset(
            discretizer, normalizer, str(root / "val_listfile.csv"), str(root / "train")
        ),
        "test": EHRdataset(
            discretizer, normalizer, str(root / "test_listfile.csv"), str(root / "test")
        ),
    }
