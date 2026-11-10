"""Independently verify MERGE paper/source cohort facts against MEDS."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPOSITORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY))

from cohort_source import (  # noqa: E402
    EXPECTED_PREFILTER_OVERLAP,
    EXPECTED_PREFILTER_SPLITS,
    FEATURE_NAMES,
    FROZEN_STAGE2_PREFILTER_OVERLAP,
    HOURS,
    LOS_HOURS,
    NOTE_CODE,
    SEED,
    TASKS,
    UPSTREAM_COMMIT,
    ensure_note_cache,
    ensure_source_cache,
    retained_notes,
    source_inventory,
)


# The released producer builds both published MIMIC-IV experiments from one
# cohort, window, split, and event stream; only the label constructor differs.
TASK_EVIDENCE = {
    "ihm": {
        "manifest_paper_cell": (
            "Table 2, MIMIC-IV IHM, MERGE, AUROC 85.40 +/- 0.24"
        ),
        "report_selected_cell": "Table 2 MIMIC-IV IHM MERGE AUROC 85.40 +/- 0.24",
        "positive_class": "hospital_expire_flag == 1",
        "label_check": (
            "exact hospital mortality labels and seed-42 split membership"
        ),
    },
    "los": {
        "manifest_paper_cell": (
            "Table 2, MIMIC-IV LOS, MERGE, AUROC 81.88 +/- 0.18"
        ),
        "report_selected_cell": "Table 2 MIMIC-IV LOS MERGE AUROC 81.88 +/- 0.18",
        "positive_class": (
            "(outtime - intime) < 96 hours AND hospital_expire_flag == 0"
        ),
        "label_check": (
            "exact released LOS labels and seed-42 split membership"
        ),
    },
}
# paper_audits/merge.md measured this on the released candidate cohort: stays
# that left the ICU in under 96 hours but died later in the same admission.
FROZEN_STAGE2_LOS_LABEL_CONFLICT = 1_022


def hash_ints(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def overlap(rows: list[dict], key: str) -> dict[str, int]:
    names = ("train", "validation", "test")
    members = {
        split: {int(row[key]) for row in rows if row["split"] == split}
        for split in names
    }
    result = {
        f"{left}_{right}": len(members[left] & members[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    values = set().union(*members.values())
    result[f"{key}s_in_multiple_splits"] = sum(
        sum(value in members[split] for split in names) > 1 for value in values
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--note-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, default="ihm")
    args = parser.parse_args()
    evidence = TASK_EVIDENCE[args.task]

    manifest = json.loads((args.task_dir / "manifest.json").read_text())
    if manifest.get("paper", {}).get("key") != "merge":
        raise AssertionError("task is not the MERGE export")
    if manifest["task"].get("modalities") != ["structured", "notes"]:
        raise AssertionError("MERGE must retain its structured and note projection")
    if manifest["task"].get("note_codes") != [NOTE_CODE]:
        raise AssertionError("MERGE note-code exception list changed")
    if manifest.get("source", {}).get("upstream_commit") != UPSTREAM_COMMIT:
        raise AssertionError("MERGE upstream commit changed")
    if manifest["source"].get("selected_paper_cell") != evidence[
        "manifest_paper_cell"
    ]:
        raise AssertionError("MERGE selected the wrong paper cell")
    if manifest["source"].get("selection_tier") != 3:
        raise AssertionError("MERGE selected the wrong modality-priority tier")
    if manifest["source"].get("selected_model_inputs") != [
        "structured",
        "notes",
        "CXR",
    ]:
        raise AssertionError("MERGE source modalities changed")
    if manifest["source"].get("retained_modalities") != [
        "structured",
        "notes",
    ]:
        raise AssertionError("MERGE retained projection changed")
    if manifest["source"].get("omitted_modalities") != ["CXR"]:
        raise AssertionError("MERGE must omit only the imaging input")

    cache = ensure_source_cache(args.mimic_root, args.cache_root)
    note_cache = ensure_note_cache(
        args.mimic_root, args.note_root, args.cache_root
    )
    if cache["source_files"] != source_inventory(args.mimic_root):
        raise AssertionError("MERGE MIMIC-IV 3.1 source inventory changed")
    wide = pd.read_parquet(
        args.cache_root / cache["wide_parquet"],
        columns=["subject_id", "hadm_id", "stay_id", "icu_time_delta"],
    )
    wide = wide.loc[
        (wide["icu_time_delta"] >= 0) & (wide["icu_time_delta"] < HOURS)
    ]
    row_counts = wide.groupby("stay_id", sort=False).size().to_dict()
    first = (
        wide.groupby("stay_id", sort=False)[["subject_id", "hadm_id"]]
        .first()
        .to_dict("index")
    )

    stays = pd.read_csv(
        args.mimic_root / "icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
    )
    stays = stays.loc[stays["los"] >= HOURS / 24.0].copy()
    if len(stays) != 46_337:
        raise AssertionError(f"paper/source LOS>=2 cohort changed: {len(stays)}")
    stay_lookup = {int(row.stay_id): row for row in stays.itertuples(index=False)}
    candidate_ids = np.asarray(sorted(stay_lookup), dtype=np.int64)
    if set(row_counts) != set(candidate_ids.tolist()):
        raise AssertionError("structured data do not cover every released candidate stay")
    shuffled = candidate_ids.copy()
    np.random.RandomState(SEED).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.70)
    validation_end = train_end + int(len(shuffled) * 0.15)
    split_ids = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    if {name: len(ids) for name, ids in split_ids.items()} != EXPECTED_PREFILTER_SPLITS:
        raise AssertionError("paper-reported seed-42 split counts changed")

    admissions = pd.read_csv(
        args.mimic_root / "hosp" / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "hospital_expire_flag", "dischtime"],
        parse_dates=["dischtime"],
    )
    labels = {
        int(row.hadm_id): (int(row.subject_id), int(row.hospital_expire_flag))
        for row in admissions.itertuples(index=False)
    }
    discharge_times = {
        int(row.hadm_id): row.dischtime
        for row in admissions.itertuples(index=False)
    }
    prefilter = []
    retained = []
    for split, ids in split_ids.items():
        for source_position, stay_id_value in enumerate(ids):
            stay_id = int(stay_id_value)
            raw_stay = stay_lookup[stay_id]
            pivot_id = first[stay_id]
            subject_id = int(pivot_id["subject_id"])
            hadm_id = int(pivot_id["hadm_id"])
            if (
                subject_id != int(raw_stay.subject_id)
                or hadm_id != int(raw_stay.hadm_id)
            ):
                raise AssertionError("MERGE pivot identifiers disagree with ICUSTAYS")
            label_subject, died = labels[hadm_id]
            if label_subject != subject_id:
                raise AssertionError("MERGE label admission ownership changed")
            icu_hours = (
                raw_stay.outtime - raw_stay.intime
            ).total_seconds() / 3600
            if icu_hours < HOURS:
                raise AssertionError(
                    "a released candidate stay ends before the prediction time"
                )
            if args.task == "los":
                label = int((icu_hours < LOS_HOURS) and (died == 0))
            else:
                label = died
            row = {
                "split": split,
                "source_position": source_position,
                "stay_id": stay_id,
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "label": label,
                "died": int(died),
                "icu_hours": icu_hours,
                "left_icu_alive": bool(
                    discharge_times[hadm_id] > raw_stay.outtime
                ),
                "irregular_rows": int(row_counts[stay_id]),
            }
            prefilter.append(row)
            if row["irregular_rows"] > 12:
                retained.append(row)

    prefilter_subject_overlap = overlap(prefilter, "subject_id")
    normalized_overlap = {
        key.replace("subject_ids", "subjects"): value
        for key, value in prefilter_subject_overlap.items()
    }
    if normalized_overlap != EXPECTED_PREFILTER_OVERLAP:
        raise AssertionError(f"paper split overlap changed: {normalized_overlap}")

    index = pq.read_table(
        args.task_dir / manifest["meds"]["task_index"]["file"]
    )
    index_by_stay = {int(row["sample_id"]): row for row in index.to_pylist()}
    if len(index_by_stay) != len(retained):
        raise AssertionError("MEDS retained cohort count changed")
    for row in retained:
        actual = index_by_stay.get(row["stay_id"])
        if actual is None:
            raise AssertionError("MEDS omitted a retained MERGE stay")
        expected = {
            "subject_id": row["subject_id"],
            "stay_id": row["stay_id"],
            "hadm_id": row["hadm_id"],
            "source_position": row["source_position"],
            "irregular_rows": row["irregular_rows"],
            "boolean_value": bool(row["label"]),
            "split": row["split"],
        }
        if any(actual[name] != value for name, value in expected.items()):
            raise AssertionError("MEDS changed a MERGE cohort row")

    retained_ids = {int(row["stay_id"]) for row in retained}
    notes = retained_notes(args.cache_root, retained_ids)
    if not set(notes["stay_id"].astype(int)).issubset(retained_ids):
        raise AssertionError("selected notes include a non-retained stay")
    split_by_stay = {int(row["stay_id"]): row["split"] for row in retained}
    expected_note_splits = {
        split: {
            "notes": int(
                notes["stay_id"].map(split_by_stay).eq(split).sum()
            ),
            "stays_with_notes": int(
                notes.loc[
                    notes["stay_id"].map(split_by_stay).eq(split), "stay_id"
                ].nunique()
            ),
        }
        for split in ("train", "validation", "test")
    }
    note_events = pq.read_table(
        args.task_dir / manifest["meds"]["events"]["file"],
        columns=["sample_id", "code"],
        filters=[("code", "=", NOTE_CODE)],
    )
    if len(note_events) != len(notes):
        raise AssertionError("MEDS radiology-note count changed")
    if set(note_events["code"].to_pylist()) != {NOTE_CODE}:
        raise AssertionError("MEDS note projection contains another event code")
    actual_note_splits = {}
    note_event_stays = note_events["sample_id"].to_numpy().astype(
        np.int64, copy=False
    )
    for split in ("train", "validation", "test"):
        selected_ids = {
            int(row["stay_id"]) for row in retained if row["split"] == split
        }
        selected_events = note_event_stays[
            np.isin(note_event_stays, np.asarray(sorted(selected_ids), dtype=np.int64))
        ]
        actual_note_splits[split] = {
            "notes": len(selected_events),
            "stays_with_notes": len(set(selected_events.tolist())),
        }
    if actual_note_splits != expected_note_splits:
        raise AssertionError("MEDS note counts by released split changed")
    note_cutoffs = notes["stay_id"].map(
        lambda value: stay_lookup[int(value)].intime + pd.Timedelta(hours=HOURS)
    )
    unavailable_notes = notes["storetime"] >= note_cutoffs
    retained_unavailable = {
        "rows": int(unavailable_notes.sum()),
        "stays": int(notes.loc[unavailable_notes, "stay_id"].nunique()),
    }
    manifest_notes = manifest["source"]["notes"]
    if manifest_notes.get("stored_at_or_after_cutoff_rows") != retained_unavailable[
        "rows"
    ] or manifest_notes.get("stored_at_or_after_cutoff_stays") != retained_unavailable[
        "stays"
    ]:
        raise AssertionError("MERGE manifest changed note availability evidence")

    split_summaries = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in retained if row["split"] == split]
        positives = sum(row["label"] for row in selected)
        actual = {
            "count": len(selected),
            "subjects": len({row["subject_id"] for row in selected}),
            "admissions": len({row["hadm_id"] for row in selected}),
            "positive": positives,
            "prevalence": positives / len(selected),
            "ordered_stay_id_sha256": hash_ints(
                [row["stay_id"] for row in selected]
            ),
            "ordered_label_sha256": hash_ints(
                [row["label"] for row in selected]
            ),
        }
        manifest_split = manifest["splits"][split]
        if any(
            manifest_split[key] != actual[key]
            for key in ("count", "subjects", "positive")
        ):
            raise AssertionError(f"MERGE {split} manifest summary changed")
        split_summaries[split] = actual

    codes = pq.read_table(
        args.task_dir / manifest["meds"]["codes"]["file"]
    )["code"].to_pylist()
    if codes != list(FEATURE_NAMES) + [NOTE_CODE]:
        raise AssertionError("MERGE MEDS code vocabulary/order changed")
    if "Potassium" in codes:
        raise AssertionError("MERGE code-faithful source must omit Potassium")

    sorted_stay_ids = sorted(row["stay_id"] for row in retained)
    post_subject_overlap = overlap(retained, "subject_id")
    post_admission_overlap = overlap(retained, "hadm_id")
    report = {
        "status": "exact",
        "paper_evidence": {
            "selected_cell": evidence["report_selected_cell"],
            "appendix_candidate_stays": 46_337,
            "appendix_split_counts": EXPECTED_PREFILTER_SPLITS,
            "positive_class": evidence["positive_class"],
            "observation_window": "first 48 ICU hours, half-open [0,48)",
        },
        "source_configuration": {
            "modalities": ["structured", "notes", "CXR"],
            "include_missing": True,
            "seed": SEED,
            "retained_projection": ["structured", "notes"],
            "omitted_modalities": ["CXR"],
            "omitted_modalities_change_membership": False,
            "membership_proof": (
                "structured stay IDs equal all 46,337 LOS>=2 valid stays before "
                "the released include_missing note/CXR union"
            ),
        },
        "prefilter": {
            "count": len(prefilter),
            "splits": EXPECTED_PREFILTER_SPLITS,
            "cross_split_subject_overlap": normalized_overlap,
            "frozen_stage2_raw_order_overlap": FROZEN_STAGE2_PREFILTER_OVERLAP,
            "frozen_stage2_conflict": (
                "the code audit reconstructed the pre-union pivot/file ordering. The "
                "selected include-missing launcher calls np.union1d for notes "
                "and CXR, sorting stay IDs before the seed-42 shuffle; MEDS "
                "preserves that executable source ordering."
            ),
        },
        "post_loader_filter": {
            "rule": "len(ts_tt) > 12",
            "count": len(retained),
            "sorted_stay_id_sha256": hash_ints(sorted_stay_ids),
            "splits": split_summaries,
            "cross_split_subject_overlap": post_subject_overlap,
            "cross_split_admission_overlap": post_admission_overlap,
        },
        "feature_projection": {
            "status": "exact",
            "structured_codes": list(FEATURE_NAMES),
            "code_count": len(FEATURE_NAMES) + 1,
            "potassium_omission_preserved": True,
            "note_codes": [NOTE_CODE],
            "imaging_retained": False,
        },
        "note_projection": {
            "status": "exact",
            "source_version": "MIMIC-IV-Note 2.2",
            "candidate_notes": note_cache["note_rows"],
            "candidate_stays_with_notes": note_cache["stays_with_notes"],
            "retained_notes": len(notes),
            "retained_stays_with_notes": int(notes["stay_id"].nunique()),
            "retained_stays_without_notes": len(retained)
            - int(notes["stay_id"].nunique()),
            "notes_removed_by_structured_loader_filter": note_cache["note_rows"]
            - len(notes),
            "splits": expected_note_splits,
            "cohort_filter": "none; include_missing retains empty-note stays",
            "notes_with_null_author_mapped_hadm_id": note_cache[
                "notes_with_null_mapped_hadm_id"
            ],
            "candidate_stored_at_or_after_cutoff": {
                "rows": note_cache["stored_at_or_after_cutoff_rows"],
                "stays": note_cache["stored_at_or_after_cutoff_stays"],
            },
            "retained_stored_at_or_after_cutoff": retained_unavailable,
            "availability_defect": (
                "preserved source behavior: selection uses charttime and ignores "
                "storetime, so these reports were unavailable at prediction"
            ),
            "frozen_stage2_direct_join_evidence": {
                "notes": 148_665,
                "stored_at_or_after_cutoff_rows": 8_323,
                "stored_at_or_after_cutoff_stays": 7_111,
                "conflict": (
                    "the frozen targeted exact-time join was not the released "
                    "add_time_delta_notes_vectorized assignment. The selected "
                    "the export path calls that mapper, which assigns at most one "
                    "first admission and first ICU match per raw note; its exact "
                    "loader-boundary counts are authoritative here."
                ),
            },
            "source": note_cache["source"],
        },
        "source_files": cache["source_files"],
        "checks": [
            "paper candidate and released prefilter split counts",
            "exact stay, admission, and subject identifiers",
            evidence["label_check"],
            "all per-split counts, positives, prevalence, and ordered hashes",
            "published source-design patient overlaps",
            "author >12 irregular-timestamp loader filter",
            "note/CXR availability cannot change the include_missing cohort",
            "exact 30-channel structured projection and Potassium omission",
            "exact retained raw-note event counts and availability by split",
            "complete singleton note_codes declaration and imaging exclusion",
        ],
    }
    if args.task == "los":
        def label_facts(rows: list[dict]) -> dict[str, int]:
            under = [row for row in rows if row["icu_hours"] < LOS_HOURS]
            conflict = [row for row in under if row["died"] == 1]
            return {
                "stays": len(rows),
                "icu_exit_under_96h": len(under),
                "released_positive": sum(row["label"] for row in rows),
                "icu_exit_under_96h_and_hospital_death": len(conflict),
                "icu_exit_under_96h_left_icu_alive_then_died": sum(
                    row["left_icu_alive"] for row in conflict
                ),
                "minimum_icu_hours": min(row["icu_hours"] for row in rows),
            }

        candidate_facts = label_facts(prefilter)
        retained_facts = label_facts(retained)
        if candidate_facts["released_positive"] != candidate_facts[
            "icu_exit_under_96h"
        ] - candidate_facts["icu_exit_under_96h_and_hospital_death"]:
            raise AssertionError("released LOS label is not the coded conjunction")
        if (
            candidate_facts["icu_exit_under_96h_left_icu_alive_then_died"]
            != FROZEN_STAGE2_LOS_LABEL_CONFLICT
        ):
            raise AssertionError(
                "frozen LOS label-conflict count changed: "
                f"{candidate_facts['icu_exit_under_96h_left_icu_alive_then_died']}"
            )
        if candidate_facts["minimum_icu_hours"] < HOURS:
            raise AssertionError("a candidate stay ends before the prediction time")
        report["released_label"] = {
            "status": "exact",
            "constructor": (
                "mimiciv/data_preprocess/create_los_task.py::get_stay_list"
            ),
            "expression": (
                "label = 1 if ((outtime - intime).total_seconds()/3600 < 96) "
                "and (hospital_expire_flag == 0) else 0"
            ),
            "horizon_hours": LOS_HOURS,
            "prediction_time": "ICU intime + 48 hours",
            "candidate_cohort": candidate_facts,
            "retained_cohort": retained_facts,
            "paper_prose_endpoint": (
                "Appendix B.1: 'leave the ICU alive in less than 96 hours'"
            ),
            "disclosed_defect": (
                "the released label additionally requires eventual hospital "
                "survival, so stays that left the ICU alive under 96 hours and "
                "died later in the same admission are labeled 0; preserved as "
                "released and not repaired (paper_audits/merge.md)"
            ),
            "frozen_stage2_label_conflict_stays": (
                FROZEN_STAGE2_LOS_LABEL_CONFLICT
            ),
            "label_is_strictly_future": (
                "every candidate stay has ICU duration >= 48 hours, so the "
                "outcome is unresolved at the prediction time"
            ),
        }
        report["checks"] = report["checks"] + [
            "released LOS label conjunction and paper-prose disagreement counts",
            "every retained stay is still in the ICU at the prediction time",
        ]
    report_path = args.task_dir / "audit" / "cohort_reconstruction.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
