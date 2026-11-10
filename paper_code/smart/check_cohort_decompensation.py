"""Assert SMART's paper-specific decompensation cohort facts for all replicas."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from cohort_source import (
    SEEDS,
    assert_mimic_release,
    ordered_text_hash,
    sha256,
    variable_names,
)
from cohort_decompensation import (
    EXPECTED_LISTFILE_COUNTS,
    EXPECTED_OBSERVED_RATE,
    EXPECTED_POSITIVES,
    EXPECTED_RETAINED_SOURCE_ROWS,
    EXPECTED_SEQUENCE_LENGTH_SUM,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TOTAL,
    MAX_NOTEBOOK_MEAN_RELATIVE_DELTA,
    MAX_NOTEBOOK_STD_RELATIVE_DELTA,
    MAX_OBSERVED_RATE_DELTA,
    MAX_POSITIVE_DELTA,
    MAX_SAMPLE_DELTA,
    NOTEBOOK_MEAN,
    NOTEBOOK_PERIOD_LENGTH,
    NOTEBOOK_STD,
    PAPER_LISTFILE_COUNTS,
    PAPER_OBSERVED_RATE,
    PAPER_POSITIVE_FRACTION,
    PAPER_POSITIVES,
    PAPER_TOTAL,
    REBUILT_MEAN,
    REBUILT_STD,
    PUBLISHED_AUROC,
    PUBLISHED_CELL,
    SMART_TASK_BUILDER,
    listfile_facts,
    load_author_notebook_boundary,
    load_build_report,
    load_cohort,
    observed_rate,
    replica_task_id,
    source_root,
    split_assignment,
)


def _overlap(index: pl.DataFrame) -> dict[str, int]:
    members = {
        split: set(index.filter(pl.col("split") == split)["subject_id"].to_list())
        for split in ("train", "validation", "test")
    }
    return {
        "train_validation": len(members["train"] & members["validation"]),
        "train_test": len(members["train"] & members["test"]),
        "validation_test": len(members["validation"] & members["test"]),
        "subjects_in_multiple_splits": sum(
            sum(subject in values for values in members.values()) > 1
            for subject in set().union(*members.values())
        ),
    }


def rederive_released_landmarks(cache_root: Path, cohort: pl.DataFrame) -> dict:
    """Recompute t and y from the producer's own all_stays.csv.

    SMART's builder uses ``t = min(24, int(los), int(lived_time))`` and
    ``y = mortality and int(lived_time - t < 24)``; both inputs come from the
    episode label row and the stay row that ``all_stays.csv`` also carries.
    """
    stays = pd.read_csv(cache_root / "benchmark" / "root" / "all_stays.csv")
    stays["INTIME"] = pd.to_datetime(stays["INTIME"])
    stays["DEATHTIME"] = pd.to_datetime(stays["DEATHTIME"])
    stays = stays.set_index("ICUSTAY_ID", verify_integrity=True)
    selected = stays.loc[cohort["sample_id"].to_list()]
    los_hours = 24.0 * selected["LOS"].to_numpy(dtype=float)
    death = selected["DEATHTIME"].to_numpy()
    intime = selected["INTIME"].to_numpy()
    lived = np.where(
        pd.isnull(selected["DEATHTIME"]).to_numpy(),
        1e18,
        (death - intime) / np.timedelta64(1, "s") / 3600.0,
    )
    landmark = np.minimum(
        24, np.minimum(los_hours.astype(np.int64), lived.astype(np.int64))
    )
    mortality = selected["MORTALITY"].to_numpy(dtype=np.int64)
    label = np.where(mortality == 0, 0, (lived - landmark < 24.0).astype(np.int64))
    exported_landmark = cohort["period_length"].to_numpy().astype(np.int64)
    exported_label = cohort["label"].to_numpy().astype(np.int64)
    landmark_mismatch = int(np.sum(landmark != exported_landmark))
    label_mismatch = int(np.sum(label != exported_label))
    if landmark_mismatch or label_mismatch:
        raise AssertionError(
            "independent re-derivation of the SMART decompensation landmark/label "
            f"disagrees ({landmark_mismatch} landmarks, {label_mismatch} labels)"
        )
    return {
        "stays_in_producer_all_stays": int(stays.shape[0]),
        "eligible_examples": int(cohort.height),
        "excluded_by_released_eligibility": int(stays.shape[0] - cohort.height),
        "landmark_mismatch": landmark_mismatch,
        "label_mismatch": label_mismatch,
        "landmark_histogram": {
            str(int(k)): int(v)
            for k, v in sorted(collections.Counter(exported_landmark.tolist()).items())
        },
        "positives_with_landmark_below_24": int(
            np.sum((exported_label == 1) & (exported_landmark < 24))
        ),
        "negatives_with_landmark_below_24": int(
            np.sum((exported_label == 0) & (exported_landmark < 24))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    report = load_build_report(cache_root)
    if assert_mimic_release(args.mimic_root.resolve()) != report["release_sha256"]:
        raise AssertionError("SMART source report does not match exact local MIMIC-III 1.4")
    cohort = load_cohort(cache_root)

    if cohort.height != EXPECTED_TOTAL:
        raise AssertionError("SMART decompensation cohort count changed")
    positives = int(cohort["label"].sum())
    if positives != EXPECTED_POSITIVES:
        raise AssertionError("SMART decompensation positive marginal changed")
    fraction = positives / cohort.height
    if abs(PAPER_TOTAL - cohort.height) > MAX_SAMPLE_DELTA:
        raise AssertionError(
            f"rebuilt cohort is too far from the paper's {PAPER_TOTAL}: {cohort.height}"
        )
    if abs(PAPER_POSITIVES - positives) > MAX_POSITIVE_DELTA:
        raise AssertionError(
            f"rebuilt positives are too far from the paper's {PAPER_POSITIVES}: {positives}"
        )
    facts = listfile_facts(source_root(cache_root))
    listfile_counts = collections.Counter(
        fact["common_split"] for fact in facts.values()
    )
    if dict(listfile_counts) != EXPECTED_LISTFILE_COUNTS:
        raise AssertionError(
            f"frozen benchmark listfile counts changed: {dict(listfile_counts)}"
        )
    if set(facts) != set(cohort["source_name"].to_list()):
        raise AssertionError("frozen listfile names differ from the cohort")

    rederivation = rederive_released_landmarks(cache_root, cohort)

    multiplicity = cohort.group_by("subject_id").len()
    repeated = multiplicity.filter(pl.col("len") > 1)
    subject_facts = {
        "subjects": int(cohort["subject_id"].n_unique()),
        "subjects_with_multiple_eligible_stays": int(repeated.height),
        "examples_from_repeated_subjects": int(repeated["len"].sum()),
        "maximum_stays_per_subject": int(multiplicity["len"].max()),
    }

    boundary = load_author_notebook_boundary(cache_root)
    rate = observed_rate(boundary["mask"])
    if rate != EXPECTED_OBSERVED_RATE:
        raise AssertionError(
            f"SMART decompensation observed rate changed: {rate!r} != {EXPECTED_OBSERVED_RATE!r}"
        )
    if abs(rate - PAPER_OBSERVED_RATE) > MAX_OBSERVED_RATE_DELTA:
        raise AssertionError(
            f"rebuilt observed rate is too far from the notebook's: {rate!r}"
        )
    if boundary["retained_source_rows"] != EXPECTED_RETAINED_SOURCE_ROWS:
        raise AssertionError("retained released source-row count changed")
    if boundary["retained_source_rows_with_no_observed_cell"] != 0:
        raise AssertionError(
            "a retained released source row has no observed cell, so the MEDS "
            "event stream no longer determines the author row structure"
        )
    if int(boundary["labels"].sum()) != EXPECTED_POSITIVES:
        raise AssertionError("released reader labels differ from the cohort")
    lengths = np.asarray([array.shape[0] for array in boundary["raw"]], dtype=np.int64)
    expected_lengths = np.minimum(
        (boundary["period_lengths"] + 1 - 1e-6).astype(np.int64),
        NOTEBOOK_PERIOD_LENGTH,
    )
    if not np.array_equal(lengths, expected_lengths):
        raise AssertionError("released notebook bin count differs from min(int(t+1-eps), 24)")
    if int(lengths.sum()) != EXPECTED_SEQUENCE_LENGTH_SUM:
        raise AssertionError("rebuilt total author sequence length changed")
    # This reconstruction's own moments are frozen and asserted exactly; the
    # authors' printed moments are a separate, unreproducible historical target
    # whose distance is measured and bounded rather than hidden by a tolerance.
    np.testing.assert_array_equal(boundary["mean"], REBUILT_MEAN)
    np.testing.assert_array_equal(boundary["std"], REBUILT_STD)
    mean_relative = np.max(np.abs((boundary["mean"] - NOTEBOOK_MEAN) / NOTEBOOK_MEAN))
    std_relative = np.max(np.abs((boundary["std"] - NOTEBOOK_STD) / NOTEBOOK_STD))
    if mean_relative > MAX_NOTEBOOK_MEAN_RELATIVE_DELTA:
        raise AssertionError(f"channel means drifted from the notebook: {mean_relative}")
    if std_relative > MAX_NOTEBOOK_STD_RELATIVE_DELTA:
        raise AssertionError(
            f"channel standard deviations drifted from the notebook: {std_relative}"
        )

    for replica, seed in enumerate(SEEDS, start=1):
        task_id = replica_task_id(replica)
        task_dir = args.task_root.resolve() / task_id
        manifest = json.loads((task_dir / "manifest.json").read_text())
        task = manifest["task"]
        source = manifest["source"]
        if task.get("modalities") != ["structured"] or task.get("note_codes") != []:
            raise AssertionError("SMART selected task must be structured-only")
        if task.get("published_auroc") != PUBLISHED_AUROC or task.get("replica_seed") != seed:
            raise AssertionError("SMART task does not identify the selected paper cell/seed")
        if task.get("source_task") != "decompensation":
            raise AssertionError("SMART task is not the released decompensation branch")
        if source.get("selection_tier") != 1 or source.get("selected_model_inputs") != ["structured"]:
            raise AssertionError("SMART selected the wrong source experiment")
        if source.get("fidelity_status") != report["status"]:
            raise AssertionError("SMART manifest lost the source-order qualification")
        if (
            source.get("locally_replaced_task_builder", {}).get("sha256")
            != sha256(SMART_TASK_BUILDER)
        ):
            raise AssertionError("SMART manifest lost the replaced task builder identity")

        index = pl.read_parquet(task_dir / manifest["meds"]["task_index"]["file"])
        expected = split_assignment(cohort, seed).select(
            "sample_id",
            pl.col("split").alias("expected_split"),
            pl.col("label").alias("expected_label"),
            pl.col("period_length").alias("expected_period_length"),
        )
        compared = index.join(expected, on="sample_id", how="inner", validate="1:1")
        if compared.height != EXPECTED_TOTAL:
            raise AssertionError("SMART MEDS task lost a cohort member")
        if compared.filter(
            (pl.col("split") != pl.col("expected_split"))
            | (pl.col("boolean_value") != pl.col("expected_label"))
            | (
                pl.col("released_period_length_hours")
                != pl.col("expected_period_length")
            )
        ).height:
            raise AssertionError(
                "SMART MEDS split/label/landmark differs from the released seed redraw"
            )
        window = (
            (index["prediction_time"] - index["observation_start"])
            .dt.total_seconds()
            .to_numpy()
            / 3600.0
        )
        if not np.array_equal(window, index["released_period_length_hours"].to_numpy()):
            raise AssertionError(
                "MEDS observation window differs from the released landmark"
            )
        counts = dict(index.group_by("split").len().iter_rows())
        if counts != EXPECTED_SPLIT_COUNTS:
            raise AssertionError(f"SMART replica split counts changed: {counts}")
        if index["sample_id"].n_unique() != EXPECTED_TOTAL:
            raise AssertionError("SMART sample IDs overlap across partitions")
        overlap = _overlap(index)
        if overlap["subjects_in_multiple_splits"] <= 0:
            raise AssertionError(
                "SMART code-faithful example split unexpectedly became patient-disjoint"
            )
        split_summary = {
            split: {
                "count": part.height,
                "subjects": part["subject_id"].n_unique(),
                "positive": int(part["boolean_value"].sum()),
            }
            for split in ("train", "validation", "test")
            for part in [index.filter(pl.col("split") == split)]
        }
        output = {
            "status": report["status"],
            "replica": replica,
            "paper_seed": seed,
            "phase": "phase 2 second published binary outcome",
            "paper_task": PUBLISHED_CELL,
            "released_task_builder": {
                "file": "data/MIMIC-III/create_decompensation.py",
                "sha256": sha256(SMART_TASK_BUILDER),
                "landmark_rule": "t = min(24, int(los), int(lived_time))",
                "label_rule": "0 if mortality == 0 else int(lived_time - t < 24)",
                "eligibility": "len(event_times) >= 2 and event_times[1] <= t",
            },
            "samples": EXPECTED_TOTAL,
            "positives": EXPECTED_POSITIVES,
            "prevalence": fraction,
            "rebuilt_positive_fraction": fraction,
            "notebook_positive_fraction": PAPER_POSITIVE_FRACTION,
            "observed_rate": rate,
            "rebuilt_observed_rate": EXPECTED_OBSERVED_RATE,
            "notebook_observed_rate": PAPER_OBSERVED_RATE,
            "notebook_moment_delta": {
                "mean_max_absolute": float(np.max(np.abs(boundary["mean"] - NOTEBOOK_MEAN))),
                "std_max_absolute": float(np.max(np.abs(boundary["std"] - NOTEBOOK_STD))),
                "mean_max_relative": float(mean_relative),
                "std_max_relative": float(std_relative),
                "worst_channels": [
                    variable_names()[int(np.argmax(np.abs((boundary["mean"] - NOTEBOOK_MEAN) / NOTEBOOK_MEAN)))],
                    variable_names()[int(np.argmax(np.abs((boundary["std"] - NOTEBOOK_STD) / NOTEBOOK_STD)))],
                ],
                "note": (
                    "this producer revision declares --reference_range_file but never "
                    "calls remove_outliers_for_variable, so the Glucose and pH moments "
                    "are dominated by a handful of extreme raw lab values and swing "
                    "with any difference in the unreleased producer runtime; the "
                    "moments only scale the author model input and never enter MEDS"
                ),
            },
            "rebuilt_listfile_counts": EXPECTED_LISTFILE_COUNTS,
            "paper_listfile_counts": PAPER_LISTFILE_COUNTS,
            "paper_targets": {
                "samples": PAPER_TOTAL,
                "positives": PAPER_POSITIVES,
                "positive_fraction": PAPER_POSITIVE_FRACTION,
                "observed_rate": PAPER_OBSERVED_RATE,
                "listfile_counts": PAPER_LISTFILE_COUNTS,
            },
            "paper_delta": {
                "samples": PAPER_TOTAL - EXPECTED_TOTAL,
                "positives": PAPER_POSITIVES - EXPECTED_POSITIVES,
                "observed_rate": rate - PAPER_OBSERVED_RATE,
                "sample_shortfall_fraction": (PAPER_TOTAL - EXPECTED_TOTAL) / PAPER_TOTAL,
                "listfile_counts": {
                    key: PAPER_LISTFILE_COUNTS[key] - EXPECTED_LISTFILE_COUNTS[key]
                    for key in PAPER_LISTFILE_COUNTS
                },
                "cause": (
                    "the authors' producer build environment is unreleased; rerunning "
                    "the pinned chain on exact MIMIC-III 1.4 keeps a slightly different "
                    "set of raw events"
                ),
            },
            "paper_text_facts": {
                "quote": (
                    "Appendix A.1: 'Decompensation: The objective is to determine "
                    "whether a patient will decease in the next 24 hours based on the "
                    "data within a 24-hour time window. ... We extract 41,744 patients "
                    "The observed rate is 43.2% and the positive rate is 3.5%.'"
                ),
                "patients": 41_744,
                "observed_rate_rounded": 0.432,
                "positive_rate_rounded": 0.035,
                "reconstructed_patients_actual": EXPECTED_TOTAL,
                "reconstructed_observed_rate_rounded": round(rate, 3),
                "reconstructed_positive_rate_rounded": round(fraction, 3),
            },
            "channels": len(variable_names()),
            "released_landmark_rederivation": rederivation,
            "subject_facts": subject_facts,
            "split_summary": split_summary,
            "cross_split_subject_overlap": overlap,
            "ordered_source_name_sha256": report["source_name_sha256"],
            "ordered_sample_id_sha256": report["sample_id_sha256"],
            "ordered_label_sha256": report["label_sha256"],
            "ordered_period_length_sha256": report["period_length_sha256"],
            "frozen_listfile_sha256": {
                mode: sha256(source_root(cache_root) / f"{mode}_listfile.csv")
                for mode in ("train", "val", "test")
            },
            "ordering_limitation": report["fidelity_limitation"],
            "released_defects_preserved": [
                "outcome-derived prediction landmark and sequence length",
                "example-level split with cross-split patient overlap",
                "pooled full-cohort normalization including validation and test",
            ],
            "assertions": {
                "rebuilt_total_prevalence_observed_rate_frozen_and_exact": True,
                "paper_total_prevalence_observed_rate_within_recorded_bounds": True,
                "rebuilt_train_val_test_listfile_counts": True,
                "independent_landmark_and_label_rederivation": True,
                "released_notebook_bin_count_rule": True,
                "source_seed_split_membership": True,
                "meds_window_equals_released_landmark": True,
                "structured_only_selected_experiment": True,
                "patient_overlap_preserved_not_repaired": True,
                "published_notebook_moments_within_historical_runtime_tolerance": True,
            },
        }
        (task_dir / "audit").mkdir(parents=True, exist_ok=True)
        (task_dir / "audit" / "cohort_reconstruction.json").write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n"
        )
        print(
            json.dumps(
                {"task": task_id, "status": output["status"], "overlap": overlap},
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
