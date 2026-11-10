"""Assert SMART's paper-specific mortality cohort facts for all replicas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import polars as pl

from cohort_source import (
    COHORT_EVIDENCE,
    EXPECTED_OBSERVED_RATE,
    EXPECTED_POSITIVES,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SUBJECTS,
    EXPECTED_TOTAL,
    SEEDS,
    assert_mimic_release,
    load_author_notebook_boundary,
    load_build_report,
    load_cohort,
    ordered_text_hash,
    replica_task_id,
    sha256,
    split_assignment,
)


PUBLISHED_MEAN = np.asarray(
    [
        0.13342736248236953,
        61.46879638121556,
        0.5394284355238489,
        3.119246761936625,
        5.290596950448792,
        11.61723736264916,
        3.180522838632143,
        143.22674480392305,
        86.3000568382371,
        168.72015948168453,
        78.73798307306927,
        97.69934906068663,
        19.29756913160995,
        120.31029497386514,
        37.0390698629806,
        83.27462812014336,
        7.282118883324198,
    ]
)
PUBLISHED_STD = np.asarray(
    [
        0.34003602959594353,
        250.38515121981368,
        0.20068257457910565,
        1.262228263075122,
        1.404701993389681,
        3.9093606844685147,
        1.897389901895581,
        69.23859216770816,
        19.169864801840426,
        15.020152083998529,
        154.8090453358367,
        1030.9393550934726,
        6.63088371639584,
        25.232502325232193,
        9.535566627978861,
        26.058995236709116,
        2.217256925266793,
    ]
)


def _inventory_names() -> set[str]:
    names = set()
    with (COHORT_EVIDENCE / "ihm.csv").open() as handle:
        for relative, _ in csv.reader(handle):
            name = Path(relative).name
            if name != "listfile.csv":
                names.add(name)
    return names


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

    if cohort.height != EXPECTED_TOTAL or cohort["subject_id"].n_unique() != EXPECTED_SUBJECTS:
        raise AssertionError("SMART paper cohort counts changed")
    if int(cohort["label"].sum()) != EXPECTED_POSITIVES:
        raise AssertionError("SMART positive marginal changed")
    if set(cohort["source_name"].to_list()) != _inventory_names():
        raise AssertionError("SMART source-name set differs from released IHM inventory")
    multiplicity = cohort.group_by("subject_id").len()
    repeated = multiplicity.filter(pl.col("len") > 1)
    if (repeated.height, int(repeated["len"].sum()), int(multiplicity["len"].max())) != (
        2_083,
        5_128,
        21,
    ):
        raise AssertionError("released repeated-subject inventory changed")

    boundary = load_author_notebook_boundary(cache_root)
    observed_rate = float(
        boundary["mask"].sum(dtype=np.float64) / boundary["mask"].size
    )
    if observed_rate != EXPECTED_OBSERVED_RATE:
        raise AssertionError(
            f"SMART notebook observed rate changed: {observed_rate} != {EXPECTED_OBSERVED_RATE}"
        )
    # SMART's exact historical pickle/runtime is absent.  The pinned producer
    # reconstruction recovers its mask exactly; the embedded notebook moments
    # are a second, numerical summary target rather than byte-level source.
    np.testing.assert_allclose(boundary["mean"], PUBLISHED_MEAN, rtol=3e-5, atol=3e-3)
    np.testing.assert_allclose(boundary["std"], PUBLISHED_STD, rtol=3e-5, atol=3e-3)

    for replica, seed in enumerate(SEEDS, start=1):
        task_id = replica_task_id(replica)
        task_dir = args.task_root.resolve() / task_id
        manifest = json.loads((task_dir / "manifest.json").read_text())
        task = manifest["task"]
        source = manifest["source"]
        if task.get("modalities") != ["structured"] or task.get("note_codes") != []:
            raise AssertionError("SMART selected task must be structured-only")
        if task.get("published_auroc") != 0.8561 or task.get("replica_seed") != seed:
            raise AssertionError("SMART task does not identify the selected paper cell/seed")
        if source.get("selection_tier") != 1 or source.get("selected_model_inputs") != ["structured"]:
            raise AssertionError("SMART selected the wrong source experiment")
        if source.get("fidelity_status") != report["status"]:
            raise AssertionError("SMART manifest lost the source-order qualification")

        index_path = task_dir / manifest["meds"]["task_index"]["file"]
        index = pl.read_parquet(index_path)
        expected = split_assignment(cohort, seed).select(
            "sample_id", pl.col("split").alias("expected_split"), pl.col("label").alias("expected_label")
        )
        compared = index.join(expected, on="sample_id", how="inner", validate="1:1")
        if compared.height != EXPECTED_TOTAL:
            raise AssertionError("SMART MEDS task lost a cohort member")
        if compared.filter(
            (pl.col("split") != pl.col("expected_split"))
            | (pl.col("boolean_value") != pl.col("expected_label"))
        ).height:
            raise AssertionError("SMART MEDS split/label differs from the released seed redraw")
        counts = dict(index.group_by("split").len().iter_rows())
        if counts != EXPECTED_SPLIT_COUNTS:
            raise AssertionError(f"SMART replica split counts changed: {counts}")
        if index["sample_id"].n_unique() != EXPECTED_TOTAL:
            raise AssertionError("SMART sample IDs overlap across partitions")
        overlap = _overlap(index)
        if overlap["subjects_in_multiple_splits"] <= 0:
            raise AssertionError("SMART code-faithful example split unexpectedly became patient-disjoint")
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
            "paper_task": "Appendix Table 3 in-hospital mortality AUROC 0.8561 +/- 0.0062",
            "samples": EXPECTED_TOTAL,
            "subjects": EXPECTED_SUBJECTS,
            "positives": EXPECTED_POSITIVES,
            "prevalence": EXPECTED_POSITIVES / EXPECTED_TOTAL,
            "observed_rate": observed_rate,
            "published_notebook_moment_delta": {
                "mean_max_absolute": float(np.max(np.abs(boundary["mean"] - PUBLISHED_MEAN))),
                "std_max_absolute": float(np.max(np.abs(boundary["std"] - PUBLISHED_STD))),
            },
            "channels": 17,
            "split_summary": split_summary,
            "cross_split_subject_overlap": overlap,
            "source_name_set_sha256": ordered_text_hash(sorted(_inventory_names())),
            "ordered_source_name_sha256": report["source_name_sha256"],
            "ordered_sample_id_sha256": report["sample_id_sha256"],
            "ordered_label_sha256": report["label_sha256"],
            "source_inventory_sha256": sha256(COHORT_EVIDENCE / "ihm.csv"),
            "ordering_limitation": report["fidelity_limitation"],
            "assertions": {
                "paper_total_prevalence_observed_rate": True,
                "released_exact_episode_name_set": True,
                "released_subject_multiplicity": True,
                "source_seed_split_membership": True,
                "structured_only_selected_experiment": True,
                "patient_overlap_preserved_not_repaired": True,
                "published_notebook_moments_within_historical_runtime_tolerance": True,
            },
        }
        output_path = task_dir / "audit" / "cohort_reconstruction.json"
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"task": task_id, "status": output["status"], "overlap": overlap}, sort_keys=True))


if __name__ == "__main__":
    main()
