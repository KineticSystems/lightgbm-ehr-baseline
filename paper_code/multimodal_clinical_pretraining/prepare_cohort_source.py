"""Rebuild the released IHM source tree from exact MIMIC-III v1.4 tables.

The upstream extractor requires uncompressed ``.csv`` files and materializes
every subject/table field.  This bounded compatibility adapter reads the
official ``.csv.gz`` payload directly, retains only item IDs that survive the
pinned producer map, and calls the pinned producer's cleaning, pivoting, and
episode functions.  It also applies the filename rendering demonstrated by
every acquired author list row; the released pandas formatting is incompatible
with those rows and with the study's pretraining parser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from cohort_source import (
    EXPECTED_SPLITS,
    LISTFILE_SHA256,
    PRODUCER_COMMIT,
    STUDY_COMMIT,
    gzip_aware_dataframe,
    import_producer,
    listfile_paths,
    load_list_rows,
    raw_path,
    sha256,
    verify_pinned_sources,
    write_json,
)


RAW_STEMS = (
    "PATIENTS",
    "ADMISSIONS",
    "ICUSTAYS",
    "CHARTEVENTS",
    "LABEVENTS",
    "OUTPUTEVENTS",
)
EVENT_STEMS = ("CHARTEVENTS", "LABEVENTS", "OUTPUTEVENTS")
EVENT_COLUMNS = (
    "SUBJECT_ID",
    "HADM_ID",
    "ICUSTAY_ID",
    "CHARTTIME",
    "ITEMID",
    "VALUE",
    "VALUEUOM",
)


def official_checksums(mimic_root: Path) -> tuple[Path, dict[str, str]]:
    manifest = mimic_root / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    entries = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name.lstrip("* ")] = digest
    selected = {}
    for stem in RAW_STEMS:
        path = raw_path(mimic_root, stem)
        expected = entries.get(path.name)
        if expected is None:
            raise AssertionError(f"official manifest omits {path.name}")
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"official MIMIC-III checksum failed: {path.name}")
        selected[path.name] = actual
    return manifest, selected


def canonical_stays(mimic_root: Path, mimic3csv) -> tuple[pd.DataFrame, dict[str, object]]:
    original_reader = mimic3csv.dataframe_from_csv
    mimic3csv.dataframe_from_csv = gzip_aware_dataframe
    try:
        patients = mimic3csv.read_patients_table(mimic_root)
        admissions = mimic3csv.read_admissions_table(mimic_root)
        stays = mimic3csv.read_icustays_table(mimic_root)
    finally:
        mimic3csv.dataframe_from_csv = original_reader

    def summary(frame: pd.DataFrame) -> dict[str, int]:
        return {
            "stays": int(frame.ICUSTAY_ID.nunique()),
            "admissions": int(frame.HADM_ID.nunique()),
            "subjects": int(frame.SUBJECT_ID.nunique()),
        }

    stages = {"raw_icustays": summary(stays)}
    stays = mimic3csv.remove_icustays_with_transfers(stays)
    stages["after_transfer_filter"] = summary(stays)
    stays = mimic3csv.merge_on_subject_admission(stays, admissions)
    stays = mimic3csv.merge_on_subject(stays, patients)
    stays = mimic3csv.filter_admissions_on_nb_icustays(stays)
    stages["after_single_stay_per_admission"] = summary(stays)
    stays = mimic3csv.add_age_to_icustays(stays)
    stays = mimic3csv.add_inunit_mortality_to_icustays(stays)
    stays = mimic3csv.add_inhospital_mortality_to_icustays(stays)
    stays = mimic3csv.filter_icustays_on_age(stays)
    stages["after_adult_filter"] = summary(stays)
    return stays, stages


def verify_rows_against_stays(
    rows: list[dict[str, object]], stays: pd.DataFrame
) -> dict[int, pd.Series]:
    indexed = stays.set_index("ICUSTAY_ID", verify_integrity=True)
    lookup = {}
    for row in rows:
        stay_id = int(row["stay_id"])
        if stay_id not in indexed.index:
            raise AssertionError(f"author stay {stay_id} was excluded by the producer")
        stay = indexed.loc[stay_id]
        if int(stay.SUBJECT_ID) != int(row["subject_id"]):
            raise AssertionError("author filename subject disagrees with producer stay")
        if int(stay.MORTALITY) != int(row["label"]):
            raise AssertionError("author IHM label disagrees with producer mortality")
        if float(stay.LOS) * 24.0 < 48.0 - 1e-6:
            raise AssertionError("author IHM row has fewer than 48 ICU hours")
        if pd.Timestamp(stay.INTIME) != pd.Timestamp(row["intime"]):
            raise AssertionError("author filename INTIME disagrees with raw ICUSTAYS")
        if pd.Timestamp(stay.OUTTIME) != pd.Timestamp(row["outtime"]):
            raise AssertionError("author filename OUTTIME disagrees with raw ICUSTAYS")
        lookup[stay_id] = stay
    return lookup


def spool_event_table(
    mimic_root: Path,
    spool_root: Path,
    stem: str,
    subject_ids: set[str],
    item_ids: set[str],
    chunksize: int,
) -> dict[str, object]:
    final = spool_root / stem.lower()
    summary_path = final / "summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text())
    partial = spool_root / f"{stem.lower()}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    total_rows = 0
    selected_subject_rows = 0
    retained_rows = 0
    nonnumeric_subjects: set[str] = set()
    path = raw_path(mimic_root, stem)
    reader = pd.read_csv(
        path,
        usecols=lambda name: name in EVENT_COLUMNS,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        in_subject = chunk["SUBJECT_ID"].isin(subject_ids)
        selected_subject_rows += int(in_subject.sum())
        subject_values = chunk.loc[in_subject, ["SUBJECT_ID", "VALUE"]]
        if not subject_values.empty:
            values = subject_values["VALUE"].str.strip()
            nonnumeric = values.ne("") & pd.to_numeric(
                values, errors="coerce"
            ).isna()
            nonnumeric_subjects.update(
                subject_values.loc[nonnumeric, "SUBJECT_ID"].tolist()
            )
        retained = chunk.loc[
            in_subject & chunk["ITEMID"].isin(item_ids), :
        ].copy()
        if retained.empty:
            continue
        if "ICUSTAY_ID" not in retained:
            retained["ICUSTAY_ID"] = ""
        for column in EVENT_COLUMNS:
            if column not in retained:
                retained[column] = ""
        retained = retained.loc[:, EVENT_COLUMNS]
        retained_rows += len(retained)
        for subject_id, group in retained.groupby("SUBJECT_ID", sort=False):
            target = partial / f"{subject_id}.csv"
            group.to_csv(
                target,
                mode="a",
                header=not target.exists(),
                index=False,
            )
        if chunk_number % 20 == 0:
            print(
                f"{stem}: raw={total_rows:,} selected-item rows={retained_rows:,}",
                flush=True,
            )
    summary = {
        "source_file": path.name,
        "source_sha256": sha256(path),
        "raw_rows": total_rows,
        "selected_subject_rows": selected_subject_rows,
        "retained_item_rows": retained_rows,
        "subjects_with_any_nonnumeric_value": sorted(
            int(value) for value in nonnumeric_subjects
        ),
    }
    write_json(partial / "summary.json", summary)
    partial.rename(final)
    return summary


def load_subject_events(
    spool_root: Path,
    subject_id: int,
    force_object_value: bool,
) -> pd.DataFrame:
    frames = []
    for stem in EVENT_STEMS:
        path = spool_root / stem.lower() / f"{subject_id}.csv"
        if path.is_file():
            frames.append(
                pd.read_csv(
                    path,
                    dtype=str,
                    keep_default_na=False,
                    na_filter=False,
                )
            )
    if not frames:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    events = pd.concat(frames, ignore_index=True)
    events = events.loc[events.VALUE.ne("")].copy()
    events.CHARTTIME = pd.to_datetime(events.CHARTTIME, errors="raise")
    events.HADM_ID = pd.to_numeric(
        events.HADM_ID.replace("", "-1"), errors="raise"
    ).astype(int)
    events.ICUSTAY_ID = pd.to_numeric(
        events.ICUSTAY_ID.replace("", "-1"), errors="raise"
    ).astype(int)
    events.ITEMID = pd.to_numeric(events.ITEMID, errors="raise").astype(int)
    events.VALUEUOM = events.VALUEUOM.astype(str)
    if force_object_value:
        events.VALUE = events.VALUE.astype(str)
    else:
        events.VALUE = pd.to_numeric(events.VALUE, errors="raise")
    return events


def materialize_task_tree(
    source_root: Path,
    list_root: Path,
    spool_root: Path,
    rows: list[dict[str, object]],
    producer_root: Path,
    table_summaries: dict[str, dict[str, object]],
    preprocessing,
    subject_module,
) -> dict[str, object]:
    final = source_root / "in-hospital-mortality"
    if (source_root / "source_build.json").is_file() and final.is_dir():
        return json.loads((source_root / "source_build.json").read_text())
    if final.is_dir():
        observed_cells = 0
        source_time_min = float("inf")
        source_time_max = float("-inf")
        for row in rows:
            partition = "test" if row["split"] == "test" else "train"
            path = final / partition / str(row["source_file"])
            if not path.is_file():
                raise AssertionError("incomplete pre-existing source task tree")
            frame = pd.read_csv(path)
            if frame.columns[0] != "Hours" or frame.shape[1] != 18 or frame.empty:
                raise AssertionError("invalid pre-existing source timeseries")
            observed_cells += int(frame.iloc[:, 1:].notna().sum().sum())
            source_time_min = min(source_time_min, float(frame.Hours.min()))
            source_time_max = max(source_time_max, float(frame.Hours.max()))
        return {
            "mimic_release": "MIMIC-III 1.4",
            "study_commit": STUDY_COMMIT,
            "producer_commit": PRODUCER_COMMIT,
            "filename_correction": (
                "render INTIME/OUTTIME as eight underscore-delimited fields with "
                "hyphenated times, exactly matching all acquired author list rows"
            ),
            "samples": len(rows),
            "observed_source_cells": observed_cells,
            "source_time_min": source_time_min,
            "source_time_max": source_time_max,
            "subjects": len({int(row["subject_id"]) for row in rows}),
        }
    partial = source_root / "in-hospital-mortality.partial"
    if partial.exists():
        shutil.rmtree(partial)
    (partial / "train").mkdir(parents=True)
    (partial / "test").mkdir(parents=True)
    for split, path in listfile_paths(list_root).items():
        target_name = "val_listfile.csv" if split == "validation" else path.name
        shutil.copyfile(path, partial / target_name)

    map_path = (
        producer_root
        / "mimic3benchmark"
        / "resources"
        / "itemid_to_variable_map.csv"
    )
    variable_map = preprocessing.read_itemid_to_variable_map(map_path)
    variables = variable_map.VARIABLE.unique()
    rows_by_subject: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_subject.setdefault(int(row["subject_id"]), []).append(row)
    nonnumeric_subjects = set()
    for summary in table_summaries.values():
        nonnumeric_subjects.update(summary["subjects_with_any_nonnumeric_value"])

    written_rows = 0
    observed_cells = 0
    source_time_min = float("inf")
    source_time_max = float("-inf")
    for subject_number, subject_id in enumerate(sorted(rows_by_subject), start=1):
        events = load_subject_events(
            spool_root,
            subject_id,
            force_object_value=subject_id in nonnumeric_subjects,
        )
        events = preprocessing.map_itemids_to_variables(events, variable_map)
        events = preprocessing.clean_events(events)
        if events.empty:
            raise AssertionError(f"selected author subject {subject_id} has no mapped events")
        timeseries = subject_module.convert_events_to_timeseries(
            events, variables=variables
        )
        for row in rows_by_subject[subject_id]:
            episode = subject_module.get_events_for_stay(
                timeseries,
                int(row["stay_id"]),
                pd.Timestamp(row["intime"]),
                pd.Timestamp(row["outtime"]),
            )
            episode = subject_module.add_hours_elpased_to_events(
                episode, pd.Timestamp(row["intime"])
            ).set_index("HOURS").sort_index(axis=0)
            episode = episode.loc[
                (episode.index > -1e-6) & (episode.index < 48.0 + 1e-6)
            ]
            if episode.empty:
                raise AssertionError("author IHM row has no first-48-hour event")
            columns = sorted(episode.columns)
            episode = episode[columns]
            source_partition = "test" if row["split"] == "test" else "train"
            target = partial / source_partition / str(row["source_file"])
            episode.to_csv(target, index_label="Hours")
            reread = pd.read_csv(target)
            if reread.columns[0] != "Hours" or reread.shape[1] != 18:
                raise AssertionError("constructed source timeseries has wrong schema")
            values = reread.iloc[:, 1:]
            observed_cells += int(values.notna().sum().sum())
            source_time_min = min(source_time_min, float(reread.Hours.min()))
            source_time_max = max(source_time_max, float(reread.Hours.max()))
            written_rows += 1
        if subject_number % 500 == 0:
            print(
                f"episodes: subjects={subject_number:,}/{len(rows_by_subject):,} "
                f"stays={written_rows:,}",
                flush=True,
            )
    if written_rows != len(rows):
        raise AssertionError("constructed task tree lost an author list row")
    if final.exists():
        raise FileExistsError(final)
    partial.rename(final)
    return {
        "mimic_release": "MIMIC-III 1.4",
        "study_commit": STUDY_COMMIT,
        "producer_commit": PRODUCER_COMMIT,
        "filename_correction": (
            "render INTIME/OUTTIME as eight underscore-delimited fields with "
            "hyphenated times, exactly matching all acquired author list rows"
        ),
        "samples": written_rows,
        "observed_source_cells": observed_cells,
        "source_time_min": source_time_min,
        "source_time_max": source_time_max,
        "subjects": len(rows_by_subject),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--list-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()

    mimic_root = args.mimic_root.resolve()
    list_root = args.list_root.resolve()
    producer_root = args.producer_root.resolve()
    source_root = args.output_root.resolve()
    source_root.mkdir(parents=True, exist_ok=True)
    source_files = verify_pinned_sources(producer_root)
    mimic3csv, preprocessing, subject_module = import_producer(producer_root)
    rows = load_list_rows(list_root)
    for split, expected in EXPECTED_SPLITS.items():
        selected = [row for row in rows if row["split"] == split]
        actual = {
            "rows": len(selected),
            "subjects": len({int(row["subject_id"]) for row in selected}),
            "positive": sum(int(row["label"]) for row in selected),
        }
        if actual != expected:
            raise AssertionError(f"{split} acquired list summary changed: {actual}")

    manifest_path, raw_hashes = official_checksums(mimic_root)
    stays, construction_stages = canonical_stays(mimic_root, mimic3csv)
    verify_rows_against_stays(rows, stays)
    subject_ids = {str(row["subject_id"]) for row in rows}
    map_path = (
        producer_root
        / "mimic3benchmark"
        / "resources"
        / "itemid_to_variable_map.csv"
    )
    variable_map = preprocessing.read_itemid_to_variable_map(map_path)
    item_ids = set(variable_map.index.astype(str))
    spool_root = source_root / "selected_event_spool"
    spool_root.mkdir(exist_ok=True)
    table_summaries = {
        stem: spool_event_table(
            mimic_root,
            spool_root,
            stem,
            subject_ids,
            item_ids,
            args.chunksize,
        )
        for stem in EVENT_STEMS
    }
    build = materialize_task_tree(
        source_root,
        list_root,
        spool_root,
        rows,
        producer_root,
        table_summaries,
        preprocessing,
        subject_module,
    )
    build.update(
        {
            "official_manifest": {
                "file": manifest_path.name,
                "sha256": sha256(manifest_path),
            },
            "official_raw_sha256": raw_hashes,
            "producer_source_sha256": source_files,
            "study_listfile_sha256": LISTFILE_SHA256,
            "construction_stages": construction_stages,
            "event_scan": {
                stem: {
                    key: value
                    for key, value in summary.items()
                    if key != "subjects_with_any_nonnumeric_value"
                }
                | {
                    "subjects_with_any_nonnumeric_value": len(
                        summary["subjects_with_any_nonnumeric_value"]
                    ),
                    "nonnumeric_subject_id_sha256": hashlib.sha256(
                        "\n".join(
                            str(value)
                            for value in summary[
                                "subjects_with_any_nonnumeric_value"
                            ]
                        ).encode()
                    ).hexdigest(),
                }
                for stem, summary in table_summaries.items()
            },
        }
    )
    write_json(source_root / "source_build.json", build)
    print(json.dumps({key: build[key] for key in ("samples", "subjects")}, indent=2))


if __name__ == "__main__":
    main()
