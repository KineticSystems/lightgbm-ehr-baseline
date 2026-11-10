"""Exact SSML/MIMIC3-benchmark in-hospital-mortality boundary projected to MEDS.

The SSML release ships one reader for the MIMIC-III benchmark *multitask*
listfiles (``reader.MultitaskReader``).  It parses and returns the
``in-hospital mortality task (pos;mask;label)`` triple for every stay, so the
cohort, labels, split, and prediction point of the Table 4 in-hospital
mortality column are fully determined inside the released boundary.  The
released trainer, however, wires only decompensation; see
``released_ihm_training_path`` in :func:`expected_source` for that disclosed
release defect.

Nothing in this module is imported by the sealed ``decompensation`` exporter:
it only imports from ``cohort_source`` so the raw occurrence stream, the
released 59/76-channel transform spec, and the MEDS event schema are literally
the same code paths the sealed task used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl

from cohort_source import (
    ANCHOR_US,
    COMMON_SPLIT,
    CONFIG_PATH,
    EPS,
    HOUR_US,
    M3B_COMMIT,
    NORMALIZER_PATH,
    SOURCE_FILENAME,
    SOURCE_SPLIT,
    SPLIT_MAP,
    STUDY_COMMIT,
    _event_frame,
    load_raw_stay,
    load_transform_spec,
    sha256_file,
)


TASK_ID = "mortality_48h"
# ``create_multitask.py`` writes ``ihm_position = 47`` for every unmasked stay
# and ``fixed_hours = 48.0``.  Both are asserted per row rather than assumed.
IHM_POSITION = 47
WINDOW_HOURS = 48
LISTFILE_HEADER = (
    "filename,length of stay,in-hospital mortality task (pos;mask;label),"
    "length of stay task (masks;labels),phenotyping task (labels),"
    "decompensation task (masks;labels)"
)
PREPARATION_LISTFILE_KEY = {
    "train": "train_listfile",
    "val": "validation_listfile",
    "test": "test_listfile",
}
RELEASED_LABEL_DEFINITION = (
    "mimic3-benchmarks create_multitask.py: ihm_label = the episode 'Mortality' "
    "flag; ihm_mask = 1 unless the ICU length of stay is below 48 hours "
    "(los < 48 - 1e-6) or the first in-ICU event is later than 48 hours "
    "(event_times[0] > 48 + 1e-6); ihm_position = 47 whenever the mask is 1 and "
    "0 otherwise.  The released SSML reader.MultitaskReader.read_example returns "
    "exactly that (pos;mask;label) triple per stay; this task keeps every "
    "mask == 1 row and uses its third subfield as the binary label."
)


def source_paths(source_root: Path) -> dict[str, Path]:
    """Resolve the shared attributable MIMIC-III benchmark multitask build."""

    root = Path(source_root).resolve()
    multitask = root / "m3b_data" / "multitask"
    return {
        "preparation": root / "preparation.json",
        "multitask": multitask,
        "train_listfile": multitask / "train_listfile.csv",
        "val_listfile": multitask / "val_listfile.csv",
        "test_listfile": multitask / "test_listfile.csv",
        "train": multitask / "train",
        "test": multitask / "test",
    }


def assert_source_ready(source_root: Path) -> dict[str, Path]:
    paths = source_paths(source_root)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    preparation = json.loads(paths["preparation"].read_text())
    if preparation.get("status") != "exact_source":
        raise AssertionError("shared benchmark preparation is not the exact source build")
    if preparation.get("commits", {}).get("mimic3_benchmarks") != M3B_COMMIT:
        raise AssertionError("shared benchmark preparation commit changed")
    for split in SPLIT_MAP:
        expected = preparation["source_artifacts"][PREPARATION_LISTFILE_KEY[split]]["sha256"]
        if sha256_file(paths[f"{split}_listfile"]) != expected:
            raise AssertionError(f"{split} multitask listfile differs from the sealed build")
    return paths


def listfile_path(paths: dict[str, Path], split: str) -> Path:
    return paths[f"{split}_listfile"]


def stay_path(paths: dict[str, Path], common_split: str, filename: str) -> Path:
    return paths[SOURCE_SPLIT[common_split]] / filename


def _iter_listfile(path: Path) -> Iterator[tuple[int, str, float, str]]:
    """Stream the released multitask listfile without materializing its lists."""

    with path.open() as handle:
        header = handle.readline().rstrip("\n")
        if header != LISTFILE_HEADER:
            raise AssertionError(f"released multitask listfile header changed in {path}")
        for row_index, line in enumerate(handle):
            fields = line.rstrip("\n").split(",")
            if len(fields) != 6:
                raise AssertionError(f"released multitask listfile row shape changed in {path}")
            yield row_index, fields[0], float(fields[1]), fields[2]


def build_samples(source_root: Path) -> tuple[pl.DataFrame, dict[str, object]]:
    """One sample per released multitask row whose in-hospital mortality mask is 1."""

    paths = assert_source_ready(source_root)
    frames: list[pl.DataFrame] = []
    summaries: dict[str, dict[str, object]] = {}
    tuple_hash = hashlib.sha256()
    order_start = 0
    for split in SPLIT_MAP:
        common = COMMON_SPLIT[split]
        filenames: list[str] = []
        labels: list[int] = []
        list_rows: list[int] = []
        stay_hours: list[float] = []
        rows = masked_out = 0
        for row_index, filename, length_of_stay, ihm in _iter_listfile(
            listfile_path(paths, split)
        ):
            rows += 1
            parts = ihm.split(";")
            if len(parts) != 3:
                raise AssertionError("released ihm field is not pos;mask;label")
            position, mask, label = (int(part) for part in parts)
            if mask not in (0, 1) or label not in (0, 1):
                raise AssertionError("released ihm mask/label is not binary")
            if mask == 0:
                if position != 0:
                    raise AssertionError("released producer changed the masked ihm position")
                masked_out += 1
                continue
            if position != IHM_POSITION:
                raise AssertionError(
                    f"released ihm prediction position is {position}, not {IHM_POSITION}"
                )
            if length_of_stay < WINDOW_HOURS - EPS:
                raise AssertionError("released ihm mask kept a stay shorter than the window")
            filenames.append(filename)
            labels.append(label)
            list_rows.append(row_index)
            stay_hours.append(length_of_stay)
            tuple_hash.update(
                f"{split}|{filename}|{row_index}|{position}|{label}\n".encode()
            )
        count = len(filenames)
        if not count:
            raise AssertionError(f"{split} multitask listfile has no eligible ihm row")
        frame = pl.DataFrame(
            {
                "source_filename": pl.Series(filenames, dtype=pl.String),
                "boolean_value": pl.Series(
                    [bool(label) for label in labels], dtype=pl.Boolean
                ),
                "source_list_row": pl.Series(list_rows, dtype=pl.Int64),
            }
        )
        match = frame["source_filename"].str.extract_groups(SOURCE_FILENAME.pattern)
        if match.struct.field("subject").null_count():
            raise AssertionError(f"{split} listfile has an unrecognized stay filename")
        frame = frame.with_columns(
            match.struct.field("subject").cast(pl.Int64).alias("subject_id"),
            match.struct.field("episode").cast(pl.Int32).alias("episode_number"),
            pl.lit(COMMON_SPLIT[split]).alias("split"),
            pl.lit(ANCHOR_US).cast(pl.Datetime("us")).alias("observation_start"),
            pl.lit(ANCHOR_US + WINDOW_HOURS * HOUR_US)
            .cast(pl.Datetime("us"))
            .alias("prediction_time"),
            pl.lit(WINDOW_HOURS, dtype=pl.Int32).alias("source_cutoff_hour"),
            pl.lit(IHM_POSITION, dtype=pl.Int32).alias("ihm_position"),
            pl.Series(
                "source_order",
                np.arange(order_start, order_start + count),
                dtype=pl.Int64,
            ),
        )
        if frame["source_filename"].n_unique() != count:
            raise AssertionError(f"{split} eligible ihm rows repeat a stay filename")
        summaries[common] = {
            "source_rows": rows,
            "ihm_masked_out": masked_out,
            "samples": count,
            "positive": int(sum(labels)),
            "negative": count - int(sum(labels)),
            "subjects": frame["subject_id"].n_unique(),
            "stays": count,
            "minimum_length_of_stay_hours": min(stay_hours),
            "maximum_length_of_stay_hours": max(stay_hours),
        }
        frames.append(frame)
        order_start += count
    samples = pl.concat(frames, how="vertical").sort("split", "source_filename")
    samples = samples.with_columns(
        pl.Series(
            "sample_id", np.arange(1, samples.height + 1, dtype=np.int64), dtype=pl.Int64
        )
    ).select(
        "subject_id",
        "sample_id",
        "observation_start",
        "prediction_time",
        "boolean_value",
        "split",
        "source_filename",
        "source_list_row",
        "source_cutoff_hour",
        "ihm_position",
        "source_order",
        "episode_number",
    )
    return samples, {
        "splits": summaries,
        "samples": samples.height,
        "ihm_position": IHM_POSITION,
        "observation_window_hours": float(WINDOW_HOURS),
        "ordered_eligible_tuple_sha256": tuple_hash.hexdigest(),
    }


def _ordered(samples: pl.DataFrame) -> pl.DataFrame:
    return samples.sort("sample_id")


def count_events(
    source_root: Path, samples: pl.DataFrame | None = None
) -> tuple[int, dict[str, int]]:
    paths = assert_source_ready(source_root)
    spec = load_transform_spec()
    if samples is None:
        samples, _ = build_samples(source_root)
    totals = {split: 0 for split in ("train", "validation", "test")}
    beyond_window = 0
    for row in _ordered(samples).iter_rows(named=True):
        raw = load_raw_stay(
            stay_path(paths, row["split"], row["source_filename"]), spec, WINDOW_HOURS
        )
        totals[row["split"]] += int(raw["times"].size)
        beyond_window += int((raw["times"] > WINDOW_HOURS).sum())
    if beyond_window:
        raise AssertionError(
            f"{beyond_window} retained occurrences postdate the {WINDOW_HOURS}h "
            "prediction time within the released 1e-6 hour reader tolerance"
        )
    return sum(totals.values()), totals


def iter_events(
    source_root: Path,
    samples: pl.DataFrame | None = None,
    batch_rows: int = 2_000_000,
) -> Iterator[pl.DataFrame]:
    paths = assert_source_ready(source_root)
    spec = load_transform_spec()
    if samples is None:
        samples, _ = build_samples(source_root)
    buffer = {
        key: []
        for key in (
            "subject_id", "time", "code", "numeric_value", "numeric_valid",
            "sample_id", "source_row", "sequence_row", "source_time",
        )
    }
    buffered = emitted = 0
    for row in _ordered(samples).iter_rows(named=True):
        raw = load_raw_stay(
            stay_path(paths, row["split"], row["source_filename"]), spec, WINDOW_HOURS
        )
        total = int(raw["times"].size)
        buffer["subject_id"].append(
            np.full(total, int(row["subject_id"]), dtype=np.int64)
        )
        buffer["time"].append(
            ANCHOR_US + np.rint(raw["times"] * HOUR_US).astype(np.int64)
        )
        buffer["code"].append(raw["codes"])
        buffer["numeric_value"].append(raw["numeric_values"])
        buffer["numeric_valid"].append(raw["numeric_valid"])
        buffer["sample_id"].append(np.full(total, int(row["sample_id"]), dtype=np.int64))
        buffer["source_row"].append(raw["source_rows"])
        buffer["sequence_row"].append(raw["sequence_rows"])
        buffer["source_time"].append(raw["times"])
        buffered += total
        previous = emitted
        emitted += 1
        if buffered >= batch_rows:
            yield _event_frame(buffer)
            for value in buffer.values():
                value.clear()
            buffered = 0
        if emitted // 2_000 != previous // 2_000:
            print(f"exported event prefixes for {emitted:,} stays")
    if buffered:
        yield _event_frame(buffer)


def expected_source(source_root: Path, event_count: int) -> dict[str, object]:
    paths = assert_source_ready(source_root)
    return {
        "repositories": [
            {
                "url": "https://github.com/lidazhang/ssml-time-series-heterogeneity.git",
                "commit": STUDY_COMMIT,
            },
            {
                "url": "https://github.com/YerevaNN/mimic3-benchmarks.git",
                "commit": M3B_COMMIT,
            },
        ],
        "dataset": "MIMIC-III 1.4",
        "shared_m3b_preparation_sha256": sha256_file(paths["preparation"]),
        "listfile_sha256": {
            COMMON_SPLIT[split]: sha256_file(listfile_path(paths, split))
            for split in SPLIT_MAP
        },
        "discretizer_config_sha256": sha256_file(CONFIG_PATH),
        "normalizer_sha256": sha256_file(NORMALIZER_PATH),
        "event_count_precomputed": event_count,
        "released_label_definition": RELEASED_LABEL_DEFINITION,
        "retained_boundary": (
            "earliest attributable raw 17-channel occurrence stream truncated at the "
            "released 48-hour in-hospital-mortality prediction point, with exact "
            "fractional source times, repeated within-bin observations, and CSV row "
            "order; the hourly overwrite, 59-channel encoding, 17 masks, "
            "previous-value imputation, and normalization are recreated only in "
            "check_batch"
        ),
        "released_ihm_training_path": {
            "ihm_trainer_released": False,
            "reader_returns_ihm": True,
            "defect": (
                "The released study code wires only decompensation: data.py builds "
                "reader.DecompensationReader and loads the decompensation normalizer, "
                "and train.py hard-codes a decompensation DATASET_PATH.  Outside "
                "reader.py, in-hospital mortality appears only in the eval.py comment "
                "over the shared binary-metric function.  The released "
                "reader.MultitaskReader nonetheless parses and returns the exact "
                "(pos;mask;label) in-hospital mortality triple, so the data boundary "
                "-- cohort, labels, split, and prediction point -- is fully released "
                "even though the training entry point for this Table 4 column is not."
            ),
            "released_wiring_references": {
                "decompensation": [
                    "data.py:16 reader.DecompensationReader",
                    "data.py:27 decomp_ts1.0 normalizer state",
                    "train.py:79 DATASET_PATH .../data/decompensation",
                ],
                "in_hospital_mortality": [
                    "reader.py:279 process_ihm",
                    "reader.py:297 MultitaskReader row parse",
                    "reader.py:343 read_example returns 'ihm'",
                    "eval.py:4 shared binary-metric comment",
                ],
            },
        },
        "normalizer_choice": (
            "the released study loads exactly one normalizer state "
            "(decomp_ts1.0.input_str-previous.n1e5.start_time-zero) in data.py; because "
            "the in-hospital mortality trainer is unreleased, that released state is "
            "used for the deterministic transform recreated in check_batch.  MEDS "
            "stores raw pre-normalization occurrences, so the exported task does not "
            "depend on this choice"
        ),
        "paper_run_evaluation": {
            "exact_membership_recoverable": False,
            "heldout_test_exercised": False,
            "reason": (
                "no released in-hospital-mortality entry point exists; the released "
                "decompensation entrypoint freshly and without a seed shuffles 800 "
                "source rows for both train and validation and maps its evaluation "
                "call to the validation list"
            ),
        },
        "canonical_export": (
            "every released multitask train/validation/test row whose in-hospital "
            "mortality mask is 1"
        ),
        "modalities": ["structured"],
        "excluded_modalities": ["notes", "imaging"],
    }
