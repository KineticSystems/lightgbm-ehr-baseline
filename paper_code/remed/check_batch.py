"""Recreate every plain MEDS emission from the snub h5 and reconcile.

The export decodes the model's own [3,128] events into plain codes/values
(docs/remed_plain_timeline.md). This check independently rebuilds the same
decoder (fresh survey over TRAIN stays) and verifies, event for event in
export order, that the task's events Parquet is exactly the decoder's
output — codes, numeric values, times, and sequence bookkeeping — and that
the declared vocabulary equals the emitted code set.

With ``--mimic-raw-root`` it additionally runs an h5-independent spot
reconciliation: for a deterministic subset of stays, our decoded numeric
lab events must appear in the raw MIMIC-IV ``hosp/labevents`` rows joined
on the stay's admission (matching label, minute, and value), proving the
decode chain reproduces the source tables and not merely the h5.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from export_source import (  # noqa: E402
    BERT_VOCAB_PATH,
    CLAMP_INTERVAL_DURATIONS,
    PREDICTION_MINUTES,
    build_decoder,
)
from plain_events import clamp_duration  # noqa: E402
from cohort_source import (  # noqa: E402
    PREDICTION_HOURS,
    load_samples,
    source_paths,
)

SPOT_STAYS = 200
SPOT_MINIMUM_COVERAGE = 0.99


def actual_event_rows(path: Path):
    """Yield exported rows in source_row order.

    The shared writer's sample-attach join shuffles rows within each write
    batch, so physical order is not emission order. Rows are drained
    through a bounded reorder buffer: emit whenever the next expected
    source_row is present; anything left at the end is a hard failure.
    """
    parquet = pq.ParquetFile(path)
    columns = [
        "subject_id",
        "time",
        "code",
        "numeric_value",
        "sample_id",
        "source_row",
        "sequence_row",
        "source_time",
    ]
    buffer: dict[int, tuple] = {}
    expected_row = 0
    maximum_buffer = 0
    for batch in parquet.iter_batches(batch_size=250_000, columns=columns):
        values = batch.to_pydict()
        for position in range(batch.num_rows):
            row = tuple(values[name][position] for name in columns)
            source_row = row[5]
            if source_row in buffer or source_row < expected_row:
                raise AssertionError(f"duplicate source_row {source_row}")
            buffer[source_row] = row
            while expected_row in buffer:
                stored = buffer.pop(expected_row)
                yield dict(zip(columns, stored))
                expected_row += 1
        maximum_buffer = max(maximum_buffer, len(buffer))
        if len(buffer) > 5_000_000:
            raise AssertionError(
                "source_row reorder buffer exceeded its bound; the events "
                "file is not batch-locally ordered"
            )
    while expected_row in buffer:
        yield dict(zip(columns, buffer.pop(expected_row)))
        expected_row += 1
    if buffer:
        raise AssertionError(
            f"{len(buffer)} rows have out-of-range source_row values"
        )


def expected_emission_rows(source_root: Path, samples, decoder):
    source_row = 0
    paths = source_paths(source_root)
    with h5py.File(paths["h5"], "r") as source:
        ehr = source["ehr"]
        for sample in samples:
            group = ehr[str(sample.stay_id)]
            times = np.asarray(group["time"][:], dtype=np.int64)
            icu_us = (
                sample.prediction_time_us - PREDICTION_HOURS * 3_600_000_000
            )
            events = np.asarray(group["hi"][:], dtype=np.int16)
            for sequence_row, (event, minute) in enumerate(zip(events, times)):
                for code, numeric, clamp in decoder.emissions(event):
                    if (
                        clamp
                        and CLAMP_INTERVAL_DURATIONS
                        and numeric is not None
                    ):
                        numeric = clamp_duration(
                            numeric, float(minute), PREDICTION_MINUTES
                        )
                    yield {
                        "subject_id": sample.subject_id,
                        "time_us": icu_us + int(minute) * 60_000_000,
                        "code": code,
                        "numeric_value": numeric,
                        "sample_id": sample.stay_id,
                        "source_row": source_row,
                        "sequence_row": sequence_row,
                        "source_time": float(minute),
                    }
                    source_row += 1


def spot_check_labevents(
    mimic_raw_root: Path, source_root: Path, samples, task_dir: Path
) -> dict[str, object]:
    """Verify decoded LAB numerics against raw hosp/labevents rows."""
    chosen = sorted(samples, key=lambda s: s.stay_id)[::
        max(1, len(samples) // SPOT_STAYS)][:SPOT_STAYS]
    by_hadm = {sample.hadm_id: sample for sample in chosen}
    stay_ids = {sample.stay_id for sample in chosen}

    def label_key(label: str) -> str:
        # Decoded labels pass through the BERT tokenizer, which inserts
        # spaces at letter/digit/punctuation boundaries ("pO2" -> "po 2");
        # compare with all whitespace removed, lowercased.
        return "".join(label.lower().split())

    labels = {}
    with gzip.open(mimic_raw_root / "hosp" / "d_labitems.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            labels[row["itemid"]] = label_key(row["label"])

    raw_pairs: dict[int, set[tuple[str, int, float]]] = defaultdict(set)
    with gzip.open(mimic_raw_root / "hosp" / "labevents.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            hadm = row.get("hadm_id", "")
            if not hadm or int(hadm) not in by_hadm:
                continue
            valuenum = row.get("valuenum", "")
            if not valuenum:
                continue
            sample = by_hadm[int(hadm)]
            icu_us = (
                sample.prediction_time_us - PREDICTION_HOURS * 3_600_000_000
            )
            chart_us = np.datetime64(row["charttime"], "us").astype(np.int64)
            minute = int((chart_us - icu_us) // 60_000_000)
            label = labels.get(row["itemid"], "")
            value4 = round(float(valuenum), 4)
            # Cover floor-vs-truncate off-by-one on negative pre-ICU minutes.
            for shifted in (minute - 1, minute, minute + 1):
                raw_pairs[sample.stay_id].add((label, shifted, value4))

    checked = 0
    matched = 0
    unmatched_examples: list[tuple] = []
    parquet = pq.ParquetFile(task_dir / "data" / "events.parquet")
    for batch in parquet.iter_batches(
        batch_size=250_000,
        columns=["sample_id", "code", "numeric_value", "source_time"],
    ):
        for sample_id, code, numeric, minute in zip(
            batch["sample_id"].to_pylist(),
            batch["code"].to_pylist(),
            batch["numeric_value"].to_pylist(),
            batch["source_time"].to_pylist(),
        ):
            if sample_id not in stay_ids or numeric is None:
                continue
            if not code.startswith("MIMICIV//LAB//"):
                continue
            label = code[len("MIMICIV//LAB//"):].split("//")[0]
            checked += 1
            key = (
                label_key(label),
                int(minute),
                round(float(np.float32(numeric)), 4),
            )
            if key in raw_pairs.get(sample_id, ()):
                matched += 1
            elif len(unmatched_examples) < 12:
                unmatched_examples.append((sample_id,) + key)
    coverage = (matched / checked) if checked else 1.0
    if coverage < SPOT_MINIMUM_COVERAGE:
        for example in unmatched_examples:
            print("UNMATCHED:", example, file=sys.stderr)
        raise AssertionError(
            f"raw labevents spot coverage {coverage:.4f} below "
            f"{SPOT_MINIMUM_COVERAGE}"
        )
    return {
        "stays": len(chosen),
        "lab_numeric_events_checked": checked,
        "matched_in_raw_labevents": matched,
        "coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--mimic-raw-root", type=Path, default=None)
    args = parser.parse_args()

    samples, _ = load_samples(args.source_root)
    decoder = build_decoder(args.source_root, samples)

    events_path = args.task_dir / "data" / "events.parquet"
    total_events = 0
    total_emissions = 0
    emitted_codes: set[str] = set()
    expected = expected_emission_rows(args.source_root, samples, decoder)
    last_sequence: tuple[int, int] | None = None
    for actual in actual_event_rows(events_path):
        try:
            reference = next(expected)
        except StopIteration:
            raise AssertionError("MEDS has more rows than the decoder emits")
        actual_time_us = np.datetime64(actual["time"], "us").astype(np.int64)
        expected_numeric = (
            None
            if reference["numeric_value"] is None
            else float(np.float32(reference["numeric_value"]))
        )
        actual_numeric = (
            None
            if actual["numeric_value"] is None
            else float(np.float32(actual["numeric_value"]))
        )
        if (
            actual["subject_id"] != reference["subject_id"]
            or int(actual_time_us) != reference["time_us"]
            or actual["code"] != reference["code"]
            or actual_numeric != expected_numeric
            or actual["sample_id"] != reference["sample_id"]
            or actual["source_row"] != reference["source_row"]
            or actual["sequence_row"] != reference["sequence_row"]
            or float(actual["source_time"]) != reference["source_time"]
        ):
            raise AssertionError(
                f"row {actual['source_row']} disagrees with the decoder: "
                f"{actual} vs {reference}"
            )
        emitted_codes.add(actual["code"])
        total_emissions += 1
        key = (actual["sample_id"], actual["sequence_row"])
        if key != last_sequence:
            total_events += 1
            last_sequence = key
    try:
        next(expected)
        raise AssertionError("the decoder emits more rows than MEDS holds")
    except StopIteration:
        pass

    declared = set(
        pq.read_table(args.task_dir / "metadata" / "codes.parquet")
        .column(0)
        .to_pylist()
    )
    if declared != emitted_codes:
        raise AssertionError(
            f"declared vocabulary ({len(declared)}) differs from emitted "
            f"codes ({len(emitted_codes)})"
        )

    spot = None
    if args.mimic_raw_root is not None:
        spot = spot_check_labevents(
            args.mimic_raw_root, args.source_root, samples, args.task_dir
        )

    report = {
        "status": "exact_at_released_ehrdataset_input_plain_decode",
        "samples": len(samples),
        "events": total_events,
        "emissions": total_emissions,
        "codes": len(emitted_codes),
        "decode_statistics": decoder.statistics.as_dict(),
        "raw_labevents_spot_check": spot
        if spot is not None
        else "skipped (no --mimic-raw-root)",
        "checks": [
            "every MEDS row equals the fresh decoder's emission, in order",
            "declared vocabulary equals the emitted code set",
            "bert vocabulary pinned: " + str(BERT_VOCAB_PATH.name),
        ]
        + (
            ["decoded lab numerics found in raw hosp/labevents"]
            if spot is not None
            else []
        ),
    }
    output = args.task_dir / "audit" / "full_batch_recreation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
