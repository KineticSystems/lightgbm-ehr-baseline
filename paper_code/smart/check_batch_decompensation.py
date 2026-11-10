"""Recreate SMART's complete retained decompensation loader/collate boundary from MEDS."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from export_decompensation import event_batches  # noqa: E402
from cohort_source import (  # noqa: E402
    SEEDS,
    array_hash,
    assert_mimic_release,
    sha256,
    channel_info,
    code_vocabulary,
    configuration,
    ordered_text_hash,
    variable_names,
)
from cohort_decompensation import (  # noqa: E402
    EXPECTED_OBSERVED_RATE,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TOTAL,
    NOTEBOOK_PERIOD_LENGTH,
    load_author_notebook_boundary,
    load_build_report,
    load_cohort,
    observed_rate,
    replica_task_id,
)


# Bounds for the float32 storage rounding described in reconstruct_boundary_from_meds.
MEDS_FLOAT32_RTOL = 1e-6
MEDS_FLOAT32_ATOL = 1e-4

EVENT_COLUMNS = (
    "subject_id",
    "time",
    "code",
    "numeric_value",
    "sample_id",
    "source_row",
    "sequence_row",
    "source_time",
)


def compare_all_source_events(cache_root: Path, cohort: pl.DataFrame, events_path: Path):
    """Compare every source event against MEDS in deterministic source-row blocks."""
    parquet = pq.ParquetFile(events_path)
    expected = event_batches(cache_root, cohort)
    row_groups = iter(range(parquet.num_row_groups))
    checked = 0
    for expected_frame in expected:
        try:
            row_group = next(row_groups)
        except StopIteration as error:
            raise AssertionError("MEDS has fewer event row groups than the source") from error
        actual = pl.from_arrow(parquet.read_row_group(row_group, columns=list(EVENT_COLUMNS)))
        actual = actual.sort("source_row").select(EVENT_COLUMNS)
        expected_frame = expected_frame.select(EVENT_COLUMNS)
        if actual.schema != expected_frame.schema:
            raise AssertionError(f"SMART MEDS event schema differs: {actual.schema}")
        if not actual.to_arrow().equals(expected_frame.to_arrow()):
            raise AssertionError(f"SMART MEDS events differ in row group {row_group}")
        checked += actual.height
    try:
        extra = next(row_groups)
    except StopIteration:
        extra = None
    if extra is not None:
        raise AssertionError("MEDS has extra event row groups")
    return checked


def _notebook_arrays_from_rows(
    rows: list[tuple[int, list[tuple[int, float]]]],
    n_bins: int,
    variables: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the released decompensation notebook cell over reconstructed rows.

    ``rows`` is the ordered per-sample list of ``(hour, [(variable_index, value)])``
    for the rows SMART's reader returns; a variable absent from a row is an
    empty cell in the released CSV.
    """
    config = configuration()
    info = channel_info()
    categorical = config["is_categorical_channel"]
    normal_values = config["normal_values"]
    data = np.zeros((len(variables), n_bins), dtype=np.float32)
    mask = np.zeros((len(variables), n_bins), dtype=np.float32)
    last_time = -1
    for hour, present in rows:
        if hour == n_bins:
            hour -= 1
        if hour > n_bins:
            break
        observed = dict(present)
        for variable_index, variable in enumerate(variables):
            if variable_index not in observed:
                if mask[variable_index, hour] == 0 and hour - last_time > 0:
                    if categorical[variable]:
                        fill = info[variable]["values"][normal_values[variable]]
                    else:
                        fill = float(normal_values[variable])
                    data[variable_index, last_time + 1 : hour + 1] = fill
            else:
                mask[variable_index, hour] += 1
                data[variable_index, hour] += observed[variable_index]
        last_time = hour
    with np.errstate(divide="ignore", invalid="ignore"):
        data = np.where(mask > 0, data / mask, data)
    mask = np.where(mask > 0, 1, 0)
    return data.T, np.ascontiguousarray(mask.T, dtype=np.int64)


def reconstruct_boundary_from_meds(
    events_path: Path, cohort: pl.DataFrame, boundary: dict
) -> dict:
    """Rebuild every author notebook array from MEDS alone and compare."""
    variables = tuple(variable_names())
    config = configuration()
    info = channel_info()
    categorical = config["is_categorical_channel"]
    variable_index = {name: index for index, name in enumerate(variables)}
    position_of = {
        int(sample_id): position
        for position, sample_id in enumerate(cohort["sample_id"].to_list())
    }
    period_of = {
        int(sample_id): float(period)
        for sample_id, period in zip(
            cohort["sample_id"].to_list(), cohort["period_length"].to_list()
        )
    }
    parquet = pq.ParquetFile(events_path)
    pending_sample: int | None = None
    pending_rows: dict[int, list[tuple[int, float]]] = {}
    pending_hours: dict[int, float] = {}
    checked = 0
    exact_samples = 0
    max_absolute = 0.0
    max_relative = 0.0

    def flush(sample_id: int) -> None:
        nonlocal checked, exact_samples, max_absolute, max_relative
        n_bins = min(
            int(period_of[sample_id] + 1 - 1e-6), NOTEBOOK_PERIOD_LENGTH
        )
        ordered = [
            (int(np.floor(pending_hours[key])), pending_rows[key])
            for key in sorted(pending_rows)
        ]
        data, mask = _notebook_arrays_from_rows(ordered, n_bins, variables)
        position = position_of[sample_id]
        # The observation mask must be bit-exact.
        np.testing.assert_array_equal(mask, boundary["mask"][position])
        expected = boundary["raw"][position]
        if np.array_equal(data, expected):
            exact_samples += 1
        else:
            # The MEDS contract stores numeric_value as float32, so a value whose
            # released decimal text is not a float32 changes the last ULP of the
            # author cell mean.  Bound it rather than pretending it is exact.
            difference = np.abs(data.astype(np.float64) - expected.astype(np.float64))
            scale = np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
            max_absolute = max(max_absolute, float(difference.max()))
            max_relative = max(max_relative, float((difference / scale).max()))
            np.testing.assert_allclose(data, expected, rtol=MEDS_FLOAT32_RTOL, atol=MEDS_FLOAT32_ATOL)
        checked += 1

    for group in range(parquet.num_row_groups):
        # The writer sorts each batch before writing it, so the deterministic
        # generator order (and therefore the contiguous per-sample blocks) is
        # recovered by sorting each row group on source_row.
        frame = pl.from_arrow(
            parquet.read_row_group(
                group,
                columns=[
                    "sample_id",
                    "code",
                    "numeric_value",
                    "sequence_row",
                    "source_time",
                    "source_row",
                ],
            )
        ).sort("source_row")
        for sample_id, code, numeric, sequence_row, source_time in zip(
            frame["sample_id"].to_list(),
            frame["code"].to_list(),
            frame["numeric_value"].to_list(),
            frame["sequence_row"].to_list(),
            frame["source_time"].to_list(),
        ):
            sample_id = int(sample_id)
            if pending_sample is None:
                pending_sample = sample_id
            elif sample_id != pending_sample:
                flush(pending_sample)
                pending_sample = sample_id
                pending_rows = {}
                pending_hours = {}
            if "//" in code:
                variable, value = code.split("//", 1)
                scalar = float(info[variable]["values"][value])
                if not categorical[variable]:
                    raise AssertionError(f"{variable} is not a released categorical channel")
            else:
                variable = code
                if categorical[variable]:
                    raise AssertionError(f"{variable} lost its released categorical encoding")
                scalar = float(numeric)
            row_key = int(sequence_row)
            pending_rows.setdefault(row_key, []).append(
                (variable_index[variable], scalar)
            )
            pending_hours[row_key] = float(source_time)
    if pending_sample is not None:
        flush(pending_sample)
    if checked != EXPECTED_TOTAL:
        raise AssertionError(
            f"MEDS reconstruction covered {checked} of {EXPECTED_TOTAL} samples"
        )
    return {
        "samples_reconstructed": checked,
        "samples_bit_exact": exact_samples,
        "samples_within_float32_storage_rounding": checked - exact_samples,
        "max_absolute_deviation": max_absolute,
        "max_relative_deviation": max_relative,
        "deviation_cause": (
            "the shared MEDS contract stores numeric_value as float32; a released "
            "decimal that is not exactly a float32 shifts the author per-bin mean "
            "by at most one float32 ULP.  Observation masks, labels, landmarks, "
            "row structure and codes are bit-exact."
        ),
    }


def _author_modules():
    from data import mimiciii as author_loader
    from data.dataloader import collate_fn

    return author_loader, collate_fn


def verify_author_loader_and_collate(boundary: dict, cohort: pl.DataFrame, seed: int):
    """Invoke the exact released loader and collate every one of its examples."""
    author_loader, collate_fn = _author_modules()
    # Keep the per-example arrays as numpy (the released loader only indexes and
    # measures them, and collate_fn calls torch.tensor on each): building Python
    # lists of ~16M floats would cost gigabytes for no fidelity gain.
    payload = (
        boundary["normalized"],
        boundary["labels"],
        boundary["mask"],
        boundary["names"],
    )
    previous_pickle_load = author_loader.pickle.load
    previous_open = getattr(author_loader, "open", None)
    author_loader.pickle.load = lambda _: payload
    author_loader.open = lambda *_args, **_kwargs: io.BytesIO(b"")
    try:
        random.seed(seed)
        datasets = author_loader.load_mimic_iii_decompensation()
    finally:
        author_loader.pickle.load = previous_pickle_load
        if previous_open is None:
            delattr(author_loader, "open")
        else:
            author_loader.open = previous_open

    indices = list(range(EXPECTED_TOTAL))
    random.Random(seed).shuffle(indices)
    split_indices = {
        "train": indices[: EXPECTED_SPLIT_COUNTS["train"]],
        "validation": indices[
            EXPECTED_SPLIT_COUNTS["train"] : EXPECTED_SPLIT_COUNTS["train"]
            + EXPECTED_SPLIT_COUNTS["validation"]
        ],
        "test": indices[
            EXPECTED_SPLIT_COUNTS["train"] + EXPECTED_SPLIT_COUNTS["validation"] :
        ],
    }
    lengths = np.asarray(
        [array.shape[0] for array in boundary["normalized"]], dtype=np.int64
    )
    result = {}
    sample_ids = cohort["sample_id"].to_numpy()
    for split, dataset in zip(("train", "validation", "test"), datasets):
        expected_indices = split_indices[split]
        if len(dataset) != len(expected_indices):
            raise AssertionError("released SMART loader split count changed")
        x_digest = hashlib.sha256()
        mask_digest = hashlib.sha256()
        label_digest = hashlib.sha256()
        lens_digest = hashlib.sha256()
        for start in range(0, len(dataset), 512):
            stop = min(start + 512, len(dataset))
            batch = collate_fn([dataset[index] for index in range(start, stop)])
            selected = np.asarray(expected_indices[start:stop], dtype=np.int64)
            expected_x = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(boundary["normalized"][index]) for index in selected],
                True,
            )
            expected_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(boundary["mask"][index]) for index in selected], True
            )
            expected_labels = torch.tensor(boundary["labels"][selected].tolist())
            expected_lens = torch.tensor(lengths[selected].tolist())
            if batch["x"].dtype != torch.float32:
                raise AssertionError("released SMART x collate dtype changed")
            if batch["mask"].dtype != torch.int64:
                raise AssertionError(
                    "released SMART decompensation mask collate dtype changed "
                    "(the notebook pickles an integer mask for this task)"
                )
            if batch["labels"].dtype != torch.int64 or batch["lens"].dtype != torch.int64:
                raise AssertionError("released SMART label/lens collate dtype changed")
            for key, expected in (
                ("x", expected_x),
                ("mask", expected_mask),
                ("labels", expected_labels),
                ("lens", expected_lens),
            ):
                if batch[key].shape != expected.shape:
                    raise AssertionError(
                        f"released SMART {split} collate shape differs: {key}"
                    )
                if not torch.equal(batch[key], expected):
                    raise AssertionError(f"released SMART {split} collate field differs: {key}")
            x_digest.update(batch["x"].contiguous().numpy().tobytes())
            mask_digest.update(batch["mask"].contiguous().numpy().tobytes())
            label_digest.update(batch["labels"].contiguous().numpy().tobytes())
            lens_digest.update(batch["lens"].contiguous().numpy().tobytes())
        result[split] = {
            "count": len(dataset),
            "ordered_sample_id_sha256": ordered_text_hash(
                sample_ids[np.asarray(expected_indices, dtype=np.int64)]
            ),
            "x_float32_sha256": x_digest.hexdigest(),
            "mask_float32_sha256": mask_digest.hexdigest(),
            "label_int64_sha256": label_digest.hexdigest(),
            "lens_int64_sha256": lens_digest.hexdigest(),
        }
    return result, split_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    task_root = args.task_root.resolve()
    build = load_build_report(cache_root)
    if assert_mimic_release(args.mimic_root.resolve()) != build["release_sha256"]:
        raise AssertionError("SMART source report does not match exact local MIMIC-III 1.4")
    cohort = load_cohort(cache_root)
    boundary = load_author_notebook_boundary(cache_root)
    vocabulary = code_vocabulary()

    if len(boundary["normalized"]) != EXPECTED_TOTAL:
        raise AssertionError("SMART normalized example count changed")
    lengths = np.asarray([array.shape[0] for array in boundary["raw"]], dtype=np.int64)
    if lengths.max() > NOTEBOOK_PERIOD_LENGTH or lengths.min() < 1:
        raise AssertionError("SMART decompensation sequence length left [1, 24]")
    for array in boundary["mask"]:
        if not set(np.unique(array)) <= {0.0, 1.0}:
            raise AssertionError("SMART observation mask is not binary")
    rate = observed_rate(boundary["mask"])
    if rate != EXPECTED_OBSERVED_RATE:
        raise AssertionError("SMART decompensation observed rate changed")

    reference_events_sha256: str | None = None
    reference_event_count: int | None = None
    reference_reconstruction: dict | None = None
    for replica, seed in enumerate(SEEDS, start=1):
        task_id = replica_task_id(replica)
        task_dir = task_root / task_id
        manifest = json.loads((task_dir / "manifest.json").read_text())
        if manifest["task"].get("modalities") != ["structured"]:
            raise AssertionError("SMART MEDS task retained a non-structured modality")
        if manifest["task"].get("note_codes") != []:
            raise AssertionError("SMART MEDS task declared note codes")
        if manifest["source"].get("selected_model_inputs") != ["structured"]:
            raise AssertionError("SMART batch check selected the wrong model input")
        if manifest["source"].get("paper_seed") != seed:
            raise AssertionError("SMART batch check selected the wrong replica")
        codes = tuple(
            pl.read_parquet(task_dir / manifest["meds"]["codes"]["file"])["code"].to_list()
        )
        if codes != vocabulary:
            raise AssertionError("SMART MEDS vocabulary differs from released channels/categories")

        events_path = task_dir / manifest["meds"]["events"]["file"]
        events_sha256 = manifest["meds"]["events"]["sha256"]
        if sha256(events_path) != events_sha256:
            raise AssertionError("SMART MEDS event file differs from its manifest hash")
        # The three replicas differ only in their seed-specific split; their event
        # streams are produced by the same generator.  Compare the stream in full
        # once, then require byte identity for the remaining replicas.
        if reference_events_sha256 is None or events_sha256 != reference_events_sha256:
            checked_events = compare_all_source_events(cache_root, cohort, events_path)
            reconstruction = reconstruct_boundary_from_meds(events_path, cohort, boundary)
            reconstruction["compared_in_full"] = True
            reference_events_sha256 = events_sha256
            reference_event_count = checked_events
            reference_reconstruction = reconstruction
        else:
            checked_events = reference_event_count
            reconstruction = dict(reference_reconstruction)
            reconstruction["compared_in_full"] = False
            reconstruction["identical_event_file_sha256"] = events_sha256
        if checked_events != manifest["event_count"]:
            raise AssertionError("SMART event parity count differs from manifest")

        loader_hashes, split_indices = verify_author_loader_and_collate(
            boundary, cohort, seed
        )
        index = pl.read_parquet(task_dir / manifest["meds"]["task_index"]["file"])
        sample_ids = cohort["sample_id"].to_numpy()
        for split, indices in split_indices.items():
            expected_ids = set(sample_ids[np.asarray(indices, dtype=np.int64)].tolist())
            actual_ids = set(index.filter(pl.col("split") == split)["sample_id"].to_list())
            if actual_ids != expected_ids:
                raise AssertionError(f"SMART MEDS {split} membership differs from author loader")

        output = {
            "status": build["status"],
            "replica": replica,
            "paper_seed": seed,
            "phase": "phase 2 second published binary outcome",
            "events_compared": checked_events,
            "event_vocabulary_sha256": ordered_text_hash(vocabulary),
            "meds_to_author_boundary_reconstruction": reconstruction,
            "author_boundary": {
                "examples": EXPECTED_TOTAL,
                "sequence_lengths": {
                    "min": int(lengths.min()),
                    "max": int(lengths.max()),
                    "sum": int(lengths.sum()),
                },
                "channels": 17,
                "normalized_dtype": "float32",
                "normalized_sha256": array_hash(np.concatenate(boundary["normalized"], axis=0)),
                "mask_dtype": "int64",
                "mask_sha256": array_hash(np.concatenate(boundary["mask"], axis=0)),
                "labels_dtype": "int64",
                "labels_sha256": array_hash(boundary["labels"]),
                "observed_rate": rate,
                "lens": "per-example min(int(t + 1 - 1e-6), 24)",
                "retained_source_rows": boundary["retained_source_rows"],
                "retained_source_rows_with_no_observed_cell": boundary[
                    "retained_source_rows_with_no_observed_cell"
                ],
            },
            "split_loader_collate_hashes": loader_hashes,
            "ordering_limitation": build["fidelity_limitation"],
            "assertions": {
                "all_source_events_equal_meds": True,
                "meds_alone_rebuilds_every_author_notebook_array": True,
                "exact_vocabulary_and_note_exception_list": True,
                "every_non_note_code_is_released_structured_input": True,
                "no_imaging_or_notes_retained": True,
                "all_author_loader_examples_recreated": True,
                "x_mask_label_lens_shape_dtype_value_equal": True,
                "released_example_split_preserved": True,
            },
        }
        (task_dir / "audit").mkdir(parents=True, exist_ok=True)
        (task_dir / "audit" / "full_batch_recreation.json").write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n"
        )
        print(
            json.dumps(
                {"task": task_id, "status": output["status"], "events": checked_events},
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
