"""Recreate KEDGN's complete retained structured loader boundary from MEDS."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

import utils as kedgn_utils  # noqa: E402
from cohort_source import (  # noqa: E402
    AVERAGE_INTERVAL_SUFFIX,
    EXPECTED_MAX_LENGTH,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TOTAL,
    HOURS,
    SEQUENCE_LENGTH_CODE,
    SPLITS,
    array_hash,
    assert_mimic_release,
    code_vocabulary,
    export_code_vocabulary,
    iter_source_examples,
    load_build_report,
    load_cohort,
    ordered_text_hash,
    sha256,
    variable_names,
)


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


def _iter_all_source_examples(cache_root: Path):
    for split in SPLITS:
        yield from iter_source_examples(cache_root, split)


def _object_rows(examples):
    rows = np.empty(len(examples), dtype=object)
    labels = np.empty((len(examples), 1), dtype=np.int64)
    raw_times = []
    raw_values = []
    raw_masks = []
    for index, example in enumerate(examples):
        row = np.empty(5, dtype=object)
        row[:] = example.kedgn_tuple()
        rows[index] = row
        labels[index, 0] = example.label
        raw_times.append(example.time)
        raw_values.append(row[2])
        raw_masks.append(example.measurements)
    return rows, labels, raw_times, raw_values, raw_masks


def reconstruct_meds_boundary(
    cache_root: Path,
    cohort: pl.DataFrame,
    index: pl.DataFrame,
    events_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict:
    """Decode MEDS independently and compare it with the released tensorizer."""
    parquet = pq.ParquetFile(events_path)
    position_by_id = dict(
        cohort.select("sample_id", "source_position").iter_rows()
    )
    index_by_id = {
        row["sample_id"]: row
        for row in index.select(
            "sample_id", "subject_id", "observation_start", "boolean_value"
        ).iter_rows(named=True)
    }
    clinical_codes = code_vocabulary()
    clinical_channel = {code: channel for channel, code in enumerate(clinical_codes)}
    interval_channel = {
        f"{code}{AVERAGE_INTERVAL_SUFFIX}": channel
        for channel, code in enumerate(clinical_codes)
    }
    allowed_codes = {*clinical_channel, *interval_channel, SEQUENCE_LENGTH_CODE}
    source_iterator = _iter_all_source_examples(cache_root)
    next_source_position = 0
    next_source_row = 0
    checked_events = 0
    checked_samples = 0
    effective_digests = {
        name: hashlib.sha256()
        for name in ("P", "time", "average", "length", "label")
    }

    for row_group in range(parquet.num_row_groups):
        actual = pl.from_arrow(
            parquet.read_row_group(row_group, columns=list(EVENT_COLUMNS))
        )
        if actual.is_empty():
            raise AssertionError("KEDGN MEDS contains an empty event row group")
        if set(actual["code"].unique().to_list()) - allowed_codes:
            raise AssertionError("KEDGN MEDS contains a non-author input code")
        if actual["numeric_value"].null_count():
            raise AssertionError("KEDGN retained input event lacks a numeric value")
        source_rows = np.sort(actual["source_row"].to_numpy())
        np.testing.assert_array_equal(
            source_rows,
            np.arange(next_source_row, next_source_row + actual.height, dtype=np.int64),
        )
        next_source_row += actual.height

        actual = actual.sort("sample_id", "source_row")
        groups = actual.partition_by("sample_id", maintain_order=True)
        sample_ids = [int(group["sample_id"][0]) for group in groups]
        if len(sample_ids) != len(set(sample_ids)):
            raise AssertionError("KEDGN MEDS split one sample into multiple groups")
        try:
            source_positions = sorted(position_by_id[sample_id] for sample_id in sample_ids)
        except KeyError as error:
            raise AssertionError("KEDGN MEDS references a non-source sample") from error
        expected_positions = list(
            range(next_source_position, next_source_position + len(sample_ids))
        )
        if source_positions != expected_positions:
            raise AssertionError(
                "KEDGN MEDS row-group membership is not the next complete source block"
            )
        examples = list(itertools.islice(source_iterator, len(sample_ids)))
        if len(examples) != len(sample_ids):
            raise AssertionError("KEDGN MEDS has more samples than the source reader")
        examples_by_id = {example.sample_id: example for example in examples}
        if set(examples_by_id) != set(sample_ids):
            raise AssertionError("KEDGN MEDS row-group sample membership differs from source")

        source_rows_object, labels, _, _, _ = _object_rows(examples)
        P, static, average, length, time, y = (
            kedgn_utils.tensorize_normalize_exact_feature_mimic3(
                source_rows_object, labels, mean, std
            )
        )
        if static is not None:
            raise AssertionError("KEDGN MIMIC structured boundary unexpectedly gained static data")
        local_by_id = {example.sample_id: local for local, example in enumerate(examples)}
        for group in groups:
            sample_id = int(group["sample_id"][0])
            example = examples_by_id[sample_id]
            local = local_by_id[sample_id]
            index_row = index_by_id[sample_id]
            if group["subject_id"].n_unique() != 1 or int(group["subject_id"][0]) != example.subject_id:
                raise AssertionError("KEDGN MEDS event subject differs from source")
            if (
                int(index_row["subject_id"]) != example.subject_id
                or index_row["observation_start"] != example.intime
                or bool(index_row["boolean_value"]) != bool(example.label)
            ):
                raise AssertionError("KEDGN task index differs from the released source example")

            raw_values = np.zeros((EXPECTED_MAX_LENGTH, 16), dtype=np.float64)
            model_mask = np.zeros((EXPECTED_MAX_LENGTH, 16), dtype=bool)
            interval_values = np.zeros((EXPECTED_MAX_LENGTH, 16), dtype=np.float32)
            interval_seen = np.zeros((EXPECTED_MAX_LENGTH, 16), dtype=bool)
            effective_time = np.zeros((EXPECTED_MAX_LENGTH, 16), dtype=np.float32)
            length_value = None
            for event in group.iter_rows(named=True):
                code = event["code"]
                sequence_row = int(event["sequence_row"])
                numeric_value = event["numeric_value"]
                if code == SEQUENCE_LENGTH_CODE:
                    if length_value is not None or sequence_row != -1:
                        raise AssertionError("KEDGN MEDS length event is duplicated or misplaced")
                    if event["source_time"] != 0.0 or event["time"] != example.intime:
                        raise AssertionError("KEDGN MEDS length event has the wrong coordinate")
                    length_value = float(numeric_value)
                    continue
                if not 0 <= sequence_row < example.length:
                    raise AssertionError("KEDGN MEDS event has an invalid sequence row")
                expected_source_time = float(example.time[sequence_row])
                if float(event["source_time"]) != expected_source_time:
                    raise AssertionError("KEDGN MEDS source_time differs from released input")
                expected_time = example.intime + timedelta(hours=expected_source_time)
                if event["time"] != expected_time:
                    raise AssertionError("KEDGN MEDS timestamp differs from released input")

                if code in clinical_channel:
                    channel = clinical_channel[code]
                    if model_mask[sequence_row, channel]:
                        raise AssertionError("KEDGN MEDS duplicates a clinical model input")
                    model_mask[sequence_row, channel] = True
                    raw_values[sequence_row, channel] = float(numeric_value)
                    effective_time[sequence_row, channel] = np.float32(
                        example.time[sequence_row] / HOURS
                    )
                elif code in interval_channel:
                    channel = interval_channel[code]
                    if interval_seen[sequence_row, channel]:
                        raise AssertionError("KEDGN MEDS duplicates an average interval")
                    interval_seen[sequence_row, channel] = True
                    interval_values[sequence_row, channel] = np.float32(numeric_value)
                else:
                    raise AssertionError("KEDGN MEDS uses an unknown retained code")

            if length_value != float(example.length):
                raise AssertionError("KEDGN MEDS sequence length differs from released input")
            np.testing.assert_array_equal(interval_seen, model_mask)
            released_mask = P[local, :, 16:].numpy().astype(bool)
            np.testing.assert_array_equal(model_mask, released_mask)
            source_raw = example.kedgn_tuple()[2]
            np.testing.assert_array_equal(
                raw_values[model_mask].astype(np.float32),
                source_raw[model_mask[: example.length]],
            )
            reconstructed_values = np.empty_like(raw_values)
            for channel in range(16):
                reconstructed_values[:, channel] = (
                    raw_values[:, channel] - mean[channel]
                ) / (std[channel] + 1e-18)
            reconstructed_values *= model_mask
            reconstructed_P = np.concatenate(
                [reconstructed_values, model_mask.astype(np.float64)], axis=1
            ).astype(np.float32)
            np.testing.assert_allclose(
                reconstructed_P, P[local].numpy(), rtol=0, atol=1e-6
            )
            np.testing.assert_array_equal(
                effective_time, time[local].numpy() * released_mask
            )
            np.testing.assert_allclose(
                interval_values,
                average[local].numpy() * released_mask,
                rtol=0,
                atol=1e-7,
            )
            np.testing.assert_array_equal(
                np.asarray([[length_value]], dtype=np.float32),
                length[local : local + 1].numpy(),
            )
            np.testing.assert_array_equal(
                np.asarray([example.label], dtype=np.int64), y[local : local + 1].numpy()
            )
            if group.height != 2 * int(model_mask.sum()) + 1:
                raise AssertionError("KEDGN MEDS sample has the wrong retained event count")

            for name, array in (
                ("P", reconstructed_P),
                ("time", effective_time),
                ("average", interval_values),
                ("length", np.asarray([length_value], dtype=np.float32)),
                ("label", np.asarray([example.label], dtype=np.int64)),
            ):
                effective_digests[name].update(np.ascontiguousarray(array).tobytes())
            checked_events += group.height
            checked_samples += 1

        next_source_position += len(sample_ids)

    if next(source_iterator, None) is not None:
        raise AssertionError("KEDGN MEDS has fewer samples than the source reader")
    if checked_samples != EXPECTED_TOTAL or next_source_position != EXPECTED_TOTAL:
        raise AssertionError("KEDGN MEDS reconstruction did not cover every source sample")
    return {
        "samples": checked_samples,
        "events": checked_events,
        "effective_tensor_sha256": {
            name: digest.hexdigest() for name, digest in effective_digests.items()
        },
    }


def _training_statistics(cache_root: Path):
    examples = list(iter_source_examples(cache_root, "train"))
    if len(examples) != EXPECTED_SPLIT_COUNTS["train"]:
        raise AssertionError("KEDGN training source count changed")
    dense = np.zeros((len(examples), EXPECTED_MAX_LENGTH, 16))
    for index, example in enumerate(examples):
        dense[index, : example.length] = example.kedgn_tuple()[2]
    mean, std = kedgn_utils.getStats(dense)
    if mean.shape != (16, 1) or std.shape != (16, 1):
        raise AssertionError("KEDGN getStats shape changed")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise AssertionError("KEDGN training statistics became non-finite")
    return mean, std


def _expected_auxiliary(raw_times, raw_masks):
    batch = len(raw_times)
    expected_time = np.zeros((batch, EXPECTED_MAX_LENGTH, 16), dtype=np.float64)
    expected_average = np.zeros_like(expected_time)
    expected_length = np.zeros((batch, 1), dtype=np.float64)
    for index, (time, mask) in enumerate(zip(raw_times, raw_masks)):
        length = len(time)
        normalized_time = time / 48.0
        expected_time[index, :length] = np.tile(normalized_time, (16, 1)).T
        expected_length[index, 0] = length
        for channel in range(16):
            positions = np.where(mask[:, channel])[0]
            if len(positions) == 1:
                expected_average[index, positions, channel] = length / 2
            elif len(positions) > 1:
                observed_time = normalized_time[positions]
                gaps = observed_time[1:] - observed_time[:-1]
                right = np.insert(gaps, -1, gaps[-1])
                left = np.insert(gaps, 0, gaps[0])
                expected_average[index, positions, channel] = (left + right) / 2
    return expected_time, expected_average, expected_length


def _update_digest(digest, tensor: torch.Tensor) -> None:
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())


def recreate_kedgn_boundary(cache_root: Path, mean: np.ndarray, std: np.ndarray):
    output = {}
    total = 0
    total_source_observed = 0
    total_model_observed = 0
    total_zero_filled = 0
    total_suppressed_zero = 0
    total_suppressed_negative = 0
    total_invented = 0
    for split in SPLITS:
        digests = {name: hashlib.sha256() for name in ("P", "time", "average", "length", "label", "source_mask")}
        split_count = 0
        source_iterator = iter_source_examples(cache_root, split)
        for start in range(0, EXPECTED_SPLIT_COUNTS[split], 256):
            stop = min(start + 256, EXPECTED_SPLIT_COUNTS[split])
            examples = list(itertools.islice(source_iterator, stop - start))
            if len(examples) != stop - start:
                raise AssertionError(f"KEDGN {split} reader batch is incomplete")
            rows, labels, raw_times, raw_values, raw_masks = _object_rows(examples)
            expected_time, expected_average, expected_length = _expected_auxiliary(
                raw_times, raw_masks
            )
            P, static, average, length, time, y = (
                kedgn_utils.tensorize_normalize_exact_feature_mimic3(
                    rows, labels, mean, std
                )
            )
            if static is not None:
                raise AssertionError("KEDGN MIMIC structured boundary unexpectedly gained static data")
            expected_shape = (len(examples), EXPECTED_MAX_LENGTH, 32)
            if tuple(P.shape) != expected_shape or P.dtype != torch.float32:
                raise AssertionError("KEDGN value/model-mask boundary shape or dtype changed")
            if (
                tuple(time.shape) != (len(examples), EXPECTED_MAX_LENGTH, 16)
                or time.dtype != torch.float32
            ):
                raise AssertionError("KEDGN time boundary shape or dtype changed")
            if (
                tuple(average.shape) != (len(examples), EXPECTED_MAX_LENGTH, 16)
                or average.dtype != torch.float32
            ):
                raise AssertionError("KEDGN interval boundary shape or dtype changed")
            if tuple(length.shape) != (len(examples), 1) or tuple(y.shape) != (
                len(examples),
            ) or length.dtype != torch.float32 or y.dtype != torch.int64:
                raise AssertionError("KEDGN length/label boundary shape or dtype changed")
            if not torch.isfinite(P).all() or not torch.isfinite(time).all() or not torch.isfinite(average).all():
                raise AssertionError("KEDGN mechanically serialized boundary is non-finite")
            np.testing.assert_array_equal(time.numpy(), expected_time.astype(np.float32))
            np.testing.assert_allclose(
                average.numpy(), expected_average.astype(np.float32), rtol=0, atol=1e-7
            )
            np.testing.assert_array_equal(length.numpy(), expected_length.astype(np.float32))
            np.testing.assert_array_equal(y.numpy(), labels[:, 0])

            dense = np.zeros((len(examples), EXPECTED_MAX_LENGTH, 16), dtype=np.float64)
            source_mask = np.zeros_like(dense, dtype=bool)
            for local, (values, mask) in enumerate(zip(raw_values, raw_masks)):
                dense[local, : len(values)] = values
                source_mask[local, : len(values)] = mask
            model_mask = dense > 0
            expected_values = np.empty_like(dense)
            for channel in range(16):
                expected_values[:, :, channel] = (
                    dense[:, :, channel] - mean[channel]
                ) / (std[channel] + 1e-18)
            expected_values *= model_mask
            np.testing.assert_allclose(
                P[:, :, :16].numpy(), expected_values.astype(np.float32), rtol=0, atol=1e-6
            )
            np.testing.assert_array_equal(P[:, :, 16:].numpy(), model_mask.astype(np.float32))
            if np.any(model_mask & ~source_mask):
                raise AssertionError("KEDGN positivity mask invented a source observation")

            for name, tensor in (
                ("P", P),
                ("time", time),
                ("average", average),
                ("length", length),
                ("label", y),
            ):
                _update_digest(digests[name], tensor)
            digests["source_mask"].update(source_mask.tobytes())
            split_count += len(examples)
            total += len(examples)
            total_source_observed += int(source_mask.sum())
            total_model_observed += int(model_mask.sum())
            total_zero_filled += int((~source_mask).sum())
            total_suppressed_zero += int((source_mask & (dense == 0)).sum())
            total_suppressed_negative += int((source_mask & (dense < 0)).sum())
            total_invented += int((model_mask & ~source_mask).sum())

        if split_count != EXPECTED_SPLIT_COUNTS[split]:
            raise AssertionError(f"KEDGN {split} tensor traversal is incomplete")
        output[split] = {
            "count": split_count,
            "P_float32_sha256": digests["P"].hexdigest(),
            "time_float32_sha256": digests["time"].hexdigest(),
            "average_interval_float32_sha256": digests["average"].hexdigest(),
            "length_float32_sha256": digests["length"].hexdigest(),
            "label_int64_sha256": digests["label"].hexdigest(),
            "source_mask_bool_sha256": digests["source_mask"].hexdigest(),
        }
    if total != EXPECTED_TOTAL:
        raise AssertionError("KEDGN exhaustive tensor recreation is incomplete")
    return output, {
        "source_finite_cells": total_source_observed,
        "released_positive_model_mask_cells": total_model_observed,
        "source_missing_or_padding_cells_zero_filled": total_zero_filled,
        "source_observed_zero_cells_suppressed": total_suppressed_zero,
        "source_observed_negative_cells_suppressed": total_suppressed_negative,
        "model_mask_invented_cells": total_invented,
    }


def _check_released_cache_loader() -> None:
    sentinels = {
        "mimic3_train_x.npy": np.asarray(["train_x"], dtype=object),
        "mimic3_val_x.npy": np.asarray(["val_x"], dtype=object),
        "mimic3_test_x.npy": np.asarray(["test_x"], dtype=object),
        "mimic3_train_y.npy": np.asarray([0], dtype=np.int64),
        "mimic3_val_y.npy": np.asarray([1], dtype=np.int64),
        "mimic3_test_y.npy": np.asarray([0], dtype=np.int64),
    }
    original = kedgn_utils.np.load
    seen = []

    def fake_load(path, allow_pickle=False):
        name = Path(path).name
        seen.append((name, allow_pickle))
        return sentinels[name]

    kedgn_utils.np.load = fake_load
    try:
        result = kedgn_utils.get_data_split("unused", dataset="mimic3")
    finally:
        kedgn_utils.np.load = original
    if [name for name, _ in seen] != list(sentinels) or not all(
        allow_pickle for _, allow_pickle in seen
    ):
        raise AssertionError("KEDGN released cache loader filename protocol changed")
    if [array.shape[0] for array in result] != [1] * 6:
        raise AssertionError("KEDGN released cache loader output protocol changed")


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
        raise AssertionError("KEDGN source report does not match exact MIMIC-III 1.4")
    cohort = load_cohort(cache_root)
    manifest = json.loads((task_root / "manifest.json").read_text())
    if manifest["task"].get("modalities") != ["structured"]:
        raise AssertionError("KEDGN MEDS task retained a non-structured modality")
    if manifest["task"].get("note_codes") != []:
        raise AssertionError("KEDGN MEDS task declared note codes")
    if (
        manifest["source"].get("selected_model_inputs") != ["structured"]
        or manifest["source"].get("selection_tier") != 3
    ):
        raise AssertionError("KEDGN batch check selected the wrong experiment")

    codes = tuple(
        pl.read_parquet(task_root / manifest["meds"]["codes"]["file"])["code"].to_list()
    )
    if codes != export_code_vocabulary() or len(variable_names()) != 16:
        raise AssertionError("KEDGN event/semantic channel vocabulary changed")
    events_path = task_root / manifest["meds"]["events"]["file"]
    index = pl.read_parquet(task_root / manifest["meds"]["task_index"]["file"])
    mean, std = _training_statistics(cache_root)
    split_hashes, missingness = recreate_kedgn_boundary(cache_root, mean, std)
    meds_reconstruction = reconstruct_meds_boundary(
        cache_root, cohort, index, events_path, mean, std
    )
    checked_events = meds_reconstruction["events"]
    if checked_events != manifest["event_count"]:
        raise AssertionError("KEDGN MEDS reconstruction count differs from manifest")
    expected_events = (
        2 * missingness["released_positive_model_mask_cells"] + EXPECTED_TOTAL
    )
    if checked_events != expected_events:
        raise AssertionError("KEDGN MEDS does not contain exactly the author-visible fields")
    _check_released_cache_loader()

    semantic_path = (
        PAPER_ROOT / "data" / "mimic3" / "mimic3_bert_var_rep_gpt_source.pt"
    )
    semantic = torch.load(semantic_path, map_location="cpu")
    if tuple(semantic.shape) != (16, 768) or semantic.dtype != torch.float32:
        raise AssertionError("KEDGN default semantic tensor shape/dtype changed")
    if not torch.isfinite(semantic).all():
        raise AssertionError("KEDGN default semantic tensor became non-finite")

    for split in SPLITS:
        expected_ids = set(
            cohort.filter(pl.col("split") == split)["sample_id"].to_list()
        )
        actual_ids = set(index.filter(pl.col("split") == split)["sample_id"].to_list())
        if actual_ids != expected_ids:
            raise AssertionError(f"KEDGN MEDS {split} membership differs from source reader")

    output = {
        "status": "exact_structured_source_boundary",
        "events_compared": checked_events,
        "event_vocabulary_sha256": ordered_text_hash(codes),
        "source_channel_order_sha256": ordered_text_hash(variable_names()),
        "training_positive_only_mean_float64_sha256": array_hash(mean),
        "training_positive_only_std_float64_sha256": array_hash(std),
        "split_loader_tensor_hashes": split_hashes,
        "meds_to_released_boundary": meds_reconstruction,
        "missing_value_accounting": missingness,
        "author_boundary": {
            "P_shape": [EXPECTED_TOTAL, EXPECTED_MAX_LENGTH, 32],
            "P_dtype": "float32",
            "time_shape": [EXPECTED_TOTAL, EXPECTED_MAX_LENGTH, 16],
            "time_dtype": "float32",
            "average_interval_shape": [EXPECTED_TOTAL, EXPECTED_MAX_LENGTH, 16],
            "average_interval_dtype": "float32",
            "length_shape": [EXPECTED_TOTAL, 1],
            "length_dtype": "float32",
            "label_dtype": "int64",
            "source_missing": "NaN with explicit finite-cell mask",
            "cache_bridge": (
                "non-conflicting equal-Hours rows coalesced; NaN-to-zero with source "
                "mask retained"
            ),
            "reader_runtime_compatibility": (
                "released literal GCS category None-to-1 mapping preserved across "
                "current pandas default-NA behavior"
            ),
            "model_mask": "zero-filled raw value > 0",
        },
        "shared_semantic_input": {
            "shape": [16, 768],
            "dtype": "float32",
            "sha256": sha256(semantic_path),
            "value_sha256": array_hash(semantic.numpy()),
            "not_exported_as_patient_data": True,
        },
        "source_qualification": build["fidelity_limitation"],
        "assertions": {
            "all_author_visible_events_equal_meds": True,
            "exact_vocabulary_and_empty_note_exception_list": True,
            "every_non_note_code_is_an_intended_structured_input": True,
            "no_notes_or_imaging_retained": True,
            "all_medical_ts_reader_examples_recreated": True,
            "all_kedgn_tensorizer_fields_recreated": True,
            "value_mask_time_interval_length_label_shape_dtype_value_equal": True,
            "source_nan_and_released_positivity_mask_behavior_preserved": True,
            "source_observed_nonpositive_values_not_exposed_to_lightgbm": True,
            "released_split_membership_preserved": True,
            "released_cache_filename_protocol_invoked": True,
        },
    }
    output_path = task_root / "audit" / "full_batch_recreation.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "events": checked_events}))


if __name__ == "__main__":
    main()
