"""Recreate SMART's complete retained loader/collate boundary from MEDS."""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from export_source import event_batches  # noqa: E402
from cohort_source import (  # noqa: E402
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TOTAL,
    SEEDS,
    array_hash,
    assert_mimic_release,
    code_vocabulary,
    load_author_notebook_boundary,
    load_build_report,
    load_cohort,
    ordered_text_hash,
    replica_task_id,
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


def _author_modules():
    from data import mimiciii as author_loader
    from data.dataloader import collate_fn

    return author_loader, collate_fn


def verify_author_loader_and_collate(boundary: dict, cohort: pl.DataFrame, seed: int):
    """Invoke the exact released loader and collate every one of its examples."""
    author_loader, collate_fn = _author_modules()
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
        datasets = author_loader.load_mimic_iii_mortality()
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
    result = {}
    sample_ids = cohort["sample_id"].to_numpy()
    for split, dataset in zip(("train", "validation", "test"), datasets):
        expected_indices = split_indices[split]
        if len(dataset) != len(expected_indices):
            raise AssertionError("released SMART loader split count changed")
        x_digest = __import__("hashlib").sha256()
        mask_digest = __import__("hashlib").sha256()
        label_digest = __import__("hashlib").sha256()
        for start in range(0, len(dataset), 512):
            stop = min(start + 512, len(dataset))
            batch = collate_fn([dataset[index] for index in range(start, stop)])
            selected = np.asarray(expected_indices[start:stop], dtype=np.int64)
            expected_x = torch.from_numpy(boundary["normalized"][selected])
            expected_mask = torch.from_numpy(boundary["mask"][selected])
            expected_labels = torch.from_numpy(boundary["labels"][selected])
            expected_lens = torch.full((len(selected),), 48, dtype=torch.int64)
            if batch["x"].dtype != torch.float32 or batch["mask"].dtype != torch.float32:
                raise AssertionError("released SMART x/mask collate dtype changed")
            if batch["labels"].dtype != torch.int64 or batch["lens"].dtype != torch.int64:
                raise AssertionError("released SMART label/lens collate dtype changed")
            for key, expected in (
                ("x", expected_x),
                ("mask", expected_mask),
                ("labels", expected_labels),
                ("lens", expected_lens),
            ):
                if not torch.equal(batch[key], expected):
                    raise AssertionError(f"released SMART {split} collate field differs: {key}")
            x_digest.update(batch["x"].contiguous().numpy().tobytes())
            mask_digest.update(batch["mask"].contiguous().numpy().tobytes())
            label_digest.update(batch["labels"].contiguous().numpy().tobytes())
        result[split] = {
            "count": len(dataset),
            "ordered_sample_id_sha256": ordered_text_hash(
                sample_ids[np.asarray(expected_indices, dtype=np.int64)]
            ),
            "x_float32_sha256": x_digest.hexdigest(),
            "mask_float32_sha256": mask_digest.hexdigest(),
            "label_int64_sha256": label_digest.hexdigest(),
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

    if boundary["normalized"].shape != (EXPECTED_TOTAL, 48, 17):
        raise AssertionError("SMART normalized tensor shape changed")
    if boundary["mask"].shape != (EXPECTED_TOTAL, 48, 17):
        raise AssertionError("SMART mask tensor shape changed")
    if not set(np.unique(boundary["mask"])) <= {0.0, 1.0}:
        raise AssertionError("SMART observation mask is not binary")

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
        checked_events = compare_all_source_events(cache_root, cohort, events_path)
        if checked_events != manifest["event_count"]:
            raise AssertionError("SMART event parity count differs from manifest")

        loader_hashes, split_indices = verify_author_loader_and_collate(boundary, cohort, seed)
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
            "events_compared": checked_events,
            "event_vocabulary_sha256": ordered_text_hash(vocabulary),
            "author_boundary": {
                "normalized_shape": [EXPECTED_TOTAL, 48, 17],
                "normalized_dtype": "float32",
                "normalized_sha256": array_hash(boundary["normalized"]),
                "mask_shape": [EXPECTED_TOTAL, 48, 17],
                "mask_dtype": "float32",
                "mask_sha256": array_hash(boundary["mask"]),
                "labels_dtype": "int64",
                "labels_sha256": array_hash(boundary["labels"]),
                "lens": 48,
            },
            "split_loader_collate_hashes": loader_hashes,
            "ordering_limitation": build["fidelity_limitation"],
            "assertions": {
                "all_source_events_equal_meds": True,
                "exact_vocabulary_and_note_exception_list": True,
                "every_non_note_code_is_released_structured_input": True,
                "no_imaging_or_notes_retained": True,
                "all_author_loader_examples_recreated": True,
                "x_mask_label_lens_shape_dtype_value_equal": True,
                "released_example_split_preserved": True,
            },
        }
        output_path = task_dir / "audit" / "full_batch_recreation.json"
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"task": task_id, "status": output["status"], "events": checked_events}, sort_keys=True))


if __name__ == "__main__":
    main()
