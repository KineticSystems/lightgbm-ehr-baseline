"""Exhaustively recreate DiPro's retained Dataset/collator fields from MEDS."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(PAPER_ROOT / "mimic_data_extract"))

from dataset_module.dipro_dataset import CollateFunc, DiproDataset  # noqa: E402
from export_source import (  # noqa: E402
    CODES,
    DEMOGRAPHIC_CODES,
    MASK_CODES,
    SPLIT_FILES,
    VALUE_CODES,
    carry_in_source_times,
)
from mimic3benchmark.constants import (  # noqa: E402
    DEMOGRAPHIC,
    PROGRESSION_DISEASE,
    SELECTED_REGIONS,
    SELECTED_VAR,
)
from cohort_source import TASK_ID, UPSTREAM_COMMIT  # noqa: E402


BATCH_SIZE = 4


def tensor_digest(digest, tensor: torch.Tensor) -> None:
    values = tensor.detach().cpu().contiguous().numpy()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())


def prepare_loader_splits(cache_root: Path) -> Path:
    root = cache_root / "loader_projection" / "mortality_los"
    root.mkdir(parents=True, exist_ok=True)
    empty_label = {disease: "not mentioned" for disease in PROGRESSION_DISEASE}
    for _, filename in SPLIT_FILES:
        frame = pd.read_csv(cache_root / "split" / "mortality_los" / filename)
        for organ in SELECTED_REGIONS:
            values = []
            for value in frame["cxr_list"]:
                pairs = max(len(ast.literal_eval(value)) - 1, 0)
                values.append(str([dict(empty_label) for _ in range(pairs)]))
            frame[f"{organ}_progression"] = values
        frame.to_csv(root / filename, index=False)
    bbox = cache_root / "loader_projection" / "unused_bbox.csv"
    pd.DataFrame({"cxr_id": []}).to_csv(bbox, index=False)
    return root.parent


def _source_records(cache_root: Path) -> list[dict]:
    stays = pd.read_csv(
        cache_root / "all_stays.csv",
        usecols=["subject_id", "stay_id", "intime"],
        parse_dates=["intime"],
    ).set_index("stay_id")
    result = []
    for split, filename in SPLIT_FILES:
        frame = pd.read_csv(cache_root / "split" / "mortality_los" / filename)
        for source_split_row, row in frame.iterrows():
            stay_id = int(row["stay_id"])
            result.append(
                {
                    "split": split,
                    "filename": filename,
                    "source_split_row": int(source_split_row),
                    "sample_id": stay_id,
                    "subject_id": int(row["subject_id"]),
                    "label": int(row["mortality_inhospital"]),
                    "intime": pd.Timestamp(stays.loc[stay_id, "intime"]),
                }
            )
    return result


def reconstruct_meds(
    task_dir: Path,
    manifest: dict,
    records: list[dict],
    cache_root: Path,
) -> dict:
    sample_ids = np.asarray([row["sample_id"] for row in records], dtype=np.int64)
    subjects = np.asarray([row["subject_id"] for row in records], dtype=np.int64)
    intimes = np.asarray(
        [np.datetime64(row["intime"], "us") for row in records],
        dtype="datetime64[us]",
    )
    order = np.argsort(sample_ids)
    sorted_ids = sample_ids[order]
    values = np.full((len(records), 49, len(SELECTED_VAR)), np.nan, dtype=np.float32)
    value_source_times = np.full(values.shape, np.nan, dtype=np.float64)
    masks = np.full((len(records), 49, len(SELECTED_VAR)), -1, dtype=np.int8)
    demographics = np.full(
        (len(records), len(DEMOGRAPHIC)), np.nan, dtype=np.float32
    )
    value_seen = np.zeros_like(values, dtype=bool)
    mask_seen = np.zeros_like(masks, dtype=bool)
    demographic_seen = np.zeros_like(demographics, dtype=bool)
    source_rows = [set() for _ in records]
    episode_lengths_by_stay = {
        int(stay_id): int(length)
        for stay_id, length in json.loads(
            (cache_root / "episode_lengths.json").read_text()
        ).items()
    }
    episode_lengths = np.asarray(
        [episode_lengths_by_stay[int(row["sample_id"])] for row in records],
        dtype=np.int64,
    )
    if np.any(episode_lengths < 1) or np.any(episode_lengths > 49):
        raise AssertionError("DiPro released episode length is outside [1,49]")
    expected_time_cells = (
        np.arange(49, dtype=np.int64)[None, :] < episode_lengths[:, None]
    )
    expected_ehr_cells = np.broadcast_to(
        expected_time_cells[:, :, None], masks.shape
    )
    record_position = {
        int(row["sample_id"]): position for position, row in enumerate(records)
    }
    carry_in = np.zeros_like(values, dtype=bool)
    carry_in_times = np.full(values.shape, np.nan, dtype=np.float64)
    for (stay_id, variable_index), hours in carry_in_source_times(
        cache_root
    ).items():
        position = record_position.get(int(stay_id))
        if position is not None:
            carry_in[position, 0, int(variable_index)] = True
            carry_in_times[position, 0, int(variable_index)] = float(hours)
    code_index = {code: index for index, code in enumerate(CODES)}
    actual_codes = set()
    event_count = 0

    event_path = task_dir / manifest["meds"]["events"]["file"]
    for batch in pq.ParquetFile(event_path).iter_batches(batch_size=300_000):
        event_count += len(batch)
        event_codes = batch["code"].to_pylist()
        actual_codes.update(event_codes)
        if batch["numeric_value"].null_count:
            raise AssertionError("DiPro MEDS contains an occurrence-only event")
        if "text_value" in batch.schema.names and batch["text_value"].null_count != len(batch):
            raise AssertionError("DiPro structured task retained text")
        event_sample_ids = batch["sample_id"].to_numpy().astype(np.int64, copy=False)
        found = np.searchsorted(sorted_ids, event_sample_ids)
        if np.any(found == len(sorted_ids)) or not np.array_equal(
            sorted_ids[found], event_sample_ids
        ):
            raise AssertionError("DiPro MEDS contains an unknown sample")
        positions = order[found]
        event_subjects = batch["subject_id"].to_numpy().astype(np.int64, copy=False)
        if not np.array_equal(event_subjects, subjects[positions]):
            raise AssertionError("DiPro MEDS changed an event subject")
        times = batch["time"].to_numpy()
        sequence_rows = batch["sequence_row"].to_numpy().astype(np.int64, copy=False)
        source_times = batch["source_time"].to_numpy(zero_copy_only=False)
        numeric = batch["numeric_value"].to_numpy().astype(np.float32, copy=False)
        rows = batch["source_row"].to_numpy().astype(np.int64, copy=False)
        for offset, code in enumerate(event_codes):
            position = int(positions[offset])
            source_row = int(rows[offset])
            if source_row in source_rows[position]:
                raise AssertionError("DiPro MEDS duplicated source_row within a sample")
            source_rows[position].add(source_row)
            index = code_index.get(code)
            if index is None:
                raise AssertionError("DiPro MEDS contains an undeclared code")
            if index < len(VALUE_CODES):
                time_index = int(sequence_rows[offset])
                variable_index = index
                if not 0 <= time_index <= 48:
                    raise AssertionError("DiPro EHR event has invalid sequence_row")
                event_source_time = float(source_times[offset])
                if event_source_time != float(time_index) and not (
                    time_index == 0
                    and carry_in[position, 0, variable_index]
                    and event_source_time
                    == carry_in_times[position, 0, variable_index]
                ):
                    raise AssertionError("DiPro EHR source coordinate changed")
                if times[offset] != intimes[position] + np.timedelta64(time_index, "h"):
                    raise AssertionError("DiPro EHR event time changed")
                if value_seen[position, time_index, variable_index]:
                    raise AssertionError("DiPro duplicated an observed EHR cell")
                values[position, time_index, variable_index] = numeric[offset]
                value_source_times[position, time_index, variable_index] = (
                    event_source_time
                )
                value_seen[position, time_index, variable_index] = True
            elif index < len(VALUE_CODES) + len(MASK_CODES):
                time_index = int(sequence_rows[offset])
                variable_index = index - len(VALUE_CODES)
                if not 0 <= time_index <= 48:
                    raise AssertionError("DiPro mask event has invalid sequence_row")
                if source_times[offset] != float(time_index):
                    raise AssertionError("DiPro mask source coordinate changed")
                if times[offset] != intimes[position] + np.timedelta64(time_index, "h"):
                    raise AssertionError("DiPro mask event time changed")
                if numeric[offset] not in (0.0, 1.0):
                    raise AssertionError("DiPro observation mask is not binary")
                if mask_seen[position, time_index, variable_index]:
                    raise AssertionError("DiPro duplicated an observation mask")
                masks[position, time_index, variable_index] = int(numeric[offset])
                mask_seen[position, time_index, variable_index] = True
            else:
                demographic_index = index - len(VALUE_CODES) - len(MASK_CODES)
                if int(sequence_rows[offset]) != -1 or not np.isnan(source_times[offset]):
                    raise AssertionError("DiPro demographic source coordinate changed")
                if times[offset] != intimes[position]:
                    raise AssertionError("DiPro demographic event time changed")
                if demographic_seen[position, demographic_index]:
                    raise AssertionError("DiPro duplicated a demographic cell")
                demographics[position, demographic_index] = numeric[offset]
                demographic_seen[position, demographic_index] = True

    if actual_codes != set(CODES):
        raise AssertionError("DiPro actual event vocabulary is incomplete")
    if not np.all(mask_seen[expected_ehr_cells]) or not np.all(demographic_seen):
        raise AssertionError("DiPro MEDS omitted a mask or demographic cell")
    if np.any(mask_seen[~expected_ehr_cells]) or np.any(value_seen[~expected_ehr_cells]):
        raise AssertionError("DiPro MEDS added EHR cells beyond the released episode")
    expected_values = masks.astype(bool) | (carry_in & (masks == 0))
    if not np.array_equal(
        value_seen[expected_ehr_cells], expected_values[expected_ehr_cells]
    ):
        raise AssertionError("DiPro value events disagree with released masks")
    carry_only = carry_in & (masks == 0)
    if not np.array_equal(
        value_source_times[carry_only], carry_in_times[carry_only]
    ):
        raise AssertionError("DiPro pre-window carry-in coordinates changed")
    for position, seen in enumerate(source_rows):
        if seen != set(range(len(seen))):
            raise AssertionError(f"DiPro sample {sample_ids[position]} source_row is not contiguous")
    if event_count != manifest["event_count"]:
        raise AssertionError("DiPro manifest event count changed")

    moments = pd.read_csv(cache_root / "mean_std.csv").set_index("variables")
    ranges = pd.read_csv(
        PAPER_ROOT
        / "mimic_data_extract"
        / "mimic3benchmark"
        / "resources"
        / "variable_ranges.csv"
    )
    ranges["LEVEL2"] = ranges["LEVEL2"].str.lower()
    ranges = ranges.drop_duplicates("LEVEL2", keep="first").set_index("LEVEL2")
    baselines = np.asarray(
        [
            (
                float(ranges.loc[variable.lower(), "IMPUTE"])
                - float(moments.loc[variable, "mean"])
            )
            / float(moments.loc[variable, "std"])
            for variable in SELECTED_VAR
        ],
        dtype=np.float32,
    )
    reconstructed = np.empty_like(values)
    for position in range(len(records)):
        current = baselines.copy()
        for time_index in range(int(episode_lengths[position])):
            updates = value_seen[position, time_index]
            current[updates] = values[position, time_index, updates]
            reconstructed[position, time_index] = current
    return {
        "sample_ids": sample_ids,
        "values": values,
        "masks": masks,
        "demographics": demographics,
        "reconstructed": reconstructed,
        "episode_lengths": episode_lengths,
        "event_count": event_count,
        "value_events": int(value_seen.sum()),
        "mask_positive_value_events": int((masks == 1).sum()),
        "pre_window_carry_in_cells": int(carry_only.sum()),
        "mask_zero_cells": int(((masks == 0) & expected_ehr_cells).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxr-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.task_dir / "manifest.json").read_text())
    if manifest.get("paper", {}).get("key") != "dipro":
        raise AssertionError("task is not DiPro")
    if manifest.get("task", {}).get("id") != TASK_ID:
        raise AssertionError("DiPro selected task changed")
    if manifest["task"].get("modalities") != ["structured"]:
        raise AssertionError("DiPro retained an undeclared modality")
    if manifest["task"].get("note_codes") != []:
        raise AssertionError("DiPro unexpectedly retained notes")
    source = manifest.get("source", {})
    if source.get("upstream_commit") != UPSTREAM_COMMIT:
        raise AssertionError("DiPro upstream commit changed")
    if source.get("selection_tier") != 4:
        raise AssertionError("DiPro selected the wrong modality tier")
    if source.get("selected_model_inputs") != [
        "structured",
        "longitudinal_regional_cxr_images",
    ]:
        raise AssertionError("DiPro selected model inputs changed")
    if source.get("retained_modalities") != ["structured"]:
        raise AssertionError("DiPro retained projection changed")
    if source.get("omitted_modalities") != ["imaging"]:
        raise AssertionError("DiPro omitted-imaging declaration changed")

    records = _source_records(args.cache_root)
    recreated = reconstruct_meds(args.task_dir, manifest, records, args.cache_root)
    loader_root = prepare_loader_splits(args.cache_root)
    empty_bbox = args.cache_root / "loader_projection" / "unused_bbox.csv"
    cxr_metadata = args.cxr_root / "mimic-cxr-2.0.0-metadata.csv.gz"
    demographic_path = args.cache_root / "demographic_processed.csv"
    ehr_path = args.cache_root / "ehr_preprocessed"
    collate = CollateFunc()
    digests = {
        name: hashlib.sha256()
        for name in (
            "ehr",
            "ehr_observation_mask",
            "ehr_sequence_mask",
            "demographic",
            "label",
            "ehr_time",
        )
    }
    item_digest = hashlib.sha256()
    total_batches = 0
    total_examples = 0
    record_position = {int(row["sample_id"]): index for index, row in enumerate(records)}

    for split, filename in SPLIT_FILES:
        dataset = DiproDataset(
            partition="val" if split == "validation" else split,
            task_name="mortality",
            label_path=str(loader_root),
            demographic_path=str(demographic_path),
            ehr_time_series_path=str(ehr_path),
            bbox_csv=str(empty_bbox),
            mimic_cxr_path=str(args.cxr_root / "files"),
            cxr_meta_path=str(cxr_metadata),
            input_size=1,
        )
        dataset.ensemble_cxr_regions = lambda paths: torch.zeros(
            (len(paths), len(SELECTED_REGIONS), 3, 1, 1), dtype=torch.float32
        )
        source_frame = pd.read_csv(args.cache_root / "split" / "mortality_los" / filename)
        if len(dataset) != len(source_frame):
            raise AssertionError("DiPro released Dataset length changed")
        for start in range(0, len(dataset), BATCH_SIZE):
            source_items = [dataset[index] for index in range(start, min(start + BATCH_SIZE, len(dataset)))]
            recreated_items = []
            for local_offset, source_item in enumerate(source_items):
                source_index = start + local_offset
                stay_id = int(source_frame.loc[source_index, "stay_id"])
                position = record_position[stay_id]
                length = source_item[0].shape[0]
                expected_ehr = torch.from_numpy(
                    recreated["reconstructed"][position, :length]
                ).float()
                expected_demographic = torch.from_numpy(
                    recreated["demographics"][position]
                ).float()
                expected_label = torch.tensor(records[position]["label"]).float()
                source_episode = pd.read_csv(
                    ehr_path / f"New_episode{stay_id}_timeseries.csv"
                )
                source_observation_mask = source_episode[
                    [f"{variable}_mask" for variable in SELECTED_VAR]
                ].to_numpy(np.int8)
                if not np.array_equal(
                    source_observation_mask,
                    recreated["masks"][position, :length],
                ):
                    raise AssertionError(
                        f"DiPro observation mask differs for stay {stay_id}"
                    )
                source_values = source_episode[list(SELECTED_VAR)].to_numpy(
                    np.float32
                )
                observed = source_observation_mask.astype(bool)
                if not np.array_equal(
                    source_values[observed],
                    recreated["values"][position, :length][observed],
                ):
                    raise AssertionError(
                        f"DiPro observed EHR values differ for stay {stay_id}"
                    )
                carry_only = (
                    recreated["masks"][position, :length] == 0
                ) & np.isfinite(recreated["values"][position, :length])
                if not np.array_equal(
                    source_values[carry_only],
                    recreated["values"][position, :length][carry_only],
                ):
                    raise AssertionError(
                        f"DiPro pre-window carry-in differs for stay {stay_id}"
                    )
                if source_item[0].dtype != expected_ehr.dtype or not torch.equal(
                    source_item[0], expected_ehr
                ):
                    raise AssertionError(f"DiPro EHR item differs for stay {stay_id}")
                if source_item[2].dtype != expected_demographic.dtype or not torch.equal(
                    source_item[2], expected_demographic
                ):
                    raise AssertionError(f"DiPro demographic item differs for stay {stay_id}")
                if not torch.equal(source_item[3], expected_label):
                    raise AssertionError(f"DiPro label differs for stay {stay_id}")
                item_digest.update(np.asarray([stay_id], dtype=np.int64).tobytes())
                tensor_digest(item_digest, source_item[0])
                tensor_digest(item_digest, source_item[2])
                tensor_digest(item_digest, source_item[3])
                tensor_digest(
                    digests["ehr_observation_mask"],
                    torch.from_numpy(source_observation_mask),
                )
                recreated_items.append(
                    (
                        expected_ehr,
                        source_item[1],
                        expected_demographic,
                        expected_label,
                        source_item[4],
                        torch.arange(0, 49, dtype=torch.float32),
                        source_item[6],
                    )
                )
            source_batch = collate(source_items)
            recreated_batch = collate(recreated_items)
            for name, batch_index in (
                ("ehr", 0),
                ("ehr_sequence_mask", 1),
                ("demographic", 4),
                ("label", 5),
                ("ehr_time", 8),
            ):
                left = source_batch[batch_index]
                right = recreated_batch[batch_index]
                if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
                    raise AssertionError(f"DiPro {name} collate shape/dtype changed")
                if not torch.equal(left, right):
                    raise AssertionError(f"DiPro {name} collate values differ")
                tensor_digest(digests[name], left)
            total_batches += 1
            total_examples += len(source_items)

    if total_examples != 2720:
        raise AssertionError("DiPro exhaustive loader traversal changed")
    lengths = json.loads((args.cache_root / "episode_lengths.json").read_text())
    report = {
        "status": "exact",
        "selected_task": {
            "id": TASK_ID,
            "paper_cell": "Table 2 / Table 12 longitudinal DiPro mortality",
            "source_model_inputs": ["structured", "longitudinal_regional_cxr_images"],
            "selection_tier": 4,
            "retained_projection": ["structured"],
            "omitted_modalities": ["imaging"],
            "note_codes": [],
        },
        "author_boundary": {
            "dataset": "dataset_module.dipro_dataset.DiproDataset.__getitem__",
            "collator": "dataset_module.dipro_dataset.CollateFunc",
            "exhaustive_examples": total_examples,
            "released_batch_size": BATCH_SIZE,
            "batches_checked": total_batches,
            "retained_item_fields": ["ehr", "demographic", "y"],
            "retained_collate_fields": [
                "ehr",
                "ehr_sequence_mask",
                "demographic",
                "y",
                "ehr_time",
            ],
            "omitted_fields": ["cxr_regions", "cxr_sequence_mask", "y_aux", "y_aux_mask", "interval_times"],
            "episode_length_counts": {
                str(length): sum(int(value) == length for value in lengths.values())
                for length in sorted({int(value) for value in lengths.values()})
            },
            "tensor_sha256": {name: digest.hexdigest() for name, digest in digests.items()},
            "item_projection_sha256": item_digest.hexdigest(),
        },
        "meds_boundary": {
            "event_count": recreated["event_count"],
            "ehr_value_events": recreated["value_events"],
            "mask_positive_value_events": recreated[
                "mask_positive_value_events"
            ],
            "pre_window_carry_in_value_events": recreated[
                "pre_window_carry_in_cells"
            ],
            "mask_zero_cells": recreated["mask_zero_cells"],
            "complete_code_count": len(CODES),
            "value_code_count": len(VALUE_CODES),
            "mask_code_count": len(MASK_CODES),
            "demographic_code_count": len(DEMOGRAPHIC_CODES),
            "all_non_note_codes_are_intended_structured_inputs": True,
            "imaging_or_auxiliary_events": 0,
        },
        "fidelity": {
            "values": "torch.equal",
            "masks": "torch.equal",
            "demographics": "torch.equal",
            "labels": "torch.equal",
            "row_order": "exact released split CSV order via source_split_row",
            "missingness": (
                "MEDS retains normalized values only where the reliable released "
                "pre-imputation mask is one, plus real elapsed [-1,0) observations "
                "carried into row zero; all other mask-zero values are reconstructed "
                "by the released normal-initialized forward fill"
            ),
            "image_instrumentation": (
                "DiproDataset.__getitem__ was called exhaustively; only its image "
                "reader was replaced by a shape-compatible zero sentinel because "
                "imaging fields are forbidden from common-task retention. The "
                "selected source model does consume these omitted image fields"
            ),
            "auxiliary_instrumentation": (
                "Regional progression dictionaries were supplied as shape-compatible "
                "not-mentioned sentinels solely so the unchanged Dataset could return; "
                "y_aux is privileged training supervision and is neither retained nor "
                "compared"
            ),
        },
    }
    output = args.task_dir / "audit" / "full_batch_recreation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
