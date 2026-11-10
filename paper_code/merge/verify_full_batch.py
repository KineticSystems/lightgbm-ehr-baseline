"""Assert exact reconstruction of MERGE's structured/raw-note projection."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch


REPOSITORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "mimiciv"))

from mimiciv_rus_multimodal import align_multimodal_irg_ts  # noqa: E402
from cohort_source import (  # noqa: E402
    FEATURE_NAMES,
    HOURS,
    NOTE_CODE,
    NOTE_SOURCE_ROW_OFFSET,
    SEED,
    TASKS,
    ensure_note_cache,
    reconstruct_author_boundary,
    retained_notes,
)
from train_mimiciv_multimodal import (  # noqa: E402
    MultimodalMIMICIVDataset,
    collate_multimodal,
)


BATCH_SIZE = 512
# Both published MIMIC-IV cells come from one released producer and loader; the
# released task script fixes the label and the pickle the trainer consumes.
TASK_SOURCE = {
    "ihm": {
        "paper_cell": "Table 2, MIMIC-IV IHM, MERGE, AUROC 85.40 +/- 0.24",
        "task_script": "create_ihm_task.py",
    },
    "los": {
        "paper_cell": "Table 2, MIMIC-IV LOS, MERGE, AUROC 81.88 +/- 0.18",
        "task_script": "create_los_task.py",
    },
}


def tensor_digest(digest: hashlib._Hash, tensor: torch.Tensor) -> None:
    values = tensor.detach().cpu().contiguous().numpy()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())


def note_projection_digest(notes) -> str:
    digest = hashlib.sha256()
    for row in notes.itertuples(index=False):
        digest.update(np.asarray([row.stay_id], dtype=np.int64).tobytes())
        digest.update(np.asarray([row.note_position], dtype=np.int64).tobytes())
        digest.update(np.asarray([row.icu_time_delta], dtype=np.float64).tobytes())
        encoded = str(row.text).encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def loader_samples(
    structured: np.ndarray,
    note_values: np.ndarray,
    note_masks: np.ndarray,
) -> list[dict]:
    result = []
    for index, sample in enumerate(structured):
        result.append(
            {
                "cxr": (
                    np.zeros((HOURS, 1), dtype=np.float32),
                    np.zeros(HOURS, dtype=bool),
                ),
                "labs_vitals": (
                    np.asarray(sample, dtype=np.float32),
                    np.ones(HOURS, dtype=bool),
                ),
                "notes": (note_values[index], note_masks[index]),
            }
        )
    return result


def compare_loader_batches(
    source_values: np.ndarray,
    recreated_values: np.ndarray,
    source_note_values: np.ndarray,
    recreated_note_values: np.ndarray,
    source_note_masks: np.ndarray,
    recreated_note_masks: np.ndarray,
    labels: np.ndarray,
    digests: dict[str, hashlib._Hash],
) -> int:
    rus = {
        "U": torch.zeros((3, HOURS), dtype=torch.float32),
        "R": torch.zeros((3, 3, HOURS), dtype=torch.float32),
        "S": torch.zeros((3, 3, HOURS), dtype=torch.float32),
    }
    modality_names = ["labs_vitals", "cxr", "notes"]
    modality_dims = {"labs_vitals": 30, "cxr": 1, "notes": 1}
    batches = 0
    for start in range(0, len(labels), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(labels))
        with contextlib.redirect_stdout(io.StringIO()):
            source_dataset = MultimodalMIMICIVDataset(
                loader_samples(
                    source_values[start:stop],
                    source_note_values[start:stop],
                    source_note_masks[start:stop],
                ),
                labels[start:stop].tolist(),
                rus,
                modality_names,
                modality_dims,
                max_seq_len=HOURS,
            )
            recreated_dataset = MultimodalMIMICIVDataset(
                loader_samples(
                    recreated_values[start:stop],
                    recreated_note_values[start:stop],
                    recreated_note_masks[start:stop],
                ),
                labels[start:stop].tolist(),
                rus,
                modality_names,
                modality_dims,
                max_seq_len=HOURS,
            )
        positions = range(stop - start)
        source = collate_multimodal([source_dataset[index] for index in positions])
        recreated = collate_multimodal(
            [recreated_dataset[index] for index in positions]
        )
        if len(source) != 4 or len(recreated) != 4:
            raise AssertionError("released MERGE collate arity changed")
        comparisons = (
            ("labs_vitals", source[0][1], recreated[0][1]),
            ("notes", source[0][2], recreated[0][2]),
            ("labs_vitals_mask", source[1][1], recreated[1][1]),
            ("notes_mask", source[1][2], recreated[1][2]),
            ("label", source[3], recreated[3]),
        )
        for name, left, right in comparisons:
            if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
                raise AssertionError(f"{name} shape/dtype mismatch")
            if not torch.equal(left, right):
                maximum = (
                    float(torch.max(torch.abs(left - right)).item())
                    if left.is_floating_point()
                    else None
                )
                raise AssertionError(f"{name} differs; maximum={maximum}")
            tensor_digest(digests[name], left)
        batches += 1
    return batches


def align_note_markers(
    notes,
    recreated_times: np.ndarray,
    record_position: dict[int, int],
    example_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_values = np.zeros((example_count, HOURS, 1), dtype=np.float64)
    source_masks = np.zeros((example_count, HOURS), dtype=bool)
    recreated_values = np.zeros_like(source_values)
    recreated_masks = np.zeros_like(source_masks)
    regular = np.zeros((HOURS, 1), dtype=np.float32)
    dimensions = {"notes": 1}
    for stay_id, selected in notes.groupby("stay_id", sort=False):
        position = record_position[int(stay_id)]
        row_positions = selected.index.to_numpy(np.int64)
        markers = selected["note_position"].to_numpy(np.float64) + 1.0
        source_features = [
            (float(time), np.asarray([marker], dtype=np.float64))
            for time, marker in zip(
                selected["icu_time_delta"].to_numpy(np.float64), markers
            )
        ]
        recreated_features = [
            (float(time), np.asarray([marker], dtype=np.float64))
            for time, marker in zip(recreated_times[row_positions], markers)
        ]
        source = align_multimodal_irg_ts(
            regular, {"notes": source_features}, dimensions
        )["notes"]
        recreated = align_multimodal_irg_ts(
            regular, {"notes": recreated_features}, dimensions
        )["notes"]
        source_values[position], source_masks[position] = source
        recreated_values[position], recreated_masks[position] = recreated
    return source_values, source_masks, recreated_values, recreated_masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--note-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, default="ihm")
    args = parser.parse_args()
    selected_source = TASK_SOURCE[args.task]

    manifest = json.loads((args.task_dir / "manifest.json").read_text())
    if manifest["task"].get("modalities") != ["structured", "notes"]:
        raise AssertionError("MERGE must retain structured data and raw notes")
    if manifest["task"].get("note_codes") != [NOTE_CODE]:
        raise AssertionError("MERGE note-code exception list changed")
    source_metadata = manifest.get("source", {})
    if source_metadata.get("selected_model_inputs") != [
        "structured",
        "notes",
        "CXR",
    ]:
        raise AssertionError("MERGE selected the wrong published experiment")
    if source_metadata.get("selection_tier") != 3:
        raise AssertionError("MERGE selected the wrong modality-priority tier")
    if source_metadata.get("retained_modalities") != ["structured", "notes"]:
        raise AssertionError("MERGE retained the wrong source projection")
    if source_metadata.get("omitted_modalities") != ["CXR"]:
        raise AssertionError("MERGE must omit only its imaging branch")
    if source_metadata.get("selected_paper_cell") != selected_source["paper_cell"]:
        raise AssertionError("MERGE batch check targets a different paper cell")

    note_cache = ensure_note_cache(
        args.mimic_root, args.note_root, args.cache_root
    )
    boundary = reconstruct_author_boundary(
        args.mimic_root,
        args.cache_root,
        include_irregular_scaler=False,
        task=args.task,
    )
    records = boundary["records"]
    expected_ids = np.asarray([row["sample_id"] for row in records], dtype=np.int64)
    record_subjects = np.asarray(
        [row["subject_id"] for row in records], dtype=np.int64
    )
    record_source_positions = np.asarray(
        [row["source_position"] for row in records], dtype=np.int64
    )
    record_intimes = np.asarray(
        [np.datetime64(row["intime"], "us") for row in records],
        dtype="datetime64[us]",
    )
    source_values = np.stack([row["values"] for row in records]).astype(
        np.float32, copy=False
    )
    labels = np.asarray([row["label"] for row in records], dtype=np.int64)
    notes = retained_notes(args.cache_root, set(expected_ids.tolist())).reset_index(
        drop=True
    )
    sorted_order = np.argsort(expected_ids)
    sorted_ids = expected_ids[sorted_order]
    recreated_values = np.full_like(source_values, np.nan)
    structured_seen = np.zeros(source_values.shape, dtype=bool)
    note_seen = np.zeros(len(notes), dtype=bool)
    recreated_note_times = np.full(len(notes), np.nan, dtype=np.float64)
    expected_note_lookup = {
        (int(row.stay_id), int(row.note_position)): position
        for position, row in enumerate(notes.itertuples(index=False))
    }
    expected_note_subjects = notes["subject_id"].to_numpy(np.int64)
    expected_note_times = notes["charttime"].to_numpy(dtype="datetime64[us]")
    expected_note_source_times = notes["icu_time_delta"].to_numpy(np.float64)
    expected_note_texts = notes["text"].astype(str).tolist()

    code_values = pa.array(FEATURE_NAMES)
    actual_codes = set()
    structured_event_count = 0
    note_event_count = 0
    event_path = args.task_dir / manifest["meds"]["events"]["file"]
    for batch in pq.ParquetFile(event_path).iter_batches(
        batch_size=500_000,
        columns=[
            "subject_id",
            "time",
            "sample_id",
            "code",
            "numeric_value",
            "source_row",
            "sequence_row",
            "source_time",
            "text_value",
        ],
    ):
        actual_codes.update(pc.unique(batch["code"]).to_pylist())
        is_note = pc.equal(batch["code"], NOTE_CODE)
        structured = batch.filter(pc.invert(is_note))
        if len(structured):
            if structured["numeric_value"].null_count:
                raise AssertionError("MEDS nullified a MERGE structured value")
            if structured["text_value"].null_count != len(structured):
                raise AssertionError("structured MERGE events contain text")
            sample_ids = structured["sample_id"].to_numpy().astype(
                np.int64, copy=False
            )
            found = np.searchsorted(sorted_ids, sample_ids)
            if np.any(found == len(sorted_ids)) or not np.array_equal(
                sorted_ids[found], sample_ids
            ):
                raise AssertionError("MEDS contains an unknown MERGE sample")
            record_positions = sorted_order[found]
            code_index_arrow = pc.index_in(
                structured["code"], value_set=code_values
            )
            if code_index_arrow.null_count:
                raise AssertionError("MEDS contains an undeclared structured code")
            code_index = code_index_arrow.to_numpy().astype(np.int64, copy=False)
            rows = structured["sequence_row"].to_numpy().astype(
                np.int64, copy=False
            )
            if np.any(rows < 0) or np.any(rows >= HOURS):
                raise AssertionError("MEDS contains an invalid sequence row")
            source_time = structured["source_time"].to_numpy()
            if not np.array_equal(source_time, rows.astype(np.float64)):
                raise AssertionError("MEDS changed structured source coordinates")
            if structured_seen[record_positions, rows, code_index].any():
                raise AssertionError("MEDS duplicates a structured cell")
            expected_source_rows = (
                record_source_positions[record_positions]
                * HOURS
                * len(FEATURE_NAMES)
                + rows * len(FEATURE_NAMES)
                + code_index
            )
            if not np.array_equal(
                structured["source_row"].to_numpy().astype(np.int64, copy=False),
                expected_source_rows,
            ):
                raise AssertionError("MEDS changed structured source-row ordering")
            if not np.array_equal(
                structured["subject_id"].to_numpy().astype(np.int64, copy=False),
                record_subjects[record_positions],
            ):
                raise AssertionError("MEDS changed a structured event subject")
            expected_times = record_intimes[record_positions] + rows.astype(
                "timedelta64[h]"
            )
            if not np.array_equal(structured["time"].to_numpy(), expected_times):
                raise AssertionError("MEDS changed a structured event time")
            recreated_values[record_positions, rows, code_index] = structured[
                "numeric_value"
            ].to_numpy().astype(np.float32, copy=False)
            structured_seen[record_positions, rows, code_index] = True
            structured_event_count += len(structured)

        selected_notes = batch.filter(is_note)
        if len(selected_notes):
            if selected_notes["numeric_value"].null_count != len(selected_notes):
                raise AssertionError("a MERGE note has a numeric value")
            if selected_notes["text_value"].null_count:
                raise AssertionError("a MERGE note lost its raw text")
            note_stays = selected_notes["sample_id"].to_numpy().astype(
                np.int64, copy=False
            )
            note_rows = selected_notes["source_row"].to_numpy().astype(
                np.int64, copy=False
            )
            note_positions = note_rows - NOTE_SOURCE_ROW_OFFSET
            expected_positions = np.asarray(
                [
                    expected_note_lookup.get((int(stay), int(position)), -1)
                    for stay, position in zip(note_stays, note_positions)
                ],
                dtype=np.int64,
            )
            if np.any(expected_positions < 0):
                raise AssertionError("MEDS contains an unknown raw note")
            if note_seen[expected_positions].any():
                raise AssertionError("MEDS duplicates a raw note")
            actual_source_times = selected_notes["source_time"].to_numpy()
            if not np.array_equal(
                actual_source_times, expected_note_source_times[expected_positions]
            ):
                raise AssertionError("MEDS changed a raw-note ICU coordinate")
            if not np.array_equal(
                selected_notes["sequence_row"].to_numpy().astype(np.int64, copy=False),
                np.floor(expected_note_source_times[expected_positions]).astype(
                    np.int64
                ),
            ):
                raise AssertionError("MEDS changed a raw-note hourly alignment")
            if not np.array_equal(
                selected_notes["subject_id"].to_numpy().astype(np.int64, copy=False),
                expected_note_subjects[expected_positions],
            ):
                raise AssertionError("MEDS changed a raw-note subject")
            if not np.array_equal(
                selected_notes["time"].to_numpy(),
                expected_note_times[expected_positions],
            ):
                raise AssertionError("MEDS changed a raw-note charttime")
            actual_texts = selected_notes["text_value"].to_pylist()
            if any(
                actual != expected_note_texts[position]
                for actual, position in zip(actual_texts, expected_positions)
            ):
                raise AssertionError("MEDS changed raw radiology-note text")
            recreated_note_times[expected_positions] = actual_source_times
            note_seen[expected_positions] = True
            note_event_count += len(selected_notes)

    if not structured_seen.all():
        raise AssertionError("MEDS omitted a MERGE structured cell")
    if not note_seen.all():
        raise AssertionError("MEDS omitted a selected raw note")
    if actual_codes != set(FEATURE_NAMES) | {NOTE_CODE}:
        raise AssertionError("MEDS MERGE event vocabulary is incomplete")
    if not np.array_equal(source_values, recreated_values):
        maximum = float(np.max(np.abs(source_values - recreated_values)))
        raise AssertionError(f"MEDS changed MERGE float32 values; maximum={maximum}")
    if note_event_count != len(notes):
        raise AssertionError("MEDS raw-note count differs from retained source")

    codes = pq.read_table(
        args.task_dir / manifest["meds"]["codes"]["file"]
    )["code"].to_pylist()
    if codes != list(FEATURE_NAMES) + [NOTE_CODE]:
        raise AssertionError("MEDS codes metadata changed")

    index = pq.read_table(args.task_dir / manifest["meds"]["task_index"]["file"])
    index_by_id = {int(row["sample_id"]): row for row in index.to_pylist()}
    if len(index_by_id) != len(records):
        raise AssertionError("MEDS task index size differs from author boundary")
    for row in records:
        actual = index_by_id.get(int(row["sample_id"]))
        if actual is None:
            raise AssertionError("MEDS omitted an author-boundary sample")
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
            raise AssertionError("MEDS changed an identifier, label, or split")

    record_position = {
        int(row["stay_id"]): position for position, row in enumerate(records)
    }
    (
        source_note_values,
        source_note_masks,
        recreated_note_values,
        recreated_note_masks,
    ) = align_note_markers(notes, recreated_note_times, record_position, len(records))
    digests = {
        name: hashlib.sha256()
        for name in (
            "labs_vitals",
            "notes",
            "labs_vitals_mask",
            "notes_mask",
            "label",
        )
    }
    split_batches = {}
    batches = 0
    for split in ("train", "validation", "test"):
        selected = np.asarray(
            [position for position, row in enumerate(records) if row["split"] == split],
            dtype=np.int64,
        )
        count = compare_loader_batches(
            source_values[selected],
            recreated_values[selected],
            source_note_values[selected],
            recreated_note_values[selected],
            source_note_masks[selected],
            recreated_note_masks[selected],
            labels[selected],
            digests,
        )
        split_batches[split] = count
        batches += count

    report = {
        "status": "exact",
        "comparison": (
            "all retained MERGE structured examples and selected raw-note "
            "events, including released note time alignment and "
            "MultimodalMIMICIVDataset/collate_multimodal views"
        ),
        "examples": len(records),
        "batches": batches,
        "batch_size": BATCH_SIZE,
        "event_count": structured_event_count + note_event_count,
        "structured_event_count": structured_event_count,
        "note_event_count": note_event_count,
        "stays_with_notes": int(notes["stay_id"].nunique()),
        "stays_without_notes": len(records) - int(notes["stay_id"].nunique()),
        "split_batches": split_batches,
        "batch_order": (
            "fixed source order within each released split; training shuffle is "
            "stochastic and immaterial to collate equality"
        ),
        "selected_source_model_inputs": ["structured", "notes", "CXR"],
        "retained_modalities": ["structured", "notes"],
        "omitted_modalities": ["CXR"],
        "imaging_retained": False,
        "author_modality_order": ["cxr", "labs_vitals", "notes"],
        "retained_branch_indices": {"labs_vitals": 1, "notes": 2},
        "note_projection": {
            "source": (
                "preprocess_notes.py::add_time_delta_notes_vectorized then "
                + selected_source["task_script"]
                + " raw text_data/text_time selection"
            ),
            "raw_fields_compared": [
                "subject_id",
                "stay_id",
                "charttime",
                "icu_time_delta",
                "source order",
                "raw text",
            ],
            "raw_projection_sha256": note_projection_digest(notes),
            "author_alignment": (
                "mimiciv_rus_multimodal.py::align_multimodal_irg_ts, exact "
                "hour bins, within-bin averaging, and missing masks"
            ),
            "embedding_boundary": (
                "raw text is retained before the release's unpinned "
                "dmis-lab/biobert-v1.1 transformation; embeddings are excluded"
            ),
        },
        "compared_fields": {
            "labs_vitals": {"shape": ["N", 48, 30], "dtype": "torch.float32"},
            "notes_alignment_marker": {
                "shape": ["N", 48, 1],
                "dtype": "torch.float32",
            },
            "labs_vitals_mask": {"shape": ["N", 48], "dtype": "torch.bool"},
            "notes_mask": {"shape": ["N", 48], "dtype": "torch.bool"},
            "label": {"shape": ["N"], "dtype": "torch.int64"},
        },
        "field_sha256": {
            name: digest.hexdigest() for name, digest in digests.items()
        },
        "ordered_source_boundary_sha256": boundary["boundary_sha256"],
        "lightgbm_boundary_audit": {
            "status": "exact",
            "included_codes": list(FEATURE_NAMES) + [NOTE_CODE],
            "note_codes": [NOTE_CODE],
            "structured_codes": list(FEATURE_NAMES),
            "excluded_persistent_modalities": [
                "CXR pixels",
                "CXR embeddings",
                "CXR timestamps/masks/flags",
                "BioBERT embeddings",
                "RUS",
            ],
            "auxiliary_event_count": 0,
            "rule": (
                "every MEDS event is an intended labs/vitals input or one "
                "selected raw radiology note"
            ),
        },
        "source_seed": SEED,
    }
    if args.task == "los":
        report["selected_paper_cell"] = selected_source["paper_cell"]
        report["released_label_constructor"] = (
            "mimiciv/data_preprocess/create_los_task.py::get_stay_list"
        )
        report["released_trainer"] = (
            "mimiciv/train_mimiciv_los.sh -> train_mimiciv_multimodal.py "
            "--task los --seq_len 48 --truncate_from_end"
        )
    report_path = args.task_dir / "audit" / "full_batch_recreation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
