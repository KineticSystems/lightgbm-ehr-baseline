#!/usr/bin/env python3
"""Verify every released MERGE IHM pickle association and loader tensor."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(PAPER_ROOT / "mimiciv"))

from mimiciv_rus_multimodal import preprocess_mimiciv_data  # noqa: E402
from cohort_source import FEATURE_NAMES, HOURS, reconstruct_author_boundary  # noqa: E402
from train_mimiciv_multimodal import (  # noqa: E402
    MultimodalMIMICIVDataset,
    collate_multimodal,
)


BASE_NAME = "ihm-48-cxr-notes-missingInd-standardized"
BATCH_SIZE = 512
EXPECTED_PREFILTER = {"train": 32_435, "validation": 6_950, "test": 6_952}
EXPECTED_POSTFILTER = {"train": 32_393, "validation": 6_940, "test": 6_944}
EXPECTED_POSITIVES = {"train": 4_768, "validation": 998, "test": 1_053}
MODALITY_DIMS = {"labs_vitals": 30, "cxr": 1024, "notes": 768}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def tensor_digest(digest: hashlib._Hash, tensor: torch.Tensor) -> None:
    values = tensor.detach().cpu().contiguous().numpy()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())


def assert_array_equal(name: str, actual: object, expected: object) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.dtype != expected_array.dtype:
        # Released pandas/scikit-learn paths may preserve equivalent platform
        # integer widths. Scientific floating-point values must remain exact.
        if actual_array.dtype.kind == expected_array.dtype.kind == "i":
            expected_array = expected_array.astype(actual_array.dtype)
        else:
            raise AssertionError(
                f"{name} dtype differs: {actual_array.dtype} != {expected_array.dtype}"
            )
    if actual_array.shape != expected_array.shape:
        raise AssertionError(
            f"{name} shape differs: {actual_array.shape} != {expected_array.shape}"
        )
    equal = (
        np.array_equal(actual_array, expected_array, equal_nan=True)
        if actual_array.dtype.kind == "f"
        else np.array_equal(actual_array, expected_array)
    )
    if not equal:
        if actual_array.dtype.kind == "f":
            finite = np.isfinite(actual_array) & np.isfinite(expected_array)
            maximum = (
                float(np.max(np.abs(actual_array[finite] - expected_array[finite])))
                if finite.any()
                else None
            )
        else:
            maximum = None
        raise AssertionError(f"{name} differs; maximum={maximum}")


def check_scaler(actual: object, expected: object, name: str) -> dict[str, object]:
    for field in ("mean_", "var_", "scale_"):
        assert_array_equal(
            f"{name}.{field}", getattr(actual, field), getattr(expected, field)
        )
    if int(actual.n_features_in_) != int(expected.n_features_in_):
        raise AssertionError(f"{name}.n_features_in_ differs")
    digest = hashlib.sha256()
    for field in ("mean_", "var_", "scale_"):
        digest.update(field.encode())
        digest.update(np.asarray(getattr(actual, field), dtype=np.float64).tobytes())
    return {
        "sha256": digest.hexdigest(),
        "n_features": int(actual.n_features_in_),
    }


def group_positions(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    return {
        int(key): np.asarray(value, dtype=np.int64)
        for key, value in frame.groupby("stay_id", sort=False).indices.items()
    }


def expected_alignment(
    times: object, features: object, width: int
) -> tuple[np.ndarray, np.ndarray]:
    aligned = np.zeros((HOURS, width), dtype=np.float64)
    mask = np.zeros(HOURS, dtype=bool)
    by_hour: dict[int, list[np.ndarray]] = {}
    for time, feature in zip(np.asarray(times, dtype=np.float64), features):
        hour = int(float(time))
        if hour < 0 or hour >= HOURS:
            raise AssertionError(f"out-of-window modality time reached loader: {time}")
        vector = np.asarray(feature)
        if vector.shape != (width,):
            raise AssertionError(
                f"modality vector width differs: {vector.shape} != {(width,)}"
            )
        if not np.isfinite(vector).all():
            raise AssertionError("nonfinite modality vector reached loader")
        by_hour.setdefault(hour, []).append(vector)
    for hour, vectors in by_hour.items():
        aligned[hour] = np.mean(vectors, axis=0)
        mask[hour] = True
    return aligned, mask


def verify_raw_associations(
    *,
    stay: dict,
    expected_record: dict,
    irregular: pd.DataFrame,
    irregular_scaled: np.ndarray,
    irregular_positions: dict[int, np.ndarray],
    note_frame: pd.DataFrame,
    note_positions: dict[int, np.ndarray],
    cxr_frame: pd.DataFrame,
    cxr_positions: dict[int, np.ndarray],
    post_record: dict | None,
) -> tuple[int, int, int]:
    stay_id = int(expected_record["stay_id"])
    scalar_fields = {
        "name": expected_record["subject_id"],
        "hadm_id": expected_record["hadm_id"],
        "stay_id": stay_id,
        "label": expected_record["label"],
    }
    for field, expected in scalar_fields.items():
        if int(stay[field]) != int(expected):
            raise AssertionError(f"stay {field} association changed")
    if tuple(stay["feature_names"]) != tuple(FEATURE_NAMES):
        raise AssertionError("structured feature order changed")

    positions = irregular_positions[stay_id]
    selected_irregular = irregular.iloc[positions]
    assert_array_equal(
        "ts_tt",
        np.asarray(stay["ts_tt"]),
        selected_irregular["icu_time_delta"].to_numpy(),
    )
    expected_mask = selected_irregular[list(FEATURE_NAMES)].notnull().to_numpy(int)
    expected_irregular = irregular_scaled[positions].copy()
    expected_irregular[np.isnan(expected_irregular)] = 0.0
    assert_array_equal("irg_ts_mask", np.asarray(stay["irg_ts_mask"]), expected_mask)
    assert_array_equal("irg_ts", np.asarray(stay["irg_ts"]), expected_irregular)
    regular = np.asarray(stay["reg_ts"])
    if regular.shape != (HOURS, len(FEATURE_NAMES)) or not np.isfinite(regular).all():
        raise AssertionError("regular structured branch shape/finiteness changed")
    if post_record is not None:
        assert_array_equal(
            "postfilter reg_ts float32",
            regular.astype(np.float32),
            post_record["values"],
        )

    selected_note_positions = note_positions.get(stay_id)
    if selected_note_positions is None:
        if int(stay["text_missing"]) != 1:
            raise AssertionError("note missing flag changed")
        if stay["text_data"] or len(stay["text_time"]) or stay["text_embeddings"]:
            raise AssertionError("missing note branch is not empty")
        note_count = 0
    else:
        selected_notes = note_frame.iloc[selected_note_positions]
        if int(stay["text_missing"]) != 0:
            raise AssertionError("note presence flag changed")
        if list(stay["text_data"]) != selected_notes["text"].tolist():
            raise AssertionError("raw note text/stay association changed")
        assert_array_equal(
            "text_time",
            np.asarray(stay["text_time"]),
            selected_notes["icu_time_delta"].to_numpy(),
        )
        if len(stay["text_embeddings"]) != len(selected_notes):
            raise AssertionError("note embedding count changed")
        for actual, chunks in zip(
            stay["text_embeddings"], selected_notes["biobert_embeddings"]
        ):
            expected = np.mean(np.stack(chunks), axis=0)
            assert_array_equal("note mean embedding", np.asarray(actual), expected)
        note_count = len(selected_notes)

    selected_cxr_positions = cxr_positions.get(stay_id)
    if selected_cxr_positions is None:
        if int(stay["cxr_missing"]) != 1:
            raise AssertionError("CXR missing flag changed")
        if len(stay["cxr_feats"]) or len(stay["cxr_time"]):
            raise AssertionError("missing CXR branch is not empty")
        cxr_count = 0
    else:
        selected_cxrs = cxr_frame.iloc[selected_cxr_positions]
        if int(stay["cxr_missing"]) != 0:
            raise AssertionError("CXR presence flag changed")
        assert_array_equal(
            "cxr_subject_ids",
            np.asarray(stay["cxr_subject_ids"]),
            selected_cxrs["subject_id"].to_numpy(),
        )
        assert_array_equal(
            "cxr_study_ids",
            np.asarray(stay["cxr_study_ids"]),
            selected_cxrs["study_id"].to_numpy(),
        )
        assert_array_equal(
            "cxr_dicom_ids",
            np.asarray(stay["cxr_dicom_ids"]),
            selected_cxrs["dicom_id"].to_numpy(),
        )
        assert_array_equal(
            "cxr_time",
            np.asarray(stay["cxr_time"]),
            selected_cxrs["icu_time_delta"].to_numpy(),
        )
        if len(stay["cxr_feats"]) != len(selected_cxrs):
            raise AssertionError("CXR feature count changed")
        for actual, expected in zip(stay["cxr_feats"], selected_cxrs["densefeatures"]):
            assert_array_equal("CXR dense feature", np.asarray(actual), expected)
        cxr_count = len(selected_cxrs)

    return len(positions), note_count, cxr_count


def verify_loader_chunk(
    stays: list[dict],
    expected_records: list[dict],
    digests: dict[str, hashlib._Hash],
) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        processed, labels = preprocess_mimiciv_data(stays, MODALITY_DIMS)
    if len(processed) != len(stays) or len(labels) != len(stays):
        raise AssertionError("released preprocessing omitted an eligible stay")

    expected_modalities: dict[str, list[np.ndarray]] = {
        name: [] for name in sorted(MODALITY_DIMS)
    }
    expected_masks: dict[str, list[np.ndarray]] = {
        name: [] for name in sorted(MODALITY_DIMS)
    }
    for stay, record, sample, label in zip(stays, expected_records, processed, labels):
        if int(label) != int(record["label"]):
            raise AssertionError("released preprocessing changed a label")
        if set(sample) != set(MODALITY_DIMS):
            raise AssertionError("released preprocessing modality set changed")
        cxr, cxr_mask = expected_alignment(
            stay["cxr_time"], stay["cxr_feats"], MODALITY_DIMS["cxr"]
        )
        notes, notes_mask = expected_alignment(
            stay["text_time"], stay["text_embeddings"], MODALITY_DIMS["notes"]
        )
        labs = np.asarray(stay["reg_ts"])
        labs_mask = np.ones(HOURS, dtype=bool)
        expected = {
            "cxr": (cxr, cxr_mask),
            "labs_vitals": (labs, labs_mask),
            "notes": (notes, notes_mask),
        }
        for modality in sorted(MODALITY_DIMS):
            actual_values, actual_mask = sample[modality]
            expected_values, expected_mask = expected[modality]
            assert_array_equal(
                f"aligned {modality}", actual_values, expected_values
            )
            assert_array_equal(
                f"aligned {modality} mask", actual_mask, expected_mask
            )
            expected_modalities[modality].append(expected_values)
            expected_masks[modality].append(expected_mask)

    rus = {
        "U": torch.zeros((3, HOURS), dtype=torch.float32),
        "R": torch.zeros((3, 3, HOURS), dtype=torch.float32),
        "S": torch.zeros((3, 3, HOURS), dtype=torch.float32),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = MultimodalMIMICIVDataset(
            processed,
            labels,
            rus,
            list(MODALITY_DIMS),
            MODALITY_DIMS,
            max_seq_len=HOURS,
        )
    batch = collate_multimodal([dataset[index] for index in range(len(dataset))])
    modality_batches, mask_batches, _, label_batch = batch
    modality_order = sorted(MODALITY_DIMS)
    for index, modality in enumerate(modality_order):
        expected_values = torch.from_numpy(
            np.stack(expected_modalities[modality])
        ).float()
        expected_mask = torch.from_numpy(np.stack(expected_masks[modality])).bool()
        if not torch.equal(modality_batches[index], expected_values):
            raise AssertionError(f"released Dataset/collate changed {modality}")
        if not torch.equal(mask_batches[index], expected_mask):
            raise AssertionError(f"released Dataset/collate changed {modality} mask")
        tensor_digest(digests[modality], modality_batches[index])
        tensor_digest(digests[f"{modality}_mask"], mask_batches[index])
    expected_labels = torch.tensor(
        [int(record["label"]) for record in expected_records], dtype=torch.long
    )
    if not torch.equal(label_batch, expected_labels):
        raise AssertionError("released Dataset/collate changed labels")
    tensor_digest(digests["label"], label_batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--repro-root", type=Path, required=True)
    parser.add_argument("--export-task", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.repro_root / "data"
    task_root = data_root / "ihm"
    audit_root = args.repro_root / "audit"
    embedding_reports = {
        name: json.loads((audit_root / f"{name}_embeddings.json").read_text())
        for name in ("notes", "cxr")
    }
    note_path = data_root / "rad_notes_text_embeddings.parquet"
    cxr_path = data_root / "mimic_cxr_embeddings.parquet"
    if embedding_reports["notes"]["output_sha256"] != sha256_file(note_path):
        raise AssertionError("note embedding artifact changed after materialization")
    if embedding_reports["cxr"]["output_sha256"] != sha256_file(cxr_path):
        raise AssertionError("CXR embedding artifact changed after materialization")

    boundary = reconstruct_author_boundary(args.mimic_root, args.cache_root)
    prefilter = {
        split: [row for row in boundary["prefilter_records"] if row["split"] == split]
        for split in EXPECTED_PREFILTER
    }
    postfilter = {
        split: [row for row in boundary["records"] if row["split"] == split]
        for split in EXPECTED_POSTFILTER
    }
    if {split: len(rows) for split, rows in prefilter.items()} != EXPECTED_PREFILTER:
        raise AssertionError("export prefilter counts changed")
    if {split: len(rows) for split, rows in postfilter.items()} != EXPECTED_POSTFILTER:
        raise AssertionError("export postfilter counts changed")

    irregular = pd.read_parquet(args.cache_root / "ts_labs_vitals.parquet")
    irregular = irregular.loc[
        (irregular["icu_time_delta"] >= 0) & (irregular["icu_time_delta"] < HOURS)
    ].reset_index(drop=True)
    if tuple(irregular.columns) != (
        "subject_id",
        "hadm_id",
        "stay_id",
        "hosp_time_delta",
        "icu_time_delta",
    ) + tuple(FEATURE_NAMES):
        raise AssertionError("structured cache schema/order changed")
    irregular_positions = group_positions(irregular)
    irregular_scaled = boundary["irregular_scaler"].transform(
        irregular[list(FEATURE_NAMES)]
    )

    notes = pd.read_parquet(note_path)
    cxrs = pd.read_parquet(cxr_path)
    note_positions = group_positions(notes)
    cxr_positions = group_positions(cxrs)
    if len(notes) != 105_197 or len(cxrs) != 19_940:
        raise AssertionError("materialized modality row count changed")

    irg_scaler_path = task_root / f"{BASE_NAME}_irg_scaler.pkl"
    imputed_scaler_path = task_root / f"{BASE_NAME}_imputed_scaler.pkl"
    with irg_scaler_path.open("rb") as handle:
        actual_irg_scaler = pickle.load(handle)
    with imputed_scaler_path.open("rb") as handle:
        actual_imputed_scaler = pickle.load(handle)
    scaler_report = {
        "irregular": check_scaler(
            actual_irg_scaler, boundary["irregular_scaler"], "irregular scaler"
        ),
        "imputed": check_scaler(
            actual_imputed_scaler, boundary["imputed_scaler"], "imputed scaler"
        ),
    }

    export_batch = json.loads(
        (args.export_task / "audit" / "full_batch_recreation.json").read_text()
    )
    digests = {
        name: hashlib.sha256()
        for name in (
            "cxr",
            "labs_vitals",
            "notes",
            "cxr_mask",
            "labs_vitals_mask",
            "notes_mask",
            "label",
        )
    }
    pickle_hashes = {}
    split_report = {}
    total_irregular = total_notes = total_cxrs = 0
    for split, file_prefix in (
        ("train", "train"),
        ("validation", "val"),
        ("test", "test"),
    ):
        pickle_path = task_root / f"{file_prefix}_{BASE_NAME}_stays.pkl"
        pickle_hashes[split] = sha256_file(pickle_path)
        with pickle_path.open("rb") as handle:
            stays = pickle.load(handle)
        expected_prefilter = prefilter[split]
        if len(stays) != len(expected_prefilter):
            raise AssertionError(f"{split} author pickle count changed")
        actual_ids = np.asarray([int(stay["stay_id"]) for stay in stays], dtype=np.int64)
        expected_ids = np.asarray(
            [int(row["stay_id"]) for row in expected_prefilter], dtype=np.int64
        )
        assert_array_equal(f"{split} ordered IDs", actual_ids, expected_ids)

        post_lookup = {int(row["stay_id"]): row for row in postfilter[split]}
        eligible_stays = []
        eligible_records = []
        for stay, expected_record in zip(stays, expected_prefilter):
            post_record = post_lookup.get(int(expected_record["stay_id"]))
            counts = verify_raw_associations(
                stay=stay,
                expected_record=expected_record,
                irregular=irregular,
                irregular_scaled=irregular_scaled,
                irregular_positions=irregular_positions,
                note_frame=notes,
                note_positions=note_positions,
                cxr_frame=cxrs,
                cxr_positions=cxr_positions,
                post_record=post_record,
            )
            total_irregular += counts[0]
            total_notes += counts[1]
            total_cxrs += counts[2]
            if len(stay["ts_tt"]) > 12:
                if post_record is None:
                    raise AssertionError("eligible author stay absent from the export")
                eligible_stays.append(stay)
                eligible_records.append(post_record)
            elif post_record is not None:
                raise AssertionError("ineligible author stay present in the export")
        if len(eligible_stays) != EXPECTED_POSTFILTER[split]:
            raise AssertionError(f"{split} postfilter count changed")
        if sum(int(row["label"]) for row in eligible_records) != EXPECTED_POSITIVES[split]:
            raise AssertionError(f"{split} positive count changed")
        for start in range(0, len(eligible_stays), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(eligible_stays))
            verify_loader_chunk(
                eligible_stays[start:stop], eligible_records[start:stop], digests
            )
        split_report[split] = {
            "prefilter_count": len(stays),
            "postfilter_count": len(eligible_stays),
            "positive": sum(int(row["label"]) for row in eligible_records),
            "ordered_stay_id_sha256": array_sha256(actual_ids),
        }

    field_hashes = {name: digest.hexdigest() for name, digest in digests.items()}
    if field_hashes["labs_vitals"] != export_batch["field_sha256"]["labs_vitals"]:
        raise AssertionError("full author pickle labs loader hash differs from the export")
    if field_hashes["label"] != export_batch["field_sha256"]["label"]:
        raise AssertionError("full author pickle label loader hash differs from the export")
    if total_irregular != len(irregular):
        raise AssertionError("author pickles did not cover every structured source row")
    if total_notes != len(notes):
        raise AssertionError("author pickles did not cover every note embedding row")
    if total_cxrs != len(cxrs):
        raise AssertionError("author pickles did not cover every CXR embedding row")

    artifacts = {
        "notes_embeddings": sha256_file(note_path),
        "cxr_embeddings": sha256_file(cxr_path),
        "irregular_scaler": sha256_file(irg_scaler_path),
        "imputed_scaler": sha256_file(imputed_scaler_path),
        "pickles": pickle_hashes,
    }
    log_path = audit_root / "create_ihm_task.log"
    if log_path.is_file():
        artifacts["create_log"] = sha256_file(log_path)
    report = {
        "status": "exact",
        "author_entrypoints": {
            "task": "mimiciv/data_preprocess/create_ihm_task.py",
            "alignment": "mimiciv/mimiciv_rus_multimodal.py::preprocess_mimiciv_data",
            "dataset": "mimiciv/train_mimiciv_multimodal.py::MultimodalMIMICIVDataset",
            "collate": "mimiciv/train_mimiciv_multimodal.py::collate_multimodal",
        },
        "checks": [
            "every prefilter stay identifier, subject, admission, label, and source order",
            "every irregular labs/vitals row, timestamp, value, observation mask, and feature name",
            "every note text/time and mean-pooled pinned BioBERT embedding",
            "every CXR subject/study/DICOM/time and pinned DenseNet feature",
            "all missing-modality flags and empty branches",
            "released >12-timestamp postfilter and all three split orders",
            "independent hourly modality alignment for every retained stay",
            "released Dataset/collate tensors in exact batches of 512",
            "structured and label loader hashes against the frozen boundary",
            "both fitted StandardScaler states against independent reconstruction",
        ],
        "source_rows": {
            "irregular_labs_vitals": total_irregular,
            "notes": total_notes,
            "cxr": total_cxrs,
        },
        "splits": split_report,
        "modality_order": sorted(MODALITY_DIMS),
        "modality_dimensions": MODALITY_DIMS,
        "batch_size": BATCH_SIZE,
        "field_sha256": field_hashes,
        "export_crosscheck": {
            "status": "exact",
            "labs_vitals_sha256": field_hashes["labs_vitals"],
            "label_sha256": field_hashes["label"],
        },
        "scalers": scaler_report,
        "artifacts": artifacts,
    }
    output_path = audit_root / "ihm_inputs.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
