#!/usr/bin/env python3
"""Independently verify every reconstructed FuseMoE Table 4 input row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(PAPER_ROOT / "src"))

from export_source import load_source, loader_args  # noqa: E402
from preprocessing.data_mimiciv import TSNote_Irg  # noqa: E402


SPLIT_FILES = {
    "train": "train_ihm-48-table4_stays.pkl",
    "validation": "val_ihm-48-table4_stays.pkl",
    "test": "test_ihm-48-table4_stays.pkl",
}
MODEL_FIELDS = (
    "ts",
    "ts_mask",
    "ts_tt",
    "reg_ts",
    "text_embeddings",
    "note_time",
    "text_time_mask",
    "label",
)


class UnusedTokenTokenizer:
    def encode_plus(self, _text, **_kwargs):
        return {"input_ids": [0], "attention_mask": [0]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def update_array_hash(digest, value) -> None:
    array = np.ascontiguousarray(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def model_field_hashes(rows) -> dict[str, str]:
    dataset = TSNote_Irg(loader_args(), "test", UnusedTokenTokenizer(), data=rows)
    digests = {field: hashlib.sha256() for field in MODEL_FIELDS}
    for position in range(len(dataset)):
        item = dataset[position]
        values = {
            "ts": item["ts"],
            "ts_mask": item["ts_mask"],
            "ts_tt": item["ts_tt"],
            "reg_ts": item["reg_ts"],
            "text_embeddings": item["text_embeddings"],
            "note_time": item["note_time"],
            "text_time_mask": item["text_time_mask"],
            "label": item["label"],
        }
        for field, value in values.items():
            digests[field].update(np.asarray([position], dtype=np.int64).tobytes())
            update_array_hash(digests[field], value)
    return {field: digest.hexdigest() for field, digest in digests.items()}


def ordered_hash(rows, key: str) -> str:
    values = np.asarray([int(row[key]) for row in rows], dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def independently_fit(source_train):
    result = []
    for feature in range(30):
        values = np.concatenate(
            [
                np.asarray(row["irg_ts"])[
                    np.asarray(row["irg_ts_mask"])[:, feature].astype(bool), feature
                ].astype(np.float64)
                for row in source_train
            ]
        )
        result.append(StandardScaler().fit(values.reshape(-1, 1)))
    return result


def verify_row(source, prepared, scalers) -> None:
    for key in ("name", "hadm_id", "stay_id", "label"):
        if int(source[key]) != int(prepared[key]):
            raise AssertionError(f"identifier/label mismatch in {key}")
    if prepared["reg_ts"] is not None:
        raise AssertionError("serialized regular branch must be an ignored placeholder")
    np.testing.assert_array_equal(source["ts_tt"], prepared["ts_tt"])
    np.testing.assert_array_equal(source["irg_ts_mask"], prepared["irg_ts_mask"])
    np.testing.assert_array_equal(source["text_time_to_end"], prepared["text_time_to_end"])
    if source["text_data"] != prepared["text_data"]:
        raise AssertionError("raw note sequence changed")
    if len(source["text_embeddings"]) != len(prepared["text_embeddings"]):
        raise AssertionError("note embedding count changed")
    for left, right in zip(source["text_embeddings"], prepared["text_embeddings"]):
        np.testing.assert_array_equal(left, right)

    values = np.asarray(source["irg_ts"])
    mask = np.asarray(source["irg_ts_mask"]).astype(bool)
    expected = np.zeros(values.shape, dtype=np.float64)
    for feature, scaler in enumerate(scalers):
        chosen = mask[:, feature]
        expected[chosen, feature] = (
            values[chosen, feature].astype(np.float64) - float(scaler.mean_[0])
        ) / float(scaler.scale_[0])
    np.testing.assert_array_equal(expected, prepared["irg_ts"])


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    source_splits, inventory = load_source(args.source_root)
    source_lookup = dict(source_splits)
    if inventory != manifest["source_artifacts"]:
        raise AssertionError("source artifact inventory changed")
    scalers = independently_fit(source_lookup["train"])
    np.testing.assert_array_equal(
        np.asarray([scaler.mean_[0] for scaler in scalers]),
        np.asarray(manifest["scaling"]["mean"]),
    )
    np.testing.assert_array_equal(
        np.asarray([scaler.scale_[0] for scaler in scalers]),
        np.asarray(manifest["scaling"]["scale"]),
    )

    verified = {}
    total = 0
    for split in ("train", "validation", "test"):
        path = args.input_dir / SPLIT_FILES[split]
        if sha256_file(path) != manifest["outputs"][split]["sha256"]:
            raise AssertionError(f"prepared pickle hash changed for {split}")
        with path.open("rb") as handle:
            rows = pickle.load(handle)
        expected_rows = source_lookup[split]
        if len(rows) != len(expected_rows):
            raise AssertionError(f"prepared row count changed for {split}")
        for source, prepared in zip(expected_rows, rows):
            verify_row(source, prepared, scalers)
        stay_hash = ordered_hash(rows, "stay_id")
        label_hash = ordered_hash(rows, "label")
        field_hashes = model_field_hashes(rows)
        expected_manifest = manifest["outputs"][split]
        if stay_hash != expected_manifest["ordered_stay_id_sha256"]:
            raise AssertionError(f"ordered stay hash mismatch for {split}")
        if label_hash != expected_manifest["ordered_label_sha256"]:
            raise AssertionError(f"ordered label hash mismatch for {split}")
        if field_hashes != expected_manifest["model_field_sha256"]:
            raise AssertionError(f"model-field hashes mismatch for {split}")
        verified[split] = {
            "rows": len(rows),
            "positive": int(sum(int(row["label"]) for row in rows)),
            "pickle_sha256": sha256_file(path),
            "ordered_stay_id_sha256": stay_hash,
            "ordered_label_sha256": label_hash,
            "model_field_sha256": field_hashes,
        }
        total += len(rows)
        del rows

    report = {
        "status": "exact",
        "paper": "fusemoe",
        "task": "ihm-48-table4",
        "verified_rows": total,
        "checks": [
            "independently reconstructed training-only channel scalers",
            "exact reconstructed split order, identifiers, and labels",
            "exact structured values, masks, and timestamps",
            "exact raw notes, note times, and released note embeddings",
            "exact released-loader model-field hashes including F_impute",
        ],
        "splits": verified,
    }
    atomic_json(args.input_dir / "verification.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

