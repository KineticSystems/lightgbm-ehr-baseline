#!/usr/bin/env python3
"""Prepare the reconstructed Table 4 FuseMoE model-input pickles.

The public ``missingInd`` artifacts were standardized on the broader cohort.
This script selects the paper's 8,770-stay complete-case cohort and applies a
training-only StandardScaler to each observed irregular channel.  The released
loader subsequently reconstructs the regular branch with ``F_impute``.

Outputs contain MIMIC-derived text and identifiers and must remain under the
ignored ``.export_assets`` tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.preprocessing import StandardScaler


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(PAPER_ROOT / "src"))

from export_source import load_source, loader_args  # noqa: E402
from preprocessing.data_mimiciv import TSNote_Irg  # noqa: E402


TASK = "ihm-48-table4"
SPLIT_FILE_NAMES = {
    "train": f"train_{TASK}_stays.pkl",
    "validation": f"val_{TASK}_stays.pkl",
    "test": f"test_{TASK}_stays.pkl",
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
    """Minimal tokenizer for hashing fields used by the pretrained-embedding path."""

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


def field_hashes(rows) -> dict[str, str]:
    dataset = TSNote_Irg(loader_args(), "train", UnusedTokenTokenizer(), data=rows)
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


def fit_scalers(train_rows):
    observed = []
    for feature in range(30):
        parts = []
        for row in train_rows:
            values = np.asarray(row["irg_ts"])
            mask = np.asarray(row["irg_ts_mask"]).astype(bool, copy=False)
            parts.append(np.asarray(values[mask[:, feature], feature], dtype=np.float64))
        channel = np.concatenate(parts)
        if len(channel) == 0 or not np.isfinite(channel).all():
            raise AssertionError(f"invalid observed training values for feature {feature}")
        scaler = StandardScaler().fit(channel.reshape(-1, 1))
        if not np.isfinite(scaler.mean_[0]) or not np.isfinite(scaler.scale_[0]):
            raise AssertionError(f"invalid scaler for feature {feature}")
        if scaler.scale_[0] <= 0:
            raise AssertionError(f"nonpositive scaler for feature {feature}")
        observed.append((channel, scaler))
    return observed


def transform_row(row, scalers):
    source_values = np.asarray(row["irg_ts"])
    source_mask = np.asarray(row["irg_ts_mask"])
    if source_values.ndim != 2 or source_values.shape[1] != 30:
        raise AssertionError("unexpected irregular tensor shape")
    if source_mask.shape != source_values.shape:
        raise AssertionError("irregular value/mask shape mismatch")
    mask = source_mask.astype(bool, copy=False)
    if not np.isfinite(source_values[mask]).all():
        raise AssertionError("nonfinite observed value")
    transformed = np.zeros(source_values.shape, dtype=np.float64)
    for feature, (_channel, scaler) in enumerate(scalers):
        selected = mask[:, feature]
        transformed[selected, feature] = (
            source_values[selected, feature].astype(np.float64)
            - float(scaler.mean_[0])
        ) / float(scaler.scale_[0])
    if np.count_nonzero(transformed[~mask]):
        raise AssertionError("masked values were not restored to zero")
    if int(row["text_missing"]) != 0 or int(row["cxr_missing"]) != 0:
        raise AssertionError("row is outside the complete-case cohort")
    if len(row["text_data"]) != len(row["text_time_to_end"]):
        raise AssertionError("note text/time length mismatch")
    if len(row["text_data"]) != len(row["text_embeddings"]):
        raise AssertionError("note text/embedding length mismatch")
    return {
        "name": int(row["name"]),
        "hadm_id": int(row["hadm_id"]),
        "stay_id": int(row["stay_id"]),
        "ts_tt": np.asarray(row["ts_tt"]).copy(),
        "irg_ts": transformed,
        "irg_ts_mask": source_mask.copy(),
        # The released loader overwrites this field whenever --reg_ts is set.
        "reg_ts": None,
        "text_data": row["text_data"],
        "text_time_to_end": np.asarray(row["text_time_to_end"]).copy(),
        "text_embeddings": row["text_embeddings"],
        "text_missing": 0,
        "label": int(row["label"]),
    }


def atomic_pickle(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=4)
    os.replace(temporary, path)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_data, inventory = load_source(args.source_root)
    split_lookup = dict(split_data)
    if [len(split_lookup[name]) for name in ("train", "validation", "test")] != [
        6139,
        1315,
        1316,
    ]:
        raise AssertionError("unexpected reconstructed split counts")
    scalers = fit_scalers(split_lookup["train"])

    scaler_manifest = {
        "observed_count": [int(len(channel)) for channel, _scaler in scalers],
        "mean": [float(scaler.mean_[0]) for _channel, scaler in scalers],
        "scale": [float(scaler.scale_[0]) for _channel, scaler in scalers],
    }
    transformed_train = [
        np.concatenate(
            [
                ((np.asarray(row["irg_ts"])[np.asarray(row["irg_ts_mask"])[:, feature].astype(bool), feature].astype(np.float64) - scaler_manifest["mean"][feature]) / scaler_manifest["scale"][feature])
                for row in split_lookup["train"]
            ]
        )
        for feature in range(30)
    ]
    scaler_manifest["transformed_train_max_abs_mean"] = float(
        max(abs(channel.mean()) for channel in transformed_train)
    )
    scaler_manifest["transformed_train_max_abs_std_error"] = float(
        max(abs(channel.std(ddof=0) - 1.0) for channel in transformed_train)
    )

    outputs = {}
    total_rows = 0
    for split in ("train", "validation", "test"):
        rows = [transform_row(row, scalers) for row in split_lookup[split]]
        path = args.output_dir / SPLIT_FILE_NAMES[split]
        atomic_pickle(path, rows)
        outputs[split] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "rows": len(rows),
            "positive": int(sum(int(row["label"]) for row in rows)),
            "ordered_stay_id_sha256": ordered_hash(rows, "stay_id"),
            "ordered_label_sha256": ordered_hash(rows, "label"),
            "model_field_sha256": field_hashes(rows),
        }
        total_rows += len(rows)
        del rows

    manifest = {
        "status": "prepared",
        "paper": "fusemoe",
        "target": "Table 4 MIMIC-IV 48-IHM vital signs + notes Laplace FuseMoE",
        "task": TASK,
        "cohort_rows": total_rows,
        "cohort_id_sha256": "8ed7fd4e61846ff62fc0eee0b73683734205f3a9321af15b211f5f759d74728b",
        "source_artifacts": inventory,
        "scaling": {
            "fit_split": "train",
            "fit_rows": 6139,
            "fit_rule": "one sklearn StandardScaler per channel over observed irregular cells only; masked cells restored to zero",
            "reconstruction_deviation": "selected-cohort scaling reconstructed from positive-affine-equivalent missingInd values because the historical 8770-row pickle is unavailable",
            **scaler_manifest,
        },
        "model_input_boundary": {
            "loader": "src/preprocessing/data_mimiciv.py::TSNote_Irg",
            "modeltype": "TS_Text",
            "regular_branch": "released F_impute recomputation from scaled irregular values",
            "notes": "released embeddings; last at most five selected by loader",
            "token_fields": "not included in model-field hashes because use_pt_text_embeddings=True; exact token parity is covered by the frozen full-batch report",
        },
        "outputs": outputs,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

