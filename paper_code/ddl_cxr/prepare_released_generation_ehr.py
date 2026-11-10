#!/usr/bin/env python3
"""Call the exact released EHR_Pickler for the source-faithful latent policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ldm.data.dynamic_data import EHR_Pickler


PAPER_ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validate", "test")
EXPECTED_IDENTITY_MATCHES = {"train": 2, "validate": 14, "test": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_link(target: Path, source: Path) -> None:
    source = source.resolve(strict=True)
    if target.is_symlink():
        assert target.resolve(strict=True) == source, (target, source)
        return
    if target.exists():
        raise FileExistsError(f"refusing non-symlink input: {target}")
    target.symlink_to(source)


def materialize_dm_with_prediction_cha(target: Path, source: Path, pred_path: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing existing repaired DM input: {target}")
    source_dm = pd.read_csv(source)
    pred = pd.read_csv(pred_path)
    assert "cha" not in source_dm.columns
    assert source_dm["index"].is_unique and pred["index"].is_unique
    source_by_index = source_dm.set_index("index", drop=False)
    missing = set(pred["index"]) - set(source_by_index.index)
    assert not missing, sorted(missing)[:10]
    # The released consumer addresses only prediction-local keys. Keep the DM
    # row at each such key, and supply the prediction gap required by the
    # producer's otherwise absent `cha` field.
    repaired = source_by_index.loc[pred["index"]].copy()
    repaired["cha"] = pred.set_index("index").loc[repaired.index, "cha"].to_numpy()
    repaired.to_csv(target, index=False)


def inspect_split(root: Path, split: str) -> dict[str, object]:
    pred_path = root / f"{split}_pred.csv"
    dm_path = root / f"{split}_dm_labels.csv"
    source_dm_path = PAPER_ROOT / "data_pre" / f"{split}_dm_labels.csv"
    pickle_path = root / f"{split}_ehr_gen.pkl"
    pred = pd.read_csv(pred_path)
    dm = pd.read_csv(dm_path)
    assert pred["index"].is_unique and dm["index"].is_unique

    pred_by_index = pred.set_index("index")
    dm_by_index = dm.set_index("index")
    shared = pred_by_index.index.intersection(dm_by_index.index)
    assert len(shared) == len(pred)
    identity_matches = int(
        (
            pred_by_index.loc[shared, ["subject_id", "stay_id"]].to_numpy()
            == dm_by_index.loc[shared, ["subject_id", "stay_id"]].to_numpy()
        ).all(axis=1).sum()
    )
    assert identity_matches == EXPECTED_IDENTITY_MATCHES[split]

    with pickle_path.open("rb") as handle:
        ehr = pickle.load(handle)
    assert len(ehr) == len(dm)
    assert set(map(str, pred["index"])) <= set(ehr)

    lengths: Counter[int] = Counter()
    dtypes: Counter[str] = Counter()
    finite_cells = 0
    total_cells = 0
    for key, value in ehr.items():
        assert isinstance(key, str)
        assert isinstance(value, torch.Tensor), type(value)
        assert value.ndim == 2 and value.shape[1] == 79, value.shape
        array = value.detach().cpu().numpy()
        assert np.isfinite(array).all(), key
        lengths[int(value.shape[0])] += 1
        dtypes[str(value.dtype)] += 1
        finite_cells += int(np.isfinite(array).sum())
        total_cells += int(array.size)
    assert finite_cells == total_cells

    return {
        "dm_csv": {
            "repair": "prediction-addressed rows plus prediction cha",
            "rows": len(dm),
            "sha256": sha256(dm_path),
            "source_rows": len(pd.read_csv(source_dm_path)),
            "source_sha256": sha256(source_dm_path),
        },
        "ehr_pickle": {
            "bytes": pickle_path.stat().st_size,
            "dtype_counts": dict(sorted(dtypes.items())),
            "entries": len(ehr),
            "length_counts": {str(key): value for key, value in sorted(lengths.items())},
            "sha256": sha256(pickle_path),
        },
        "identity_matches_at_shared_local_index": identity_matches,
        "prediction_csv": {"rows": len(pred), "sha256": sha256(pred_path)},
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    subjects_root = args.subjects_root.resolve(strict=True)
    output_root.mkdir(parents=True, exist_ok=True)

    prediction_root = (
        PAPER_ROOT / ".export_assets/author_source/prediction_cohort"
    )
    for split in SPLITS:
        ensure_link(
            output_root / f"{split}_pred.csv",
            prediction_root / f"{split}_pred.csv",
        )
        materialize_dm_with_prediction_cha(
            output_root / f"{split}_dm_labels.csv",
            PAPER_ROOT / "data_pre" / f"{split}_dm_labels.csv",
            output_root / f"{split}_pred.csv",
        )
        pickle_path = output_root / f"{split}_ehr_gen.pkl"
        if pickle_path.exists():
            raise FileExistsError(f"refusing to overwrite generated pickle: {pickle_path}")

    # This is the exact source class and method. In particular, stage
    # 'generation' selects *_dm_labels.csv and preserves the released local-key
    # identity defect rather than silently correcting it.
    producer = EHR_Pickler(
        stage="generation",
        metadata_path=os.fspath(output_root),
        mimic_iv_subjects_dir=os.fspath(subjects_root),
        normalizer_state=os.fspath(PAPER_ROOT / "ldm/data/normalizers/dm_normalizer"),
    )
    producer.pickle_ehr()

    report = {
        "normalizer_sha256": sha256(PAPER_ROOT / "ldm/data/normalizers/dm_normalizer"),
        "policy": (
            "released *_dm_labels.csv row at each *_pred.csv split-local index; "
            "missing producer cha supplied from that prediction row"
        ),
        "producer_sha256": sha256(PAPER_ROOT / "ldm/data/dynamic_data.py"),
        "splits": {split: inspect_split(output_root, split) for split in SPLITS},
        "status": "exact_released_policy",
    }
    report_path = output_root / "released_generation_ehr.json"
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_path.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
