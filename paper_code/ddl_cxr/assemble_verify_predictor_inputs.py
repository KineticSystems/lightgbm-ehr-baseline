#!/usr/bin/env python3
"""Assemble and independently verify every released DDL-CXR predictor item."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import struct
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image

from ldm.data.dynamic_data import PredictDataset


EXPECTED = {
    "train": {
        "rows": 5142,
        "positives": 746,
        "csv": "3c8d8dae94e32230295555799b371de2b8b517d4343e5d50c75bd7f500dd31db",
        "ehr": "1a013a64b31a026446284f5d3790e2f4dd41094fa84a9c3c0156b4889728bc58",
        "hdf5": "f8f78f7a4968b12036ea60d2a4ccd36ba26a9ac8463076c80aef38316b509359",
    },
    "validate": {
        "rows": 861,
        "positives": 124,
        "csv": "bfdb351f2079b1e812bcfa7de735e3ecc8f115ef723a54e914cf65b60f668fea",
        "ehr": "2fc2037ca95f9aa4d68b1db11c241aead2be52acf32478cdcbd0e29a61ed2e93",
        "hdf5": "afabd1e2657aacb242b3d1af9758af23948c149036a31d8f7aefeaa616fff17d",
    },
    "test": {
        "rows": 1483,
        "positives": 219,
        "csv": "d7f194d5308b61b40aebeabdd803df8367ec5063d623c745d54cd3f7aceffc07",
        "ehr": "0ccf8165605a1bf29b7c9ae0f0bdec134f11ebc9adef4d50e7bfe490aa097df1",
        "hdf5": "24e404aefd22b213b613bed62c6dd31c0a04a37b66a28c91bd72130992aea1ed",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_array(digest: "hashlib._Hash", value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(struct.pack("<I", array.ndim))
    digest.update(struct.pack("<" + "Q" * array.ndim, *array.shape))
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))


def symlink_exact(target: Path, link: Path) -> None:
    target = target.resolve(strict=True)
    if link.is_symlink():
        if link.resolve(strict=True) != target:
            raise AssertionError(f"existing link has wrong target: {link}")
        return
    if link.exists():
        raise FileExistsError(f"refusing non-link predictor input: {link}")
    link.symlink_to(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cxr-root", type=Path, required=True)
    args = parser.parse_args()

    author_root = args.author_root.resolve(strict=True)
    generation_root = args.generation_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    cxr_root = args.cxr_root.resolve(strict=True)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "verification.json"
    if report_path.exists():
        raise FileExistsError(f"refusing prior verification report: {report_path}")

    for partition, expected in EXPECTED.items():
        csv = author_root / f"{partition}_pred.csv"
        ehr = author_root / f"{partition}_ehr_pred.pkl"
        latent = generation_root / partition / f"{partition}_z1_for_pred.h5"
        assert sha256(csv) == expected["csv"]
        assert sha256(ehr) == expected["ehr"]
        assert sha256(latent) == expected["hdf5"]
        symlink_exact(csv, output_root / csv.name)
        symlink_exact(ehr, output_root / ehr.name)
        symlink_exact(latent, output_root / latent.name)

    report: dict[str, object] = {
        "cxr_root": "exact MIMIC-CXR-JPG 2.0.0 files payload",
        "partitions": {},
        "status": "in_progress",
    }
    combined = hashlib.sha256()
    for partition, expected in EXPECTED.items():
        csv_path = output_root / f"{partition}_pred.csv"
        ehr_path = output_root / f"{partition}_ehr_pred.pkl"
        hdf_path = output_root / f"{partition}_z1_for_pred.h5"
        frame = pd.read_csv(csv_path)
        assert len(frame) == expected["rows"]
        assert frame.index.equals(pd.RangeIndex(expected["rows"]))
        assert int(frame["mortality_inhospital"].sum()) == expected["positives"]
        assert frame["index"].tolist() == list(range(expected["rows"]))
        with ehr_path.open("rb") as handle:
            expected_ehr = pickle.load(handle)
        assert set(expected_ehr) == {str(index) for index in range(expected["rows"])}
        with h5py.File(hdf_path, "r") as latent_handle:
            assert set(latent_handle) == {str(index) for index in range(expected["rows"])}
            for index in range(expected["rows"]):
                latent = latent_handle[str(index)][...]
                assert latent.shape == (4, 28, 28)
                assert latent.dtype == np.float32
                assert np.isfinite(latent).all()

        dataset = PredictDataset(
            partition=partition,
            task="mortality",
            metadata_path=os.fspath(output_root),
            mimic_cxr_jpg_dir=os.fspath(cxr_root),
        )
        assert len(dataset) == expected["rows"]
        digest = hashlib.sha256()
        with h5py.File(hdf_path, "r") as latent_handle:
            for index in range(expected["rows"]):
                row = frame.loc[index]
                item_seed = 910_000 + index
                random.seed(item_seed)
                np.random.seed(item_seed)
                torch.manual_seed(item_seed)
                ehr, label, image, sample_id, latent = dataset[index]
                assert int(sample_id) == index
                assert int(label.item()) == int(row["mortality_inhospital"])
                expected_ehr_value = np.asarray(expected_ehr[str(index)])
                assert ehr.shape == (48, 79)
                assert np.array_equal(np.asarray(ehr), expected_ehr_value)
                assert tuple(image.shape) == (3, 224, 224)
                assert torch.isfinite(image).all()
                assert tuple(latent.shape) == (4, 28, 28)
                assert latent.dtype == torch.float32
                assert torch.isfinite(latent).all()
                assert np.array_equal(latent.numpy(), latent_handle[str(index)][...])

                image_path = (
                    cxr_root
                    / f"p{str(int(row['subject_id']))[:2]}"
                    / f"p{int(row['subject_id'])}"
                    / f"s{int(row['study_id'])}"
                    / f"{row['dicom_id']}.jpg"
                )
                assert image_path.is_file()
                random.seed(item_seed)
                np.random.seed(item_seed)
                torch.manual_seed(item_seed)
                with Image.open(image_path) as expected_pil:
                    expected_image = dataset.transform[partition](
                        expected_pil.convert("RGB")
                    )
                assert torch.equal(image, expected_image)

                digest.update(struct.pack("<qB", index, int(label.item())))
                update_array(digest, np.asarray(ehr))
                update_array(digest, image.numpy())
                update_array(digest, latent.numpy())
        dataset.gen_h5file.close()
        partition_digest = digest.hexdigest()
        combined.update(partition.encode("ascii") + b"\0")
        combined.update(bytes.fromhex(partition_digest))
        report["partitions"][partition] = {
            "csv_sha256": expected["csv"],
            "ehr_sha256": expected["ehr"],
            "hdf5_sha256": expected["hdf5"],
            "item_digest_sha256": partition_digest,
            "labels": "exact",
            "loader_associations": "exact",
            "positives": expected["positives"],
            "rows": expected["rows"],
            "tensor_shapes": {"ehr": [48, 79], "image": [3, 224, 224], "latent": [4, 28, 28]},
        }

    report["combined_item_digest_sha256"] = combined.hexdigest()
    report["rows"] = sum(value["rows"] for value in EXPECTED.values())
    report["status"] = "exact"
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_path.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
