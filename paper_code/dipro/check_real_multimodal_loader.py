#!/usr/bin/env python3
"""Fail-closed smoke check for the unchanged DiPro multimodal test loader."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision


PAPER_DIR = Path(__file__).resolve().parent
ASSETS = PAPER_DIR / ".export_assets"
PREPARED = ASSETS / "multimodal_reproduction"
SOURCE = ASSETS / "source_cache"
DEFAULT_OUTPUT = PREPARED / "REAL_LOADER_CHECK.json"
EXPECTED_AUTHOR_DATASET_SHA256 = "3a11c545da29818471e6a99907f877a91679fb8e3294d77ff26139bf48b8888d"
EXPECTED_PREPARATION_SHA256 = "53d702eedc8c5030805c053d12dbe10e39f0c210eeafebf4fcc6e5d7aeec36d0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_cxr_list(value: str) -> List[Tuple[str, float]]:
    parsed = ast.literal_eval(value)
    return [(str(dicom), float(time)) for dicom, time in parsed]


def expected_image_path(root: Path, metadata: pd.DataFrame, dicom: str) -> Path:
    row = metadata.loc[dicom]
    require(isinstance(row, pd.Series), "MIMIC-CXR metadata has duplicate dicom_id: {}".format(dicom))
    subject_id = int(row["subject_id"])
    study_id = int(row["study_id"])
    return root / "files" / ("p" + str(subject_id)[:2]) / ("p" + str(subject_id)) / ("s" + str(study_id)) / (dicom + ".jpg")


def check_batch(
    dataset: Any,
    collate: Any,
    indices: Sequence[int],
    labels: pd.DataFrame,
    episode_lengths: Dict[str, int],
) -> Dict[str, Any]:
    items = [dataset[index] for index in indices]
    batch = collate(items)
    require(len(batch) == 10, "author collator returned the wrong field count")
    (
        ehr,
        ehr_mask,
        regions,
        region_mask,
        demographics,
        targets,
        y_aux,
        y_aux_mask,
        ehr_times,
        interval_times,
    ) = batch

    cxr_lengths = [len(parse_cxr_list(labels.loc[index, "cxr_list"])) for index in indices]
    ehr_lengths = [episode_lengths[str(int(labels.loc[index, "stay_id"]))] for index in indices]
    expected_targets = torch.tensor(labels.loc[list(indices), "mortality_inhospital"].to_numpy()).float()
    require(torch.equal(targets, expected_targets), "batch labels or order changed")
    require(ehr.shape == (len(indices), max(ehr_lengths), 38), "unexpected collated EHR shape")
    require(ehr_mask.shape == ehr.shape[:2], "unexpected EHR mask shape")
    require(ehr_mask.sum(dim=1).tolist() == ehr_lengths, "EHR masks differ from source episode lengths")
    require(regions.shape == (len(indices), max(cxr_lengths), 10, 3, 224, 224), "unexpected image tensor shape")
    require(region_mask.sum(dim=1).tolist() == cxr_lengths, "CXR sequence masks differ from cxr_list")
    require(y_aux.shape == (len(indices), max(cxr_lengths) - 1, 10, 7), "unexpected auxiliary tensor shape")
    require(y_aux_mask.sum(dim=1).tolist() == [length - 1 for length in cxr_lengths], "auxiliary masks differ")
    require(ehr_times.shape == (len(indices), 49), "released fixed EHR-time length changed")
    require(interval_times.shape == (len(indices), max(cxr_lengths) - 1, 2), "unexpected interval-time shape")
    for name, tensor in (("ehr", ehr), ("regions", regions), ("demographics", demographics), ("interval_times", interval_times)):
        require(bool(torch.isfinite(tensor).all()), "non-finite values in {}".format(name))
    require(float(regions.min()) >= 0.0 and float(regions.max()) <= 1.0, "ToTensor image range changed")

    return {
        "indices": list(indices),
        "stay_ids": [int(labels.loc[index, "stay_id"]) for index in indices],
        "labels": [int(value) for value in targets.tolist()],
        "ehr_lengths": ehr_lengths,
        "cxr_lengths": cxr_lengths,
        "ehr_shape": list(ehr.shape),
        "region_shape": list(regions.shape),
        "y_aux_shape": list(y_aux.shape),
        "region_min": float(regions.min()),
        "region_max": float(regions.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-cxr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    mimic_root = args.mimic_cxr_root.resolve()
    output = args.output.resolve()
    require(output.parent == PREPARED.resolve(), "loader-check output must remain in prepared artifact root")
    require(not output.exists(), "refusing to overwrite existing loader-check result")
    metadata_candidates = (
        mimic_root / "mimic-cxr-2.0.0-metadata.csv",
        mimic_root / "mimic-cxr-2.0.0-metadata.csv.gz",
    )
    metadata_path = next((path for path in metadata_candidates if path.is_file()), None)
    require((mimic_root / "files").is_dir(), "missing MIMIC-CXR-JPG 2.0.0 files directory")
    require(metadata_path is not None, "missing MIMIC-CXR-JPG 2.0.0 metadata")
    require(
        sha256(PAPER_DIR / "dataset_module" / "dipro_dataset.py") == EXPECTED_AUTHOR_DATASET_SHA256,
        "author dataset source changed",
    )
    require(
        sha256(PREPARED / "PREPARATION_COMPLETE.json") == EXPECTED_PREPARATION_SHA256,
        "prepared multimodal manifest changed",
    )

    labels_path = PREPARED / "split" / "mortality_los" / "test.csv"
    labels = pd.read_csv(labels_path)
    require(len(labels) == 546, "unexpected released test length")
    require(int(labels["mortality_inhospital"].sum()) == 110, "unexpected released test positives")
    cxr_lists = [parse_cxr_list(value) for value in labels["cxr_list"]]
    all_dicoms = [dicom for cxrs in cxr_lists for dicom, _ in cxrs]
    require(len(all_dicoms) == 1247 and len(set(all_dicoms)) == 1247, "unexpected exact test CXR set")

    metadata = pd.read_csv(metadata_path, usecols=["dicom_id", "subject_id", "study_id"])
    require(not metadata["dicom_id"].duplicated().any(), "duplicate dicom IDs in MIMIC-CXR metadata")
    metadata = metadata.set_index("dicom_id")
    missing_images: List[str] = []
    image_bytes = 0
    for dicom in all_dicoms:
        require(dicom in metadata.index, "test dicom absent from MIMIC-CXR metadata: {}".format(dicom))
        image_path = expected_image_path(mimic_root, metadata, dicom)
        if not image_path.is_file():
            missing_images.append(dicom)
        else:
            require(image_path.stat().st_size > 0, "empty MIMIC-CXR JPEG: {}".format(dicom))
            image_bytes += image_path.stat().st_size
    require(not missing_images, "missing {} exact test JPEGs".format(len(missing_images)))

    bbox = pd.read_csv(PREPARED / "cxr_bbox.csv", usecols=["cxr_id"])
    bbox_ids = set(bbox["cxr_id"].astype(str))
    require(set(all_dicoms).issubset(bbox_ids), "test CXR absent from author bbox output")

    episode_lengths = json.loads((SOURCE / "episode_lengths.json").read_text())
    test_ehr_lengths = [episode_lengths[str(int(stay))] for stay in labels["stay_id"]]
    require(min(test_ehr_lengths) == 46 and max(test_ehr_lengths) == 49, "unexpected test EHR length range")
    all_short_batches = [
        start
        for start in range(0, len(labels), 4)
        if max(test_ehr_lengths[start : start + 4]) < 49
    ]
    require(not all_short_batches, "released size-four ordering contains a model-incompatible all-short batch")

    sys.path.insert(0, str(PAPER_DIR))
    from dataset_module.dipro_dataset import CollateFunc, DiproDataset

    dataset = DiproDataset(
        partition="test",
        task_name="mortality",
        label_path=str(PREPARED / "split"),
        demographic_path=str(SOURCE / "demographic_processed.csv"),
        ehr_time_series_path=str(SOURCE / "ehr_preprocessed"),
        bbox_csv=str(PREPARED / "cxr_bbox.csv"),
        mimic_cxr_path=str(mimic_root / "files"),
        cxr_meta_path=str(metadata_path),
    )
    require(len(dataset) == 546, "unchanged author dataset returned the wrong test length")
    collate = CollateFunc(dim=0)

    # First exact released batch through DataLoader, without changing its row
    # composition.  Zero workers is the sole resource-control substitution.
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate, drop_last=False)
    first_from_loader = next(iter(loader))
    first_direct = collate([dataset[index] for index in range(4)])
    require(len(first_from_loader) == len(first_direct), "DataLoader/collator field count mismatch")
    for left, right in zip(first_from_loader, first_direct):
        require(torch.equal(left, right), "DataLoader altered first-batch author tensors")

    length_five_index = next(index for index, cxrs in enumerate(cxr_lists) if len(cxrs) == 5)
    short_index = next(index for index, length in enumerate(test_ehr_lengths) if length < 49)
    starts = sorted({0, (length_five_index // 4) * 4, (short_index // 4) * 4})
    batch_checks = []
    for start in starts:
        indices = list(range(start, min(start + 4, len(dataset))))
        batch_checks.append(check_batch(dataset, collate, indices, labels, episode_lengths))

    result = {
        "status": "pass",
        "test_rows": len(labels),
        "test_positive": int(labels["mortality_inhospital"].sum()),
        "test_cxr_count": len(all_dicoms),
        "test_jpeg_bytes": image_bytes,
        "test_ehr_length_counts": {
            str(length): int(test_ehr_lengths.count(length)) for length in sorted(set(test_ehr_lengths))
        },
        "all_short_size_four_batches": all_short_batches,
        "num_workers": 0,
        "batch_size": 4,
        "shuffle": False,
        "drop_last": False,
        "metadata_storage": "gzip" if metadata_path.suffix == ".gz" else "uncompressed",
        "batch_checks": batch_checks,
        "source_sha256": {
            "author_dataset": sha256(PAPER_DIR / "dataset_module" / "dipro_dataset.py"),
            "prepared_manifest": sha256(PREPARED / "PREPARATION_COMPLETE.json"),
            "test_labels": sha256(labels_path),
            "bbox": sha256(PREPARED / "cxr_bbox.csv"),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "output": str(output), "sha256": sha256(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
