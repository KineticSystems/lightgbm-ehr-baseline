#!/usr/bin/env python3
"""Prepare the released DiPro mortality inputs without rebuilding its split.

This driver is additive reproduction infrastructure.  It deliberately calls the
authors' released preprocessing functions for the operations that affect model
inputs:

* ``correct_scenegraph``
* ``get_bbox_for_cxrs``
* ``get_disease_progression_labels``
* ``apply_order_map``
* ``get_organ_progression_labels_mort_los``

Only scene graphs named by the already reconstructed released ``cxr_stay.csv``
are extracted.  That subset is sufficient because mortality progression labels
are looked up only for consecutive CXR pairs in the frozen mortality split, and
every such CXR is in ``cxr_stay.csv``.  The script verifies this precondition,
the immutable source hashes, the exact released split, every extracted member,
and the semantic identity of all base split columns after enrichment.
"""

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import zipfile

import pandas as pd


PAPER_DIR = Path(__file__).resolve().parent
ASSET_DIR = PAPER_DIR / ".export_assets"
SOURCE_CACHE = ASSET_DIR / "source_cache"
EXTERNAL = ASSET_DIR / "external_artifacts" / "physionet_chest_imagenome_1.0.0"
ARCHIVE = EXTERNAL / "silver_dataset" / "scene_graph.zip"
GOLD = EXTERNAL / "gold_dataset" / "gold_object_comparison_with_coordinates.txt"
ORDER_MAP = (
    PAPER_DIR
    / "mimic_data_extract"
    / "mimic3benchmark"
    / "resources"
    / "all_comparison_order_map.csv"
)
CXR_STAY = SOURCE_CACHE / "cxr_stay.csv"
BASE_SPLIT_DIR = SOURCE_CACHE / "split" / "mortality_los"
DEFAULT_OUTPUT = ASSET_DIR / "multimodal_reproduction"

EXPECTED_SHA256 = {
    ARCHIVE: "1baccf0523944755c0e041b5ade3e6f7f41bcc1faca1cc1cfce457e6c29a11c3",
    GOLD: "7efc6d779705aee0770f3c474baa3fc7cfd486333ad5c39717ad1ff9d95772b0",
    ORDER_MAP: "ef52dfb01bff1859877a1e0235ef73ab62210eb2684341fe9edfe642bfe1ed6f",
    CXR_STAY: "c7e93b80c0cff53e4b6eef420a45fd18cd8c742588c6380fb30539de497211ce",
    BASE_SPLIT_DIR / "train.csv": "163a6d802d41227b2cfe6ee33a6b64886c9461931e4230b1b752749bf2a3f859",
    BASE_SPLIT_DIR / "val.csv": "8d23187b34f2f29bf96f6309a9d40b8268f1229972fad6e54bc284642a322d7e",
    BASE_SPLIT_DIR / "test.csv": "c0be5891d59e307a8f50ce43bd9681b0c58d1fb85a4635a6df645132f0b700f2",
}
EXPECTED_SPLITS = {
    "train": {"rows": 1889, "positive": 316, "cxr_slots": 4358},
    "val": {"rows": 285, "positive": 39, "cxr_slots": 666},
    "test": {"rows": 546, "positive": 110, "cxr_slots": 1247},
}
EXPECTED_CXR_STAY_ROWS = 13684
EXPECTED_EXACT_SPLIT_CXRS = 6271
CORRECTION_DICOMS = (
    "3e953c67-28b78460-4f783437-379841d3-a79ec530",
    "8bd72ea4-f6e5470f-3ddbc81e-a1da43d8-b6e39876",
    "38bbebbd-b67908dc-387567bf-e3f965b7-77a18536",
    "038c0310-98178aef-3414521f-1d34c309-8b94b78e",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_source_hashes() -> Dict[str, str]:
    observed: Dict[str, str] = {}
    for path, expected in EXPECTED_SHA256.items():
        require(path.is_file(), "missing required source: {}".format(path))
        actual = sha256(path)
        require(actual == expected, "SHA-256 mismatch for {}: {} != {}".format(path, actual, expected))
        observed[str(path.relative_to(PAPER_DIR))] = actual
    return observed


def parse_cxr_list(value: Any) -> List[Tuple[str, float]]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    require(isinstance(parsed, list), "cxr_list is not a list")
    result: List[Tuple[str, float]] = []
    for entry in parsed:
        require(isinstance(entry, (list, tuple)) and len(entry) == 2, "invalid cxr_list entry")
        result.append((str(entry[0]), float(entry[1])))
    return result


def validate_frozen_inputs() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any]]:
    cxr_stay = pd.read_csv(CXR_STAY)
    require(len(cxr_stay) == EXPECTED_CXR_STAY_ROWS, "unexpected cxr_stay row count")
    require(cxr_stay["dicom_id"].nunique() == EXPECTED_CXR_STAY_ROWS, "cxr_stay dicom IDs are not unique")
    selected = set(cxr_stay["dicom_id"].astype(str))
    require(set(CORRECTION_DICOMS).issubset(selected), "author correction case absent from cxr_stay")

    frames: Dict[str, pd.DataFrame] = {}
    split_subjects: Dict[str, set] = {}
    all_exact_dicoms: List[str] = []
    split_summary: Dict[str, Any] = {}
    for split, expected in EXPECTED_SPLITS.items():
        frame = pd.read_csv(BASE_SPLIT_DIR / (split + ".csv"))
        require(len(frame) == expected["rows"], "unexpected {} row count".format(split))
        require(int(frame["mortality_inhospital"].sum()) == expected["positive"], "unexpected {} positives".format(split))
        parsed_lists = [parse_cxr_list(value) for value in frame["cxr_list"]]
        dicoms = [dicom for cxrs in parsed_lists for dicom, _ in cxrs]
        require(len(dicoms) == expected["cxr_slots"], "unexpected {} CXR-slot count".format(split))
        require(set(dicoms).issubset(selected), "{} split contains a CXR absent from cxr_stay".format(split))
        require(frame["stay_id"].nunique() == len(frame), "{} stay IDs are not unique".format(split))
        frames[split] = frame
        split_subjects[split] = set(frame["subject_id"].astype(int))
        all_exact_dicoms.extend(dicoms)
        split_summary[split] = {
            "rows": len(frame),
            "positive": int(frame["mortality_inhospital"].sum()),
            "cxr_slots": len(dicoms),
            "cxr_length_counts": {
                str(length): int(sum(len(cxrs) == length for cxrs in parsed_lists))
                for length in sorted(set(map(len, parsed_lists)))
            },
        }

    require(not (split_subjects["train"] & split_subjects["val"]), "train/val subject overlap")
    require(not (split_subjects["train"] & split_subjects["test"]), "train/test subject overlap")
    require(not (split_subjects["val"] & split_subjects["test"]), "val/test subject overlap")
    require(len(all_exact_dicoms) == EXPECTED_EXACT_SPLIT_CXRS, "unexpected exact-split CXR-slot total")
    require(len(set(all_exact_dicoms)) == EXPECTED_EXACT_SPLIT_CXRS, "exact split repeats a CXR")
    return cxr_stay, frames, split_summary


def write_inventory(scene_dir: Path, dicoms: Sequence[str], output: Path) -> Tuple[str, Dict[str, str]]:
    hashes: Dict[str, str] = {}
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["dicom_id", "bytes", "sha256"])
        for dicom in sorted(dicoms):
            path = scene_dir / (dicom + "_SceneGraph.json")
            require(path.is_file(), "missing extracted scene graph: {}".format(dicom))
            digest = sha256(path)
            hashes[dicom] = digest
            writer.writerow([dicom, path.stat().st_size, digest])
    return sha256(output), hashes


def extract_selected_graphs(archive: Path, dicoms: Sequence[str], scene_dir: Path) -> Dict[str, Any]:
    scene_dir.mkdir(parents=True)
    total_bytes = 0
    total_compressed_bytes = 0
    with zipfile.ZipFile(str(archive), "r") as bundle:
        infos = {info.filename: info for info in bundle.infolist() if not info.is_dir()}
        require(len(infos) == 243310, "unexpected number of files in scene_graph.zip")
        for dicom in sorted(dicoms):
            member = "scene_graph/{}_SceneGraph.json".format(dicom)
            require(member in infos, "selected scene graph absent from archive: {}".format(dicom))
            info = infos[member]
            target = scene_dir / (dicom + "_SceneGraph.json")
            with bundle.open(info, "r") as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            require(target.stat().st_size == info.file_size, "extracted size mismatch: {}".format(dicom))
            total_bytes += info.file_size
            total_compressed_bytes += info.compress_size
    return {
        "selected_graphs": len(dicoms),
        "uncompressed_bytes": total_bytes,
        "compressed_bytes": total_compressed_bytes,
    }


def import_author_functions():
    mimic_extract = PAPER_DIR / "mimic_data_extract"
    sys.path.insert(0, str(PAPER_DIR))
    sys.path.insert(0, str(mimic_extract))
    # Each released module parses required command-line arguments at import time.
    # Supply real, inert paths during import and restore this driver's argv before
    # doing any work.  No author module or function is changed.
    original_argv = sys.argv[:]
    sys.argv = [
        original_argv[0],
        "--all_stay_csv_path", str(SOURCE_CACHE / "all_stays.csv"),
        "--mimic_cxr_meta", str(SOURCE_CACHE / "unused_metadata.csv"),
        "--chexpert_label", str(SOURCE_CACHE / "unused_chexpert.csv"),
        "--ImaGenome", str(DEFAULT_OUTPUT / "scene_graph"),
        "--output_dir", str(DEFAULT_OUTPUT),
        "--scene_graph", str(DEFAULT_OUTPUT / "scene_graph"),
        "--golden_dir", str(GOLD),
        "--save_path", str(DEFAULT_OUTPUT),
        "--cxr_stay", str(CXR_STAY),
        "--progression_task_label_dir", str(DEFAULT_OUTPUT / "disease_progressions"),
        "--split_save_dir", str(DEFAULT_OUTPUT / "split"),
        "--all_comparison_path", str(DEFAULT_OUTPUT / "all_comparison.csv"),
    ]
    try:
        from mimic3benchmark.constants import PROGRESSION_DICT, PROGRESSION_DISEASE, REGIONS, SELECTED_REGIONS
        from mimic3benchmark.scripts.get_cxr import correct_scenegraph, get_bbox_for_cxrs
        from mimic3benchmark.scripts.process_ImaGenome import get_disease_progression_labels, apply_order_map
        from mimic3benchmark.scripts.split_train_and_test import get_organ_progression_labels_mort_los
    finally:
        sys.argv = original_argv

    return {
        "PROGRESSION_DICT": PROGRESSION_DICT,
        "PROGRESSION_DISEASE": PROGRESSION_DISEASE,
        "REGIONS": REGIONS,
        "SELECTED_REGIONS": SELECTED_REGIONS,
        "correct_scenegraph": correct_scenegraph,
        "get_bbox_for_cxrs": get_bbox_for_cxrs,
        "get_disease_progression_labels": get_disease_progression_labels,
        "apply_order_map": apply_order_map,
        "get_organ_progression_labels_mort_los": get_organ_progression_labels_mort_los,
    }


def verify_order_map_coverage(all_comparison: Path) -> Dict[str, int]:
    keys = ["subject_id", "study_id", "cxr1_dicom_id", "cxr2_dicom_id"]
    comparison = pd.read_csv(all_comparison)
    order = pd.read_csv(ORDER_MAP)
    require(not order.duplicated(keys).any(), "released order map has duplicate keys")
    merged = comparison.merge(order, on=keys, how="left")
    missing = int(merged["order_index"].isna().sum())
    require(missing == 0, "selected comparison rows missing from released order map")
    require(merged["order_index"].nunique() == len(merged), "selected comparison order indices are not unique")
    return {"rows": len(comparison), "missing_order_indices": missing}


def verify_bbox(path: Path, cxr_stay: pd.DataFrame, regions: Sequence[str]) -> Dict[str, Any]:
    bbox = pd.read_csv(path)
    expected_ids = list(cxr_stay["dicom_id"].astype(str).drop_duplicates())
    require(list(bbox.columns) == ["cxr_id"] + list(regions), "unexpected bbox columns")
    require(list(bbox["cxr_id"].astype(str)) == expected_ids, "bbox row order or IDs differ from cxr_stay")
    missing_counts: Dict[str, int] = {}
    for region in regions:
        parsed = [ast.literal_eval(value) for value in bbox[region]]
        require(all(isinstance(value, list) and len(value) == 4 for value in parsed), "invalid bbox value")
        missing_counts[region] = int(sum(value == [0, 0, 0, 0] for value in parsed))
    return {"rows": len(bbox), "missing_region_counts": missing_counts}


def verify_enriched_splits(
    output_split_dir: Path,
    base_frames: Mapping[str, pd.DataFrame],
    selected_regions: Sequence[str],
    progression_diseases: Sequence[str],
    progression_values: Iterable[str],
) -> Dict[str, Any]:
    allowed = set(progression_values)
    result: Dict[str, Any] = {}
    for split, base in base_frames.items():
        enriched_path = output_split_dir / (split + ".csv")
        enriched = pd.read_csv(enriched_path)
        require(list(enriched.columns[: len(base.columns)]) == list(base.columns), "base columns moved in {}".format(split))
        require(len(enriched) == len(base), "row count changed in {}".format(split))
        for column in base.columns:
            if column == "cxr_list":
                left = [parse_cxr_list(value) for value in base[column]]
                right = [parse_cxr_list(value) for value in enriched[column]]
                require(left == right, "CXR sequence changed in {}".format(split))
            else:
                pd.testing.assert_series_equal(
                    base[column].reset_index(drop=True),
                    enriched[column].reset_index(drop=True),
                    check_dtype=False,
                    check_names=False,
                    check_exact=True,
                )

        cxr_lists = [parse_cxr_list(value) for value in enriched["cxr_list"]]
        value_counts = {name: 0 for name in sorted(allowed)}
        for row_index, cxrs in enumerate(cxr_lists):
            expected_pairs = len(cxrs) - 1
            for region in selected_regions:
                column = region + "_progression"
                require(column in enriched, "missing progression column: {}".format(column))
                progression = ast.literal_eval(enriched.loc[row_index, column])
                require(len(progression) == expected_pairs, "wrong progression-pair length in {}".format(split))
                for disease_map in progression:
                    require(set(disease_map) == set(progression_diseases), "wrong disease keys in {}".format(split))
                    require(set(disease_map.values()).issubset(allowed), "unknown progression value in {}".format(split))
                    for value in disease_map.values():
                        value_counts[value] += 1
        result[split] = {
            "rows": len(enriched),
            "sha256": sha256(enriched_path),
            "progression_value_counts": value_counts,
        }
    return result


def author_source_hashes() -> Dict[str, str]:
    paths = (
        PAPER_DIR / "mimic_data_extract" / "mimic3benchmark" / "scripts" / "get_cxr.py",
        PAPER_DIR / "mimic_data_extract" / "mimic3benchmark" / "scripts" / "process_ImaGenome.py",
        PAPER_DIR / "mimic_data_extract" / "mimic3benchmark" / "scripts" / "split_train_and_test.py",
        PAPER_DIR / "mimic_data_extract" / "mimic3benchmark" / "constants.py",
    )
    return {str(path.relative_to(PAPER_DIR)): sha256(path) for path in paths}


def prepare(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    require(output_dir.parent == ASSET_DIR.resolve(), "output must be an immediate child of .export_assets")
    require(not output_dir.exists(), "refusing to overwrite existing output: {}".format(output_dir))

    source_hashes = validate_source_hashes()
    cxr_stay, base_frames, split_summary = validate_frozen_inputs()
    selected_dicoms = sorted(set(cxr_stay["dicom_id"].astype(str)))
    author = import_author_functions()

    temporary = Path(tempfile.mkdtemp(prefix="." + output_dir.name + ".", dir=str(ASSET_DIR)))
    try:
        scene_dir = temporary / "scene_graph"
        raw_corrections = temporary / "raw_correction_graphs"
        raw_corrections.mkdir(parents=True)
        extraction = extract_selected_graphs(ARCHIVE, selected_dicoms, scene_dir)
        raw_inventory_sha, raw_hashes = write_inventory(
            scene_dir, selected_dicoms, temporary / "scene_graph_raw_inventory.tsv"
        )
        for dicom in CORRECTION_DICOMS:
            shutil.copy2(
                scene_dir / (dicom + "_SceneGraph.json"),
                raw_corrections / (dicom + "_SceneGraph.json"),
            )

        author["correct_scenegraph"](str(scene_dir))
        corrected_inventory_sha, corrected_hashes = write_inventory(
            scene_dir, selected_dicoms, temporary / "scene_graph_corrected_inventory.tsv"
        )
        changed = sorted(dicom for dicom in selected_dicoms if raw_hashes[dicom] != corrected_hashes[dicom])
        require(changed == sorted(CORRECTION_DICOMS), "author correction changed an unexpected file set")

        author["get_bbox_for_cxrs"](str(CXR_STAY), str(scene_dir), str(temporary))
        bbox_summary = verify_bbox(temporary / "cxr_bbox.csv", cxr_stay, author["REGIONS"])

        author["get_disease_progression_labels"](str(scene_dir), str(temporary))
        comparison_path = temporary / "all_comparison.csv"
        comparison_summary = verify_order_map_coverage(comparison_path)
        author["apply_order_map"](str(comparison_path), str(ORDER_MAP))

        split_root = temporary / "split"
        output_split_dir = split_root / "mortality_los"
        output_split_dir.mkdir(parents=True)
        for split in EXPECTED_SPLITS:
            shutil.copy2(BASE_SPLIT_DIR / (split + ".csv"), output_split_dir / (split + ".csv"))
        author["get_organ_progression_labels_mort_los"](
            str(GOLD), str(comparison_path), str(split_root), str(split_root)
        )
        enriched_summary = verify_enriched_splits(
            output_split_dir,
            base_frames,
            author["SELECTED_REGIONS"],
            author["PROGRESSION_DISEASE"],
            author["PROGRESSION_DICT"].keys(),
        )

        outputs = {
            "cxr_bbox.csv": sha256(temporary / "cxr_bbox.csv"),
            "all_comparison.csv": sha256(comparison_path),
            "scene_graph_raw_inventory.tsv": raw_inventory_sha,
            "scene_graph_corrected_inventory.tsv": corrected_inventory_sha,
        }
        manifest = {
            "format_version": 1,
            "purpose": "DiPro released mortality-checkpoint input preparation",
            "selection_rule": "all unique dicom_id values in the frozen released cxr_stay.csv",
            "selection_equivalence": (
                "Every mortality cxr_list CXR is asserted to be selected; each comparison is stored in "
                "the current CXR's graph; gold overrides are read independently. Nonselected graphs "
                "therefore cannot affect mortality split inputs."
            ),
            "source_sha256": source_hashes,
            "author_source_sha256": author_source_hashes(),
            "split_summary": split_summary,
            "extraction": extraction,
            "author_correction_dicoms": list(CORRECTION_DICOMS),
            "changed_by_author_correction": changed,
            "raw_correction_sha256": {dicom: raw_hashes[dicom] for dicom in CORRECTION_DICOMS},
            "corrected_correction_sha256": {dicom: corrected_hashes[dicom] for dicom in CORRECTION_DICOMS},
            "bbox": bbox_summary,
            "comparison": comparison_summary,
            "enriched_splits": enriched_summary,
            "output_sha256": outputs,
        }
        manifest_path = temporary / "PREPARATION_COMPLETE.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest["manifest_sha256_before_self_entry"] = sha256(manifest_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        temporary.rename(output_dir)
        print(json.dumps({
            "status": "complete",
            "output": str(output_dir),
            "manifest_sha256": sha256(output_dir / "PREPARATION_COMPLETE.json"),
            "selected_graphs": len(selected_dicoms),
            "comparison_rows": comparison_summary["rows"],
        }, sort_keys=True))
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.output_dir)


if __name__ == "__main__":
    main()
