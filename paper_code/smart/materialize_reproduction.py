"""Materialize and verify SMART's released mortality pickle boundary.

The authors did not release mortality_normalized.pkl. The export reconstructed
the notebook boundary and exhaustively checked it against all three released
seed-specific loader/collate splits. This script repeats those hash checks and
writes the tuple only beneath ignored paper_results for direct author-code use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import polars as pl

from check_batch import verify_author_loader_and_collate
from cohort_source import (
    SEEDS,
    array_hash,
    load_author_notebook_boundary,
    load_build_report,
    load_cohort,
    ordered_text_hash,
    replica_task_id,
    sha256,
)


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
EXPECTED_BOUNDARY = {
    "normalized_sha256": "1435196765e3b379e9fef50627711ea20e166b6170d8b4e6410a47f30a7865f2",
    "mask_sha256": "576e04f51449aa65b49ba828db2c7fbef93fb5091f10481a920253143852324e",
    "labels_sha256": "b5c777523f9383ac9236b768cd63dd77f83e1b23e3f34d7780bbf10863be34d7",
    "names_sha256": "3871add3d48e700c69e5423e9878d9db101686995f9c916f6b27838066f1f56f",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_hashes(task_root: Path, cache_root: Path) -> dict:
    code_files = (
        "main_pretrain.py",
        "main_finetune.py",
        "models/smart.py",
        "data/mimiciii.py",
        "data/dataloader.py",
        "utils/metrics.py",
        "utils/utils.py",
        "cohort_source.py",
    )
    result = {
        "code": {name: file_sha256(PAPER_ROOT / name) for name in code_files},
        "source_build": file_sha256(cache_root / "source_build.json"),
        "source_cohort": file_sha256(cache_root / "source_cohort.parquet"),
        "replicas": {},
    }
    for replica in (1, 2, 3):
        directory = task_root / replica_task_id(replica)
        result["replicas"][str(replica)] = {
            "manifest": file_sha256(directory / "manifest.json"),
            "task_index": file_sha256(directory / "audit/task_index.parquet"),
            "cohort_audit": file_sha256(directory / "audit/cohort_reconstruction.json"),
            "batch_audit": file_sha256(directory / "audit/full_batch_recreation.json"),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=PAPER_ROOT / ".export_assets/source"
    )
    parser.add_argument(
        "--task-root", type=Path, default=WORKSPACE_ROOT / "paper_tasks/smart"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "paper_results/smart/runtime/data/MIMIC-III/mortality_normalized.pkl"
        ),
    )
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    task_root = args.task_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    build = load_build_report(cache_root)
    cohort = load_cohort(cache_root)
    boundary = load_author_notebook_boundary(cache_root)
    actual_boundary = {
        "normalized_sha256": array_hash(boundary["normalized"]),
        "mask_sha256": array_hash(boundary["mask"]),
        "labels_sha256": array_hash(boundary["labels"]),
        "names_sha256": ordered_text_hash(boundary["names"]),
    }
    if actual_boundary != EXPECTED_BOUNDARY:
        raise AssertionError(
            f"SMART author boundary changed: {actual_boundary} != {EXPECTED_BOUNDARY}"
        )

    replica_checks = {}
    sample_ids = cohort["sample_id"].to_numpy()
    for replica, seed in enumerate(SEEDS, start=1):
        actual_hashes, split_indices = verify_author_loader_and_collate(
            boundary, cohort, seed
        )
        directory = task_root / replica_task_id(replica)
        audit = json.loads((directory / "audit/full_batch_recreation.json").read_text())
        if actual_hashes != audit["split_loader_collate_hashes"]:
            raise AssertionError(f"replica {replica} released loader hashes changed")
        index = pl.read_parquet(directory / "audit/task_index.parquet")
        for split, indices in split_indices.items():
            expected_ids = set(sample_ids[indices].tolist())
            actual_ids = set(
                index.filter(pl.col("split") == split)["sample_id"].to_list()
            )
            if actual_ids != expected_ids:
                raise AssertionError(f"replica {replica} {split} membership changed")
        replica_checks[str(replica)] = {
            "seed": seed,
            "split_loader_collate_hashes": actual_hashes,
            "task_index_membership": "exact",
        }

    payload = (
        boundary["normalized"],
        boundary["labels"],
        boundary["mask"],
        boundary["names"],
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)
    with output.open("rb") as handle:
        reloaded = pickle.load(handle)
    reloaded_hashes = {
        "normalized_sha256": array_hash(reloaded[0]),
        "labels_sha256": array_hash(reloaded[1]),
        "mask_sha256": array_hash(reloaded[2]),
        "names_sha256": ordered_text_hash(reloaded[3]),
    }
    if reloaded_hashes != actual_boundary:
        raise AssertionError("serialized SMART runtime tuple changed values or order")

    record = {
        "status": "exact_verified_runtime_boundary",
        "source_status": build["status"],
        "fidelity_limitation": build["fidelity_limitation"],
        "output": str(output),
        "pickle_sha256": sha256(output),
        "pickle_bytes": output.stat().st_size,
        "shapes": {
            "normalized": list(boundary["normalized"].shape),
            "labels": list(boundary["labels"].shape),
            "mask": list(boundary["mask"].shape),
            "names": len(boundary["names"]),
        },
        "boundary_hashes": actual_boundary,
        "replica_checks": replica_checks,
        "source_hashes": source_hashes(task_root, cache_root),
    }
    record_path = output.parents[2] / "materialization.json"
    write_json(record_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
