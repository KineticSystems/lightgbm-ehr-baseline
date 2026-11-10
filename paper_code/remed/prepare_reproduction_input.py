#!/usr/bin/env python3
"""Create the exact paper-local REMed input layout without copying source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXPECTED = {
    "source_build": "1a0d9f59d9715d484ec2945636e0171e7b3e7a6dd69ec0c9e5d98fddfd31c2c1",
    "author_task_manifest": "4112503d57535334dec6bb57fe9ffae5d1fe5a0864244aeaee84ff544438c7e0",
    "cohort_report": "966df1d6f75374d9ec809c27a5512bef47748537f3f540398076009a291f72b6",
    "author_batch_report": "3baae1b7f31f591983b5b0e90f5d943367f892621f4f51f1ac45b1cdd2f9e39d",
}
EXPECTED_BYTES = {
    "mimiciv.h5": 13_029_597_233,
    "mimiciv_cohort.csv": 2_004_901_440,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_workspace(path: Path, workspace: Path) -> str:
    return str(path.resolve().relative_to(workspace.resolve()))


def main() -> None:
    paper_root = Path(__file__).resolve().parent
    workspace = paper_root.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=paper_root / ".export_work" / "source_48h",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=paper_root / ".export_assets" / "reproduction" / "input",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            paper_root
            / ".export_assets"
            / "reproduction"
            / "audit"
            / "input_shim.json"
        ),
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    if output_root.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite an existing input shim or report")

    source_hdf = source_root / "mimiciv.h5"
    source_cohort = source_root / "mimiciv_cohort.csv"
    source_build = source_root / "source_build.json"
    archived_task = workspace / "paper_tasks" / "remed" / "los_7d_at_48h.hashed_vocab_20260901"
    current_task = workspace / "paper_tasks" / "remed" / "los_7d_at_48h"
    author_manifest = archived_task / "manifest.json"
    cohort_report = archived_task / "audit" / "cohort_reconstruction.json"
    author_batch_report = archived_task / "audit" / "full_batch_recreation.json"
    current_manifest = current_task / "manifest.json"
    current_batch_report = current_task / "audit" / "full_batch_recreation.json"

    for path in (
        source_hdf,
        source_cohort,
        source_build,
        author_manifest,
        cohort_report,
        author_batch_report,
        current_manifest,
        current_batch_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (source_hdf, source_cohort):
        if path.stat().st_size != EXPECTED_BYTES[path.name]:
            raise AssertionError((path, path.stat().st_size, EXPECTED_BYTES[path.name]))

    observed_hashes = {
        "source_build": sha256(source_build),
        "author_task_manifest": sha256(author_manifest),
        "cohort_report": sha256(cohort_report),
        "author_batch_report": sha256(author_batch_report),
    }
    if observed_hashes != EXPECTED:
        raise AssertionError((observed_hashes, EXPECTED))
    if json.loads(source_build.read_text()).get("status") != "complete":
        raise AssertionError("accepted source build is not complete")
    if json.loads(cohort_report.read_text()).get("status") != "exact_source_with_small_paper_token_length_mismatch":
        raise AssertionError("accepted cohort report status changed")
    if json.loads(author_batch_report.read_text()).get("status") != "exact_at_released_ehrdataset_input":
        raise AssertionError("archived author-loader batch report status changed")

    input_48h = output_root / "48h"
    input_48h.mkdir(parents=True)
    linked = {}
    for source in (source_hdf, source_cohort, source_build):
        destination = input_48h / source.name
        target = os.path.relpath(source, start=input_48h)
        destination.symlink_to(target)
        if destination.resolve() != source:
            raise AssertionError((destination, destination.resolve(), source))
        linked[source.name] = {
            "bytes": source.stat().st_size,
            "source": relative_to_workspace(source, workspace),
            "target": target,
        }

    current_batch = json.loads(current_batch_report.read_text())
    report = {
        "accepted_author_boundary": {
            "archived_task": relative_to_workspace(archived_task, workspace),
            "hashes": observed_hashes,
            "status": "exact_at_released_ehrdataset_input",
        },
        "current_common_task": {
            "batch_report_sha256": sha256(current_batch_report),
            "batch_status": current_batch.get("status"),
            "manifest_sha256": sha256(current_manifest),
            "path": relative_to_workspace(current_task, workspace),
        },
        "input_layout": relative_to_workspace(output_root, workspace),
        "linked": linked,
        "status": "exact_paper_local_input_shim",
        "test_accesses": 0,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
