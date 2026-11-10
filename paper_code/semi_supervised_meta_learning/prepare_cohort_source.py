"""Build the exact attributable M3B decompensation task from the shared cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
RESOURCE_ROOT = (
    PAPER_ROOT
    / ".export_assets"
    / "external_artifacts"
    / "yerevann_resources"
)
PRODUCER_COMMIT = "ea0314c7cbd369f62e2237ace6f683740f867e3a"
STUDY_COMMIT = "426c1766e287dec5782855a430ec14c5cacc3a94"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def assert_outside_repository(path: Path) -> None:
    try:
        path.resolve().relative_to(WORKSPACE_ROOT.resolve())
    except ValueError:
        return
    raise ValueError("derived credentialed data must remain outside the repository")


def run_step(markers: Path, name: str, command: list[str], cwd: Path) -> None:
    marker = markers / f"{name}.json"
    if marker.is_file():
        payload = json.loads(marker.read_text())
        if payload.get("command") != command or payload.get("status") != "complete":
            raise AssertionError(f"resume marker changed for {name}")
        print(f"reuse complete step: {name}")
        return
    subprocess.run(command, cwd=cwd, check=True)
    write_json(
        marker,
        {
            "status": "complete",
            "command": command,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def summarize_listfile(path: Path) -> dict[str, object]:
    total = positive = eligible = eligible_positive = below = above = 0
    subjects: set[int] = set()
    eligible_subjects: set[int] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            cutoff = float(row["period_length"])
            label = int(row["y_true"])
            subject = int(row["stay"].split("_", 1)[0])
            positive += label
            subjects.add(subject)
            if cutoff < 24:
                below += 1
            elif cutoff >= 2000:
                above += 1
            else:
                eligible += 1
                eligible_positive += label
                eligible_subjects.add(subject)
    return {
        "rows": total,
        "positive": positive,
        "negative": total - positive,
        "subjects": len(subjects),
        "eligible_24_to_2000_rows": eligible,
        "eligible_positive": eligible_positive,
        "eligible_negative": eligible - eligible_positive,
        "eligible_subjects": len(eligible_subjects),
        "excluded_below_24": below,
        "excluded_at_or_above_2000": above,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-source-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args()
    shared_root = args.shared_source_root.resolve()
    source_root = args.source_root.resolve()
    assert_outside_repository(source_root)

    shared_preparation_path = shared_root / "preparation.json"
    shared_preparation = json.loads(shared_preparation_path.read_text())
    if shared_preparation.get("status") != "exact_source":
        raise AssertionError("shared M3B source is not exact")
    if shared_preparation.get("commits", {}).get("mimic3_benchmarks") != PRODUCER_COMMIT:
        raise AssertionError("shared M3B producer commit changed")
    m3b_root = shared_root / "m3b_data" / "root"
    for split in ("train", "test"):
        if not (m3b_root / split).is_dir():
            raise FileNotFoundError(m3b_root / split)

    source_root.mkdir(parents=True, exist_ok=True)
    markers = source_root / "markers"
    markers.mkdir(exist_ok=True)
    output = source_root / "decompensation"
    producer = RESOURCE_ROOT / "mimic3benchmark" / "scripts" / "create_decompensation.py"
    splitter = RESOURCE_ROOT / "mimic3models" / "split_train_val.py"
    run_step(
        markers,
        "01_create_decompensation",
        [sys.executable, str(producer), str(m3b_root), str(output)],
        PAPER_ROOT,
    )
    run_step(
        markers,
        "02_split_train_validation",
        [sys.executable, str(splitter), str(output)],
        PAPER_ROOT,
    )

    listfiles = {
        split: output / f"{split}_listfile.csv"
        for split in ("train", "val", "test")
    }
    summaries = {split: summarize_listfile(path) for split, path in listfiles.items()}
    train_validation = {
        key: summaries["train"][key] + summaries["val"][key]
        for key in ("rows", "positive", "negative")
    }
    if train_validation != {"rows": 2_908_414, "positive": 61_013, "negative": 2_847_401}:
        raise AssertionError(f"M3B train+validation inventory changed: {train_validation}")
    test_inventory = {
        key: summaries["test"][key] for key in ("rows", "positive", "negative")
    }
    if test_inventory != {"rows": 523_208, "positive": 9_683, "negative": 513_525}:
        raise AssertionError(f"M3B test inventory changed: {test_inventory}")

    write_json(
        source_root / "preparation.json",
        {
            "status": "exact_attributable_benchmark",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "commits": {
                "study": STUDY_COMMIT,
                "mimic3_benchmarks": PRODUCER_COMMIT,
            },
            "shared_m3b_preparation": {
                "path": str(shared_preparation_path),
                "sha256": sha256_file(shared_preparation_path),
            },
            "producer": {"path": str(producer), "sha256": sha256_file(producer)},
            "splitter": {"path": str(splitter), "sha256": sha256_file(splitter)},
            "listfiles": summaries,
            "paper_loader_cutoff_filter": {
                "minimum_inclusive_hours": 24,
                "maximum_exclusive_hours": 2000,
            },
            "train_list_order_note": "the exact producer shuffles after filesystem-order traversal; membership and labels are fixed but train row order is realization-specific",
        },
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
