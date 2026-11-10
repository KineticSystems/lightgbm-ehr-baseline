"""End-to-end synthetic test of CLUDA's direct-MEDS exporter and both checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

import cohort_source  # noqa: E402
import check_batch as batch_check  # noqa: E402
import export_source as source_exporter  # noqa: E402

from cohort_source import (  # noqa: E402
    AGE_MEAN,
    AGE_SCALING_ABS_TOLERANCE,
    AGE_SD,
    RAW_REQUIRED,
    TASK_ID,
    TS_FEATURES,
    UPSTREAM_COMMIT,
    assert_loader_age_scaling,
    load_author_source,
    sha256,
    source_files,
    validate_raw_release,
    write_json,
)


def expect_failure(function, exception_type, message: str) -> None:
    try:
        function()
    except exception_type as error:
        if message not in str(error):
            raise AssertionError(f"unexpected failure message: {error}") from error
    else:
        raise AssertionError(f"expected {exception_type.__name__}: {message}")


def test_fidelity_guards(root: Path) -> None:
    # The production guard is anchored to the independently acquired official
    # v1.0 manifest digest, rather than trusting an adjacent manifest file.
    assert cohort_source.TRUSTED_MIMIC_IV_1_0_MANIFEST_SHA256 == (
        "fa9fe89ecfb5b156fb5ed27bd420476d1c22b1e45fd26ae66aef0368c8bca06b"
    )
    raw_root = root / "raw_guard_fixture"
    for position, name in enumerate(RAW_REQUIRED[1:], start=1):
        path = raw_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((f"guard-{position}\n").encode())
    manifest_path = raw_root / "SHA256SUMS.txt"
    manifest_path.write_text(
        "".join(f"{sha256(raw_root / name)} {name}\n" for name in RAW_REQUIRED[1:])
    )
    expect_failure(
        lambda: validate_raw_release(raw_root),
        AssertionError,
        "not the pinned official MIMIC-IV v1.0 manifest",
    )

    fixture_sizes = {
        name: (raw_root / name).stat().st_size for name in RAW_REQUIRED
    }
    trusted_fixture_manifest = sha256(manifest_path)
    with patch.object(
        cohort_source,
        "TRUSTED_MIMIC_IV_1_0_MANIFEST_SHA256",
        trusted_fixture_manifest,
    ), patch.object(cohort_source, "V1_0_REQUIRED_FILE_SIZES", fixture_sizes):
        validated = validate_raw_release(raw_root)
        assert validated["manifest_sha256"] == trusted_fixture_manifest

        # Rewriting a payload and its adjacent manifest remains internally
        # consistent, but must fail because the independent manifest pin did
        # not change with it.
        (raw_root / RAW_REQUIRED[1]).write_bytes(b"substituted-release\n")
        manifest_path.write_text(
            "".join(
                f"{sha256(raw_root / name)} {name}\n" for name in RAW_REQUIRED[1:]
            )
        )
        expect_failure(
            lambda: validate_raw_release(raw_root),
            AssertionError,
            "not the pinned official MIMIC-IV v1.0 manifest",
        )

    assert_loader_age_scaling(AGE_MEAN, AGE_SD)
    expect_failure(
        lambda: assert_loader_age_scaling(
            AGE_MEAN + 2 * AGE_SCALING_ABS_TOLERANCE, AGE_SD
        ),
        AssertionError,
        "differs from loader",
    )


def make_source(root: Path) -> Path:
    source_root = root / "source"
    paths = source_files(source_root)
    paths["admissions"].parent.mkdir(parents=True, exist_ok=True)
    rows = []
    static_rows = []
    value_rows = []
    mask_rows = []
    stay_id = 30000000
    definitions = {
        "train": [(30, 0, 5), (35, 1, 52), (40, 0, 49), (44, 1, 7), (55, 0, 6)],
        "val": [(25, 0, 5), (33, 1, 50), (38, 0, 8), (42, 1, 48)],
        "test": [(50, 0, 5), (55, 1, 53), (60, 0, 49), (64, 1, 9), (35, 0, 6)],
    }
    for split_type, items in definitions.items():
        for age, label, length in items:
            stay_id += 1
            subject_id = stay_id - 20000000
            rows.append(
                {
                    "stay_id": stay_id,
                    "subject_id": subject_id,
                    "intime": "2150-01-01 00:00:00",
                    "hospital_expire_flag": label,
                    "splitType": split_type,
                }
            )
            static_rows.append(
                {
                    "stay_id": stay_id,
                    "age": (age - AGE_MEAN) / AGE_SD,
                    "bmi": np.float32((age % 7) / 3.0),
                    "height": np.float32((age % 5) / 2.0),
                    "weight": np.float32((age % 11) / 4.0),
                    "adm_categ": stay_id % 3,
                    "sex_categ": stay_id % 2,
                }
            )
            values = np.zeros((length, len(TS_FEATURES)), dtype=np.float64)
            masks = np.zeros((length, len(TS_FEATURES)), dtype=np.int64)
            values[0, 0] = 1.25  # pre-ICU carry, unobserved at hour zero
            for hour in range(length):
                for feature in (1, 5, 20, 40):
                    if (hour + feature + stay_id) % (feature % 4 + 3) == 0:
                        masks[hour, feature] = 1
                        values[hour, feature] = (hour + 1) * (feature + 1) / 17.0
                if hour:
                    missing = masks[hour] == 0
                    values[hour, missing] = values[hour - 1, missing]
            for hour in range(length):
                value_rows.append(
                    {"stay_id": stay_id, "charttime": hour, **dict(zip(TS_FEATURES, values[hour]))}
                )
                mask_rows.append(
                    {"stay_id": stay_id, "charttime": hour, **dict(zip(TS_FEATURES, masks[hour]))}
                )
    pd.DataFrame(rows).to_csv(paths["admissions"], index=False)
    pd.DataFrame(value_rows).to_csv(paths["timeseries"], index=False)
    pd.DataFrame(mask_rows).to_csv(paths["mask"], index=False)
    pd.DataFrame(static_rows).to_csv(paths["static"], index=False)
    pd.DataFrame({"feature": TS_FEATURES, "mean": 0.0, "sd": 1.0}).to_csv(
        paths["ts_scaling"], index=False
    )
    pd.DataFrame(
        {"feature": ["age", "bmi", "height", "weight"], "mean": 0.0, "sd": 1.0}
    ).to_csv(paths["static_scaling"], index=False)
    paths["session_info"].write_text("synthetic\n")
    paths["ricu_version"].write_text("0.5.2\n")
    paths["channel_order"].write_text("\n".join(TS_FEATURES) + "\n")
    paths["static_order"].write_text(
        "age\nbmi\nheight\nweight\nadm_0\nadm_1\nadm_2\nsex\n"
    )
    manifest = {
        "status": "synthetic_test_source",
        "dataset_version": "synthetic",
        "upstream_commit": UPSTREAM_COMMIT,
        "files": {
            name: {"file": str(path.relative_to(source_root)), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "summary": {},
    }
    write_json(source_root / "source_manifest.json", manifest)
    return source_root


def test_released_charttime_semantics(root: Path) -> None:
    source_root = make_source(root / "charttime_fixture")
    paths = source_files(source_root)
    admissions = pd.read_csv(paths["admissions"])
    stay_id = int(admissions.loc[admissions["splitType"] == "train", "stay_id"].iloc[0])

    # The released notebook sorts within stay by charttime, then drops the
    # stay_id/charttime columns. Row position—not absolute ICU hour—is the
    # loader's sequence coordinate. A contiguous grid may therefore begin
    # after hour zero and must retain its absolute source coordinate separately.
    for name in ("timeseries", "mask"):
        frame = pd.read_csv(paths[name])
        selected = frame["stay_id"] == stay_id
        frame.loc[selected, "charttime"] += 3
        frame.to_csv(paths[name], index=False)
    manifest_path = source_root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("timeseries", "mask"):
        manifest["files"][name]["sha256"] = sha256(paths[name])
    write_json(manifest_path, manifest)

    source = load_author_source(source_root, allow_synthetic=True)
    train = source["train"]
    position = int(np.flatnonzero(train.stay_ids == stay_id)[0])
    np.testing.assert_array_equal(
        train.source_charttime[position], np.arange(3.0, 8.0)
    )
    assert len(train.sequence[position]) == 5
    assert len(train.sequence_mask[position]) == 5
    selected = cohort_source.filtered_split(train, age_group=2)
    selected_position = int(np.flatnonzero(selected.stay_ids == stay_id)[0])
    np.testing.assert_array_equal(
        selected.source_charttime[selected_position], np.arange(3.0, 8.0)
    )
    task_selected = {
        "train": selected,
        "validation": cohort_source.filtered_split(source["validation"], age_group=2),
        "test": cohort_source.filtered_split(source["test"], age_group=3),
    }
    sample = next(
        row
        for row in cohort_source.iter_sample_rows(task_selected)
        if row["stay_id"] == stay_id and row["stay_hour"] == 4
    )
    intime = train.intimes[position]
    assert sample["observation_start"] == intime + np.timedelta64(3, "h")
    assert sample["prediction_time"] == intime + np.timedelta64(7, "h")
    sample_events = []
    for frame in source_exporter.event_batches(task_selected):
        sample_events.extend(
            frame.filter(frame["sample_id"] == sample["sample_id"]).to_dicts()
        )
    assert sample_events
    for event in sample_events:
        if event["code"].startswith("STATIC/"):
            assert event["source_time"] == 3.0
            assert np.datetime64(event["time"], "us") == intime + np.timedelta64(3, "h")
        else:
            source_index = (
                int(event["source_row"]) - (500 if event["code"].startswith("VALUE/") else 1_000)
            ) // len(TS_FEATURES)
            assert event["source_time"] == float(3 + source_index)
            assert np.datetime64(event["time"], "us") == intime + np.timedelta64(
                3 + source_index, "h"
            )

    # A true gap is not an hourly fill_gaps() grid and must still fail closed.
    for name in ("timeseries", "mask"):
        frame = pd.read_csv(paths[name])
        rows = frame.index[frame["stay_id"] == stay_id]
        frame.loc[rows[1], "charttime"] += 1
        frame.to_csv(paths[name], index=False)
        manifest["files"][name]["sha256"] = sha256(paths[name])
    write_json(manifest_path, manifest)
    expect_failure(
        lambda: load_author_source(source_root, allow_synthetic=True),
        AssertionError,
        "released hourly grid is not finite, integral, and contiguous",
    )


def run(command: list[str], *, environment: Optional[Dict[str, str]] = None) -> None:
    subprocess.run(command, check=True, cwd=PAPER_ROOT, env=environment)


def test_event_coordinate_guards(source_root: Path, task_dir: Path) -> None:
    manifest = json.loads((task_dir / "manifest.json").read_text())
    assert manifest["task"]["positive_class"] == (
        "hospital_expire_flag == 1 (death during the hospital admission)"
    )
    index = pq.read_table(
        task_dir / manifest["meds"]["task_index"]["file"]
    ).to_pandas().set_index("sample_id", verify_integrity=True)
    sid, rows = next(batch_check.iter_event_groups(
        task_dir / manifest["meds"]["events"]["file"]
    ))
    index_row = index.loc[sid]
    selected = cohort_source.selected_splits(
        load_author_source(source_root, allow_synthetic=True)
    )
    source_charttime = selected[index_row["split"]].source_charttime[
        int(index_row["loader_patient_id"])
    ]
    batch_check.reconstruct(rows, index_row, source_charttime)

    def changed(field: str, value) -> dict:
        result = {
            name: values.copy() if hasattr(values, "copy") else list(values)
            for name, values in rows.items()
        }
        result[field][0] = value
        return result

    expect_failure(
        lambda: batch_check.reconstruct(
            changed("subject_id", int(rows["subject_id"][0]) + 1),
            index_row,
            source_charttime,
        ),
        AssertionError,
        "event subject differs",
    )
    expect_failure(
        lambda: batch_check.reconstruct(
            changed("time", int(rows["time"][0]) + 1),
            index_row,
            source_charttime,
        ),
        AssertionError,
        "static event duplication/position changed",
    )
    expect_failure(
        lambda: batch_check.reconstruct(
            changed("source_row", int(rows["source_row"][0]) + 1),
            index_row,
            source_charttime,
        ),
        AssertionError,
        "canonical source-row order",
    )


def test_task_source_manifest_guards(source_root: Path, task_dir: Path) -> None:
    manifest_path = task_dir / "manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    manifest["source"]["source_manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        for checker in ("check_cohort.py", "check_batch.py"):
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PAPER_ROOT / checker),
                    "--source-root",
                    str(source_root),
                    "--task-dir",
                    str(task_dir),
                    "--allow-synthetic",
                ],
                cwd=PAPER_ROOT,
                capture_output=True,
                text=True,
            )
            assert completed.returncode != 0
            assert "not sealed to the exact CLUDA source manifest" in completed.stderr
    finally:
        manifest_path.write_bytes(original)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cluda-export-test-") as directory:
        root = Path(directory)
        test_fidelity_guards(root)
        test_released_charttime_semantics(root)
        source_root = make_source(root)
        task_dir = root / TASK_ID
        run(
            [
                sys.executable,
                str(PAPER_ROOT / "export_source.py"),
                "--source-root",
                str(source_root),
                "--output-dir",
                str(task_dir),
                "--allow-synthetic",
            ]
        )
        test_event_coordinate_guards(source_root, task_dir)
        test_task_source_manifest_guards(source_root, task_dir)
        environment = os.environ.copy()
        shared_python = PAPER_ROOT.parents[1] / "lightgbm_code" / ".venv" / "bin" / "python"
        run(
            [
                str(shared_python),
                "-m",
                "missing_xgboost_lightgbm.validate_task",
                str(task_dir),
            ],
            environment=environment,
        )
        run(
            [
                sys.executable,
                str(PAPER_ROOT / "check_cohort.py"),
                "--source-root",
                str(source_root),
                "--task-dir",
                str(task_dir),
                "--allow-synthetic",
            ]
        )
        run(
            [
                sys.executable,
                str(PAPER_ROOT / "check_batch.py"),
                "--source-root",
                str(source_root),
                "--task-dir",
                str(task_dir),
                "--allow-synthetic",
            ]
        )
        cohort = json.loads((task_dir / "audit" / "cohort_reconstruction.json").read_text())
        batch = json.loads((task_dir / "audit" / "full_batch_recreation.json").read_text())
        assert cohort["status"] == "synthetic_exact_loader_projection"
        assert batch["status"] == "synthetic_exact_loader_projection"
        assert batch["retained_projection"]["early_mask_bug_samples"] > 0
        print(json.dumps({"status": "passed", "samples": batch["retained_projection"]["all_samples_checked"]}))


if __name__ == "__main__":
    main()
