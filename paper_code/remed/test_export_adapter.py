"""Small synthetic regression test for the reversible REMed the adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from check_batch import main as check_batch_main
from export_source import main as export_main
from cohort_source import (
    decode_hi_event,
    encode_hi_event,
    event_code,
    event_code_from_payload,
)


def event(digit_token: int) -> np.ndarray:
    """A decodable triage temperature event: 'triage temperature 9 8 . <d>'.

    Token ids resolve through the pinned Bio_ClinicalBERT vocabulary:
    [CLS]=101, t=189/##ria=3464/##ge=2176 (table run), temperature=4143
    (column run), 9=130/8=129/.=119 (value run start), [SEP]=102. The
    trailing digit token varies per event so values differ.
    """
    value = np.zeros((3, 128), dtype=np.int16)
    tokens = (101, 189, 3464, 2176, 4143, 130, 129, 119, digit_token, 102)
    types = (5, 1, 1, 1, 2, 3, 3, 3, 3, 6)
    value[0, : len(tokens)] = tokens
    value[1, : len(types)] = types
    return value


def make_source(root: Path) -> None:
    rows = []
    with h5py.File(root / "mimiciv.h5", "w") as handle:
        ehr = handle.create_group("ehr")
        for position, (split, label) in enumerate(
            (
                ("train", 0),
                ("train", 1),
                ("valid", 0),
                ("valid", 1),
                ("test", 0),
                ("test", 1),
            )
        ):
            stay_id = 3000 + position
            hi = np.stack((event(122 + position), event(123 + position)))
            times = np.asarray((-60 + position, 60 + position), dtype=np.int32)
            group = ehr.create_group(str(stay_id))
            group.create_dataset("hi", data=hi, dtype="i2")
            group.create_dataset("time", data=times, dtype="i")
            group.create_dataset(
                "fl", data=np.ones((3, 8 + position), dtype=np.int16), dtype="i2"
            )
            group.attrs["los_7"] = label
            rows.append(
                {
                    "subject_id": 1000 + position,
                    "hadm_id": 2000 + position,
                    "stay_id": stay_id,
                    "ADMITTIME": "2200-01-01 00:00:00+00:00",
                    "INTIME": 120,
                    "los_7": label,
                    "split_2020": split,
                }
            )
    pd.DataFrame(rows).to_csv(root / "mimiciv_cohort.csv", index=False)
    (root / "source_build.json").write_text(
        json.dumps({"status": "complete", "fixture": True}) + "\n"
    )


def test_reversible_event_code() -> None:
    expected = event(777)
    payload = encode_hi_event(expected)
    assert np.array_equal(decode_hi_event(payload), expected)
    assert event_code_from_payload(payload) == event_code(expected)


def test_export_and_batch_recreation(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    make_source(source)
    task = tmp_path / "los_7d_at_48h"
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_source.py", "--source-root", str(source), "--output-dir", str(task)],
    )
    export_main()
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_batch.py", "--source-root", str(source), "--task-dir", str(task)],
    )
    check_batch_main()
    report = json.loads((task / "audit" / "full_batch_recreation.json").read_text())
    assert report["status"] == "exact_at_released_ehrdataset_input_plain_decode"
    assert report["samples"] == 6
    assert report["events"] == 12
    assert report["emissions"] == 12
    import pyarrow.parquet as _pq
    declared = _pq.read_table(task / "metadata" / "codes.parquet")["code"].to_pylist()
    assert declared == ["MIMICIV//TRIAGE//temperature"]
    events_table = _pq.read_table(task / "data" / "events.parquet")
    values = [v for v in events_table["numeric_value"].to_pylist()]
    assert all(v is not None and 98.0 < v < 99.0 for v in values)
