"""Pinned export source helpers for the released full-label IHM experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd


REPOSITORY = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPOSITORY.parents[1]
HOURS = 48.0
TASK_ID = "mortality_48h"
STUDY_COMMIT = "655c26a23880950cc270df5681b981e6869e26df"
PRODUCER_COMMIT = "ea0314c7cbd369f62e2237ace6f683740f867e3a"
TORCHMIMIC_COMMIT = "b3610fc98025ec42903e7646c715b4a5faeac403"
LISTFILE_SHA256 = {
    "train": "802bf8e91912c195ce1800b248f967c4cb0d77b55c38358deaddcaf265ed88d5",
    "validation": "7b0fa183e44b5dde12f081cb3578b9228f6cbfe27786ead478f6252890815d32",
    "test": "828c0915b34c41d2fa231fe1f0a443e5b2e71e712225eab631727b902c5604d3",
}
EXPECTED_SPLITS = {
    "train": {"rows": 14_681, "subjects": 12_565, "positive": 1_987},
    "validation": {"rows": 3_222, "subjects": 2_766, "positive": 436},
    "test": {"rows": 3_236, "subjects": 2_763, "positive": 374},
}
ORDERED_NAME_SHA256 = {
    "train": "48b934c28d4da9befa67b33021fa142cbe7e575fa29d9cf55d12e109ba5d6a62",
    "validation": "97c65822561dd7258ce60a8f257a3e7b7d1fafada25818645deefcdbca2da84b",
    "test": "3d53a28036dc9df2e36c5689884b6ede4d6cb5a75b4e38aca733d4cf79ce24d2",
}
ORDERED_LABEL_SHA256 = {
    "train": "798e1914be06b15e810cee73b3e9c75ee6886411828089d0f18e068d1849fb16",
    "validation": "348f4ce6a5f781eca8fad41aad9d485301f0ffb8b33c6b114711b4d6f89d8727",
    "test": "7185f1fb6972b3db9d1f055a773d6581b6305b4649deb501e7a26d5f40f79bcd",
}
FROZEN_AUDIT_REPORTED_ORDERED_NAME_SHA256 = {
    "train": "dda951a2734c3924c47891fa220b4c62e553dd9593dbb96eff0e7411d1ac4514",
    "validation": "aac77e455ee40ba30be0f9eafd145e2a1245ef4c36727bef82b0b85eaf8d1d15",
    "test": "083c5a14712dff28d843a07dab07afc89a1616a9d79db2f119d463aca4ee6e9b",
}
FROZEN_AUDIT_REPORTED_ORDERED_LABEL_SHA256 = {
    "train": "9da9e4c0a4e9fc3f76ea06f941ada109686f3d669ab5b04150431fe30e0bdc3b",
    "validation": "e3556b746d3e14d6585b9ce547cc9b52ae47e42f66a99dba32a3b9b12389a9bf",
    "test": "5ff5566cedefaafaf56775580f96c87aa38ebed9e674e3cd83001b83860a3fb6",
}
PRODUCER_SHA256 = {
    "mimic3benchmark/mimic3csv.py": "b09ca596162f97b0477ef6fa4a2c65dd365bf566bd1b25f0db0d5b2c4006e19c",
    "mimic3benchmark/preprocessing.py": "231cef670f58a999d5c3424f69893c38d9ebb2b8071c27fd1a739e17fff90053",
    "mimic3benchmark/subject.py": "0d7aef5dd2492283697ea9a01f94f8a8027dade036aa902c0c72330ecd08f5db",
    "mimic3benchmark/resources/itemid_to_variable_map.csv": "86ca1228eabd4731449740916bad5f7b03e9b8bc84c0648e49bab9e72ba5397e",
    "mimic3benchmark/resources/testset.csv": "206d879065dd7ae245e8d8b2bddf962b300f1c21c0351dc7b2ceaf9c308b243a",
    "mimic3models/resources/valset.csv": "e4d70f15f5926b8512ee3542b427333cc7f8bf2109d98ee6568eaee54f650430",
}
STUDY_SOURCE_SHA256 = {
    "mimic3-benchmarks/create_decompensation.py": "d413c88d9a6df0a95fad88ba523cd41ba05e80f393b928b8fc7e3827b1ae652a",
    "mimic3-benchmarks/create_in_hospital_mortality.py": "14484c6f6d6184de3f05e5bfd3ca8d694b381411ea3eae160b1ea76174a726f9",
    "mimic3-benchmarks/create_length_of_stay.py": "60e294439dd4e39a51ce5e27df5991898be9f96e6c183bf14769e9a26f94821d",
    "mimic3-benchmarks/create_multitask.py": "8695922eaef1d7402c2b76350052cbf7d4ac82a132c8bcab2d86fc306cc96dee",
    "mimic3-benchmarks/create_phenotyping.py": "c8f3bd725dea8fc8cc83c9fa1a01635469ae0bb6011495a8cf7cdee77ec67299",
    "mimic3-benchmarks/extract_episodes_from_subjects.py": "e3c2ae07019aeb0d2cc29cf5bc871e04b2e9b9c9805c83515a4bc88dde0dd096",
}
NORMALIZER_SHA256 = "47f898f711f44347c082f299af5359dc89464dbb9dedafe548d66d8efdbe4341"
DISCRETIZER_SHA256 = "fe920aae287359c357ec6731e631dc27e1f24b60541156b9ecf3a1ab7c82cff3"
NAME_PATTERN = re.compile(
    r"^(?P<subject>[0-9]+)_episode(?P<stay>[0-9]+)_timeseries_"
    r"intime_(?P<intime>[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2})_"
    r"outtime_(?P<outtime>[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2})\.csv$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def hash_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_pinned_sources(producer_root: Path) -> dict[str, str]:
    actual = {}
    for relative, expected in PRODUCER_SHA256.items():
        path = producer_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[relative] = sha256(path)
        if actual[relative] != expected:
            raise AssertionError(f"pinned producer file changed: {relative}")
    for relative, expected in STUDY_SOURCE_SHA256.items():
        path = REPOSITORY / relative
        actual[relative] = sha256(path)
        if actual[relative] != expected:
            raise AssertionError(f"released study file changed: {relative}")
    return actual


def import_producer(producer_root: Path):
    verify_pinned_sources(producer_root)
    sys.path.insert(0, str(producer_root))
    from mimic3benchmark import mimic3csv, preprocessing, subject  # noqa: PLC0415

    return mimic3csv, preprocessing, subject


def raw_path(root: Path, stem: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        path = root / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing {stem}.csv[.gz] below {root}")


def gzip_aware_dataframe(path: str | Path, *args, **kwargs):
    path = Path(path)
    if not path.is_file() and path.suffix == ".csv":
        compressed = path.with_suffix(".csv.gz")
        if compressed.is_file():
            path = compressed
    return pd.read_csv(path, *args, **kwargs)


def parse_name(name: str) -> dict[str, object]:
    match = NAME_PATTERN.fullmatch(name)
    if match is None:
        raise AssertionError(f"unexpected author IHM filename: {name!r}")
    return {
        "subject_id": int(match.group("subject")),
        "stay_id": int(match.group("stay")),
        "intime": pd.to_datetime(match.group("intime"), format="%Y-%m-%d_%H-%M-%S"),
        "outtime": pd.to_datetime(match.group("outtime"), format="%Y-%m-%d_%H-%M-%S"),
    }


def listfile_paths(root: Path) -> dict[str, Path]:
    return {
        "train": root / "train_listfile.csv",
        "validation": root / "val_listfile.csv",
        "test": root / "test_listfile.csv",
    }


def load_list_rows(list_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, path in listfile_paths(list_root).items():
        if sha256(path) != LISTFILE_SHA256[split]:
            raise AssertionError(f"{split} author listfile hash changed")
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["stay", "y_true"]:
                raise AssertionError(f"{split} author listfile header changed")
            for position, record in enumerate(reader):
                name = str(record["stay"])
                label = int(record["y_true"])
                if label not in (0, 1):
                    raise AssertionError("IHM labels must be binary")
                rows.append(
                    {
                        **parse_name(name),
                        "split": split,
                        "source_position": position,
                        "source_file": name,
                        "label": label,
                    }
                )
    if len({int(row["stay_id"]) for row in rows}) != len(rows):
        raise AssertionError("author full lists contain a duplicate ICU stay")
    return rows


def load_source_rows(source_root: Path) -> list[dict[str, object]]:
    task_root = source_root / "in-hospital-mortality"
    rows = load_list_rows(task_root)
    for row in rows:
        source_partition = "test" if row["split"] == "test" else "train"
        path = task_root / source_partition / str(row["source_file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        row["timeseries_path"] = path
    return rows


def torchmimic_artifacts() -> dict[str, Path]:
    import torchmimic  # noqa: PLC0415

    package = Path(torchmimic.__file__).resolve().parent
    return {
        "discretizer": package / "data" / "discretizers" / "discretizer_config.json",
        "normalizer": package
        / "data"
        / "normalizers"
        / "ihm_ts1.0.input_str:previous.start_time:zero.normalizer",
    }


def configuration() -> dict[str, object]:
    artifacts = torchmimic_artifacts()
    if sha256(artifacts["discretizer"]) != DISCRETIZER_SHA256:
        raise AssertionError("torchmimic discretizer artifact changed")
    return json.loads(artifacts["discretizer"].read_text())


def variable_names() -> tuple[str, ...]:
    return tuple(configuration()["id_to_channel"])


def code_vocabulary() -> tuple[str, ...]:
    config = configuration()
    result = []
    for variable in config["id_to_channel"]:
        if config["is_categorical_channel"][variable]:
            result.extend(
                f"{variable}//{value}" for value in config["possible_values"][variable]
            )
        else:
            result.append(variable)
    if len(result) != 59 or len(set(result)) != 59:
        raise AssertionError("released discretizer must define 59 value channels")
    return tuple(result)


def author_datasets(source_root: Path):
    from torchmimic.data import IHMDataset  # noqa: PLC0415

    artifacts = torchmimic_artifacts()
    if sha256(artifacts["normalizer"]) != NORMALIZER_SHA256:
        raise AssertionError("torchmimic IHM normalizer artifact changed")
    root = source_root / "in-hospital-mortality"
    return {
        "train": IHMDataset(
            str(root), customListFile=str(root / "train_listfile.csv"), train=True
        ),
        "validation": IHMDataset(
            str(root), customListFile=str(root / "val_listfile.csv"), train=True
        ),
        "test": IHMDataset(
            str(root), customListFile=str(root / "test_listfile.csv"), train=False
        ),
    }


def author_preprocessors():
    from torchmimic.data.preprocessing import Discretizer, Normalizer  # noqa: PLC0415

    artifacts = torchmimic_artifacts()
    discretizer = Discretizer(
        timestep=1.0,
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
    )
    header = discretizer.transform(
        [["0"] + [""] * len(discretizer._id_to_channel)], end=HOURS
    )[1].split(",")
    continuous = [index for index, name in enumerate(header) if "->" not in name]
    normalizer = Normalizer(fields=continuous)
    normalizer.load_params(artifacts["normalizer"])
    return discretizer, normalizer, header
