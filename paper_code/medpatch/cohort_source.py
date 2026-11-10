"""Shared exact-source helpers for the MedPatch the exporter/checkers."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parent
CONFIG_PATH = REPOSITORY / "ehr_utils" / "resources" / "discretizer_config.json"
NORMALIZER_PATH = (
    REPOSITORY
    / "normalizers"
    / "ihm_ts1.0.input_str_previous.start_time_zero.normalizer"
)
ITEMID_MAP_PATH = (
    REPOSITORY
    / "mimic4extract"
    / "mimic3benchmark"
    / "resources"
    / "itemid_to_variable_map.csv"
)
TEST_MEMBERSHIP_PATH = (
    REPOSITORY
    / "mimic4extract"
    / "mimic3benchmark"
    / "resources"
    / "testset_iv.csv"
)
VALIDATION_MEMBERSHIP_PATH = (
    REPOSITORY / "ehr_utils" / "resources" / "valset_iv.csv"
)
HOURS = 48.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def configuration() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def variable_names() -> tuple[str, ...]:
    return tuple(configuration()["id_to_channel"])


def code_vocabulary() -> tuple[str, ...]:
    config = configuration()
    result = []
    for variable in config["id_to_channel"]:
        if config["is_categorical_channel"][variable]:
            result.extend(
                f"{variable}//{value}"
                for value in config["possible_values"][variable]
            )
        else:
            result.append(variable)
    return tuple(result)


def split_members(path: Path) -> set[int]:
    members = set()
    with path.open() as handle:
        for line in handle:
            subject, selected = line.rstrip("\n").split(",")
            if int(selected) == 1:
                members.add(int(subject))
    return members


def load_source_rows(source_root: Path) -> list[dict[str, object]]:
    source_root = source_root.resolve()
    stays_path = source_root.parent / "subjects" / "all_stays.csv"
    stays = pd.read_csv(
        stays_path,
        usecols=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "los",
            "mortality",
        ],
    )
    stays["intime"] = pd.to_datetime(stays["intime"], errors="raise")
    stays["outtime"] = pd.to_datetime(stays["outtime"], errors="raise")
    stay_lookup = {
        int(row.stay_id): row for row in stays.itertuples(index=False)
    }
    result = []
    specifications = (
        ("train", source_root / "train_listfile.csv", source_root / "train"),
        ("validation", source_root / "val_listfile.csv", source_root / "train"),
        ("test", source_root / "test_listfile.csv", source_root / "test"),
    )
    for split, listfile, data_dir in specifications:
        with listfile.open(newline="") as handle:
            for source_position, record in enumerate(csv.DictReader(handle)):
                stay_id = int(record["stay_id"])
                stay = stay_lookup.get(stay_id)
                if stay is None:
                    raise AssertionError("mortality listfile contains an unknown ICU stay")
                subject_id = int(record["stay"].split("_", 1)[0])
                if subject_id != int(stay.subject_id):
                    raise AssertionError("filename subject disagrees with all_stays.csv")
                label = int(record["y_true"])
                if label != int(stay.mortality):
                    raise AssertionError("mortality listfile label disagrees with extraction")
                timeseries_path = data_dir / record["stay"]
                if not timeseries_path.is_file():
                    raise FileNotFoundError(timeseries_path)
                result.append(
                    {
                        "split": split,
                        "source_position": source_position,
                        "source_file": record["stay"],
                        "timeseries_path": timeseries_path,
                        "subject_id": subject_id,
                        "hadm_id": int(stay.hadm_id),
                        "stay_id": stay_id,
                        "label": label,
                        "intime": stay.intime.to_pydatetime(),
                        "outtime": stay.outtime.to_pydatetime(),
                        "los_days": float(stay.los),
                    }
                )
    stay_ids = [int(row["stay_id"]) for row in result]
    if len(stay_ids) != len(set(stay_ids)):
        raise AssertionError("released mortality listfiles repeat a stay")
    return result


def author_preprocessors():
    sys.path.insert(0, str(REPOSITORY))
    from ehr_utils.preprocessing import Discretizer, Normalizer

    discretizer = Discretizer(
        timestep=1.0,
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
        config_path=str(CONFIG_PATH),
    )
    config = configuration()
    dummy = np.asarray(
        [
            ["0"]
            + [str(config["normal_values"][name]) for name in config["id_to_channel"]]
        ]
    )
    header = ["Hours"] + list(config["id_to_channel"])
    _, output_header = discretizer.transform(dummy, header=header, end=HOURS)
    output_header = output_header.split(",")
    continuous = [
        position for position, name in enumerate(output_header) if "->" not in name
    ]
    normalizer = Normalizer(fields=continuous)
    normalizer.load_params(str(NORMALIZER_PATH))
    if len(output_header) != 76 or len(continuous) != 12:
        raise AssertionError("unexpected released MedPatch EHR channel layout")
    return discretizer, normalizer, tuple(output_header)


def _install_import_stubs() -> None:
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch_utils = types.ModuleType("torch.utils")
        torch_data = types.ModuleType("torch.utils.data")

        class Dataset:
            pass

        class DataLoader:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("the export does not instantiate the PyTorch DataLoader")

        torch_data.Dataset = Dataset
        torch_data.DataLoader = DataLoader
        torch_utils.data = torch_data
        torch.utils = torch_utils
        sys.modules["torch"] = torch
        sys.modules["torch.utils"] = torch_utils
        sys.modules["torch.utils.data"] = torch_data
    if "PIL" not in sys.modules:
        pil = types.ModuleType("PIL")
        pil.Image = types.ModuleType("PIL.Image")
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = pil.Image
    if "torchvision" not in sys.modules:
        torchvision = types.ModuleType("torchvision")
        transforms = types.ModuleType("torchvision.transforms")
        torchvision.transforms = transforms
        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.transforms"] = transforms


def author_loader_module():
    _install_import_stubs()
    sys.path.insert(0, str(REPOSITORY))
    from datasets import ehr_dataset

    return ehr_dataset


def author_datasets(source_root: Path):
    module = author_loader_module()
    discretizer, normalizer, header = author_preprocessors()
    args = SimpleNamespace(task="in-hospital-mortality")
    source_root = source_root.resolve()
    datasets = {
        "train": module.EHRdataset(
            args,
            discretizer,
            normalizer,
            str(source_root / "train_listfile.csv"),
            str(source_root / "train"),
        ),
        "validation": module.EHRdataset(
            args,
            discretizer,
            normalizer,
            str(source_root / "val_listfile.csv"),
            str(source_root / "train"),
        ),
        "test": module.EHRdataset(
            args,
            discretizer,
            normalizer,
            str(source_root / "test_listfile.csv"),
            str(source_root / "test"),
        ),
    }
    return module, datasets, header
