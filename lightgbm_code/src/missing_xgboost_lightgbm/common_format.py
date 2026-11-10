"""Shared constants and small utilities for one MEDS-backed binary task."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


FORMAT_NAME = "missing_xgboost_meds"
SCHEMA_VERSION = 3
MEDS_VERSION = "0.4"
SPLITS = ("train", "validation", "test")
TASK_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_strings(values: list[str] | tuple[str, ...]) -> str:
    """Hash an ordered string sequence without delimiter ambiguity."""
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def open_task(root: Path):
    """Open the single supported MEDS event contract."""
    path = root.resolve() / "manifest.json"
    if not path.is_file():
        raise ValueError(f"task manifest is missing: {path}")
    from .meds_format import MedsTask

    format_name = json.loads(path.read_text()).get("format")
    if format_name != FORMAT_NAME:
        raise ValueError(f"unsupported task format: {format_name!r}")
    return MedsTask.open(root)
