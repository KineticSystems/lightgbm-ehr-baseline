"""Validate a completed direct MEDS export."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common_format import open_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir", type=Path)
    args = parser.parse_args()
    task = open_task(args.task_dir)
    print(f"Validated {task.task_id}: {task.modalities}")


if __name__ == "__main__":
    main()
