"""Build (or resume) DPMM's released MIMIC-IV 2.2 source reconstruction cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))

from cohort_source_mimic4 import ensure_source  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = ensure_source(args.mimic_root, args.cache_root)
    print(json.dumps({k: manifest[k] for k in (
        "counts", "mortality_positives", "readmission_positives",
        "observed_source_rows", "retained_source_rows",
        "readmission_defect_measurement",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
