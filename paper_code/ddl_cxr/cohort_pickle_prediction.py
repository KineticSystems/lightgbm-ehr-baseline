"""Invoke DDL-CXR's released EHR_Pickler for only the selected predictor."""

from __future__ import annotations

import argparse
from pathlib import Path

from ldm.data.dynamic_data import EHR_Pickler


REPOSITORY = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--subjects-root", type=Path, required=True)
    args = parser.parse_args()
    pickler = EHR_Pickler(
        stage="prediction",
        normalizer_state=str(
            REPOSITORY / "ldm" / "data" / "normalizers" / "predict_normalizer"
        ),
        metadata_path=str(args.metadata_path),
        mimic_iv_subjects_dir=str(args.subjects_root),
    )
    pickler.pickle_ehr()


if __name__ == "__main__":
    main()
