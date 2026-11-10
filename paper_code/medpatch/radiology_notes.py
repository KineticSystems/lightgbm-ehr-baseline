"""Reconstruct MedPatch's released first-48-hour radiology-report text per stay.

The trimodal mortality experiment (`--modalities EHR-CXR-RR --data_pairs
partial`) builds its radiology channel in `datasets/DataFusion.py`:

    rr = icu_stay_metadata[columns].merge(
        radiology_reports[['subject_id', 'hadm_id', 'charttime', 'text']],
        how='inner', on=['subject_id', 'hadm_id'])
    rr = rr.loc[(rr.radiology_charttime >= rr.intime)
                & (rr.radiology_charttime <= rr.intime + DateOffset(hours=48))]
    rr = rr.groupby(group_columns, as_index=False).agg(
        {'radiology_text': lambda texts: ' '.join(texts),
         'radiology_charttime': 'max'})

and then left-joins it onto every EHR stay, so a stay without an in-window
report keeps a null text rather than leaving the cohort. This module reproduces
exactly that projection and returns one concatenated string per `stay_id`.

`all_stays.csv` carries one row per stay with a unique `hadm_id`, so grouping
by the released `group_columns` is equivalent to grouping by `stay_id`; that
equivalence is asserted rather than assumed. Grouping by `stay_id` also avoids
the released code's need to `fillna` the grouping columns, which exists only
because pandas drops null group keys.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


WINDOW_HOURS = 48
NOTE_COLUMNS = ["subject_id", "hadm_id", "charttime", "text"]
STAY_COLUMNS = ["subject_id", "hadm_id", "stay_id", "intime"]


def radiology_source_path(note_root: Path) -> Path:
    """Return the released radiology note table, allowing the gzip payload."""
    note_root = Path(note_root)
    for candidate in (
        note_root / "note" / "radiology.csv",
        note_root / "note" / "radiology.csv.gz",
        note_root / "radiology.csv",
        note_root / "radiology.csv.gz",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no radiology note table under {note_root}")


def radiology_text_by_stay(
    all_stays_path: Path,
    note_root: Path,
) -> dict[int, str]:
    """Map stay_id to the released space-joined in-window radiology text."""
    stays = pd.read_csv(all_stays_path, usecols=STAY_COLUMNS)
    if stays["stay_id"].duplicated().any():
        raise AssertionError("all_stays.csv does not have one row per stay")
    if stays["hadm_id"].duplicated().any():
        raise AssertionError(
            "hadm_id is not unique per stay; grouping by stay_id is no longer "
            "equivalent to the released group_columns"
        )
    stays["intime"] = pd.to_datetime(stays["intime"], errors="raise")

    reports = pd.read_csv(
        radiology_source_path(note_root), usecols=NOTE_COLUMNS
    )
    reports["charttime"] = pd.to_datetime(reports["charttime"], errors="raise")
    # Preserve released file order; pandas groupby keeps within-group order and
    # the author's ' '.join therefore concatenates in this order.
    reports = reports.reset_index(drop=True)

    merged = stays.merge(reports, how="inner", on=["subject_id", "hadm_id"])
    window_end = merged["intime"] + pd.DateOffset(hours=WINDOW_HOURS)
    merged = merged.loc[
        (merged["charttime"] >= merged["intime"])
        & (merged["charttime"] <= window_end)
    ]
    merged["text"] = merged["text"].fillna("").astype(str)

    grouped = merged.groupby("stay_id", as_index=False, sort=False).agg(
        {"text": lambda texts: " ".join(texts)}
    )
    return {
        int(row.stay_id): str(row.text)
        for row in grouped.itertuples(index=False)
    }
