"""Thin training view over DPMM's verified export source artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from dataset_mimic3.fusion import MIMIC_NOTE_EHR, load_note_ehr
from cohort_source import TOKENIZER, author_datasets


PAPER_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PAPER_ROOT.parent.parent
CACHE_ROOT = PAPER_ROOT / ".export_assets/source_cache"
TASK_ROOT = WORKSPACE_ROOT / "paper_tasks/dpmm/mortality_48h_partial_ehr_note"
SOURCE_MANIFEST = CACHE_ROOT / "source_manifest.json"
FULL_BATCH_REPORT = TASK_ROOT / "audit/full_batch_recreation.json"
COHORT_REPORT = TASK_ROOT / "audit/cohort_reconstruction.json"
EXPECTED_SPLIT_COUNTS = {"train": 14681, "validation": 3222, "test": 3236}
EXPECTED_PAIRED_COUNTS = {"train": 3652, "validation": 815, "test": 806}
EXPECTED_TEST_POSITIVES = 374


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenNoteDataset(Dataset):
    """Expose reconstructed notes through the released MIMICNOTE interface."""

    def __init__(
        self,
        split: str,
        cohort: pd.DataFrame,
        notes: pd.DataFrame,
        tokenizer,
    ):
        paired = cohort.loc[
            (cohort["split"] == split) & cohort["note_present"]
        ].sort_values("fusion_position")
        expected = EXPECTED_PAIRED_COUNTS[split]
        if len(paired) != expected:
            raise AssertionError(
                f"DPMM {split} paired count changed: {len(paired)} != {expected}"
            )
        split_notes = notes.loc[notes["split"] == split].sort_values(
            ["stay_id", "slot"]
        )
        grouped = {int(key): value for key, value in split_notes.groupby("stay_id")}
        records: dict[str, tuple[list[str], float]] = {}
        for row in paired.itertuples(index=False):
            stay_id = int(row.stay_id)
            if stay_id not in grouped:
                raise AssertionError(f"paired stay {stay_id} has no reconstructed notes")
            group = grouped[stay_id].sort_values("slot")
            if group["slot"].tolist() != [0, 1, 2, 3, 4]:
                raise AssertionError(f"paired stay {stay_id} does not have slots 0..4")
            records[str(row.source_name)] = (
                group["text"].astype(str).tolist(),
                float(row.label),
            )
        if len(records) * 5 != len(split_notes):
            raise AssertionError(f"DPMM {split} note rows are not exactly five per stay")
        self.filtered_names = paired["source_name"].astype(str).tolist()
        self.names = list(self.filtered_names)
        self._records = records
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.filtered_names)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.filtered_names[index]
        texts, label = self._records[str(index)]
        tokens = []
        masks = []
        for text in texts:
            encoded = self.tokenizer.encode_plus(
                text,
                padding="max_length",
                max_length=512,
                add_special_tokens=True,
                return_attention_mask=True,
                truncation=True,
            )
            tokens.append(encoded["input_ids"])
            masks.append(encoded["attention_mask"])
        token_tensor = torch.tensor(tokens, dtype=torch.int64)
        mask_tensor = torch.tensor(masks, dtype=torch.int64)
        if token_tensor.shape != (5, 512) or mask_tensor.shape != (5, 512):
            raise AssertionError("DPMM TinyBERT note boundary changed")
        return token_tensor, mask_tensor, torch.tensor(label, dtype=torch.float32)


@dataclass
class AuthorDataBundle:
    cohort: pd.DataFrame
    structured: dict
    notes: dict[str, FrozenNoteDataset]
    fusion: dict[str, MIMIC_NOTE_EHR]
    source_manifest: dict

    def ordered_rows(self, split: str) -> pd.DataFrame:
        return (
            self.cohort.loc[self.cohort["split"] == split]
            .sort_values("fusion_position")
            .reset_index(drop=True)
        )


def _load_frames(cache_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = cache_root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "exact_reconstruction":
        raise AssertionError("DPMM source cache is not the verified reconstruction")
    cohort_path = cache_root / manifest["files"]["cohort"]
    notes_path = cache_root / manifest["files"]["selected_notes"]
    if sha256(cohort_path) != manifest["file_sha256"]["cohort"]:
        raise AssertionError("DPMM source cohort hash changed")
    if sha256(notes_path) != manifest["file_sha256"]["selected_notes"]:
        raise AssertionError("DPMM selected-note hash changed")
    cohort = pd.read_parquet(cohort_path)
    notes = pd.read_parquet(notes_path)
    counts = cohort.groupby("split").size().to_dict()
    if counts != EXPECTED_SPLIT_COUNTS:
        raise AssertionError(f"DPMM split counts changed: {counts}")
    paired = cohort.loc[cohort["note_present"]].groupby("split").size().to_dict()
    if paired != EXPECTED_PAIRED_COUNTS:
        raise AssertionError(f"DPMM paired counts changed: {paired}")
    test_positive = int(cohort.loc[cohort["split"] == "test", "label"].sum())
    if test_positive != EXPECTED_TEST_POSITIVES:
        raise AssertionError(f"DPMM test positives changed: {test_positive}")
    return cohort, notes, manifest


def build_author_bundle(args, cache_root: Path = CACHE_ROOT) -> AuthorDataBundle:
    cache_root = cache_root.resolve()
    cohort, note_frame, manifest = _load_frames(cache_root)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    structured = author_datasets(cache_root)
    note_datasets = {
        split: FrozenNoteDataset(split, cohort, note_frame, tokenizer)
        for split in ("train", "validation", "test")
    }
    fusion = {
        "train": MIMIC_NOTE_EHR(args, structured["train"], note_datasets["train"]),
        "validation": MIMIC_NOTE_EHR(
            args,
            structured["validation"],
            note_datasets["validation"],
            split="val",
        ),
        "test": MIMIC_NOTE_EHR(
            args, structured["test"], note_datasets["test"], split="test"
        ),
    }
    bundle = AuthorDataBundle(cohort, structured, note_datasets, fusion, manifest)
    for split, dataset in fusion.items():
        expected_rows = bundle.ordered_rows(split)
        actual_order = list(dataset.files_paired) + list(dataset.files_unpaired)
        expected_order = expected_rows["source_name"].astype(str).tolist()
        if actual_order != expected_order:
            raise AssertionError(f"DPMM {split} fusion order changed")
        if len(dataset) != EXPECTED_SPLIT_COUNTS[split]:
            raise AssertionError(f"DPMM {split} fusion length changed")
    return bundle


def released_loaders(args, bundle: AuthorDataBundle):
    train, validation, test = load_note_ehr(
        args,
        bundle.structured["train"],
        bundle.structured["validation"],
        bundle.notes["train"],
        bundle.notes["validation"],
        bundle.structured["test"],
        bundle.notes["test"],
    )
    if train.dataset.files_paired != bundle.fusion["train"].files_paired:
        raise AssertionError("DPMM released train loader paired order changed")
    if validation.dataset.files_unpaired != bundle.fusion["validation"].files_unpaired:
        raise AssertionError("DPMM released validation loader unpaired order changed")
    if test.dataset.files_unpaired != bundle.fusion["test"].files_unpaired:
        raise AssertionError("DPMM released test loader unpaired order changed")
    return train, validation, test
