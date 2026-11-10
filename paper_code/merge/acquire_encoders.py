#!/usr/bin/env python3
"""Acquire and hash the exact public encoder artifacts used for MERGE."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil

from huggingface_hub import HfApi, snapshot_download
import torchxrayvision as xrv


EXPECTED_BIOBERT_FILES = {
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
}
CXR_ALIAS = "densenet121-res224-chex"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(root).parts:
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def copy_exact(source: Path, target: Path) -> None:
    if target.exists():
        if sha256(target) != sha256(source):
            raise RuntimeError(f"refusing to overwrite mismatched artifact: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--biobert-revision", required=True)
    parser.add_argument("--cxr-source", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    biobert_root = output_root / "biobert-v1.1"
    cxr_root = output_root / "torchxrayvision-chex"
    hf_cache = output_root / "huggingface-cache"
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.local_files_only:
        resolved = HfApi().model_info(
            "dmis-lab/biobert-v1.1", revision=args.biobert_revision
        ).sha
        if resolved != args.biobert_revision:
            raise RuntimeError(
                f"BioBERT revision mismatch: {resolved} != {args.biobert_revision}"
            )

    snapshot_download(
        repo_id="dmis-lab/biobert-v1.1",
        revision=args.biobert_revision,
        local_dir=biobert_root,
        cache_dir=hf_cache,
        allow_patterns=sorted(EXPECTED_BIOBERT_FILES),
        local_files_only=args.local_files_only,
    )

    actual_biobert = {
        path.relative_to(biobert_root).as_posix()
        for path in biobert_root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(biobert_root).parts
    }
    missing = EXPECTED_BIOBERT_FILES - actual_biobert
    if missing:
        raise RuntimeError(f"BioBERT snapshot is missing: {sorted(missing)}")

    if not args.cxr_source.is_file():
        raise FileNotFoundError(args.cxr_source)
    cxr_filename = Path(
        xrv.models.model_urls[CXR_ALIAS]["weights_url"]
    ).name
    cxr_target = cxr_root / cxr_filename
    copy_exact(args.cxr_source.resolve(), cxr_target)

    manifest = {
        "biobert": {
            "repo_id": "dmis-lab/biobert-v1.1",
            "revision": args.biobert_revision,
            "files": inventory(biobert_root),
        },
        "cxr": {
            "alias": CXR_ALIAS,
            "weights_url": xrv.models.model_urls[CXR_ALIAS]["weights_url"],
            "files": inventory(cxr_root),
        },
        "packages": {
            "huggingface-hub": version("huggingface-hub"),
            "torchxrayvision": version("torchxrayvision"),
            "transformers": version("transformers"),
        },
    }
    manifest_path = output_root / "encoder_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
