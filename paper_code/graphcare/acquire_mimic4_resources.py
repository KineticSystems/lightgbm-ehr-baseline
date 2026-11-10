#!/usr/bin/env python3
"""Acquire and pin the four extra PyHealth resources the MIMIC-IV path needs.

`data_prepare.py:65-77` adds `ICD10CM -> CCSCM` and `ICD10PROC -> CCSPROC` to
the MIMIC-IV code mapping.  PyHealth 1.1.2 downloads those unversioned CSVs
from `storage.googleapis.com/pyhealth/resource/` at first use.  The code audit pinned
the nine MIMIC-III resources the same way; this script does the same for the
four MIMIC-IV-only resources, recording size, line count, server metadata, the
MD5 the bucket reports as the object ETag, and a SHA-256 for every later run.

The downloaded bytes are the *currently served* payload, not proven identical
to the crosswalk bytes GraphCare used in 2023.  That residual historical
uncertainty is recorded in the exported audits.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parent
TARGET = (
    PAPER_ROOT / ".export_assets" / "external_artifacts" / "pyhealth_resource_mimic4"
)
BASE_URL = "https://storage.googleapis.com/pyhealth/resource/"
NAMES = (
    "ICD10CM.csv",
    "ICD10PROC.csv",
    "ICD10CM_to_CCSCM.csv",
    "ICD10PROC_to_CCSPROC.csv",
)


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    report = {"retrieved": datetime.now(timezone.utc).isoformat(), "files": {}}
    digests = {}
    for name in NAMES:
        path = TARGET / name
        with urllib.request.urlopen(BASE_URL + name, timeout=300) as response:
            headers = dict(response.headers)
            payload = response.read()
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        digests[name] = digest
        report["files"][name] = {
            "bytes": len(payload),
            "lines": payload.count(b"\n"),
            "header": payload.split(b"\n", 1)[0].decode("utf-8").strip(),
            "md5": hashlib.md5(payload).hexdigest(),
            "etag": headers.get("ETag", "").strip('"'),
            "last_modified": headers.get("Last-Modified"),
            "sha256": digest,
        }
        if report["files"][name]["md5"] != report["files"][name]["etag"]:
            raise AssertionError(f"{name}: MD5 does not match the bucket ETag")
    report["sha256"] = digests
    (TARGET / "SHA256SUMS.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
