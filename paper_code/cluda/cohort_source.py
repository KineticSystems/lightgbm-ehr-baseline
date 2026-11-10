"""Exact-source helpers for CLUDA's MIMIC-IV age-shift mortality task.

The released R producer is the scientific source.  This module performs only
the released notebook's deterministic CSV-to-array work and the export
projection.  It deliberately preserves the released loader's short-window
mask defect; check_batch.py asserts that behavior against ICUDataset itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


UPSTREAM_URL = "https://github.com/oezyurty/CLUDA"
UPSTREAM_COMMIT = "60e0b10f0e967fbada08ee6c164fc7c97932c57b"
RICU_VERSION = "0.5.2"
RICU_COMMIT = "98fcbc6ecfd06bcb26cb78a2e29c7ccd1b3f9cbd"
MIMIC_VERSION = "1.0"
MIMIC_DOI = "10.13026/s6n6-xd98"
SOURCE_STATUS = (
    "paper_artifact_triangulated_split_with_integrity_pinned_scalar_los_"
    "optimization_and_mechanical_static_tail"
)

# Independently acquired from the authenticated PhysioNet AWS access point for
# ``mimiciv/1.0`` before the local release tree was validated. Pinning the
# manifest bytes here prevents an adjacent, self-consistent replacement from
# becoming its own trust anchor.
TRUSTED_MIMIC_IV_1_0_MANIFEST_SHA256 = (
    "fa9fe89ecfb5b156fb5ed27bd420476d1c22b1e45fd26ae66aef0368c8bca06b"
)

# The exact byte-size signature is a secondary check in addition to the
# trusted official manifest digest above.
V1_0_REQUIRED_FILE_SIZES = {
    "SHA256SUMS.txt": 2459,
    "core/admissions.csv.gz": 17_208_966,
    "core/patients.csv.gz": 2_955_582,
    "hosp/services.csv.gz": 9_565_293,
    "hosp/labevents.csv.gz": 2_096_142_623,
    "icu/icustays.csv.gz": 2_848_628,
    "icu/chartevents.csv.gz": 2_350_783_547,
}

# RICU 0.5.2 supports MIMIC-IV v1.0, and its configured raw-table counts plus
# the paper's 49,351-stay cohort identify v1.0 as the executable/evaluated
# release. The upstream README's v0.4 access URL is stale provenance and is
# retained as a disclosed documentation conflict rather than followed.
DATASET_VERSION_RESOLUTION = {
    "selected": "1.0",
    "released_ricu_configuration": "1.0",
    "paper_reported_processed_stays": 49_351,
    "v1_initial_filtered_stays": 49_356,
    "v0_4_initial_filtered_stays": 46_816,
    "conflict": "upstream README links MIMIC-IV v0.4",
}
RICU_CONFIGURED_ROWS = {
    "admissions": 523_740,
    "patients": 382_278,
    "services": 562_892,
    "labevents": 122_103_667,
    "icustays": 76_540,
    "chartevents": 329_499_788,
}

UPSTREAM_FILE_SHA256 = {
    "load_ICU_datasets/process_miiv.R": "1b19d6c61f012dd538003128fc81b9d8e519189cd8da4480af6cf89a1a262d46",
    "utils/dataset.py": "54486b0182ac8aaa818e71e1449023d57859d266a7c4f2c7adacbc75ddefe3ef",
    "load_ICU_datasets/process_aumc.R": "bb44bf01166766d4091de5c8055f36fd9c175ce3daed64fd2492b4123f7a1482",
    "load_ICU_datasets/feature_selection.R": "f5a289228d4d83d66f3bae45a86beb477339246cfd4c378f33dbc56f4894749f",
    "prepare_ICU_datasets/prepare_dataset_miiv.ipynb": "091cd1d9bb9825d421b24a6b18b42c173901ea9f376d393102c7504e44e64981",
    "prepare_ICU_datasets/prepare_dataset_aumc.ipynb": "5c9eadb55bca4dd5b0cdff96156d3afcd64776b45f7c48a0a3f8ba5329d16b26",
}

RELEASED_LOS_BLOCK_SHA256 = (
    "f151e16e73bc8f6e4d49fb0fccdc95e55a71aa2ab1d1f1ebe3e80b5e17c52466"
)
RELEASED_PRE_SPLIT_RNG_SHA256 = (
    "0c206a90d90b15baaaf4db06185f0716bcf8e24035561f7a4b76938ff48e3344"
)
PAPER_PRE_SPLIT_RNG_SHA256 = (
    "6dd4ba42d5e418370956f80414fafffe41a0589649706b33203454a608c140a4"
)
PAPER_SPLIT_RNG_STATUS = "paper_artifact_triangulated_split_rng_state"
PAPER_SPLIT_RNG_DRAWS = {
    "released": 37,
    "additional_time_series_loader": 35,
    "paper_evaluated": 72,
}
SOURCE_ADAPTER_FILES = (
    "analyze_split_probe.R",
    "split_rng_search_audit.json",
    "import_miiv_table.R",
    "prepare_miiv_cohort.R",
    "prepare_miiv_cohort_fast.R",
    "cluda_los_filter_fast.R",
    "cluda_split_rng_compat.R",
    "prepare_cohort_source.py",
    "cohort_source.py",
)
R_RUNTIME_VERSIONS = {
    "R": "4.1.3",
    "ricu": RICU_VERSION,
    "dplyr": "1.1.2",
    "caret": "6.0.94",
    "openssl": "2.0.6",
    "jsonlite": "1.8.5",
}
# Content that defines the cohort, and that any correct install reproduces
# byte for byte: ricu's concept dictionary and data-source configuration are
# what select and name the variables, and the package's own code and exports
# are what act on them.
#
# Deliberately not pinned here, because no independent rebuild can match them
# and none of them changes the cohort:
#   bin/Rscript
#   lib/R/library/ricu/R/ricu.rdb
#   lib/R/library/ricu/R/ricu.rdx
#   conda-meta/r-base-4.1.3-h63daf7b_11.json
#   conda-meta/r-dplyr-1.1.2-r41ha503ecb_0.json
#   conda-meta/r-caret-6.0_94-r41h133d619_0.json
#   conda-meta/r-openssl-2.0.6-r41habfbb5e_0.json
#   conda-meta/r-jsonlite-1.8.5-r41h57805ef_0.json
# bin/Rscript is a launcher with the environment prefix written into it,
# ricu.rdb/.rdx are R's lazy-load database, which is not byte-reproducible,
# and the conda-meta records embed the install path. The package versions
# below are checked by running the interpreter, which covers what these
# files were standing in for.
R_RUNTIME_FILE_SHA256 = {
    "lib/R/library/ricu/extdata/config/data-sources.json": (
        "85d2dad69cd86142fab0d9c4f20e41f8365906bf2c13640c2a29ee1b6da66cea"
    ),
    "lib/R/library/ricu/extdata/config/concept-dict.json": (
        "87e41f4e5224ffdea5bf2834efc5d832f75f2c766629a866e4778fc2dc56d369"
    ),
    "lib/R/library/ricu/R/ricu": (
        "570ca456b280cdeb201ef5ebdf22dc8f80092e2c0c68e33c7f73340e420f3759"
    ),
    "lib/R/library/ricu/NAMESPACE": (
        "3faf343c9c5f1dbc47fb266331c092ace0625b10a31a2c6b79ef023b955c0148"
    ),
}

TASK_ID = "age_mortality_g1_to_g2"
PUBLISHED_AUROC = 0.798
# Paper groups 1 and 2 are loader groups 2 and 3.  Loader group 1 is the
# pediatric 0--19.9 bin, which the paper excludes from its four-domain table.
SOURCE_AGE_GROUP = 2
TARGET_AGE_GROUP = 3
HISTORY_HOURS = 48
MIN_HOURS = 4

# Exact Table 5 presentation order.  The first 35 fields are the released
# cross-dataset availability result; the final six are the explicitly appended
# fields in feature_selection.R.
TS_FEATURES = (
    "alb",
    "alp",
    "alt",
    "ast",
    "be",
    "bicar",
    "bili",
    "bun",
    "ca",
    "cai",
    "ck",
    "cl",
    "crea",
    "dbp",
    "etco2",
    "fio2",
    "glu",
    "hct",
    "hgb",
    "hr",
    "inr_pt",
    "k",
    "lact",
    "map",
    "mg",
    "na",
    "pco2",
    "ph",
    "phos",
    "plt",
    "po2",
    "ptt",
    "sbp",
    "temp",
    "wbc",
    "basos",
    "eos",
    "lymph",
    "neut",
    "pt",
    "rbc",
)
STATIC_FEATURES = (
    "age",
    "bmi",
    "height",
    "weight",
    "adm_0",
    "adm_1",
    "adm_2",
    "sex",
)
CODES = tuple(f"STATIC/{name}" for name in STATIC_FEATURES) + tuple(
    f"CARRY/{name}" for name in TS_FEATURES
) + tuple(
    f"VALUE/{name}" for name in TS_FEATURES
) + tuple(f"TS/{name}" for name in TS_FEATURES)

AGE_MEAN = 63.55277
AGE_SD = 17.259564
# AGE_MEAN is printed to five decimal places in the released loader constant;
# half of its last printed unit is 5e-6.  The small epsilon accommodates
# decimal parsing at that rounding boundary without accepting a different
# fitted scaling distribution.
AGE_SCALING_ABS_TOLERANCE = 5.1e-6
AGE_BOUNDS = {
    1: (0.0, 19.9),
    2: (19.9, 45.0),
    3: (45.0, 65.0),
    4: (65.0, 85.0),
    5: (85.0, 120.0),
}

PAPER_PREFILTER_COUNTS = {"train": 34546, "validation": 7403, "test": 7402}
PAPER_LOADER_COUNTS = {"train": 34290, "validation": 7343, "test": 7353}
PAPER_SAMPLE_COUNTS = {
    "train": 2398546,
    "validation": 513636,
    "test": 512454,
}
PAPER_TOTAL_PREFILTER = 49351
PAPER_MORTALITY_PERCENT = 9.95

RAW_REQUIRED = (
    "SHA256SUMS.txt",
    "core/admissions.csv.gz",
    "core/patients.csv.gz",
    "hosp/services.csv.gz",
    "hosp/labevents.csv.gz",
    "icu/icustays.csv.gz",
    "icu/chartevents.csv.gz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_int_hash(values: Iterable[int]) -> str:
    return hashlib.sha256(np.asarray(list(values), dtype=np.int64).tobytes()).hexdigest()


def ordered_float32_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_json_atomic(path: Path, value: dict) -> None:
    """Write a completed manifest atomically; never expose a partial seal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_split_rng_search_audit() -> tuple[dict, dict]:
    """Validate and summarize the retained aggregate-only split search."""
    path = Path(__file__).resolve().parent / "split_rng_search_audit.json"
    if not path.is_file():
        raise FileNotFoundError("CLUDA split RNG search audit is missing")
    audit = json.loads(path.read_text())
    searched = audit.get("searched_name_draws")
    released = audit.get("released_script")
    evaluated = audit.get("paper_evaluated_state")
    probes = audit.get("runtime_probes")
    if (
        audit.get("status") != "complete_bounded_split_rng_search"
        or searched != {
            "minimum": 0,
            "maximum": 300,
            "inclusive": True,
            "candidates": 301,
        }
        or audit.get("matching_name_draws") != PAPER_SPLIT_RNG_DRAWS["paper_evaluated"]
        or not isinstance(released, dict)
        or released.get("identified_name_draws") != PAPER_SPLIT_RNG_DRAWS["released"]
        or released.get("pre_split_rng_sha256") != RELEASED_PRE_SPLIT_RNG_SHA256
        or not isinstance(evaluated, dict)
        or evaluated.get("identified_name_draws") != PAPER_SPLIT_RNG_DRAWS["paper_evaluated"]
        or evaluated.get("pre_split_rng_sha256") != PAPER_PRE_SPLIT_RNG_SHA256
        or not isinstance(probes, list)
        or len(probes) != 6
        or any(
            not isinstance(probe, dict)
            or probe.get("identified_name_draws") != PAPER_SPLIT_RNG_DRAWS["released"]
            or probe.get("pre_split_rng_sha256") != RELEASED_PRE_SPLIT_RNG_SHA256
            for probe in probes
        )
    ):
        raise AssertionError("CLUDA split RNG search audit changed or is incomplete")
    seal = {
        "file": path.name,
        "sha256": sha256(path),
        "status": audit["status"],
        "searched_name_draws": searched,
        "matching_name_draws": audit["matching_name_draws"],
        "runtime_probe_count": len(probes),
    }
    return audit, seal


def validate_upstream_files() -> dict[str, str]:
    paper_root = Path(__file__).resolve().parent
    actual = {}
    for relative, expected in UPSTREAM_FILE_SHA256.items():
        path = paper_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"sealed CLUDA upstream file is missing: {relative}")
        digest = sha256(path)
        if digest != expected:
            raise AssertionError(f"sealed CLUDA upstream file changed: {relative}")
        actual[relative] = digest
    return actual


def _workspace_relative(path: Path) -> str:
    """Report the runtime's location without embedding this checkout's path."""
    workspace = Path(__file__).resolve().parents[2]
    try:
        return str(path.resolve().relative_to(workspace))
    except ValueError:
        return str(path)


def validate_r_runtime() -> dict[str, object]:
    """Fail closed on any change to the isolated R runtime used by the export."""
    paper_root = Path(__file__).resolve().parent
    override = os.environ.get("CLUDA_R_ENV_ROOT", "").strip()
    environment_root = Path(override) if override else paper_root / ".export_assets" / "r-env"
    actual_files: dict[str, str] = {}
    for relative, expected in R_RUNTIME_FILE_SHA256.items():
        path = environment_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned CLUDA R runtime file is missing: {path}")
        digest = sha256(path)
        if digest != expected:
            raise AssertionError(f"pinned CLUDA R runtime file changed: {relative}")
        actual_files[relative] = digest

    rscript = environment_root / "bin" / "Rscript"
    if not os.access(rscript, os.X_OK):
        raise PermissionError(f"pinned CLUDA Rscript is not executable: {rscript}")
    expression = (
        'cat(as.character(getRversion()), '
        'as.character(packageVersion("ricu")), '
        'as.character(packageVersion("dplyr")), '
        'as.character(packageVersion("caret")), '
        'as.character(packageVersion("openssl")), '
        'as.character(packageVersion("jsonlite")), sep="\\t")'
    )
    completed = subprocess.run(
        [str(rscript), "--vanilla", "-e", expression],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = completed.stdout.strip().split("\t")
    if len(fields) != 6:
        raise AssertionError("pinned CLUDA R runtime returned an invalid identity")
    versions = dict(zip(("R", "ricu", "dplyr", "caret", "openssl", "jsonlite"), fields))
    if versions != R_RUNTIME_VERSIONS:
        raise AssertionError(
            f"pinned CLUDA R package versions changed: {versions!r} != {R_RUNTIME_VERSIONS!r}"
        )
    return {
        "environment": _workspace_relative(environment_root),
        "versions": versions,
        "file_sha256": actual_files,
        "ricu_source_commit": RICU_COMMIT,
        "ricu_source_config_matches_commit": True,
    }


def assert_loader_age_scaling(mean: float, sd: float) -> dict[str, float]:
    if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
        raise AssertionError("reconstructed CLUDA age scaling is not finite/positive")
    if not np.isclose(mean, AGE_MEAN, rtol=0.0, atol=AGE_SCALING_ABS_TOLERANCE):
        raise AssertionError(
            f"reconstructed training age mean {mean} differs from loader {AGE_MEAN}"
        )
    if not np.isclose(sd, AGE_SD, rtol=0.0, atol=AGE_SCALING_ABS_TOLERANCE):
        raise AssertionError(
            f"reconstructed training age SD {sd} differs from loader {AGE_SD}"
        )
    return {
        "reconstructed_training_mean": float(mean),
        "reconstructed_training_sd": float(sd),
        "released_loader_mean": AGE_MEAN,
        "released_loader_sd": AGE_SD,
        "absolute_tolerance": AGE_SCALING_ABS_TOLERANCE,
    }


def validate_raw_release(raw_root: Path) -> dict:
    """Require the v1.0 layout and verify every consumed file against its manifest."""
    raw_root = raw_root.resolve()
    if TRUSTED_MIMIC_IV_1_0_MANIFEST_SHA256 is None:
        raise RuntimeError(
            "MIMIC-IV v1.0 is unsupported until the official credentialed "
            "SHA256SUMS.txt digest is independently obtained and pinned"
        )
    missing = [name for name in RAW_REQUIRED if not (raw_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "exact MIMIC-IV v1.0 source is incomplete; missing " + ", ".join(missing)
        )
    manifest_path = raw_root / "SHA256SUMS.txt"
    manifest_digest = sha256(manifest_path)
    if manifest_digest != TRUSTED_MIMIC_IV_1_0_MANIFEST_SHA256:
        raise AssertionError("supplied manifest is not the pinned official MIMIC-IV v1.0 manifest")
    for name, expected_size in V1_0_REQUIRED_FILE_SIZES.items():
        actual_size = (raw_root / name).stat().st_size
        if actual_size != expected_size:
            raise AssertionError(
                f"MIMIC-IV v1.0 size signature differs for {name}: "
                f"{actual_size} != {expected_size}"
            )
    entries: dict[str, str] = {}
    for line in manifest_path.read_text().splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            name = fields[1].lstrip("*")
            if name.startswith("./"):
                name = name[2:]
            entries[name] = fields[0]
    hashes = {}
    for name in RAW_REQUIRED[1:]:
        expected = entries.get(name)
        if expected is None:
            raise AssertionError(f"MIMIC-IV manifest omits required v1.0 path: {name}")
        actual = sha256(raw_root / name)
        if actual != expected:
            raise AssertionError(f"MIMIC-IV source hash mismatch: {name}")
        hashes[name] = actual
    return {
        "dataset": "MIMIC-IV",
        "version": MIMIC_VERSION,
        "doi": MIMIC_DOI,
        "root": str(raw_root),
        "manifest_sha256": manifest_digest,
        "manifest_identity": "independently pinned official PhysioNet v1.0 digest",
        "size_signature": dict(V1_0_REQUIRED_FILE_SIZES),
        "consumed_file_sha256": hashes,
        "layout": "v1.0 core/hosp/icu",
        "version_resolution": dict(DATASET_VERSION_RESOLUTION),
    }


def source_files(source_root: Path) -> dict[str, Path]:
    root = source_root / "released_csv" / "miiv"
    return {
        "admissions": root / "admissions.csv",
        "timeseries": root / "ts_data_scaled.csv",
        "mask": root / "ts_data_mask.csv",
        "static": root / "static_data_scaled.csv",
        "ts_scaling": root / "ts_scaling.csv",
        "static_scaling": root / "static_scaling.csv",
        "session_info": root / "sessionInfo.txt",
        "ricu_version": root / "ricu_version.txt",
        "channel_order": root / "channel_order.txt",
        "static_order": root / "static_order.txt",
    }


def require_source(source_root: Path, *, allow_synthetic: bool = False) -> dict:
    manifest_path = source_root / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    current_upstream = validate_upstream_files()
    if manifest.get("dataset_version") != MIMIC_VERSION:
        if not (allow_synthetic and manifest.get("dataset_version") == "synthetic"):
            raise AssertionError("CLUDA export requires exact MIMIC-IV v1.0")
    if not allow_synthetic:
        if manifest.get("status") != SOURCE_STATUS:
            raise AssertionError("cached CLUDA source manifest is not complete")
        summary = manifest.get("summary")
        required_summary = {
            "prefilter_counts",
            "loader_counts",
            "sample_counts",
            "loader_positives",
            "total_prefilter",
            "ts_features",
            "static_features",
        }
        if not isinstance(summary, dict) or set(summary) != required_summary:
            raise AssertionError("cached CLUDA source has no completed source summary")
        if summary.get("ts_features") != list(TS_FEATURES) or summary.get(
            "static_features"
        ) != list(STATIC_FEATURES):
            raise AssertionError("cached CLUDA source summary changed its feature declarations")
        split_names = {"train", "validation", "test"}
        for field in ("prefilter_counts", "loader_counts", "sample_counts", "loader_positives"):
            values = summary.get(field)
            if not isinstance(values, dict) or set(values) != split_names:
                raise AssertionError(f"cached CLUDA source summary has invalid {field}")
            if any(not isinstance(value, int) or value < 0 for value in values.values()):
                raise AssertionError(f"cached CLUDA source summary has invalid {field} counts")
        if summary.get("total_prefilter") != sum(summary["prefilter_counts"].values()):
            raise AssertionError("cached CLUDA source summary total is inconsistent")

        imports = manifest.get("ricu_imports")
        if not isinstance(imports, dict) or set(imports) != set(RICU_CONFIGURED_ROWS):
            raise AssertionError("cached CLUDA source lost its six RICU import audits")
        for table, expected_rows in RICU_CONFIGURED_ROWS.items():
            audit = imports[table]
            if (
                not isinstance(audit, dict)
                or audit.get("status") != "complete"
                or audit.get("table") != table
                or int(audit.get("configured_rows", -1)) != expected_rows
                or int(audit.get("observed_rows", -1)) != expected_rows
            ):
                raise AssertionError(f"cached CLUDA RICU import audit changed: {table}")
        if TRUSTED_MIMIC_IV_1_0_MANIFEST_SHA256 is None:
            raise RuntimeError(
                "cached CLUDA sources are unsupported until the official "
                "MIMIC-IV v1.0 manifest digest is pinned"
            )
        raw_source = manifest.get("raw_source", {})
        if raw_source.get("manifest_sha256") != TRUSTED_MIMIC_IV_1_0_MANIFEST_SHA256:
            raise AssertionError("cached source is not bound to the trusted v1.0 manifest")
        if raw_source.get("version_resolution") != DATASET_VERSION_RESOLUTION:
            raise AssertionError("cached source lost the attributable dataset-version resolution")
        runtime = validate_r_runtime()
        recorded = manifest.get("r_runtime") or {}
        # `environment` is where the runtime sits, not what it is, and the
        # pinned file set has since been narrowed to the hashes an independent
        # install reproduces. Bind on the identity: package versions, the ricu
        # revision, and every file still pinned.
        if recorded.get("versions") != runtime["versions"]:
            raise AssertionError("cached source is not bound to the pinned CLUDA R versions")
        if recorded.get("ricu_source_commit") != runtime["ricu_source_commit"]:
            raise AssertionError("cached source records a different ricu revision")
        recorded_files = recorded.get("file_sha256") or {}
        for name, digest in runtime["file_sha256"].items():
            if recorded_files.get(name, digest) != digest:
                raise AssertionError(
                    f"cached source is bound to a different R runtime file: {name}"
                )
    for name, path in source_files(source_root).items():
        if not path.is_file():
            raise FileNotFoundError(f"released source boundary is missing: {path}")
        expected = manifest["files"][name]["sha256"]
        if sha256(path) != expected:
            raise AssertionError(f"released source boundary hash changed: {name}")
    if manifest.get("upstream_commit") != UPSTREAM_COMMIT:
        raise AssertionError("source manifest records the wrong CLUDA revision")
    if not allow_synthetic and manifest.get("upstream_file_sha256") != current_upstream:
        raise AssertionError("source manifest upstream-file seal changed")
    if not allow_synthetic:
        paper_root = Path(__file__).resolve().parent
        expected_adapters = manifest.get("adapter_sha256", {})
        for name in SOURCE_ADAPTER_FILES:
            expected = expected_adapters.get(name)
            if expected is None or sha256(paper_root / name) != expected:
                raise AssertionError(f"sealed source adapter changed: {name}")
        _, split_rng_search_seal = validate_split_rng_search_audit()
        if manifest.get("split_rng_search") != split_rng_search_seal:
            raise AssertionError("cached source lost its bounded split RNG search seal")
        fast_record_path = source_root / "released_csv" / "los_fast_path.json"
        fast_record = manifest.get("fast_los_filter")
        if not isinstance(fast_record, dict) or not fast_record_path.is_file():
            raise AssertionError("cached source lacks the exact fast-LOS provenance seal")
        if fast_record.get("file") != "released_csv/los_fast_path.json":
            raise AssertionError("cached source records the wrong fast-LOS audit path")
        if fast_record.get("sha256") != sha256(fast_record_path):
            raise AssertionError("cached fast-LOS audit changed")
        record = json.loads(fast_record_path.read_text())
        if record.get("status") != "released_los_loop_integrity_pinned_and_replaced":
            raise AssertionError("fast-LOS audit has the wrong status")
        if record.get("released_process_miiv_sha256") != UPSTREAM_FILE_SHA256[
            "load_ICU_datasets/process_miiv.R"
        ]:
            raise AssertionError("fast-LOS audit is not pinned to released process_miiv.R")
        if record.get("released_los_block_sha256") != RELEASED_LOS_BLOCK_SHA256:
            raise AssertionError("fast-LOS audit is not pinned to the released loop")
        split_audit = record.get("split_rng_audit")
        if (
            not isinstance(split_audit, dict)
            or split_audit.get("status") != PAPER_SPLIT_RNG_STATUS
            or split_audit.get("released_script_pre_split_rng_sha256")
            != RELEASED_PRE_SPLIT_RNG_SHA256
            or split_audit.get("paper_evaluated_pre_split_rng_sha256")
            != PAPER_PRE_SPLIT_RNG_SHA256
            or split_audit.get("released_name_draws")
            != PAPER_SPLIT_RNG_DRAWS["released"]
            or split_audit.get("additional_time_series_loader_name_draws")
            != PAPER_SPLIT_RNG_DRAWS["additional_time_series_loader"]
            or split_audit.get("paper_evaluated_name_draws")
            != PAPER_SPLIT_RNG_DRAWS["paper_evaluated"]
        ):
            raise AssertionError("cached source lost its paper-evaluated split RNG seal")
        validate_age_scaling(source_root)
    return manifest


def validate_age_scaling(source_root: Path, *, allow_synthetic: bool = False) -> dict:
    scaling = pd.read_csv(source_files(source_root)["static_scaling"])
    if scaling.columns.tolist() != ["feature", "mean", "sd"]:
        raise AssertionError("static scaling declaration changed")
    rows = scaling.loc[scaling["feature"] == "age"]
    if len(rows) != 1:
        raise AssertionError("static scaling must contain exactly one age row")
    mean = float(rows.iloc[0]["mean"])
    sd = float(rows.iloc[0]["sd"])
    if allow_synthetic:
        if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
            raise AssertionError("synthetic age scaling is invalid")
        return {"synthetic_mean": mean, "synthetic_sd": sd}
    return assert_loader_age_scaling(mean, sd)


@dataclass
class SplitArrays:
    source_split: str
    stay_ids: np.ndarray
    subject_ids: np.ndarray
    intimes: np.ndarray
    source_charttime: np.ndarray
    sequence: np.ndarray
    sequence_mask: np.ndarray
    static: np.ndarray
    mortality: np.ndarray

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray([len(value) for value in self.sequence], dtype=np.int64)


def _split_name(value: str) -> str:
    return {"train": "train", "val": "validation", "test": "test"}[value]


def _static_matrix(frame: pd.DataFrame, stay_ids: np.ndarray) -> np.ndarray:
    indexed = frame.set_index("stay_id", verify_integrity=True).loc[stay_ids]
    expected = ["age", "bmi", "height", "weight", "adm_categ", "sex_categ"]
    if indexed.columns.tolist() != expected:
        raise AssertionError(
            f"static_data_scaled.csv columns changed: {indexed.columns.tolist()}"
        )
    adm = indexed["adm_categ"].to_numpy(dtype=np.int64)
    if not np.isin(adm, [0, 1, 2]).all():
        raise AssertionError("released admission category left {0,1,2}")
    one_hot = np.eye(3, dtype=np.float64)[adm]
    return np.column_stack(
        [
            indexed[["age", "bmi", "height", "weight"]].to_numpy(np.float64),
            one_hot,
            indexed["sex_categ"].to_numpy(np.float64),
        ]
    )


def _load_author_source_files(source_root: Path) -> dict[str, SplitArrays]:
    """Parse a just-validated boundary without requiring a published manifest."""
    paths = source_files(source_root)
    admissions = pd.read_csv(paths["admissions"])
    static = pd.read_csv(paths["static"])
    values = pd.read_csv(paths["timeseries"])
    masks = pd.read_csv(paths["mask"])

    required_adm = {
        "stay_id",
        "subject_id",
        "intime",
        "hospital_expire_flag",
        "splitType",
    }
    if not required_adm.issubset(admissions.columns):
        raise AssertionError("released admissions boundary lost required identifiers")
    expected_ts = ["stay_id", "charttime", *TS_FEATURES]
    if values.columns.tolist() != expected_ts or masks.columns.tolist() != expected_ts:
        raise AssertionError("released 41-channel order differs from Table 5")
    values = values.sort_values(["stay_id", "charttime"], kind="stable").reset_index(drop=True)
    masks = masks.sort_values(["stay_id", "charttime"], kind="stable").reset_index(drop=True)
    if not np.array_equal(values[["stay_id", "charttime"]], masks[["stay_id", "charttime"]]):
        raise AssertionError("released values and mask grids are misaligned")
    grid_ids = values["stay_id"].to_numpy(np.int64)
    grid_time = values["charttime"].to_numpy(np.float64)
    grid_values = values[list(TS_FEATURES)].to_numpy(np.float64)
    grid_masks = masks[list(TS_FEATURES)].to_numpy(np.int64)
    if not np.isfinite(grid_values).all() or not np.isin(grid_masks, [0, 1]).all():
        raise AssertionError("released scaled values/masks violate their numeric domains")

    starts = np.flatnonzero(np.r_[True, grid_ids[1:] != grid_ids[:-1]])
    ends = np.r_[starts[1:], len(grid_ids)]
    ranges = {int(grid_ids[start]): (int(start), int(end)) for start, end in zip(starts, ends)}

    rng = np.random.RandomState(1234)
    result: dict[str, SplitArrays] = {}
    for released_name in ("train", "val", "test"):
        selected = admissions.loc[admissions["splitType"] == released_name]
        stay_ids = selected["stay_id"].to_numpy(np.int64).copy()
        rng.shuffle(stay_ids)
        selected = admissions.set_index("stay_id", verify_integrity=True).loc[stay_ids]
        sequence = np.empty(len(stay_ids), dtype=object)
        sequence_mask = np.empty(len(stay_ids), dtype=object)
        source_charttime = np.empty(len(stay_ids), dtype=object)
        for index, stay_id in enumerate(stay_ids):
            if int(stay_id) not in ranges:
                raise AssertionError(f"admission has no released hourly grid: {stay_id}")
            start, end = ranges[int(stay_id)]
            times = grid_time[start:end].copy()
            if not np.isfinite(times).all() or (
                len(times) > 1 and not np.equal(np.diff(times), 1.0).all()
            ) or not np.equal(times, np.rint(times)).all():
                raise AssertionError(
                    f"released hourly grid is not finite, integral, and contiguous: {stay_id}"
                )
            source_charttime[index] = times
            sequence[index] = grid_values[start:end]
            sequence_mask[index] = grid_masks[start:end]
        result[_split_name(released_name)] = SplitArrays(
            source_split=_split_name(released_name),
            stay_ids=stay_ids,
            subject_ids=selected["subject_id"].to_numpy(np.int64),
            intimes=pd.to_datetime(selected["intime"]).to_numpy(dtype="datetime64[us]"),
            source_charttime=source_charttime,
            sequence=sequence,
            sequence_mask=sequence_mask,
            static=_static_matrix(static, stay_ids),
            mortality=selected["hospital_expire_flag"].to_numpy(np.float64),
        )
    return result


def load_author_source(source_root: Path, *, allow_synthetic: bool = False) -> dict[str, SplitArrays]:
    """Recreate the released notebook arrays without persisting an NPY bundle."""
    require_source(source_root, allow_synthetic=allow_synthetic)
    return _load_author_source_files(source_root)


def age_membership(static: np.ndarray, age_group: int) -> np.ndarray:
    lower, upper = AGE_BOUNDS[age_group]
    ages = static[:, 0] * AGE_SD + AGE_MEAN
    return np.logical_and(ages > lower, ages <= upper)


def filtered_split(split: SplitArrays, age_group: int) -> SplitArrays:
    keep = np.logical_and(split.lengths >= MIN_HOURS, age_membership(split.static, age_group))
    return SplitArrays(
        source_split=split.source_split,
        stay_ids=split.stay_ids[keep],
        subject_ids=split.subject_ids[keep],
        intimes=split.intimes[keep],
        source_charttime=split.source_charttime[keep],
        sequence=split.sequence[keep],
        sequence_mask=split.sequence_mask[keep],
        static=split.static[keep],
        mortality=split.mortality[keep],
    )


def selected_splits(source: dict[str, SplitArrays]) -> dict[str, SplitArrays]:
    """Table 1 row 1: paper G1->G2, implemented as loader bins 2->3."""
    return {
        "train": filtered_split(source["train"], SOURCE_AGE_GROUP),
        "validation": filtered_split(source["validation"], SOURCE_AGE_GROUP),
        "test": filtered_split(source["test"], TARGET_AGE_GROUP),
    }


def sample_id(stay_id: int, end_index: int) -> int:
    if not 0 <= end_index < 100_000:
        raise AssertionError("released ICU sequence exceeds sample-id encoding")
    return int(stay_id) * 100_000 + int(end_index)


def iter_sample_rows(selected: dict[str, SplitArrays]):
    for split_name in ("train", "validation", "test"):
        arrays = selected[split_name]
        age_group = SOURCE_AGE_GROUP if split_name != "test" else TARGET_AGE_GROUP
        for patient_id, (
            stay_id, subject_id, intime, source_charttime, seq, mortality
        ) in enumerate(
            zip(
                arrays.stay_ids,
                arrays.subject_ids,
                arrays.intimes,
                arrays.source_charttime,
                arrays.sequence,
                arrays.mortality,
            )
        ):
            source_charttime = np.asarray(source_charttime, dtype=np.int64)
            for end_index in range(MIN_HOURS, len(seq) + 1):
                start_index = max(0, end_index - HISTORY_HOURS)
                yield {
                    "subject_id": int(subject_id),
                    "sample_id": sample_id(int(stay_id), end_index),
                    "observation_start": intime
                    + np.timedelta64(int(source_charttime[start_index]), "h"),
                    "prediction_time": intime
                    + np.timedelta64(int(source_charttime[end_index - 1]) + 1, "h"),
                    "boolean_value": bool(mortality),
                    "split": split_name,
                    "stay_id": int(stay_id),
                    "stay_hour": int(end_index),
                    "loader_patient_id": int(patient_id),
                    "source_original_split": arrays.source_split,
                    "loader_age_group": age_group,
                    "paper_age_group": age_group - 1,
                }


def source_summary(source: dict[str, SplitArrays]) -> dict:
    loader_counts = {}
    sample_counts = {}
    positives = {}
    for name, arrays in source.items():
        keep = arrays.lengths >= MIN_HOURS
        loader_counts[name] = int(keep.sum())
        sample_counts[name] = int(np.maximum(arrays.lengths[keep] - MIN_HOURS + 1, 0).sum())
        positives[name] = int(arrays.mortality[keep].sum())
    return {
        "prefilter_counts": {name: len(arrays.stay_ids) for name, arrays in source.items()},
        "loader_counts": loader_counts,
        "sample_counts": sample_counts,
        "loader_positives": positives,
        "total_prefilter": int(sum(len(value.stay_ids) for value in source.values())),
        "ts_features": list(TS_FEATURES),
        "static_features": list(STATIC_FEATURES),
    }
