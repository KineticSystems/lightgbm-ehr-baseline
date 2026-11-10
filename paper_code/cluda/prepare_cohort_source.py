"""Run CLUDA's released MIMIC-IV v1.0 producer and seal its boundary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from cohort_source import (  # noqa: E402
    DATASET_VERSION_RESOLUTION,
    MIMIC_VERSION,
    PAPER_PRE_SPLIT_RNG_SHA256,
    PAPER_SPLIT_RNG_DRAWS,
    PAPER_SPLIT_RNG_STATUS,
    RAW_REQUIRED,
    RELEASED_PRE_SPLIT_RNG_SHA256,
    RELEASED_LOS_BLOCK_SHA256,
    RICU_CONFIGURED_ROWS,
    RICU_COMMIT,
    RICU_VERSION,
    SOURCE_ADAPTER_FILES,
    SOURCE_STATUS,
    TS_FEATURES,
    STATIC_FEATURES,
    UPSTREAM_COMMIT,
    _load_author_source_files,
    sha256,
    source_files,
    source_summary,
    validate_raw_release,
    validate_r_runtime,
    validate_split_rng_search_audit,
    validate_upstream_files,
    write_json_atomic,
)


PAPER_ROOT = Path(__file__).resolve().parent

REQUIRED_TABLES = (
    "admissions",
    "patients",
    "services",
    "labevents",
    "icustays",
    "chartevents",
)
RICU_TABLE_OUTPUTS = {
    "admissions": ("admissions.fst",),
    "patients": ("patients.fst",),
    "services": ("services.fst",),
    "labevents": tuple(f"labevents/{part}.fst" for part in range(1, 11)),
    "icustays": ("icustays.fst",),
    "chartevents": tuple(f"chartevents/{part}.fst" for part in range(1, 31)),
}
def link_raw_tree(raw_root: Path, ricu_root: Path) -> None:
    target = ricu_root / "miiv"
    target.mkdir(parents=True, exist_ok=True)
    for relative_name in RAW_REQUIRED[1:]:
        relative = Path(relative_name)
        source = raw_root / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise AssertionError(f"ricu source link changed: {destination}")
        else:
            destination.symlink_to(source.resolve())


def validate_table_audit(
    table: str, audit_path: Path, miiv_dir: Path, completed_tables: tuple[str, ...]
) -> dict[str, object]:
    if not audit_path.is_file():
        raise FileNotFoundError(f"RICU import omitted table audit: {audit_path}")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "complete" or audit.get("table") != table:
        raise AssertionError(f"invalid RICU table audit for {table}")

    expected_files = RICU_TABLE_OUTPUTS[table]
    records = audit.get("files")
    if not isinstance(records, list):
        raise AssertionError(f"RICU table audit has no file list for {table}")
    observed_files = tuple(record.get("file") for record in records)
    if observed_files != expected_files:
        raise AssertionError(
            f"RICU output set changed for {table}: {observed_files!r} != {expected_files!r}"
        )
    observed_rows = 0
    for record in records:
        relative = record["file"]
        path = miiv_dir / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"RICU output is missing or is not a real file: {path}")
        rows = record.get("rows")
        if not isinstance(rows, (int, float)) or rows <= 0:
            raise AssertionError(f"RICU output has invalid row count: {path}")
        observed_rows += int(rows)
        if record.get("sha256") != sha256(path):
            raise AssertionError(f"RICU output hash changed during validation: {path}")
    if int(audit.get("observed_rows", -1)) != observed_rows:
        raise AssertionError(f"RICU total row count is inconsistent for {table}")
    configured_rows = int(audit.get("configured_rows", -1))
    if configured_rows != RICU_CONFIGURED_ROWS[table]:
        raise AssertionError(
            f"RICU 0.5.2 source configuration changed for {table}: "
            f"{configured_rows} != {RICU_CONFIGURED_ROWS[table]}"
        )
    if observed_rows != configured_rows:
        raise AssertionError(
            f"imported {table} is not the complete RICU-configured MIMIC-IV v1.0 table: "
            f"{observed_rows} != {configured_rows}"
        )

    expected_so_far = {
        relative
        for completed in completed_tables
        for relative in RICU_TABLE_OUTPUTS[completed]
    }
    actual_so_far = {
        str(path.relative_to(miiv_dir)) for path in miiv_dir.rglob("*.fst")
    }
    if actual_so_far != expected_so_far:
        raise AssertionError(
            "RICU produced an unexpected cumulative FST set: "
            f"{sorted(actual_so_far ^ expected_so_far)}"
        )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--stop-after-imports",
        action="store_true",
        help="validate/import all six exact RICU tables, then stop before cohort construction",
    )
    args = parser.parse_args()

    upstream_files = validate_upstream_files()
    r_runtime = validate_r_runtime()
    _, split_rng_search_seal = validate_split_rng_search_audit()

    manifest_path = args.source_root / "source_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("dataset_version") != MIMIC_VERSION:
            raise AssertionError("cached CLUDA source is not MIMIC-IV v1.0")
        from cohort_source import require_source

        require_source(args.source_root)
        miiv_dir = args.source_root / "ricu_data" / "miiv"
        audit_root = args.source_root / "import_audit"
        for table in REQUIRED_TABLES:
            validate_table_audit(
                table,
                audit_root / f"{table}.json",
                miiv_dir,
                REQUIRED_TABLES,
            )
        print(json.dumps({"status": "reused_exact_source", "summary": manifest["summary"]}, indent=2))
        return

    raw = validate_raw_release(args.mimic_root)
    existing_boundary = [
        path for path in source_files(args.source_root).values() if path.exists()
    ]
    if existing_boundary:
        raise FileExistsError(
            "partial unsealed CLUDA cohort boundary exists; inspect it before removing: "
            + ", ".join(str(path) for path in existing_boundary)
        )
    rscript = PAPER_ROOT / ".export_assets" / "r-env" / "bin" / "Rscript"
    ricu_root = args.source_root / "ricu_data"
    link_raw_tree(args.mimic_root.resolve(), ricu_root)
    released_root = args.source_root / "released_csv"
    released_root.mkdir(parents=True, exist_ok=True)
    r_tmp = args.source_root / "r_tmp"
    r_tmp.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "RICU_DATA_PATH": str(ricu_root),
            "RICU_SRC_LOAD": "",
            # RICU temporarily expands and repartitions the large compressed
            # tables.  Keep that scratch data beside the ignored export
            # cache instead of the workstation's quota-limited /tmp tmpfs.
            "TMPDIR": str(r_tmp),
        }
    )
    import_audit_root = args.source_root / "import_audit"
    table_audits: dict[str, dict[str, object]] = {}
    existing_audits = tuple(import_audit_root.glob("*.json"))
    unexpected_audits = {
        path.name for path in existing_audits
    } - {f"{table}.json" for table in REQUIRED_TABLES}
    if unexpected_audits:
        raise FileExistsError(f"unexpected RICU import audits: {sorted(unexpected_audits)}")
    audited_tables = [
        table
        for table in REQUIRED_TABLES
        if (import_audit_root / f"{table}.json").is_file()
    ]
    if audited_tables != list(REQUIRED_TABLES[: len(audited_tables)]):
        raise FileExistsError(
            f"RICU import audits are not a resumable prefix: {audited_tables}"
        )
    for table in audited_tables:
        audit_path = import_audit_root / f"{table}.json"
        expected_paths = tuple(
            ricu_root / "miiv" / relative for relative in RICU_TABLE_OUTPUTS[table]
        )
        if not all(path.is_file() for path in expected_paths):
            raise FileNotFoundError(f"audited RICU import is incomplete for {table}")
        table_audits[table] = validate_table_audit(
            table, audit_path, ricu_root / "miiv", tuple(audited_tables)
        )
        print(f"Revalidated completed RICU import: {table}")

    completed_tables = list(audited_tables)
    for table in REQUIRED_TABLES[len(completed_tables) :]:
        audit_path = import_audit_root / f"{table}.json"
        expected_paths = tuple(
            ricu_root / "miiv" / relative for relative in RICU_TABLE_OUTPUTS[table]
        )
        partial = [str(path) for path in expected_paths if path.exists()]
        if partial:
            raise FileExistsError(
                f"unaudited partial RICU import exists for {table}: " + ", ".join(partial)
            )
        subprocess.run(
            [
                str(rscript),
                "--vanilla",
                str(PAPER_ROOT / "import_miiv_table.R"),
                str(ricu_root),
                table,
                str(audit_path),
            ],
            check=True,
            cwd=PAPER_ROOT,
            env=environment,
        )
        completed_tables.append(table)
        table_audits[table] = validate_table_audit(
            table,
            audit_path,
            ricu_root / "miiv",
            tuple(completed_tables),
        )

    if args.stop_after_imports:
        print(
            json.dumps(
                {
                    "status": "validated_import_boundary",
                    "tables": {
                        table: int(audit["observed_rows"])
                        for table, audit in table_audits.items()
                    },
                },
                indent=2,
            )
        )
        return

    subprocess.run(
        [
            str(rscript),
            "--vanilla",
            str(PAPER_ROOT / "prepare_miiv_cohort_fast.R"),
            str(ricu_root),
            str(released_root),
        ],
        check=True,
        cwd=PAPER_ROOT,
        env=environment,
    )

    files = source_files(args.source_root)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("R producer omitted boundary files: " + ", ".join(missing))
    if files["ricu_version"].read_text().strip() != RICU_VERSION:
        raise AssertionError("R producer did not use ricu 0.5.2")
    if tuple(files["channel_order"].read_text().splitlines()) != TS_FEATURES:
        raise AssertionError("R producer channel order differs from accepted Table 5")
    if tuple(files["static_order"].read_text().splitlines()) != STATIC_FEATURES:
        raise AssertionError("R producer static order changed")

    fast_record_path = released_root / "los_fast_path.json"
    if not fast_record_path.is_file():
        raise FileNotFoundError("R producer omitted the exact fast-LOS audit")
    fast_record = json.loads(fast_record_path.read_text())
    if fast_record.get("status") != "released_los_loop_integrity_pinned_and_replaced":
        raise AssertionError("fast-LOS producer did not seal its released-loop replacement")
    if fast_record.get("released_process_miiv_sha256") != upstream_files[
        "load_ICU_datasets/process_miiv.R"
    ]:
        raise AssertionError("fast-LOS producer is not pinned to released process_miiv.R")
    if fast_record.get("released_los_block_sha256") != RELEASED_LOS_BLOCK_SHA256:
        raise AssertionError("fast-LOS producer is not pinned to the released loop")
    split_audit = fast_record.get("split_rng_audit")
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
        raise AssertionError("producer did not seal the paper-evaluated split RNG state")
    split_transform = fast_record.get("split_rng_transform")
    if (
        not isinstance(split_transform, dict)
        or split_transform.get("status")
        != "released_split_boundary_integrity_pinned_and_reconstructed"
        or split_transform.get("released_process_miiv_sha256")
        != upstream_files["load_ICU_datasets/process_miiv.R"]
        or split_transform.get("released_script_pre_split_rng_sha256")
        != RELEASED_PRE_SPLIT_RNG_SHA256
        or split_transform.get("paper_evaluated_pre_split_rng_sha256")
        != PAPER_PRE_SPLIT_RNG_SHA256
    ):
        raise AssertionError("producer did not seal its split-boundary transformation")

    manifest = {
        "status": SOURCE_STATUS,
        "dataset_version": MIMIC_VERSION,
        "dataset_version_resolution": DATASET_VERSION_RESOLUTION,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_file_sha256": upstream_files,
        "ricu": {"version": RICU_VERSION, "time_compatible_commit": RICU_COMMIT},
        "r_runtime": r_runtime,
        "ricu_imports": table_audits,
        "raw_source": raw,
        "files": {
            name: {"file": str(path.relative_to(args.source_root)), "sha256": sha256(path)}
            for name, path in files.items()
        },
        "adapter_sha256": {
            name: sha256(PAPER_ROOT / name) for name in SOURCE_ADAPTER_FILES
        },
        "fast_los_filter": {
            "file": str(fast_record_path.relative_to(args.source_root)),
            "sha256": sha256(fast_record_path),
            "filter_audit": fast_record["filter_audit"],
            "released_process_miiv_sha256": fast_record[
                "released_process_miiv_sha256"
            ],
            "released_los_block_sha256": fast_record[
                "released_los_block_sha256"
            ],
            "split_rng_audit": split_audit,
            "split_rng_transform": split_transform,
        },
        "split_rng_search": split_rng_search_seal,
        "summary": {},
    }
    source = _load_author_source_files(args.source_root)
    manifest["summary"] = source_summary(source)
    write_json_atomic(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "summary": manifest["summary"]}, indent=2))


if __name__ == "__main__":
    main()
