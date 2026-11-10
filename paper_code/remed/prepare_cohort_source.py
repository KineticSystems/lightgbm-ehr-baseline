"""Run the exact attributed snub producer for REMed's 48-hour MIMIC-IV input."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parent
SNUB_COMMIT = "bab6c07acce4a617153fba7da4f68bc32d20652f"
SNUB_MIMICIV_ORIGINAL_SHA256 = (
    "d8aabd589497b9f6efd88424212445d3657301114aaa013ecc7582353c484d7d"
)
SNUB_MIMICIV_PATCHED_SHA256 = (
    "a33827ffedc697031b8fdb29c3259ce0fdc7c603d485718be813d844543a92cd"
)
REQUIRED_MAIN = (
    "hosp/admissions.csv.gz",
    "hosp/patients.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "hosp/d_labitems.csv.gz",
    "hosp/labevents.csv.gz",
    "hosp/prescriptions.csv.gz",
    "hosp/microbiologyevents.csv.gz",
    "icu/icustays.csv.gz",
    "icu/d_items.csv.gz",
    "icu/inputevents.csv.gz",
    "icu/chartevents.csv.gz",
    "icu/outputevents.csv.gz",
    "icu/procedureevents.csv.gz",
)
REQUIRED_ED = (
    "ed/edstays.csv.gz",
    "ed/medrecon.csv.gz",
    "ed/pyxis.csv.gz",
    "ed/vitalsign.csv.gz",
    "ed/diagnosis.csv.gz",
    "ed/triage.csv.gz",
)
SPARK_EXCLUDED_OPTIMIZER_RULES = (
    "org.apache.spark.sql.catalyst.optimizer.ConvertToLocalRelation",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def root_identity(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()


def spark_worker_environment(_unused) -> tuple[str, str]:
    """Return worker facts before the expensive author pipeline starts."""
    import platform as worker_platform
    from pathlib import Path as WorkerPath

    import ehrs as worker_ehrs

    return (
        worker_platform.python_version(),
        str(WorkerPath(worker_ehrs.__file__).resolve().parent.parent),
    )


def source_configuration(threads: int) -> dict[str, object]:
    return {
        "ehr": "mimiciv",
        "obs_size": 48,
        "pred_size": 48,
        "max_patient_token_len": 2147483647,
        "max_event_size": 2147483647,
        "use_more_tables": True,
        "readmission": True,
        "diagnosis": True,
        "min_event_size": 0,
        "split_seeds": [2020, 2021, 2022, 2023, 2024],
        "use_ed": True,
        "valid_percent": 0.1,
        "threads": threads,
        "spark_driver_memory": "14g",
        "spark_optimizer_excluded_rules": list(SPARK_EXCLUDED_OPTIMIZER_RULES),
        "spark_sql_ansi_enabled": False,
        "spark_python": "paper_venv_python_3_10_for_driver_and_workers",
        "spark_pythonpath": "pinned_snub_root_for_driver_and_workers",
        "guard_tasks_max": 512,
    }


def parse_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"release manifest is missing: {path}")
    result = {}
    for line in path.read_text().splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        result[fields[1].lstrip("*./")] = fields[0].lower()
    return result


def verify_release(root: Path, required: tuple[str, ...]) -> dict[str, str]:
    manifest = parse_manifest(root / "SHA256SUMS.txt")
    verified = {}
    for relative in required:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required release file is missing: {path}")
        expected = manifest.get(relative)
        if expected is None:
            raise ValueError(f"release manifest does not name {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"release checksum differs for {path}")
        verified[relative] = actual
    return verified


def ensure_link(path: Path, target: Path) -> None:
    target = target.resolve()
    if path.is_symlink():
        if path.resolve() != target:
            raise ValueError(f"existing source link has the wrong target: {path}")
        return
    if path.exists():
        raise FileExistsError(f"refusing to replace existing source path: {path}")
    path.symlink_to(target, target_is_directory=True)


def ensure_pandas_datetime_compatibility(snub_root: Path) -> dict[str, str]:
    path = snub_root / "ehrs" / "mimiciv.py"
    before = sha256_file(path)
    if before == SNUB_MIMICIV_ORIGINAL_SHA256:
        source = path.read_text()
        old = 'icustays.loc[:, "INTIME"] = pd.to_datetime('
        new = 'icustays["INTIME"] = pd.to_datetime('
        if source.count(old) != 1 or new in source:
            raise ValueError("unexpected snub datetime-assignment source context")
        path.write_text(source.replace(old, new, 1))
    elif before != SNUB_MIMICIV_PATCHED_SHA256:
        raise ValueError("snub MIMIC-IV producer differs from the pinned source/patch")
    after = sha256_file(path)
    if after != SNUB_MIMICIV_PATCHED_SHA256:
        raise ValueError("snub pandas datetime compatibility patch has the wrong hash")
    return {
        "reason": "preserve datetime dtype under pandas 2.x direct-column assignment",
        "original_sha256": SNUB_MIMICIV_ORIGINAL_SHA256,
        "patched_sha256": SNUB_MIMICIV_PATCHED_SHA256,
    }


def expected_build(
    args, raw_hashes, artifact_hashes, compatibility_patch
) -> dict[str, object]:
    return {
        "status": "complete",
        "producer": {
            "repository": "https://github.com/Jwoo5/integrated-ehr-pipeline",
            "commit": SNUB_COMMIT,
            "entrypoint": "EHR_REGISTRY['mimiciv'].run_pipeline",
        },
        "data": {
            "mimic_iv_release": "2.0",
            "mimic_iv_root_variable": "MIMIC_IV_2_0_ROOT",
            "mimic_iv_root_identity_sha256": root_identity(args.mimic_root),
            "mimic_iv_ed_release": "2.0",
            "mimic_iv_ed_root_variable": "MIMIC_IV_ED_2_0_ROOT",
            "mimic_iv_ed_root_identity_sha256": root_identity(args.ed_root),
            "required_file_sha256": raw_hashes,
        },
        "artifacts": artifact_hashes,
        "compatibility_patches": {
            "pandas_datetime_assignment": compatibility_patch,
        },
        "configuration": source_configuration(args.threads),
        "software": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in (
                    "h5py",
                    "numpy",
                    "pandas",
                    "pyspark",
                    "transformers",
                )
            },
            "java": "Temurin-21.0.12.1+1",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--ed-root", type=Path, required=True)
    parser.add_argument("--snub-root", type=Path, required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--java-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        raise ValueError("the export permits 1-4 guarded Spark threads")
    if args.snub_root.name != (
        "integrated-ehr-pipeline-" + SNUB_COMMIT
    ):
        raise ValueError("snub source directory does not identify the pinned commit")
    compatibility_patch = ensure_pandas_datetime_compatibility(args.snub_root)

    build_path = args.source_root / "source_build.json"
    source_files = (
        args.source_root / "mimiciv.h5",
        args.source_root / "mimiciv_cohort.csv",
    )
    if build_path.is_file() and all(path.is_file() for path in source_files):
        previous = json.loads(build_path.read_text())
        data = previous.get("data", {})
        producer = previous.get("producer", {})
        configuration = previous.get("configuration", {})
        if (
            previous.get("status") != "complete"
            or producer.get("commit") != SNUB_COMMIT
            or data.get("mimic_iv_release") != "2.0"
            or data.get("mimic_iv_ed_release") != "2.0"
            or data.get("mimic_iv_root_identity_sha256")
            != root_identity(args.mimic_root)
            or data.get("mimic_iv_ed_root_identity_sha256")
            != root_identity(args.ed_root)
            or configuration != source_configuration(args.threads)
        ):
            raise ValueError(
                "existing REMed source build has different provenance/configuration"
            )
        print(f"Reusing exact attributed source build: {args.source_root}")
        return
    if build_path.exists() or any(path.exists() for path in source_files):
        raise FileExistsError("partial REMed source build exists; inspect it before retrying")

    raw_hashes = {
        "mimic_iv_2_0": verify_release(args.mimic_root, REQUIRED_MAIN),
        "mimic_iv_ed_2_0": verify_release(args.ed_root, REQUIRED_ED),
    }
    artifacts = {
        "ccs_multi_dx_tool_2015.csv": args.mapping_root
        / "ccs_multi_dx_tool_2015.csv",
        "icd10cmtoicd9gem.csv": args.mapping_root / "icd10cmtoicd9gem.csv",
        "bioclinicalbert_config.json": args.tokenizer_root / "config.json",
        "bioclinicalbert_vocab.txt": args.tokenizer_root / "vocab.txt",
        "snub_main.py": args.snub_root / "main.py",
        "snub_ehrs_ehr.py": args.snub_root / "ehrs" / "ehr.py",
        "snub_ehrs_mimiciv.py": args.snub_root / "ehrs" / "mimiciv.py",
        "temurin_release": args.java_root / "release",
        "temurin_jdk_incubator_vector.jmod": args.java_root
        / "jmods"
        / "jdk.incubator.vector.jmod",
    }
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("required producer artifacts are missing: " + ", ".join(missing))
    if 'JAVA_VERSION="21.0.12.1"' not in artifacts["temurin_release"].read_text():
        raise ValueError("paper-local Temurin runtime is not the pinned Java 21 build")
    artifact_hashes = {
        name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in artifacts.items()
    }
    build = expected_build(
        args, raw_hashes, artifact_hashes, compatibility_patch
    )
    args.work_root.mkdir(parents=True, exist_ok=True)
    combined = args.work_root / "mimiciv_2_0_with_ed_2_0"
    combined.mkdir(exist_ok=True)
    ensure_link(combined / "hosp", args.mimic_root / "hosp")
    ensure_link(combined / "icu", args.mimic_root / "icu")
    ensure_link(combined / "ed", args.ed_root / "ed")
    cache_root = args.work_root / "snub_cache"
    spark_root = args.work_root / "spark_local"
    cache_root.mkdir(exist_ok=True)
    spark_root.mkdir(exist_ok=True)
    args.source_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.snub_root))
    import ehrs.ehr as ehr_module
    from ehrs import EHR_REGISTRY
    from main import get_parser
    from pyspark.sql import SparkSession

    original_from_pretrained = ehr_module.AutoTokenizer.from_pretrained
    original_expanduser = ehr_module.os.path.expanduser

    def load_pinned_tokenizer(*_positional, **_keywords):
        return original_from_pretrained(
            str(args.tokenizer_root), local_files_only=True
        )

    def expand_export_cache(path):
        if path == "~/.cache/ehr":
            return str(cache_root)
        return original_expanduser(path)

    ehr_module.AutoTokenizer.from_pretrained = load_pinned_tokenizer
    ehr_module.os.path.expanduser = expand_export_cache
    source_args = get_parser().parse_args(
        [
            "--ehr",
            "mimiciv",
            "--data",
            str(combined),
            "--ccs",
            str(artifacts["ccs_multi_dx_tool_2015.csv"]),
            "--gem",
            str(artifacts["icd10cmtoicd9gem.csv"]),
            "--obs_size",
            "48",
            "--pred_size",
            "48",
            "--max_patient_token_len",
            "2147483647",
            "--max_event_size",
            "2147483647",
            "--use_more_tables",
            "--dest",
            str(args.source_root),
            "--num_threads",
            str(args.threads),
            "--readmission",
            "--diagnosis",
            "--min_event_size",
            "0",
            "--seed",
            "2020, 2021, 2022, 2023, 2024",
            "--use_ed",
        ]
    )
    try:
        ehr = EHR_REGISTRY["mimiciv"](source_args)
    finally:
        ehr_module.os.path.expanduser = original_expanduser
    ehr.cache_dir = str(cache_root)
    spark = (
        SparkSession.builder.master(f"local[{args.threads}]")
        .config("spark.driver.memory", "14g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.network.timeout", "300s")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.timeZone", "UTC")
        .config(
            "spark.sql.optimizer.excludedRules",
            ",".join(SPARK_EXCLUDED_OPTIMIZER_RULES),
        )
        .config("spark.local.dir", str(spark_root))
        .appName("REMed_Export_Exact_Snub")
        .getOrCreate()
    )
    try:
        worker_python, worker_snub_root = (
            spark.sparkContext.parallelize([None], 1)
            .map(spark_worker_environment)
            .collect()[0]
        )
        if worker_python != platform.python_version():
            raise RuntimeError(
                "Spark worker Python does not match the paper driver: "
                f"{worker_python} != {platform.python_version()}"
            )
        if Path(worker_snub_root).resolve() != args.snub_root.resolve():
            raise RuntimeError(
                "Spark worker imported ehrs from the wrong source: "
                f"{worker_snub_root} != {args.snub_root}"
            )
        ehr.run_pipeline(spark)
    finally:
        spark.stop()
    if not all(path.is_file() for path in source_files):
        raise RuntimeError("snub producer returned without both expected source files")
    build_path.write_text(json.dumps(build, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(cache_root, ignore_errors=True)
    shutil.rmtree(spark_root, ignore_errors=True)
    print(f"Built exact attributed source at {args.source_root}")


if __name__ == "__main__":
    main()
