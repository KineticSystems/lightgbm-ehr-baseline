#!/usr/bin/env python3
"""Independently verify DDL-CXR seed artifacts, metrics, and aggregation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


EXPECTED = {
    0: {
        "checkpoint": "0e9b399876af747587bea6b2f2af3557a0cf8385a7c58a9b5de6f021e50cc0b4",
        "console": "4e97ad88b49c0df861f2d54c1cbab6856b85dc24c351c71dbfe6a690799d705e",
        "predictions": "de96b3ffc50c0c2f4b1ff2384522f4c39a154606834953ef92cccae66dfb44df",
        "result": "6725b1d98ef8a20b5e73f6f4ce40de9739473fafbc43421423babb04fe84c73d",
        "trace": "277b16ee3762ee2554da063b500683858f16dafcb9114c729a7b0fa592364014",
    },
    1: {
        "checkpoint": "0e5b5f278fb77e7e5737d660e5ac2f545d7ad737a4fdaafdf28f0ef59ab50922",
        "console": "77d5d4056ca159a1c9a93f81e6a2848826c97f3ecf641e9fd947f76f5b790ee6",
        "predictions": "c653996533b50c142627b465a2ae3ec2d69466c257e002f1d54fcdaf895b9fa4",
        "result": "8f19039db17581f12a43b45b95d9456700a74c3286cd09eb1f865b1f5c633ec2",
        "trace": "a374ec10343b5f6213f31516242f3b580f273cba5056e39527baacf74429748f",
    },
    2: {
        "checkpoint": "afced2a06e0bffeaa4f5b9d72c968f64bf42ab70fe67af09a295d6362e25db71",
        "console": "3e4aad8c3ab2413509a13825bd0e21409b975573734d76e897b0ebe139340f31",
        "predictions": "d0bc5247ac2abe3816fb5fda0d09dc82162ca6008fac097d09aa0c3640f4c5db",
        "result": "c7db8736ae6c3f7d5073e309fff35906194e442bcaa45e76d86bc1a528a7bddc",
        "trace": "d911e97f0ddbfee23a340fbc4bd7788585187be97b1218839ef87e892b19d9dd",
    },
    3: {
        "checkpoint": "733484d6b8e2c8ad97921f569ec460aa756a00da9e8d40d09fb822a1a9a440cd",
        "console": "967bdf5543e677d86544d014642e8413dcd5ba2cd56d9e694b4b34132069f9b9",
        "predictions": "1799602e0f0dbd604fda17c89b3b2a48511c3798a8e3c61b08d45f9b6ffca3fa",
        "result": "4d354821833723fa5c96b6cfcf91c21d8890b3b3b52c7f0773d17fe992f61a9f",
        "trace": "906983b65b83ebc3e51aaae1f1751d8fc6ad72b59c36e27a62d3e27842417ea8",
    },
    4: {
        "checkpoint": "678ce2bab07eff4ff50b5a0061e34168583a245cc275cc2745c1b136f7b2b55a",
        "console": "305fa298c835bafc57c3efe76ed1f29db7a05425b0b64939c26ec2123f0bf7bb",
        "predictions": "3998ca0d00224524d417f62500c61d471b1489aff925b4158c255af16fef2f97",
        "result": "9409eb382b60d6c544352868316cd9a962e292dc7219c588ae66c27eb9ff75d2",
        "trace": "3f534bd42584eb9d168912ec0dc1450823b0fff0d5d9cf370eb0eb3158528157",
    },
}
CONFIG_SHA = "2a5599fbd9ea90b7cad4413a49968af596546851062bbfd5dda4547852f3a4cf"
TEST_CSV_SHA = "d7f194d5308b61b40aebeabdd803df8367ec5063d623c745d54cd3f7aceffc07"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "population_sd": float(values.std(ddof=0)),
        "sample_sd": float(values.std(ddof=1)),
        "sem": float(values.std(ddof=1) / math.sqrt(len(values))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs_root = args.runs_root.resolve(strict=True)
    test_csv = args.test_csv.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing prior aggregate verification: {output}")
    assert sha256(test_csv) == TEST_CSV_SHA
    expected_frame = pd.read_csv(test_csv)
    expected_ids = np.arange(1483, dtype=np.int64)
    expected_targets = expected_frame["mortality_inhospital"].to_numpy(dtype=np.int64)
    assert expected_frame["index"].to_numpy(dtype=np.int64).tolist() == expected_ids.tolist()
    assert int(expected_targets.sum()) == 219

    seeds: list[dict[str, object]] = []
    state_keys = None
    for seed, expected in EXPECTED.items():
        run_root = runs_root / f"seed_{seed}"
        artifacts = run_root / "artifacts"
        paths = {
            "checkpoint": artifacts / "best.ckpt",
            "console": run_root / "console.log",
            "predictions": artifacts / "test_predictions.csv",
            "result": artifacts / "result.json",
            "trace": artifacts / "epochs.jsonl",
        }
        for name, path in paths.items():
            assert sha256(path) == expected[name], (seed, name)
        assert sha256(artifacts / "resolved_config.yaml") == CONFIG_SHA

        result = json.loads(paths["result"].read_text())
        assert result["status"] == "complete"
        assert result["seed"] == seed
        assert result["prediction_sha256"] == expected["predictions"]
        assert result["checkpoint"]["sha256"] == expected["checkpoint"]
        assert result["test"]["rows"] == 1483
        assert result["test"]["positives"] == 219

        frame = pd.read_csv(paths["predictions"])
        assert list(frame.columns) == ["sample_id", "target", "logit", "probability"]
        ids = frame["sample_id"].to_numpy(dtype=np.int64)
        targets = frame["target"].to_numpy(dtype=np.int64)
        logits = frame["logit"].to_numpy(dtype=np.float32)
        probabilities = frame["probability"].to_numpy(dtype=np.float32)
        assert np.array_equal(ids, expected_ids)
        assert np.array_equal(targets, expected_targets)
        assert np.isfinite(logits).all() and np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert np.allclose(
            torch.sigmoid(torch.from_numpy(logits)).numpy(),
            probabilities,
            rtol=1e-6,
            atol=1e-7,
        )
        assert np.array_equal(np.argsort(logits), np.argsort(probabilities))
        auroc = float(roc_auc_score(targets, probabilities))
        auprc = float(average_precision_score(targets, probabilities))
        assert auroc == result["test"]["auroc"]
        assert auprc == result["test"]["auprc"]
        assert float(roc_auc_score(targets, logits)) == auroc
        assert float(average_precision_score(targets, logits)) == auprc

        records = [json.loads(line) for line in paths["trace"].read_text().splitlines()]
        assert len(records) == result["epochs_completed"]
        assert [record["epoch"] for record in records] == list(range(len(records)))
        for epoch, record in enumerate(records):
            assert record["global_step"] == (epoch + 1) * 81
            assert record["train_rows"] == 5142
            assert record["validation_rows"] == 861
        best = max(records, key=lambda record: record["validation_auprc"])
        assert best["epoch"] == result["best_epoch_zero_based"]
        assert best["validation_auprc"] == result["best_validation_auprc"]
        assert records[-1]["wait_count"] == 10
        assert result["stopped_early"] is True

        payload = torch.load(
            paths["checkpoint"], map_location="cpu", weights_only=True, mmap=True
        )
        assert payload["epoch"] == best["epoch"]
        assert payload["global_step"] == (best["epoch"] + 1) * 81
        assert payload["validation_auprc"] == best["validation_auprc"]
        keys = tuple(sorted(payload["state_dict"]))
        if state_keys is None:
            state_keys = keys
        else:
            assert keys == state_keys
        finite_tensors = 0
        for tensor in payload["state_dict"].values():
            if torch.is_floating_point(tensor):
                assert torch.isfinite(tensor).all()
                finite_tensors += 1
        assert finite_tensors > 0
        del payload
        gc.collect()

        seeds.append(
            {
                "artifacts": {
                    **expected,
                    "config": CONFIG_SHA,
                },
                "auprc": auprc,
                "auroc": auroc,
                "best_epoch_zero_based": best["epoch"],
                "best_validation_auprc": best["validation_auprc"],
                "checkpoint_state_keys": len(keys),
                "epochs": len(records),
                "seed": seed,
                "status": "exact",
            }
        )

    auroc_values = np.asarray([seed["auroc"] for seed in seeds], dtype=np.float64)
    auprc_values = np.asarray([seed["auprc"] for seed in seeds], dtype=np.float64)
    auroc = summary(auroc_values)
    auprc = summary(auprc_values)
    auroc_band = [0.813, 0.831]
    auprc_band = [0.512, 0.534]
    auroc_match = auroc_band[0] <= auroc["mean"] <= auroc_band[1]
    auprc_match = auprc_band[0] <= auprc["mean"] <= auprc_band[1]
    report = {
        "aggregate": {
            "auprc": {
                **auprc,
                "paper_one_sd_band": auprc_band,
                "paper_point": 0.523,
                "signed_gap": auprc["mean"] - 0.523,
                "within_band": auprc_match,
            },
            "auroc": {
                **auroc,
                "lightgbm": 0.8327011444425177,
                "signed_gap_to_lightgbm": auroc["mean"] - 0.8327011444425177,
                "paper_one_sd_band": auroc_band,
                "paper_point": 0.822,
                "signed_gap": auroc["mean"] - 0.822,
                "within_band": auroc_match,
            },
            "classification": "descriptive_match" if auroc_match and auprc_match else "nonmatch",
            "rule": "mean must lie in both prespecified paper one-SD bands",
        },
        "checkpoint_state_key_set": "identical across seeds",
        "seeds": seeds,
        "status": "exact",
        "test_csv_sha256": TEST_CSV_SHA,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
