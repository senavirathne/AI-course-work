"""One planned held-out evaluation of frozen ANN/SVM/reference/VLM comparisons."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
import torch

from fabric_inspector.common import (
    code_hash,
    environment,
    now,
    read_json,
    resolve_under,
    sha256,
    write_json,
)
from fabric_inspector.data.images import preprocess
from fabric_inspector.data.manifest import REAL, load_manifest
from fabric_inspector.evaluation.metrics import metrics, group_bootstrap
from fabric_inspector.genai.vlm import LocalVLM, benchmark
from fabric_inspector.models.classical import texture_features
from fabric_inspector.serving.predictor import adjusted_scores
from fabric_inspector.training.runner import load_checkpoint, synchronize
from xai_validate import evaluate_cases


def evaluate_run(run, name, rows, root, output, bundle=None):
    path = output / run["checkpoint"]
    is_ann = path.suffix == ".pt"
    if is_ann:
        model, metadata, device = load_checkpoint(path, expected_sha=run["checkpoint_sha256"])

        def predict(image):
            tensor = preprocess(image, metadata["preprocessing"]).unsqueeze(0).to(device)
            with torch.inference_mode():
                synchronize(device)
                started = time.perf_counter()
                score = float(torch.sigmoid(model(tensor))[0])
                synchronize(device)
                model_ms = (time.perf_counter() - started) * 1000
            return score, model_ms
    else:
        loaded = joblib.load(path)  # Own frozen artifacts only; never accept an uploaded pickle.

        def predict(image):
            features = texture_features(image, loaded["feature_mode"]).reshape(1, -1)
            started = time.perf_counter()
            score = float(loaded["pipeline"].decision_function(features)[0])
            return score, (time.perf_counter() - started) * 1000

    for row in rows[:3]:
        with Image.open(resolve_under(root, row["image_path"])) as image:
            predict(image)
    scores, labels, end_times, model_times, predictions = [], [], [], [], []
    for row in rows:
        started = time.perf_counter()
        with Image.open(resolve_under(root, row["image_path"])) as image:
            image.load()
            score, model_ms = predict(image)
        if bundle:
            score = float(adjusted_scores([score], bundle["temperature"])[0])
        threshold = bundle["classification_threshold"] if bundle else run["validation_threshold"]
        decision = int(score >= threshold)
        elapsed = (time.perf_counter() - started) * 1000
        scores.append(score)
        labels.append(int(row["label"]))
        end_times.append(elapsed)
        model_times.append(model_ms)
        predictions.append(
            {
                "patch_id": row["patch_id"],
                "label": int(row["label"]),
                "source_id": row["source_id"],
                "group_id": row["group_id"],
                "defect_pixels": int(row["defect_pixels"]),
                "score": score,
                "prediction": decision,
                "end_to_end_ms": elapsed,
                "model_only_ms": model_ms,
            }
        )
    result = {
        "model": name,
        "run_id": run["run_id"],
        "partition": "test",
        "provenance": REAL,
        "metrics": metrics(labels, [r["prediction"] for r in predictions], scores),
        "training_seconds": run["training_seconds"],
        "predictions": predictions,
        "timing": {
            "batch_size": 1,
            "warmup": min(3, len(rows)),
            "n": len(rows),
            "mean_end_to_end_ms": float(np.mean(end_times)),
            "p95_end_to_end_ms": float(np.quantile(end_times, 0.95)),
            "mean_model_only_ms": float(np.mean(model_times)),
            "scope": "End-to-end includes file decode and preprocessing; excludes process/network startup",
        },
        "uncertainty": group_bootstrap(
            labels, [r["prediction"] for r in predictions], [r["group_id"] for r in rows]
        ),
    }
    if is_ann:
        result["brier_score"] = float(np.mean((np.asarray(scores) - np.asarray(labels)) ** 2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    frozen_path = args.output / "artifacts" / "final-evaluation-manifest.json"
    frozen = read_json(frozen_path)
    if frozen["code_sha256"] != code_hash(Path(".")) or frozen["provenance"] != REAL:
        raise ValueError("Executable code or data provenance differs from the frozen protocol")
    manifest = (frozen_path.parent / frozen["manifest"]).resolve()
    if sha256(manifest) != frozen["manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    split, all_rows, root = load_manifest(manifest, verify_files=True)
    if split["split_sha256"] != frozen["split_sha256"]:
        raise ValueError("Split changed after freeze")
    rows = [r for r in all_rows if r["split"] == "test"]
    if [r["patch_id"] for r in rows] != frozen["test_patch_ids"]:
        raise ValueError("Test examples changed")
    for name, digest in frozen["artifact_sha256"].items():
        if sha256(resolve_under(frozen_path.parent, name)) != digest:
            raise ValueError(f"Frozen artifact changed: {name}")
    # Load the actual VLM before marking test access; fail early if its weights are unavailable.
    adapter = LocalVLM(frozen["vlm"]["config"])
    result_dir = args.output / "reports" / "final-evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)
    access = result_dir / "test-access.json"
    # O_EXCL is also a concurrency guard. A failed/partial attempt requires a
    # documented lecturer decision; this script cannot quietly reset it.
    fd = os.open(access, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as stream:
        import json

        json.dump(
            {
                "started_at": now(),
                "frozen_manifest_sha256": sha256(frozen_path),
                "status": "STARTED",
                "test_examples": len(rows),
            },
            stream,
        )
    results = []
    try:
        labels = [int(r["label"]) for r in rows]
        for name, constant in [("always_normal", 0), ("always_defective", 1)]:
            results.append(
                {
                    "model": name,
                    "metrics": metrics(labels, [constant] * len(rows)),
                    "training_seconds": 0,
                    "timing": "Constant output; separate from trained-model timing",
                }
            )
        for name, run in frozen["comparisons"].items():
            result = evaluate_run(
                run,
                name,
                rows,
                root,
                args.output,
                frozen["serving_bundle"] if name == "final_ann" else None,
            )
            write_json(result_dir / f"{name}.json", result)
            results.append(result)
        seed_scores = []
        for index, run in enumerate(frozen["seed_repetitions"]):
            result = evaluate_run(run, f"seed_repeat_{index}", rows, root, args.output)
            write_json(result_dir / f"seed_repeat_{index}.json", result)
            seed_scores.append(result["metrics"]["f2"])
        vlm = benchmark(
            adapter, frozen["vlm"]["config"], rows, root, result_dir / "vlm.json", "test", REAL
        )
        results.append(
            {
                "model": "vlm",
                "metrics": vlm["metrics"],
                "mean_end_to_end_ms": vlm["mean_latency_ms"],
                "failure_rate": vlm["failure_rate"],
            }
        )
        bundle = frozen["serving_bundle"]
        model, metadata, device = load_checkpoint(
            args.output / "artifacts" / bundle["checkpoint"],
            expected_sha=bundle["checkpoint_sha256"],
        )
        evaluate_cases(model, metadata, device, bundle, rows, root, result_dir / "xai", "test")
        write_json(
            result_dir / "comparison.json",
            {
                "status": "COMPLETE",
                "timestamp": now(),
                "environment": environment(),
                "provenance": REAL,
                "comparisons": results,
                "test_seed_repetitions": {
                    "f2": seed_scores,
                    "mean": float(np.mean(seed_scores)),
                    "sample_std": float(np.std(seed_scores, ddof=1)),
                },
            },
        )
        write_json(
            access,
            {
                "status": "COMPLETE",
                "completed_at": now(),
                "frozen_manifest_sha256": sha256(frozen_path),
            },
        )
    except Exception as error:
        write_json(
            access,
            {
                "status": "FAILED_PARTIAL",
                "error": f"{type(error).__name__}: {error}",
                "instruction": "Preserve partial results. Do not delete this guard to repeat silently.",
            },
        )
        raise


if __name__ == "__main__":
    main()
