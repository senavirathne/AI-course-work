"""Texture SVM with train-only scaling and a comparable 24-configuration budget."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from skimage.feature import hog, local_binary_pattern

from fabric_inspector.common import code_hash, environment, now, resolve_under, sha256, write_json
from fabric_inspector.data.manifest import load_manifest
from fabric_inspector.evaluation.metrics import select_threshold


def texture_features(image: Image.Image, mode="hog_lbp") -> np.ndarray:
    array = np.asarray(image.convert("L"))
    if array.shape != (256, 256):
        raise ValueError("Texture features require the shared 256x256 patch")
    lbp = local_binary_pattern(array, P=8, R=1, method="uniform")
    # Spatial 4x4 histograms retain local departures instead of hiding them in one mean.
    features = []
    for y in range(0, 256, 64):
        for x in range(0, 256, 64):
            histogram = np.bincount(lbp[y : y + 64, x : x + 64].astype(int).ravel(), minlength=10)[
                :10
            ]
            features.extend(histogram / histogram.sum())
    if mode == "hog_lbp":
        features.extend(
            hog(
                array,
                orientations=9,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
            )
        )
    elif mode != "lbp":
        raise ValueError("Use lbp or hog_lbp")
    return np.asarray(features, dtype=np.float64)


def feature_matrix(rows, root, mode):
    values = []
    for row in rows:
        with Image.open(resolve_under(root, row["image_path"])) as image:
            values.append(texture_features(image, mode))
    return np.stack(values)


def train_search(
    manifest: Path, output: Path, project: Path, smoke=False, start_index=0, resume=False
) -> list[dict]:
    split, rows, root = load_manifest(manifest, verify_files=True)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    ytrain, yval = (
        np.array([int(r["label"]) for r in train]),
        np.array([int(r["label"]) for r in val]),
    )
    records = []
    configs = [
        {"features": mode, "kernel": kernel, "C": c, "class_weight": weight, "gamma": "scale"}
        for mode in ["lbp", "hog_lbp"]
        for kernel in ["linear", "rbf"]
        for c in [0.1, 1.0, 10.0]
        for weight in [None, "balanced"]
    ]
    if smoke:
        configs = [
            dict(features="lbp", kernel="rbf", C=1.0, class_weight="balanced", gamma="scale")
        ]
    cache = {}
    for index, config in enumerate(configs):
        run_id = f"svm-{'smoke' if smoke else 'search'}-{index + start_index:03d}"
        record_path = output / "experiments" / f"{run_id}.json"
        if record_path.exists():
            if resume:
                from fabric_inspector.common import read_json

                prior = read_json(record_path)
                if prior["status"] != "COMPLETE" or prior["config"] != config:
                    raise ValueError("Resume requires an unchanged, completed configuration")
                if prior["split_sha256"] != split["split_sha256"]:
                    raise ValueError("Resume split mismatch")
                records.append(prior)
                continue
            raise FileExistsError(f"Run already exists: {run_id}")
        started_at = now()
        base_record = {
            "run_id": run_id,
            "status": "RUNNING",
            "started_at": started_at,
            "code_sha256": code_hash(project),
            "environment": environment(),
            "device": "cpu",
            "split_sha256": split["split_sha256"],
            "provenance": split["provenance"],
            "seed": 42,
            "config": config,
            "hypothesis": "Local texture histograms may be competitive with limited independent sources",
        }
        write_json(record_path, base_record)
        try:
            start = time.perf_counter()
            mode = config["features"]
            if mode not in cache:
                cache[mode] = (feature_matrix(train, root, mode), feature_matrix(val, root, mode))
            xtrain, xval = cache[mode]
            dual_attempt = config["kernel"] == "linear" and mode == "hog_lbp" and config["C"] >= 1
            estimator = (
                LinearSVC(
                    C=config["C"],
                    class_weight=config["class_weight"],
                    loss="squared_hinge",
                    dual=True if dual_attempt else "auto",
                    tol=1e-4,
                    max_iter=2000 if dual_attempt else 10000,
                    random_state=42,
                )
                if config["kernel"] == "linear"
                else SVC(
                    kernel="rbf",
                    C=config["C"],
                    class_weight=config["class_weight"],
                    gamma=config["gamma"],
                    probability=False,
                    cache_size=1024,
                    random_state=42,
                )
            )
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "svm",
                        estimator,
                    ),
                ]
            )
            # The scaler and any class weights are fitted on train only.
            numerical_attempts = []
            fit_start = time.perf_counter()
            with warnings.catch_warnings(record=True) as fit_warnings:
                warnings.simplefilter("always")
                model.fit(xtrain, ytrain)
            numerical_attempts.append(
                {
                    "phase": "dual_attempt" if dual_attempt else "original_solver",
                    "parameters": estimator.get_params(),
                    "seconds": time.perf_counter() - fit_start,
                    "iterations": np.asarray(estimator.n_iter_).tolist(),
                    "warnings": [str(w.message) for w in fit_warnings],
                }
            )
            if dual_attempt and any(
                issubclass(w.category, ConvergenceWarning) for w in fit_warnings
            ):
                # Numerical convergence, never validation performance, determines fallback.
                # Retain the original objective, tolerance and full iteration budget.
                estimator.set_params(dual="auto", max_iter=10000)
                fit_start = time.perf_counter()
                with warnings.catch_warnings(record=True) as fit_warnings:
                    warnings.simplefilter("always")
                    model.fit(xtrain, ytrain)
                numerical_attempts.append(
                    {
                        "phase": "original_solver_fallback",
                        "parameters": estimator.get_params(),
                        "seconds": time.perf_counter() - fit_start,
                        "iterations": np.asarray(estimator.n_iter_).tolist(),
                        "warnings": [str(w.message) for w in fit_warnings],
                    }
                )
            seconds = time.perf_counter() - start
            scores = model.decision_function(xval)
            threshold, measured = select_threshold(yval, scores)
            path = output / "artifacts" / f"{run_id}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "pipeline": model,
                    "feature_mode": mode,
                    "split_sha256": split["split_sha256"],
                    "provenance": split["provenance"],
                },
                path,
            )
        except Exception as error:
            base_record.update(
                status="FAILED_EXECUTION",
                finished_at=now(),
                error=f"{type(error).__name__}: {error}",
                elapsed_seconds=time.perf_counter() - start,
                outcome="Execution failure; not automatically a substantive scientific failure",
            )
            write_json(record_path, base_record)
            raise
        record = {
            "run_id": run_id,
            "status": "COMPLETE",
            "started_at": started_at,
            "finished_at": now(),
            "code_sha256": code_hash(project),
            "environment": environment(),
            "device": "cpu",
            "split_sha256": split["split_sha256"],
            "provenance": split["provenance"],
            "seed": 42,
            "config": config,
            "hypothesis": "Local texture histograms may be competitive when independent source coverage is limited",
            "training_seconds": seconds,
            "timing_note": "Feature extraction is charged on first use; later configurations reuse features",
            "solver": type(estimator).__name__,
            "solver_parameters": estimator.get_params(),
            "fit_warnings": [str(w.message) for w in fit_warnings],
            "iterations": np.asarray(estimator.n_iter_).tolist(),
            "numerical_solver_attempts": numerical_attempts,
            "validation_threshold": threshold,
            "validation_metrics": measured,
            "checkpoint": str(path.relative_to(output)),
            "checkpoint_sha256": sha256(path),
            "outcome": "Measured; compare with control before interpreting",
            "lesson": "PENDING evidence-based student interpretation",
        }
        write_json(record_path, record)
        records.append(record)
        print(f"{run_id}: val F2={measured['f2']:.4f}", flush=True)
    return records
