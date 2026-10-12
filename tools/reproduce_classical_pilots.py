"""Replay the four logged libsvm pilots with their original hinge-loss pipeline."""

import argparse
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fabric_inspector.common import code_hash, environment, now, read_json, sha256, write_json
from fabric_inspector.data.manifest import load_manifest
from fabric_inspector.evaluation.metrics import select_threshold
from fabric_inspector.models.classical import feature_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    templates = [read_json(Path(f"experiments/svm-search-{i:03d}.json")) for i in range(4)]
    split, rows, root = load_manifest(args.manifest, verify_files=True)
    assert all(r["split_sha256"] == split["split_sha256"] for r in templates)
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "val"]
    ytrain = np.array([int(r["label"]) for r in train])
    yval = np.array([int(r["label"]) for r in validation])
    cache = {}
    for template in templates:
        run_id = template["run_id"]
        target = args.output / "experiments" / (run_id + ".json")
        if target.exists():
            raise FileExistsError(
                "Pilot replay output already exists; preserve the measured record"
            )
        config = template["config"]
        record = {
            "run_id": run_id,
            "status": "RUNNING",
            "started_at": now(),
            "config": config,
            "seed": 42,
            "split_sha256": split["split_sha256"],
            "provenance": split["provenance"],
            "environment": environment(),
            "code_sha256": code_hash(Path(".")),
            "execution_tool_sha256": sha256(Path(__file__)),
            "original_protocol_source": "docs/source-versions/sources-957ae7314bb7.zip",
            "hypothesis": template["hypothesis"],
            "device": "cpu",
        }
        write_json(target, record)
        started = time.perf_counter()
        try:
            mode = config["features"]
            if mode not in cache:
                cache[mode] = (
                    feature_matrix(train, root, mode),
                    feature_matrix(validation, root, mode),
                )
            xtrain, xval = cache[mode]
            estimator = SVC(
                kernel=config["kernel"],
                C=config["C"],
                class_weight=config["class_weight"],
                gamma=config["gamma"],
                probability=False,
                random_state=42,
            )
            model = Pipeline([("scale", StandardScaler()), ("svm", estimator)])
            with warnings.catch_warnings(record=True) as fit_warnings:
                warnings.simplefilter("always")
                model.fit(xtrain, ytrain)
            seconds = time.perf_counter() - started
            threshold, measured = select_threshold(yval, model.decision_function(xval))
            checkpoint = args.output / "artifacts" / (run_id + ".joblib")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "pipeline": model,
                    "feature_mode": mode,
                    "split_sha256": split["split_sha256"],
                    "provenance": split["provenance"],
                },
                checkpoint,
            )
            record.update(
                status="COMPLETE",
                finished_at=now(),
                training_seconds=seconds,
                solver="SVC original libsvm pilot",
                solver_parameters=estimator.get_params(),
                fit_warnings=[str(w.message) for w in fit_warnings],
                iterations=np.asarray(estimator.n_iter_).tolist(),
                validation_threshold=threshold,
                validation_metrics=measured,
                checkpoint=checkpoint.relative_to(args.output).as_posix(),
                checkpoint_sha256=sha256(checkpoint),
                outcome="Measured replay of the historical pilot",
                lesson="PENDING evidence-based interpretation",
            )
            write_json(target, record)
            print(f"{run_id}: replay validation F2={measured['f2']:.6f}", flush=True)
        except Exception as error:
            record.update(
                status="FAILED_EXECUTION",
                finished_at=now(),
                error=f"{type(error).__name__}: {error}",
            )
            write_json(target, record)
            raise


if __name__ == "__main__":
    main()
