"""Select checkpoints, calibration and cutoffs using validation evidence only."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

from fabric_inspector.common import read_json, write_json, sha256, code_hash
from fabric_inspector.data.images import FabricDataset
from fabric_inspector.data.manifest import load_manifest
from fabric_inspector.evaluation.metrics import select_threshold, recommendation_thresholds
from fabric_inspector.serving.predictor import adjusted_scores
from fabric_inspector.training.runner import load_checkpoint, ann_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/split.json"))
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    split, rows, root = load_manifest(args.manifest, verify_files=True)
    pattern = "ann-smoke.json" if args.smoke else "ann-search-[0-9][0-9][0-9].json"
    runs = [read_json(p) for p in sorted((args.output / "experiments").glob(pattern))]
    runs = [r for r in runs if r["status"] == "COMPLETE"]
    if not runs:
        raise ValueError("No completed ANN runs")
    selected = max(
        runs,
        key=lambda r: (
            r["validation_metrics"]["f2"],
            r["validation_metrics"].get("average_precision") or 0,
            -r["architecture"]["total_parameters"],
        ),
    )
    if selected["split_sha256"] != split["split_sha256"]:
        raise ValueError("Selected run uses another split")
    checkpoint = args.output / selected["checkpoint"]
    model, metadata, device = load_checkpoint(
        checkpoint, expected_sha=selected["checkpoint_sha256"]
    )
    loader = DataLoader(
        FabricDataset([r for r in rows if r["split"] == "val"], root, metadata["preprocessing"]),
        batch_size=32,
    )
    raw, labels, ids = ann_scores(model, loader, device)
    temperature = 1.0
    if args.calibrate:

        def nll(value):
            p = np.clip(adjusted_scores(raw, value), 1e-7, 1 - 1e-7)
            return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

        fit = minimize_scalar(nll, bounds=(0.25, 4.0), method="bounded")
        if not fit.success:
            raise ValueError("Temperature calibration did not converge")
        temperature = float(fit.x)
    scores = adjusted_scores(raw, temperature)
    threshold, measured = select_threshold(labels, scores)
    recommendations = recommendation_thresholds(labels, scores)
    bins = []
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        keep = (scores >= low) & (scores <= high if high == 1 else scores < high)
        bins.append(
            {
                "low": float(low),
                "high": float(high),
                "n": int(keep.sum()),
                "mean_score": float(scores[keep].mean()) if keep.any() else None,
                "defect_fraction": float(labels[keep].mean()) if keep.any() else None,
            }
        )
    bundle = {
        "schema_version": 1,
        "run_id": selected["run_id"],
        "checkpoint": checkpoint.relative_to(args.output / "artifacts").as_posix(),
        "checkpoint_sha256": sha256(checkpoint),
        "split_sha256": split["split_sha256"],
        "provenance": split["provenance"],
        "preprocessing": metadata["preprocessing"],
        "classification_threshold": threshold,
        "recommendation_thresholds": recommendations,
        "temperature": temperature,
        "score_kind": "validation_temperature_adjusted_score"
        if args.calibrate
        else "uncalibrated_model_score",
        "selection_policy": "Validation F2, then average precision, then smaller parameter count",
        "validation_metrics": measured,
        "code_sha256_at_selection": code_hash(Path(".")),
        "initial_ann_run_id": runs[0]["run_id"] if args.smoke else "ann-search-000",
        "seed_repeat_protocol": {
            "seeds": [42, 123, 2026],
            "rule": "Repeat selected configuration on train/val before freeze; report spread and record whether variation is material",
        },
        "calibration": {
            "fit_partition": "val" if args.calibrate else "not fitted",
            "raw_brier": float(np.mean((raw - labels) ** 2)),
            "adjusted_brier": float(np.mean((scores - labels) ** 2)),
            "reliability_bins": bins,
            "limitation": "Validation reliability/calibration estimates share selection data; do not establish guaranteed confidence",
        },
    }
    write_json(args.output / "artifacts" / "serving-bundle.json", bundle)
    write_json(
        args.output / "artifacts" / "validation-predictions.json",
        {
            "partition": "val",
            "provenance": split["provenance"],
            "predictions": [
                {
                    "patch_id": name,
                    "label": int(y),
                    "raw_score": float(raw_score),
                    "score": float(s),
                }
                for name, y, raw_score, s in zip(ids, labels, raw, scores)
            ],
        },
    )
    print(
        f"Selected {selected['run_id']} using validation only. Bundle: {args.output / 'artifacts/serving-bundle.json'}"
    )


if __name__ == "__main__":
    main()
