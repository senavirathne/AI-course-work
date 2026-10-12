"""Assess the predeclared selected-model seed variability before test access."""

import argparse
import copy
from pathlib import Path

import numpy as np

from fabric_inspector.common import read_json, write_json
from fabric_inspector.training.runner import train_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/split.json"))
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    bundle = read_json(args.output / "artifacts" / "serving-bundle.json")
    selected = read_json(args.output / "experiments" / (bundle["run_id"] + ".json"))
    results = [selected]
    for seed in [123, 2026]:
        config = copy.deepcopy(selected["config"])
        config.update(
            seed=seed,
            hypothesis="Repeat the selected architecture to assess training seed variability",
        )
        run_id = f"ann-repeat-{seed}"
        results.append(
            train_run(args.manifest, args.output, config, Path("."), args.device, run_id)
        )
    scores = [r["validation_metrics"]["f2"] for r in results]
    write_json(
        args.output / "artifacts" / "seed-variability.json",
        {
            "partition": "val",
            "split_sha256": bundle["split_sha256"],
            "provenance": bundle["provenance"],
            "run_ids": [r["run_id"] for r in results],
            "seeds": [42, 123, 2026],
            "f2": scores,
            "mean": float(np.mean(scores)),
            "sample_std": float(np.std(scores, ddof=1)),
            "range": float(np.ptp(scores)),
            "material_variability_rule": "F2 range > 0.02 (predeclared coursework heuristic)",
            "material": bool(np.ptp(scores) > 0.02),
            "final_test_protocol": "Evaluate all three frozen repetitions in one final evaluation; report mean/spread. Do not choose a seed from test results.",
        },
    )


if __name__ == "__main__":
    main()
