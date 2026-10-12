"""Perform the predeclared longer validation-selected refinement before test access."""

import argparse
import copy
from pathlib import Path

from fabric_inspector.common import read_json, sha256, write_json
from fabric_inspector.training.runner import train_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/split.json"))
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (args.output / "reports/final-evaluation/test-access.json").exists():
        raise ValueError("Refinement is forbidden after final test access")
    runs = [read_json(args.output / f"experiments/ann-search-{i:03d}.json") for i in range(20)]
    if any(r["status"] != "COMPLETE" for r in runs):
        raise ValueError("All twenty screening configurations must complete first")
    best = max(
        runs,
        key=lambda r: (
            r["validation_metrics"]["f2"],
            r["validation_metrics"].get("average_precision") or 0,
            -r["architecture"]["total_parameters"],
        ),
    )
    config = copy.deepcopy(best["config"])
    config.update(
        epochs=12,
        hypothesis="Predeclared longer refinement of the best screening configuration; validation only",
    )
    write_json(
        args.output / "experiments/refinement-plan.json",
        {
            "control_run_id": best["run_id"],
            "config": config,
            "rule": "Same F2/AP/smaller-parameter selection across all 21 ANN configurations",
            "held_out_access": False,
            "execution_tool_sha256": sha256(Path(__file__)),
        },
    )
    train_run(args.manifest, args.output, config, Path("."), args.device, "ann-search-020")


if __name__ == "__main__":
    main()
