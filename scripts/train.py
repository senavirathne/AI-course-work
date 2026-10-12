"""Run a small development check or the predeclared train/validation searches."""

import argparse
from pathlib import Path

from fabric_inspector.common import write_json
from fabric_inspector.models.classical import train_search
from fabric_inspector.training.runner import default_config, search_configs, train_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=["ann", "svm"])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/split.json"))
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="SVM run numbering offset for retaining pilot evidence",
    )
    parser.add_argument(
        "--epochs", type=int, help="Change the budget before test access; logged for every run"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip completed run IDs, preserving their records"
    )
    args = parser.parse_args()
    if args.family == "svm":
        train_search(
            args.manifest, args.output, Path("."), args.smoke, args.start_index, args.resume
        )
        return
    configs = [default_config()] if args.smoke else search_configs()
    for config in configs:
        if args.epochs:
            config["epochs"] = args.epochs
        elif args.smoke:
            config["epochs"] = 3
    write_json(
        args.output / "experiments" / "ann-search-plan.json",
        {
            "configurations": configs,
            "count": len(configs),
            "held_out_access": False,
            "scope": "development smoke" if args.smoke else "assessed search",
        },
    )
    for index, config in enumerate(configs):
        run_id = "ann-smoke" if args.smoke else f"ann-search-{index:03d}"
        path = args.output / "experiments" / f"{run_id}.json"
        if args.resume and path.exists():
            from fabric_inspector.common import read_json

            prior = read_json(path)
            if prior["status"] != "COMPLETE" or prior["config"] != config:
                raise ValueError("Resume requires an unchanged, completed configuration")
            continue
        train_run(args.manifest, args.output, config, Path("."), args.device, run_id)


if __name__ == "__main__":
    main()
