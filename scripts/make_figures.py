"""Generate learning, ablation and calibration figures from actual run records."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fabric_inspector.common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    runs = [
        read_json(p)
        for p in sorted((args.output / "experiments").glob("ann-*.json"))
        if p.name != "ann-search-plan.json"
    ]
    runs = [r for r in runs if r.get("status") == "COMPLETE"]
    if not runs:
        raise ValueError("No real completed runs")
    figures = args.output / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for run in runs:
        epochs = [h["epoch"] for h in run["history"]]
        axes[0].plot(
            epochs,
            [h["train_loss"] for h in run["history"]],
            alpha=0.7,
            label=run["run_id"] + " train",
        )
        axes[0].plot(
            epochs,
            [h["val_loss"] for h in run["history"]],
            linestyle="--",
            alpha=0.7,
            label=run["run_id"] + " val",
        )
        axes[1].plot(epochs, [h["val_f2"] for h in run["history"]], alpha=0.7, label=run["run_id"])
    axes[0].set(xlabel="Epoch", ylabel="Sample-weighted mean BCE", title="Learning curves")
    axes[1].set(xlabel="Epoch", ylabel="Defect-class F2", title="Validation, selected cutoffs")
    if len(runs) <= 4:
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
    fig.suptitle(runs[0]["provenance"])
    fig.savefig(figures / "learning-curves.png", dpi=160)
    plt.close(fig)
    baseline = next((r for r in runs if r["run_id"] == "ann-search-000"), runs[0])
    comparisons = []
    for run in (r for r in runs if r["run_id"] != baseline["run_id"]):
        differences = {
            key: value
            for key, value in run["config"].items()
            if key not in {"hypothesis"} and value != baseline["config"].get(key)
        }
        comparisons.append(
            {
                "run_id": run["run_id"],
                "control_run_id": baseline["run_id"],
                "config_changes": differences,
                "validation_f2_delta": run["validation_metrics"]["f2"]
                - baseline["validation_metrics"]["f2"],
                "interpretation": "PENDING: inspect curves and control validity; do not count ties or syntax failures automatically as substantive failures",
            }
        )
    write_json(
        args.output / "experiments" / "controlled-comparisons.json",
        {"provenance": baseline["provenance"], "comparisons": comparisons},
    )
    bundle_path = args.output / "artifacts" / "serving-bundle.json"
    if bundle_path.exists():
        bundle = read_json(bundle_path)
        bins = [b for b in bundle["calibration"]["reliability_bins"] if b["n"]]
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect agreement reference")
        ax.plot(
            [b["mean_score"] for b in bins],
            [b["defect_fraction"] for b in bins],
            "o-",
            label="Validation bins",
        )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean model score",
            ylabel="Observed defect fraction",
            title="Validation reliability (selection data)",
        )
        ax.legend(fontsize=8)
        fig.savefig(figures / "validation-reliability.png", dpi=160)
        plt.close(fig)
    print("Generated figures from recorded runs; no held-out examples accessed")


if __name__ == "__main__":
    main()
