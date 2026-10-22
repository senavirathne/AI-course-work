"""Create a complete readable log and evidence-based lessons from actual runs."""

import csv
import json
import argparse
import os
from pathlib import Path

from fabric_inspector.common import read_json, write_json


def matched_classical_context(run, records):
    """Find an actual single-setting counterpart within the same fitting family."""
    cfg = run["config"]
    targets = []
    if cfg["C"] != 0.1:
        targets.append(("C", 0.1))
    if cfg["class_weight"] is not None:
        targets.append(("class_weight", None))
    targets.append(("features", "lbp" if cfg["features"] == "hog_lbp" else "hog_lbp"))
    if cfg["class_weight"] is None:
        targets.append(("class_weight", "balanced"))
    keys = {"features", "kernel", "C", "class_weight", "gamma"}
    for changed, target in targets:
        for other in records:
            if not other["run_id"].startswith("svm-"):
                continue
            oc = other["config"]
            same_solver = other.get("solver", "SVC pilot") == run.get("solver", "SVC pilot")
            if (
                same_solver
                and oc.get(changed) == target
                and all(cfg.get(k) == oc.get(k) for k in keys - {changed})
            ):
                m, om = run["validation_metrics"], other["validation_metrics"]
                return (
                    f"Matched counterpart {other['run_id']} differs in {changed} "
                    f"({oc[changed]} -> {cfg[changed]}): F2 difference {m['f2'] - om['f2']:+.4f}, "
                    f"AP difference {m['average_precision'] - om['average_precision']:+.4f}. "
                    "These descriptive differences do not establish a universal setting effect. "
                )
    return "No same-solver single-setting counterpart is claimed for this record. "


def describe(run, control, records):
    m = run["validation_metrics"]
    cfg = run["config"]
    common = (
        f"Validation F2={m['f2']:.4f}, AP={m.get('average_precision', 0):.4f}, "
        f"TP/FP/FN={m['tp']}/{m['fp']}/{m['fn']}. "
    )
    if run["run_id"].startswith("svm-"):
        return (
            common
            + matched_classical_context(run, records)
            + (
                f"{cfg['features']} with {run.get('solver', 'SVC pilot')}, "
                f"kernel={cfg['kernel']}, C={cfg['C']}, class_weight={cfg['class_weight']}. "
                "Descriptor extraction and scaling use train only; the score cutoff uses validation. "
                "Interpret kernel/weighting with matched feature/C controls. "
                + (
                    "Fit warnings are retained; convergence is not assumed. "
                    if run.get("fit_warnings")
                    else ""
                )
                + "Only one validation origin is represented, so these are descriptive selection results."
            )
        )
    diff = m["f2"] - control["validation_metrics"]["f2"]
    history = run["history"]
    prefix = (
        f"Difference from initial ANN F2={diff:+.4f}; best epoch {run['selected_epoch']} "
        f"of {run['epochs_completed']}. Recorded mean training loss "
        f"{history[0]['train_loss']:.4f} to {history[-1]['train_loss']:.4f}. "
    )
    if cfg["model"]["name"] == "mobilenet":
        reason = "Real ImageNet transfer is tested with all layers fine-tuned, replicated grayscale and no crop. Domain mismatch and this early-training budget limit the conclusion."
    elif cfg["optimizer"] == "sgd":
        reason = "Momentum SGD is judged at its own stated learning rate; compare all four SGD rates with AdamW rather than treating one unsuccessful rate as an optimiser verdict."
    elif not cfg.get("weighted_loss", True):
        reason = "Unweighted BCE changes the objective and score distribution. Raw loss magnitude is not comparable with weighted BCE; inspect F2, AP and missed defects."
    elif cfg["batch_size"] != control["config"]["batch_size"]:
        reason = "Batch size changes gradient aggregation and updates per epoch; fixed epochs are not fixed update counts. Avoid attributing the difference solely to gradient noise."
    elif cfg["model"] != control["config"]["model"]:
        reason = "Capacity/depth/dropout changed; use the recorded layer table and matched controls. More parameters are not evidence of better generalisation."
    elif not cfg.get("augment", True):
        reason = "Removing flips tests whether orientation variation helps the held-out acquisition; one origin cannot establish a universal augmentation policy."
    else:
        reason = "Assess learning rate and convergence within the recorded budget; a short-run difference does not establish eventual learnability."
    return (
        common
        + prefix
        + reason
        + " Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    os.chdir(args.output)
    out = Path("reports/experiment-log.csv")
    existing_rows = []
    if out.exists():
        with out.open(newline="") as stream:
            existing_rows = list(csv.DictReader(stream))
    records = []
    for pattern in ["ann-search-*.json", "ann-repeat-*.json", "svm-search-*.json"]:
        for p in sorted(Path("experiments").glob(pattern)):
            r = read_json(p)
            if r.get("status") == "COMPLETE":
                records.append((p, r))
    control = next(r for _, r in records if r["run_id"] == "ann-search-000")
    fields = [
        "run_id",
        "family",
        "status",
        "started_at",
        "finished_at",
        "seed",
        "split_sha256",
        "code_sha256",
        "hypothesis",
        "config",
        "epochs_completed",
        "selected_epoch",
        "validation_f2",
        "validation_ap",
        "tp",
        "fp",
        "fn",
        "training_seconds",
        "checkpoint",
        "checkpoint_sha256",
        "outcome",
        "lesson",
        "evaluation_partition",
        "test_f2",
        "test_ap",
    ]
    rows = []
    for p, r in records:
        m = r["validation_metrics"]
        r["lesson"] = describe(r, control, [record for _, record in records])
        r["outcome"] = "Measured validation evidence; no test-driven changes"
        write_json(p, r)
        row = {k: r.get(k, "") for k in fields}
        row.update(
            family="ANN" if r["run_id"].startswith("ann") else "SVM",
            evaluation_partition="validation",
            seed=r.get("seed", r["config"].get("seed")),
            config=json.dumps(r["config"], sort_keys=True),
            validation_f2=m["f2"],
            validation_ap=m.get("average_precision"),
            tp=m["tp"],
            fp=m["fp"],
            fn=m["fn"],
        )
        rows.append(row)
    selection_path = Path("artifacts/vlm-selection.json")
    selection = read_json(selection_path) if selection_path.exists() else {}
    selected_prompt = selection.get("config", {}).get("prompt_index")
    for index in range(2):
        path = Path(f"experiments/vlm-validation-{index}.json")
        if not path.exists():
            continue
        run = read_json(path)
        if run.get("status") != "COMPLETE":
            continue
        metrics = run["metrics"]
        elapsed = sum(
            example.get("end_to_end_ms", 0.0) for example in run.get("examples", [])
        ) / 1000
        rows.append(
            {
                "run_id": f"vlm-validation-{index}",
                "family": "VLM",
                "status": run["status"],
                "started_at": "",
                "finished_at": run.get("timestamp", ""),
                "seed": "",
                "split_sha256": selection.get("split_sha256", ""),
                "code_sha256": selection.get("code_sha256", ""),
                "hypothesis": f"Evaluate prompt {index} on the complete validation partition before test access.",
                "config": json.dumps(run["config"], sort_keys=True),
                "epochs_completed": "",
                "selected_epoch": "",
                "validation_f2": metrics["f2"],
                "validation_ap": "",
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "training_seconds": elapsed,
                "checkpoint": f"{run['config']['model_id']}@{run['config']['revision']}",
                "checkpoint_sha256": "",
                "outcome": "Selected on validation" if index == selected_prompt else "Not selected on validation",
                "lesson": (
                    f"Full validation F2={metrics['f2']:.4f}; unresolved outputs="
                    f"{metrics['unresolved_count']}. Prompt selection used validation F2, "
                    "then lower failure rate, never held-out test results."
                ),
                "evaluation_partition": "validation",
                "test_f2": "",
                "test_ap": "",
            }
        )
    comparison_path = Path("reports/final-evaluation/comparison.json")
    if comparison_path.exists():
        comparison = read_json(comparison_path)
        summaries = {
            item["model"]: {
                "f2": item["metrics"]["f2"],
                "average_precision": item["metrics"].get("average_precision"),
            }
            for item in comparison["comparisons"]
        }
        final_ann = next(
            item for item in comparison["comparisons"] if item["model"] == "final_ann"
        )
        access_path = Path("reports/final-evaluation/test-access.json")
        access = read_json(access_path) if access_path.exists() else {}
        rows.append(
            {
                "run_id": "frozen-final-evaluation",
                "family": "FINAL_EVALUATION",
                "status": comparison["status"],
                "started_at": "",
                "finished_at": comparison.get("timestamp", ""),
                "seed": "",
                "split_sha256": selection.get("split_sha256", ""),
                "code_sha256": "",
                "hypothesis": "Execute the frozen held-out comparison without model, prompt, or threshold selection.",
                "config": json.dumps(
                    {
                        "comparison_file": comparison_path.as_posix(),
                        "models": summaries,
                        "physical_test_computation_count": access.get(
                            "physical_test_computation_count"
                        ),
                    },
                    sort_keys=True,
                ),
                "epochs_completed": "",
                "selected_epoch": "",
                "validation_f2": "",
                "validation_ap": "",
                "tp": final_ann["metrics"]["tp"],
                "fp": final_ann["metrics"]["fp"],
                "fn": final_ann["metrics"]["fn"],
                "training_seconds": "",
                "checkpoint": comparison_path.as_posix(),
                "checkpoint_sha256": "",
                "outcome": "Frozen held-out comparison; identically recomputed after workspace loss",
                "lesson": (
                    "Final ANN, initial ANN, classical, trivial, seed-repeat, and VLM results "
                    "are retained in the referenced comparison. No test result selected a model, "
                    "prompt, or threshold."
                ),
                "evaluation_partition": "test",
                "test_f2": final_ann["metrics"]["f2"],
                "test_ap": final_ann["metrics"].get("average_precision", ""),
            }
        )
    generated = {row["run_id"]: row for row in rows}
    merged = [generated.pop(row["run_id"], row) for row in existing_rows]
    merged.extend(generated.values())
    rows = merged
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summarised {len(rows)} completed real experiment/evaluation records")


if __name__ == "__main__":
    main()
