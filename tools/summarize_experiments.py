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
            seed=r.get("seed", r["config"].get("seed")),
            config=json.dumps(r["config"], sort_keys=True),
            validation_f2=m["f2"],
            validation_ap=m.get("average_precision"),
            tp=m["tp"],
            fp=m["fp"],
            fn=m["fn"],
        )
        rows.append(row)
    out = Path("reports/experiment-log.csv")
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Complete measured experiment log",
        "",
        "All numbers below use the frozen RAW-FABRID validation partition. Full configurations, histories, environment, timestamps and hashes are in the referenced JSON records. Engineering interruptions are retained separately and are not scientific failures.",
        "",
        "| Run | Configuration | Val F2 | Val AP | TP / FP / FN | Seconds |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for _, r in records:
        cfg = r["config"]
        m = r["validation_metrics"]
        if r["run_id"].startswith("ann"):
            short = f"{cfg['model']}; {cfg['optimizer']} lr={cfg['learning_rate']}; batch={cfg['batch_size']}; seed={cfg['seed']}; aug={cfg['augment']}; weighted={cfg['weighted_loss']}"
        else:
            short = f"{cfg['features']}; {r.get('solver', 'SVC pilot')}; {cfg['kernel']}; C={cfg['C']}; weight={cfg['class_weight']}"
        lines.append(
            f"| {r['run_id']} | {short} | {m['f2']:.4f} | {m['average_precision']:.4f} | {m['tp']} / {m['fp']} / {m['fn']} | {r['training_seconds']:.1f} |"
        )
    lines.extend(["", "## Recorded lessons", ""])
    for _, r in records:
        lines.extend([f"### {r['run_id']}", "", r["lesson"], ""])
    lines.extend(
        [
            "## Actual VLM validation",
            "",
            "| Prompt | Examples | F2 | Unresolved | Mean total latency (ms) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for index in range(2):
        p = Path(f"experiments/vlm-validation-{index}.json")
        if p.exists():
            r = read_json(p)
            m = r["metrics"]
            lines.append(
                f"| {index} | {m['n']} | {m['f2']:.4f} | {m['unresolved_count']} | {r['mean_latency_ms']:.1f} |"
            )
    lines.extend(
        [
            "",
            "Raw VLM decisions, exact prompts, processor/model revision, attempts and token counts are retained in JSON and JSONL. All completed prompts are scored on every validation example. Partial execution logs never select the prompt.",
            "",
            "## Engineering history",
            "",
            "`docs/engineering-runs/` preserves actual CPU profiling, solver amendments, interrupted attempts and control-resolution notes. `development-run/` preserves the earlier artificial sample checks, explicitly excluded from assessed results.",
            "",
        ]
    )
    Path("reports/experiment-log.md").write_text("\n".join(lines))
    print(f"Summarised {len(rows)} completed real training runs")


if __name__ == "__main__":
    main()
