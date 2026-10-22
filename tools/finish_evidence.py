"""Create readable figures and post-evaluation analyses from immutable measured results."""

import argparse
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from fabric_inspector.common import now, read_json, resolve_under, write_json
from fabric_inspector.data.manifest import load_manifest, read_mask
from fabric_inspector.evaluation.metrics import metrics


def training_figures():
    _, rows, root = load_manifest(Path("data/manifests/split.json"), verify_files=False)
    train = [row for row in rows if row["split"] == "train"]
    negatives = [row for row in train if int(row["label"]) == 0]
    positives = sorted(
        (row for row in train if int(row["label"]) == 1), key=lambda row: int(row["defect_pixels"])
    )
    examples = [
        negatives[0],
        next(
            row
            for row in negatives
            if row["verified_extra_group"] != negatives[0]["verified_extra_group"]
        ),
        positives[0],
        positives[-1],
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.0), constrained_layout=True)
    for ax, row in zip(axes, examples):
        image = Image.open(resolve_under(root, row["image_path"])).convert("L")
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        ax.set_title(
            f"Label {row['label']} | {row['verified_extra_group']}\n{int(row['defect_pixels'])} annotated pixels",
            fontsize=10,
        )
    fig.suptitle("Illustrative training examples; class balance is not representative", fontsize=10)
    fig.savefig("reports/figures/compact-training-examples.png", dpi=180)
    plt.close(fig)
    bundle = read_json(Path("artifacts/serving-bundle.json"))
    initial = read_json(Path("experiments/ann-search-000.json"))
    final = read_json(Path("experiments") / (bundle["run_id"] + ".json"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for column, run in enumerate([initial, final]):
        h = run["history"]
        x = [e["epoch"] for e in h]
        axes[0, column].plot(x, [e["train_loss"] for e in h], "-o", label="Train")
        axes[0, column].plot(x, [e["val_loss"] for e in h], "--o", label="Validation")
        axes[0, column].set(
            title=run["run_id"], xlabel="Epoch", ylabel="Mean BCE (recorded objective)"
        )
        axes[0, column].legend()
        axes[1, column].plot(x, [e["val_f2"] for e in h], "-o")
        axes[1, column].axvline(
            run["selected_epoch"], color="gray", ls="--", label="Selected epoch"
        )
        axes[1, column].set(xlabel="Epoch", ylabel="Validation defect-class F2")
        axes[1, column].legend()
    fig.savefig("reports/figures/initial-final-learning.png", dpi=180)
    plt.close(fig)
    runs = [read_json(Path(f"experiments/ann-search-{i:03d}.json")) for i in range(20)]
    comparisons = [
        ([0, 1, 2, 3], "AdamW learning rate", "learning_rate"),
        ([0, 4, 5], "Batch size", "batch_size"),
        ([0, 6, 7], "Convolutional blocks", "depth"),
        ([0, 1, 2, 3, 8, 9, 10, 11], "Optimiser and its learning rate", "optimizer"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    records = []
    for ax, (ids, title, key) in zip(axes.ravel(), comparisons):
        rr = [runs[i] for i in ids]
        labels = [
            str(r["config"]["model"]["depth"])
            if key == "depth"
            else (r["config"]["optimizer"] + "\n" + str(r["config"]["learning_rate"]))
            if key == "optimizer"
            else str(r["config"][key])
            for r in rr
        ]
        ax.bar(range(len(rr)), [r["validation_metrics"]["f2"] for r in rr], color="#237f89")
        ax.set(xticks=range(len(rr)), xticklabels=labels, ylabel="Validation F2", title=title)
        if key == "optimizer":
            ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(0, max(r["validation_metrics"]["f2"] for r in rr) * 1.25)
        records.append(
            {
                "factor": title,
                "run_ids": [r["run_id"] for r in rr],
                "f2": [r["validation_metrics"]["f2"] for r in rr],
                "ap": [r["validation_metrics"]["average_precision"] for r in rr],
            }
        )
    fig.suptitle("Controlled screening: full validation partition, maximum four epochs")
    fig.savefig("reports/figures/hyperparameter-study.png", dpi=180)
    plt.close(fig)
    write_json(
        Path("reports/optimisation-analysis.json"),
        {
            "timestamp": now(),
            "partition": "val",
            "comparisons": records,
            "cautions": [
                "Fixed epochs are not fixed gradient-update counts across batch sizes.",
                "Loss magnitudes from weighted and unweighted BCE are not directly comparable.",
                "Only one validation origin: these are selection observations, not population causal effects.",
                "Each optimiser has its own tested learning-rate range; a single rate cannot establish superiority.",
            ],
        },
    )
    p = Path("experiments/controlled-comparisons.json")
    if p.exists():
        d = read_json(p)
        for r in d["comparisons"]:
            delta = r["validation_f2_delta"]
            r["interpretation"] = (
                f"Recorded validation F2 difference {delta:+.6f} against ann-search-000. Changes: {r['config_changes']}. This is a controlled single-factor comparison only when the substantive configuration difference is one factor; seed/refinement/multi-factor variants require separate interpretation. No held-out selection."
            )
        write_json(p, d)
    for run in [initial, final]:
        pd.DataFrame(run["architecture"]["layers"]).to_csv(
            Path("reports") / (run["run_id"] + "-architecture.csv"), index=False
        )


def post_test():
    assert read_json(Path("reports/final-evaluation/test-access.json"))["status"] == "COMPLETE"
    bundle = read_json(Path("artifacts/serving-bundle.json"))
    split, rows, root = load_manifest(Path("data/manifests/split.json"), verify_files=True)
    by_id = {r["patch_id"]: r for r in rows}
    final = read_json(Path("reports/final-evaluation/final_ann.json"))
    predictions = final["predictions"]
    groups = []
    for origin in sorted({by_id[p["patch_id"]]["verified_extra_group"] for p in predictions}):
        pp = [p for p in predictions if by_id[p["patch_id"]]["verified_extra_group"] == origin]
        groups.append(
            {
                "publisher_origin": origin,
                **metrics(
                    [p["label"] for p in pp],
                    [p["prediction"] for p in pp],
                    [p["score"] for p in pp],
                ),
            }
        )
    positive = [p for p in predictions if p["label"] == 1]
    sizes = []
    for label, lo, hi in [
        ("<=1% mask area", 0, 0.01),
        ("1%-5% mask area", 0.01, 0.05),
        (">5% mask area", 0.05, 1.0),
    ]:
        pp = [p for p in positive if lo < p["defect_pixels"] / 65536 <= hi]
        sizes.append(
            {
                "size_bin": label,
                "positives": len(pp),
                "missed": sum(p["prediction"] == 0 for p in pp),
                "recall": float(np.mean([p["prediction"] == 1 for p in pp])) if pp else None,
            }
        )
    cuts = bundle["recommendation_thresholds"]
    actions = {name: [] for name in ["routine_inspection", "manual_review", "priority_hold"]}
    for p in predictions:
        action = (
            "routine_inspection"
            if p["score"] < cuts["low"]
            else "manual_review"
            if p["score"] < cuts["high"]
            else "priority_hold"
        )
        actions[action].append(p)
    action_table = [
        {
            "action": name,
            "patches": len(pp),
            "fraction": len(pp) / len(predictions),
            "defects": sum(p["label"] for p in pp),
            "normal": sum(1 - p["label"] for p in pp),
        }
        for name, pp in actions.items()
    ]
    analysis = {
        "timestamp": now(),
        "posthoc_only": True,
        "selection_changed": False,
        "run_id": bundle["run_id"],
        "all_test_examples": len(predictions),
        "publisher_origin_metrics": groups,
        "positive_mask_size_recall": sizes,
        "recommendations": action_table,
        "recommendation_thresholds": cuts,
        "missed_defects_in_routine_band": sum(p["label"] for p in actions["routine_inspection"]),
        "review_or_hold_fraction": (len(actions["manual_review"]) + len(actions["priority_hold"]))
        / len(predictions),
        "brier_score": final["brier_score"],
        "limitations": [
            "Only two held-out publisher origin groups; group bootstrap intervals are coarse, not reliable factory-population coverage.",
            "Mask size is publisher annotation area, not a verified severity/cost rating.",
            "Origin identifiers do not establish individual physical rolls or unseen manufacturers.",
            "Recommendation costs are coursework assumptions, not measured company costs; routine inspection never means certified defect-free.",
        ],
    }
    write_json(Path("reports/error-analysis.json"), analysis)
    xai_root = Path("reports/final-evaluation/xai")
    xai = read_json(xai_root / "xai-validation.json")
    errors = [c for c in xai["cases"] if c["incorrect"]]
    assert len(errors) >= 3
    fns = [c for c in errors if c["error_type"] == "false_negative"]
    fps = [c for c in errors if c["error_type"] == "false_positive"]
    chosen = (fns[:2] + fps[:1]) if fns and fps else errors[:3]
    for case in errors:
        if len(chosen) >= 3:
            break
        if case not in chosen:
            chosen.append(case)
    panels = []
    for index, c in enumerate(chosen):
        row = by_id[c["patch_id"]]
        image = Image.open(resolve_under(root, row["image_path"])).convert("L")
        mask = (
            read_mask(resolve_under(root, row["mask_path"]), row.get("mask_foreground") or "binary")
            if row["mask_path"]
            else np.zeros((256, 256), dtype=np.uint8)
        )
        cam_image = Image.open(xai_root / c["overlay"])
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), constrained_layout=True)
        for ax, img, title in zip(
            axes,
            [image, mask, cam_image],
            ["Original 256x256", "Publisher foreground (255 only)", "Actual defect-logit Grad-CAM"],
        ):
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(
            f"{c['error_type'].replace('_', ' ').title()} | {c['patch_id']}\nscore {c['score']:.6f}; frozen cutoff {bundle['classification_threshold']:.6f}",
            fontsize=10,
        )
        panel = Path(f"reports/figures/xai-error-{index + 1}.png")
        fig.savefig(panel, dpi=180)
        plt.close(fig)
        drops = c["occlusion"]["logit_drops"]
        zero_cam = c["occlusion"]["degenerate_zero_cam"]
        sensitive = (
            not zero_cam
            and drops["top"] > 0
            and drops["top"] > drops["random"]
            and drops["top"] > drops["low"]
        )
        text = (
            f"This genuine {c['error_type'].replace('_', ' ')} has ground-truth label {c['label']} and prediction {c['prediction']}; score {c['score']:.6f} versus validation-frozen cutoff {bundle['classification_threshold']:.6f}. "
            f"The publisher mask covers {c['defect_pixels']} pixels ({c['defect_pixels'] / 65536:.3%}). Equal-area top/low/random occlusion changes the defect logit by {drops['top']:+.6f}/{drops['low']:+.6f}/{drops['random']:+.6f}. "
            + (
                "The map is identically zero; tied top/low rankings cannot validate a highlighted region. "
                if zero_cam
                else "Occluding the highlighted region decreases the defect logit more than either control in this case. "
                if sensitive
                else "This case does not support stronger sensitivity of the highlighted region than both controls. "
            )
            + "Mean-fill perturbation may create unnatural weave; the map does not identify a causal production fault or establish a certified defect boundary."
        )
        panels.append(
            {
                **c,
                "case_name": f"Error {index + 1}: " + c["error_type"].replace("_", " "),
                "panel": panel.as_posix(),
                "analysis": text,
                "interpretation": text,
                "source_id": row["source_id"],
                "publisher_origin": row["verified_extra_group"],
            }
        )
    write_json(
        Path("docs/three-xai-errors.json"),
        {
            "timestamp": now(),
            "partition": "test",
            "run_id": bundle["run_id"],
            "cases": panels,
            "selection": "From the predeclared deterministic explanation sample; two FN and one FP where available. No further tuning.",
        },
    )
    demo = Path("data/demo")
    demo.mkdir(parents=True, exist_ok=True)
    pp_by_id = {p["patch_id"]: p for p in predictions}
    demo_choices = []
    for label in [0, 1]:
        found = next(
            (p for p in predictions if p["label"] == label and p["prediction"] == label), None
        )
        if found:
            demo_choices.append(found)
    demo_choices.append(pp_by_id[panels[0]["patch_id"]])
    cases = []
    for index, p in enumerate(demo_choices):
        row = by_id[p["patch_id"]]
        dest = demo / f"{index + 1}-{row['patch_id']}"
        if dest.suffix.lower() != ".png":
            dest = dest.with_suffix(".png")
        shutil.copyfile(resolve_under(root, row["image_path"]), dest)
        mask_path = None
        if row["mask_path"]:
            mask_dest = demo / (dest.stem + "-mask.png")
            shutil.copyfile(resolve_under(root, row["mask_path"]), mask_dest)
            mask_path = mask_dest.as_posix()
        cases.append(
            {
                "path": dest.as_posix(),
                "label": p["label"],
                "patch_id": p["patch_id"],
                "expected_score": p["score"],
                "incorrect": p["label"] != p["prediction"],
                "mask_path": mask_path,
                "image_sha256": row["sha256"],
                "publisher_origin": row["verified_extra_group"],
            }
        )
    write_json(
        demo / "cases.json",
        {
            "provenance": split["provenance"],
            "run_id": bundle["run_id"],
            "cases": cases,
            "note": "Authentic examples copied after final evaluation solely for app verification and demonstration. They do not change any split or metric.",
        },
    )
    # Replace draft interpretation fields with bounded, measured statements.
    for directory in [xai_root, Path("reports/xai-validation")]:
        p = directory / "xai-validation.json"
        if not p.exists():
            continue
        d = read_json(p)
        for c in d["cases"]:
            drops = c["occlusion"]["logit_drops"]
            c["interpretation"] = (
                f"Measured defect-logit drops: top={drops['top']:+.6f}, low={drops['low']:+.6f}, random={drops['random']:+.6f}. Nondegenerate top-region deletion decreases the logit more than both controls: {not c['occlusion']['degenerate_zero_cam'] and drops['top'] > 0 and drops['top'] > max(drops['low'], drops['random'])}. Mask correspondence and this unnatural mean-fill perturbation do not prove causality."
            )
        d["three_error_requirement"] = (
            "ANALYSED IN docs/three-xai-errors.json"
            if directory == xai_root
            else "Validation sanity-check evidence; final error cases analysed separately"
        )
        write_json(p, d)
    print("Created full-test error analysis and three genuine XAI case panels", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-test", action="store_true")
    args = parser.parse_args()
    training_figures()
    if args.post_test:
        post_test()


if __name__ == "__main__":
    main()
