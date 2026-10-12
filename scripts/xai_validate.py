"""Validation explanations, genuine errors and matched-area occlusion evidence."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from fabric_inspector.common import read_json, write_json, resolve_under
from fabric_inspector.data.images import preprocess
from fabric_inspector.data.manifest import load_manifest, read_mask
from fabric_inspector.explainability.gradcam import (
    gradcam,
    overlay,
    spatial_metrics,
    occlusion_check,
)
from fabric_inspector.serving.predictor import adjusted_scores
from fabric_inspector.training.runner import load_checkpoint


def evaluate_cases(model, metadata, device, bundle, rows, root, output, partition):
    import torch
    from torch.utils.data import DataLoader
    from fabric_inspector.data.images import FabricDataset
    from fabric_inspector.training.runner import ann_scores

    cases = []
    output.mkdir(parents=True, exist_ok=True)
    # This is an explanation sample, not a change to the evaluation population.
    # All primary metrics use every example. Select up to nine distinct source
    # images in each confusion category, using filename order only.
    scores, _, ids = ann_scores(
        model,
        DataLoader(FabricDataset(rows, root, metadata["preprocessing"]), batch_size=32),
        device,
    )
    predicted = adjusted_scores(scores, bundle["temperature"]) >= bundle["classification_threshold"]
    by_id = dict(zip(ids, predicted))
    selected, categories = [], {}
    for label, decision in [(1, 1), (1, 0), (0, 1), (0, 0)]:
        bucket = [
            r
            for r in sorted(rows, key=lambda r: r["patch_id"])
            if int(r["label"]) == label and int(by_id[r["patch_id"]]) == decision
        ]
        categories[f"label_{label}_prediction_{decision}"] = len(bucket)
        seen = set()
        chosen = []
        for row in bucket:
            if row["source_id"] not in seen:
                chosen.append(row)
                seen.add(row["source_id"])
            if len(chosen) == 9:
                break
        if len(chosen) < 9:
            chosen_ids = {r["patch_id"] for r in chosen}
            chosen.extend(r for r in bucket if r["patch_id"] not in chosen_ids)
        selected.extend(chosen[:9])
    for index, row in enumerate(selected):
        with Image.open(resolve_under(root, row["image_path"])) as image:
            image = image.convert("L").copy()
        tensor = preprocess(image, metadata["preprocessing"]).unsqueeze(0).to(device)
        with torch.inference_mode():
            raw = float(torch.sigmoid(model(tensor))[0])
        score = float(adjusted_scores([raw], bundle["temperature"])[0])
        prediction = int(score >= bundle["classification_threshold"])
        cam = gradcam(model, tensor)
        if row["mask_path"]:
            mask = read_mask(
                resolve_under(root, row["mask_path"]), row.get("mask_foreground") or "binary"
            )
        else:
            mask = np.zeros((256, 256), dtype=np.uint8)
        filename = f"case-{index:04d}.png"
        overlay(image, cam).save(output / filename)
        cases.append(
            {
                "patch_id": row["patch_id"],
                "partition": partition,
                "label": int(row["label"]),
                "prediction": prediction,
                "score": score,
                "incorrect": prediction != int(row["label"]),
                "error_type": ("false_negative" if int(row["label"]) else "false_positive")
                if prediction != int(row["label"])
                else None,
                "defect_pixels": int(row["defect_pixels"]),
                "spatial": spatial_metrics(cam, mask),
                "occlusion": occlusion_check(model, tensor, cam),
                "overlay": filename,
                "interpretation": "PENDING student analysis of actual overlay and measured perturbations",
            }
        )
    errors = [c for c in cases if c["incorrect"]]
    result = {
        "partition": partition,
        "provenance": bundle["provenance"],
        "run_id": metadata["run_id"],
        "settings": {"target": "defect logit", "occlusion_fraction": 0.1, "seed": 42},
        "explanation_selection": "At most nine per confusion category; prefer distinct source images, deterministic filename order. Explanation sampling only; model metrics retain all examples.",
        "prediction_population": len(rows),
        "population_confusion_categories": categories,
        "cases": cases,
        "genuine_error_count": len(errors),
        "three_error_requirement": "AVAILABLE FOR ANALYSIS"
        if len(errors) >= 3
        else "PENDING: fewer than three genuine errors",
        "sanity_check_cases": len(cases),
        "limitations": "Mask correspondence is not faithfulness; coarse maps and mean-fill occlusion can mislead",
    }
    write_json(output / "xai-validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/split.json"))
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    _, rows, root = load_manifest(args.manifest, verify_files=True)
    bundle = read_json(args.output / "artifacts" / "serving-bundle.json")
    model, metadata, device = load_checkpoint(
        args.output / "artifacts" / bundle["checkpoint"], expected_sha=bundle["checkpoint_sha256"]
    )
    result = evaluate_cases(
        model,
        metadata,
        device,
        bundle,
        [r for r in rows if r["split"] == "val"],
        root,
        args.output / "reports" / "xai-validation",
        "val",
    )
    print(f"Validated {len(result['cases'])} cases, {result['genuine_error_count']} actual errors")


if __name__ == "__main__":
    main()
