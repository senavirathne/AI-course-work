"""Generate train-only EDA and exactly three observed modelling constraints."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fabric_inspector.common import resolve_under, write_json
from fabric_inspector.data.manifest import load_manifest, read_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/split.json"))
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    split, rows, root = load_manifest(args.manifest, verify_files=True)
    training = [r for r in rows if r["split"] == "train"]
    means, contrasts, textures = [], [], []
    for row in training:
        with Image.open(resolve_under(root, row["image_path"])) as image:
            pixels = np.asarray(image, dtype=float)
        means.append(float(pixels.mean()))
        contrasts.append(float(pixels.std()))
        textures.append(
            float(np.abs(np.diff(pixels, axis=0)).mean() + np.abs(np.diff(pixels, axis=1)).mean())
        )
    positive = sum(int(r["label"]) for r in training)
    sources = {r["source_id"] for r in training}
    groups = {r["group_id"] for r in training}
    areas = np.array([int(r["defect_pixels"]) / 65536 for r in training if int(r["label"])])
    stats = {
        "partition": "train",
        "provenance": split["provenance"],
        "split_sha256": split["split_sha256"],
        "patches": len(training),
        "positives": positive,
        "positive_fraction": positive / len(training),
        "independent_groups": len(groups),
        "source_images": len(sources),
        "source_images_containing_positive_patches": len(
            {r["source_id"] for r in training if int(r["label"])}
        ),
        "defect_area_fraction_quantiles": dict(
            zip(
                ["min", "q25", "median", "q75", "max"],
                np.quantile(areas, [0, 0.25, 0.5, 0.75, 1]).tolist(),
            )
        ),
        "brightness_range": [min(means), max(means)],
        "contrast_range": [min(contrasts), max(contrasts)],
        "texture_total_variation_range": [min(textures), max(textures)],
        "group_metadata_limit": (
            f"Publisher field {split['audit']['verified_group_field']} joined through source metadata; "
            "anonymised fabric-origin groups retained together. Exact physical-roll, fabric-type and "
            "manufacturer relationships beyond this field are not established."
            if split["audit"].get("verified_group_field")
            else "Only source/verified duplicate groups; no inferred roll or manufacturer claims"
        ),
    }
    findings = [
        {
            "finding": f"Training has {positive}/{len(training)} defective patches ({positive / len(training):.1%}).",
            "constraint": "Compare weighted and unweighted BCE; retain both trivial references and defect-class F2.",
            "evidence": "train class counts; no test image inspection",
        },
        {
            "finding": f"The smallest observed training defect covers {areas.min():.3%} of a 256x256 patch.",
            "constraint": "Preserve the patch resolution and avoid cropping/downsampling that can erase small defects; examine coarse CAM limits.",
            "evidence": "verified training binary masks",
        },
        {
            "finding": f"{len(training)} training patches come from {len(sources)} source images and {len(groups)} independent components.",
            "constraint": "Restrict CNN capacity, test transfer learning, and use component bootstrap rather than patch-independent uncertainty.",
            "evidence": "immutable source/duplicate manifest",
        },
    ]
    figures = args.output / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].bar(
        ["Normal", "Defective"], [len(training) - positive, positive], color=["#2d7760", "#ce7444"]
    )
    axes[0, 0].set_title("Training patch balance")
    axes[0, 0].set_ylabel("Patches")
    source_counts = [
        sum(int(r["label"]) for r in training if r["source_id"] == name) for name in sorted(sources)
    ]
    axes[0, 1].hist(source_counts, bins=min(15, len(sources)), color="#2d7760")
    axes[0, 1].set_title("Positive patches per training source")
    axes[0, 1].set_xlabel("Positive patches")
    axes[0, 1].set_ylabel("Source images")
    axes[0, 2].hist(areas * 100, bins=20, color="#ce7444")
    axes[0, 2].set_title("Training defect sizes")
    axes[0, 2].set_xlabel("Annotated defect area (%)")
    axes[1, 0].hist(means, bins=20, color="#2d7760")
    axes[1, 0].set_title("Mean grayscale brightness")
    axes[1, 1].hist(contrasts, bins=20, color="#2d7760")
    axes[1, 1].set_title("Grayscale contrast (standard deviation)")
    axes[1, 2].hist(textures, bins=20, color="#2d7760")
    axes[1, 2].set_title("Texture total variation")
    fig.suptitle(f"Train-only EDA · {split['provenance']}", fontsize=13)
    fig.savefig(figures / "eda-summary.png")
    plt.close(fig)
    sample = [r for r in training if int(r["label"])][:6] + [
        r for r in training if not int(r["label"])
    ][:6]
    fig, axes = plt.subplots(3, 4, figsize=(10, 8), constrained_layout=True)
    for ax, row in zip(axes.ravel(), sample):
        with Image.open(resolve_under(root, row["image_path"])) as image:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        if row["mask_path"]:
            binary = (
                read_mask(
                    resolve_under(root, row["mask_path"]), row.get("mask_foreground") or "binary"
                )
                > 0
            )
            ax.contour(binary, levels=[0.5], colors=["#ffb64d"], linewidths=0.8)
        ax.set_title(f"Train · {'defective' if int(row['label']) else 'normal'}", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(sample) :]:
        ax.axis("off")
    fig.suptitle("Training examples; orange contours are annotation masks")
    fig.savefig(figures / "eda-training-examples.png")
    plt.close(fig)
    write_json(args.output / "docs" / "eda-statistics.json", stats)
    write_json(
        args.output / "docs" / "eda-modelling-findings.json",
        {"provenance": split["provenance"], "findings": findings},
    )
    print(f"Generated train-only EDA and {len(findings)} observed constraints")


if __name__ == "__main__":
    main()
