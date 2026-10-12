"""Freeze real trained comparisons and a real validation-selected VLM before test access."""

import argparse
import os
from pathlib import Path

from fabric_inspector.common import code_hash, read_json, now, sha256, write_json
from fabric_inspector.data.manifest import REAL, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/split.json"))
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    target = args.output / "artifacts" / "final-evaluation-manifest.json"
    if target.exists():
        raise FileExistsError("Final evaluation manifest already frozen")
    split, rows, _ = load_manifest(args.manifest, verify_files=True)
    if split["provenance"] != REAL:
        raise ValueError("Synthetic or partial data cannot produce assessed final evidence")
    bundle_path = args.output / "artifacts" / "serving-bundle.json"
    bundle = read_json(bundle_path)
    vlm_path = args.output / "artifacts" / "vlm-selection.json"
    vlm = read_json(vlm_path)
    if vlm["status"] != "COMPLETE" or vlm["selected_on"] != "val":
        raise ValueError("A genuinely run validation-selected VLM is required")
    for record in [bundle, vlm]:
        if record["split_sha256"] != split["split_sha256"] or record["provenance"] != REAL:
            raise ValueError("Comparison split/provenance mismatch")
    ann = [
        read_json(p)
        for p in sorted((args.output / "experiments").glob("ann-search-[0-9][0-9][0-9].json"))
    ]
    svm = [read_json(p) for p in sorted((args.output / "experiments").glob("svm-search-*.json"))]
    completed_ann = [r for r in ann if r["status"] == "COMPLETE"]
    completed_svm = [r for r in svm if r["status"] == "COMPLETE"]
    if len(completed_ann) < 20 or len(completed_svm) < 24:
        raise ValueError(
            "The predeclared search budget must have logged attempts before final evaluation"
        )
    for run in ann + svm:
        if run["split_sha256"] != split["split_sha256"]:
            raise ValueError("Run from a different split in comparison logs")
    failures = read_json(args.output / "docs" / "failed-approaches.json")
    scientific = [r for r in failures.get("attempts", []) if r.get("diagnosis") and r.get("run_id")]
    logged_ids = {r["run_id"] for r in completed_ann + completed_svm}
    if len(scientific) < 4 or sum(bool(r.get("substantive")) for r in scientific) < 2:
        raise ValueError(
            "Four actual unsuccessful attempts and two diagnosed substantive failed approaches required"
        )
    if len({r["run_id"] for r in scientific}) < 4 or any(
        r["run_id"] not in logged_ids for r in scientific
    ):
        raise ValueError(
            "Failure diagnoses must reference four distinct actual completed experiments"
        )
    hypothesis = args.output / "docs" / "pre-evaluation-hypothesis.md"
    if not hypothesis.exists() or "PENDING" in hypothesis.read_text():
        raise ValueError("Record the pre-evaluation winner hypothesis before freezing")
    best_svm = max(
        completed_svm,
        key=lambda r: (
            r["validation_metrics"]["f2"],
            r["validation_metrics"].get("average_precision") or 0,
        ),
    )
    by_id = {r["run_id"]: r for r in completed_ann}
    comparisons = {
        "initial_ann": by_id[bundle["initial_ann_run_id"]],
        "final_ann": by_id[bundle["run_id"]],
        "classical": best_svm,
    }
    seed_path = args.output / "artifacts" / "seed-variability.json"
    seeds = read_json(seed_path)
    if seeds["split_sha256"] != split["split_sha256"] or seeds["provenance"] != REAL:
        raise ValueError("Seed evidence split/provenance mismatch")
    repetitions = [
        read_json(args.output / "experiments" / (run_id + ".json")) for run_id in seeds["run_ids"]
    ]
    artifact_files = {
        "serving-bundle.json": sha256(bundle_path),
        "vlm-selection.json": sha256(vlm_path),
        "seed-variability.json": sha256(seed_path),
    }
    for run in list(comparisons.values()) + repetitions:
        path = args.output / run["checkpoint"]
        if sha256(path) != run["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed since run logging")
        artifact_files[str(path.relative_to(args.output / "artifacts"))] = sha256(path)
    write_json(
        target,
        {
            "schema_version": 1,
            "frozen_at": now(),
            "code_sha256": code_hash(Path(".")),
            "provenance": REAL,
            "manifest": os.path.relpath(args.manifest, target.parent),
            "manifest_sha256": sha256(args.manifest),
            "split_sha256": split["split_sha256"],
            "test_patch_ids": [r["patch_id"] for r in rows if r["split"] == "test"],
            "artifact_sha256": artifact_files,
            "comparisons": comparisons,
            "seed_repetitions": repetitions,
            "serving_bundle": bundle,
            "vlm": vlm,
            "protocol": {
                "primary_metric": "defect-class F2",
                "thresholds": "validation-selected and frozen",
                "test_evaluation_count": 1,
                "vlm_unresolved": "Count as error, retain all examples",
                "timing": "Batch size 1, three warmup examples; end-to-end decode/preprocess included",
                "cpu_inference_threads": vlm["config"]["cpu_threads"],
                "bootstrap": "1000 source/duplicate-component resamples, seed 42",
                "xai": "Grad-CAM defect logit, fixed 10% top/low/random mean-fill occlusion",
            },
            "search_counts": {"ann_attempts": len(ann), "svm_attempts": len(svm)},
        },
    )
    print(
        f"Frozen {target}. Do not alter models, prompts, cutoffs or executable code after this point."
    )


if __name__ == "__main__":
    main()
