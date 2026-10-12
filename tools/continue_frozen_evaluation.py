"""Resume a stopped, unchanged test computation without clearing its access guard.

Use only after an infrastructure interruption. Completed scored-model files and
every valid recorded VLM response are retained. No training or selection occurs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from fabric_inspector.common import (
    code_hash,
    environment,
    now,
    read_json,
    resolve_under,
    sha256,
    write_json,
)
from fabric_inspector.data.manifest import REAL, load_manifest
from fabric_inspector.evaluation.metrics import metrics
from fabric_inspector.genai.vlm import LocalVLM, benchmark, parse_decision
from fabric_inspector.training.runner import load_checkpoint

sys.path.insert(0, str(Path("scripts").resolve()))
from xai_validate import evaluate_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason", required=True, help="Observed infrastructure interruption")
    args = parser.parse_args()
    frozen_path = Path("artifacts/final-evaluation-manifest.json")
    frozen = read_json(frozen_path)
    assert code_hash(Path(".")) == frozen["code_sha256"]
    manifest = (frozen_path.parent / frozen["manifest"]).resolve()
    assert sha256(manifest) == frozen["manifest_sha256"]
    split, all_rows, root = load_manifest(manifest, verify_files=True)
    assert split["split_sha256"] == frozen["split_sha256"] and split["provenance"] == REAL
    rows = [r for r in all_rows if r["split"] == "test"]
    assert [r["patch_id"] for r in rows] == frozen["test_patch_ids"]
    for name, digest in frozen["artifact_sha256"].items():
        assert sha256(resolve_under(frozen_path.parent, name)) == digest
    directory = Path("reports/final-evaluation")
    guard_path = directory / "test-access.json"
    guard = read_json(guard_path)
    assert guard["status"] in ["STARTED", "FAILED_PARTIAL"]
    assert not (directory / "comparison.json").exists()
    original = directory / "vlm.responses.jsonl"
    assert time.time() - original.stat().st_mtime > 90, "Response log is active; do not resume"
    original_bytes = original.read_bytes()
    prefix = [json.loads(line) for line in original_bytes.splitlines()]
    config = frozen["vlm"]["config"]

    def validate(entries, offset):
        for entry, row in zip(entries, rows[offset : offset + len(entries)], strict=True):
            assert entry["patch_id"] == row["patch_id"] and entry["image_sha256"] == row["sha256"]
            assert entry["attempts"] and len(entry["attempts"]) <= config["max_retries"] + 1
            for attempt in entry["attempts"]:
                assert attempt.get("resolved_revision") in [None, config["revision"]]
            if entry["prediction"] != -1:
                assert (
                    parse_decision(entry["attempts"][-1]["raw_response"], config["response_format"])
                    == entry["prediction"]
                )

    assert 0 < len(prefix) <= len(rows)
    validate(prefix, 0)
    engineering = Path("docs/engineering-runs")
    # A prior continuation can itself have stopped before its suffix was merged.
    for path in sorted(engineering.glob("test-continuation-*.responses.jsonl")):
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        if not entries:
            continue
        start = frozen["test_patch_ids"].index(entries[0]["patch_id"])
        validate(entries, start)
        if start == len(prefix):
            prefix.extend(entries)
        else:
            assert start + len(entries) <= len(prefix)
            assert entries == prefix[start : start + len(entries)]
    names = [
        "initial_ann",
        "final_ann",
        "classical",
        "seed_repeat_0",
        "seed_repeat_1",
        "seed_repeat_2",
    ]
    retained = {name: sha256(directory / (name + ".json")) for name in names}
    for name in names:
        result = read_json(directory / (name + ".json"))
        assert result["metrics"]["n"] == len(rows)
        assert [p["patch_id"] for p in result["predictions"]] == frozen["test_patch_ids"]
    preserved = engineering / f"test-preserved-prefix-{len(prefix):05d}.responses.jsonl"
    if preserved.exists():
        assert preserved.read_bytes() == original_bytes
    else:
        preserved.write_bytes(original_bytes)
    index = len(list(engineering.glob("test-resumption-*.json"))) + 1
    audit_path = engineering / f"test-resumption-{index:03d}.json"
    audit = {
        "status": "CONTINUING",
        "reason": args.reason,
        "started_at": now(),
        "code_sha256": frozen["code_sha256"],
        "frozen_manifest_sha256": sha256(frozen_path),
        "guard_never_cleared": True,
        "original_guard": guard,
        "preserved_prefix_count": len(prefix),
        "preserved_raw_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "retained_completed_results_sha256": retained,
        "selection_changed": False,
        "timing_note": "VLM times span unchanged CPU-one-thread execution segments; restart/model loading and segment warmups excluded. Original warmup responses may not have been persisted before interruption.",
    }
    write_json(audit_path, audit)
    adapter = LocalVLM(config)
    assert original.read_bytes() == original_bytes, "Original worker is still writing"
    tail_path = engineering / f"test-continuation-{len(prefix):05d}.json"
    if len(prefix) < len(rows):
        tail = benchmark(adapter, config, rows[len(prefix) :], root, tail_path, "test", REAL)
        entries = prefix + tail["examples"]
    else:
        tail = {"warmup_examples": 0, "warmup_responses": []}
        entries = prefix
    validate(entries, 0)
    assert len(entries) == len(rows)
    del adapter
    times = [e["end_to_end_ms"] for e in entries]
    vlm = {
        **tail,
        "status": "COMPLETE",
        "timestamp": now(),
        "partition": "test",
        "provenance": REAL,
        "config": config,
        "environment": environment(),
        "examples": entries,
        "prompt_sha256": hashlib.sha256(config["prompt"].encode()).hexdigest(),
        "metrics": metrics([int(r["label"]) for r in rows], [e["prediction"] for e in entries]),
        "mean_latency_ms": float(np.mean(times)),
        "p95_latency_ms": float(np.quantile(times, 0.95)),
        "failure_rate": float(np.mean([e["prediction"] == -1 for e in entries])),
        "labels_used_only_for_scoring": True,
        "no_probability_or_auc_claim": True,
        "latency_scope": "Decode, processor, generation/parsing/retries; excludes one-time model loading and response-log writes",
        "batch_size": 1,
        "resumption": audit_path.as_posix(),
    }
    assert original.read_bytes() == original_bytes
    staged = original.with_suffix(".jsonl.tmp")
    staged.write_text("".join(json.dumps(e, allow_nan=False) + "\n" for e in entries))
    staged.replace(original)
    write_json(directory / "vlm.json", vlm)
    labels = [int(r["label"]) for r in rows]
    results = [
        {
            "model": name,
            "metrics": metrics(labels, [constant] * len(rows)),
            "training_seconds": 0,
            "timing": "Constant output; separate from trained-model timing",
        }
        for name, constant in [("always_normal", 0), ("always_defective", 1)]
    ]
    results.extend(read_json(directory / (name + ".json")) for name in frozen["comparisons"])
    results.append(
        {
            "model": "vlm",
            "metrics": vlm["metrics"],
            "mean_end_to_end_ms": vlm["mean_latency_ms"],
            "failure_rate": vlm["failure_rate"],
            "resumption": audit_path.as_posix(),
        }
    )
    seed_scores = [
        read_json(directory / f"seed_repeat_{i}.json")["metrics"]["f2"] for i in range(3)
    ]
    bundle = frozen["serving_bundle"]
    model, metadata, device = load_checkpoint(
        Path("artifacts") / bundle["checkpoint"], expected_sha=bundle["checkpoint_sha256"]
    )
    evaluate_cases(model, metadata, device, bundle, rows, root, directory / "xai", "test")
    assert retained == {name: sha256(directory / (name + ".json")) for name in names}
    write_json(
        directory / "comparison.json",
        {
            "status": "COMPLETE",
            "timestamp": now(),
            "environment": environment(),
            "provenance": REAL,
            "comparisons": results,
            "test_seed_repetitions": {
                "f2": seed_scores,
                "mean": float(np.mean(seed_scores)),
                "sample_std": float(np.std(seed_scores, ddof=1)),
            },
            "workspace_recovery": "docs/engineering-runs/final-results-workspace-reset.json",
            "resumption": audit_path.as_posix(),
        },
    )
    audit.update(
        status="COMPLETE",
        completed_at=now(),
        final_examples=len(entries),
        completed_scored_files_unchanged=True,
    )
    write_json(audit_path, audit)
    write_json(
        guard_path,
        {
            **guard,
            "status": "COMPLETE",
            "completed_at": now(),
            "resumption": audit_path.as_posix(),
            "frozen_manifest_sha256": sha256(frozen_path),
        },
    )
    print(
        "COMPLETE: only missing VLM responses were executed; existing scored files and guard retained"
    )


if __name__ == "__main__":
    main()
