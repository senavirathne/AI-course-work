"""Complete the interrupted, unchanged validation protocol without losing raw responses."""

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from fabric_inspector.common import code_hash, environment, now, read_json, write_json
from fabric_inspector.data.manifest import load_manifest
from fabric_inspector.evaluation.metrics import metrics
from fabric_inspector.genai.vlm import LocalVLM, benchmark, default_vlm_config, parse_decision


def main():
    if Path("reports/final-evaluation/test-access.json").exists():
        raise ValueError("Validation prompt selection is forbidden after test access")
    split, rows, root = load_manifest(Path("data/manifests/split.json"), verify_files=True)
    validation = [r for r in rows if r["split"] == "val"]
    config = default_vlm_config()
    adapter = LocalVLM(config)
    output = Path("experiments/vlm-validation-0.json")
    if not output.exists():
        original = Path("experiments/vlm-validation-0.responses.jsonl")
        recovered = Path("docs/engineering-runs/vlm-validation-0-recovered.responses.jsonl")
        if original.exists() and not recovered.exists():
            original.rename(recovered)
        if recovered.exists():
            # The saved entry point loops prompts in order, starting with index 0.
            # Retain its exact source and require identical model/prompt/parser code.
            archive = Path("docs/source-versions/recovered-checkpoint-sources.zip")
            with zipfile.ZipFile(archive) as z:
                for source in ["src/fabric_inspector/genai/vlm.py", "scripts/vlm_validate.py"]:
                    if z.read(source) != Path(source).read_bytes():
                        raise ValueError("Recovered VLM protocol differs; do not reuse responses")
            prior = [json.loads(line) for line in recovered.read_text().splitlines()]
            for entry, row in zip(prior, validation, strict=False):
                if entry["patch_id"] != row["patch_id"] or entry["image_sha256"] != row["sha256"]:
                    raise ValueError("Recovered responses do not match the exact ordered images")
                for response in entry["attempts"]:
                    if response.get("resolved_revision") not in [None, config["revision"]]:
                        raise ValueError("Recovered model revision mismatch")
                if entry["prediction"] != -1:
                    if (
                        parse_decision(entry["attempts"][-1]["raw_response"], "label")
                        != entry["prediction"]
                    ):
                        raise ValueError("Recovered parsing mismatch")
            if not 0 < len(prior) < len(validation):
                raise ValueError("Expected an incomplete validation prefix")
            part = Path("experiments/vlm-validation-0-continuation.json")
            tail = (
                read_json(part)
                if part.exists()
                else benchmark(
                    adapter,
                    config,
                    validation[len(prior) :],
                    root,
                    part,
                    "val",
                    split["provenance"],
                )
            )
            entries = prior + tail["examples"]
            assert [e["patch_id"] for e in entries] == [r["patch_id"] for r in validation]
            times = [e["end_to_end_ms"] for e in entries]
            combined = dict(tail)
            combined.update(
                timestamp=now(),
                examples=entries,
                metrics=metrics(
                    [int(r["label"]) for r in validation], [e["prediction"] for e in entries]
                ),
                mean_latency_ms=float(np.mean(times)),
                p95_latency_ms=float(np.quantile(times, 0.95)),
                failure_rate=float(np.mean([e["prediction"] == -1 for e in entries])),
                recovery={
                    "preserved_prefix": len(prior),
                    "continuation": len(tail["examples"]),
                    "original_response_log": recovered.as_posix(),
                    "protocol_source_archive": archive.as_posix(),
                    "note": "Same immutable model, prompt and parser. Validation timing spans two execution sessions with concurrent training; final test timing is measured separately.",
                },
            )
            original.write_text("".join(json.dumps(e, allow_nan=False) + "\n" for e in entries))
            write_json(output, combined)
        else:
            benchmark(adapter, config, validation, root, output, "val", split["provenance"])
    second = Path("experiments/vlm-validation-1.json")
    if not second.exists():
        benchmark(
            adapter, default_vlm_config(1), validation, root, second, "val", split["provenance"]
        )
    results = [read_json(output), read_json(second)]
    selected = max(
        results,
        key=lambda r: (r["metrics"]["f2"], -r["failure_rate"], -r["config"]["prompt_index"]),
    )
    write_json(
        Path("artifacts/vlm-selection.json"),
        {
            "status": "COMPLETE",
            "split_sha256": split["split_sha256"],
            "provenance": split["provenance"],
            "config": selected["config"],
            "selected_on": "val",
            "code_sha256": code_hash(Path(".")),
            "validation_metrics": selected["metrics"],
            "environment": environment(),
            "prompt_sha256": hashlib.sha256(selected["config"]["prompt"].encode()).hexdigest(),
        },
    )
    print("Complete validation-only prompt selection", flush=True)


if __name__ == "__main__":
    main()
