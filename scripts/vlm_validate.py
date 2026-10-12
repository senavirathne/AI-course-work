"""Select a real local VLM prompt on the complete validation partition."""

import argparse
from pathlib import Path

from fabric_inspector.common import code_hash, write_json
from fabric_inspector.data.manifest import load_manifest
from fabric_inspector.genai.vlm import LocalVLM, benchmark, default_vlm_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/split.json"))
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    split, rows, root = load_manifest(args.manifest, verify_files=True)
    validation = [r for r in rows if r["split"] == "val"]
    config = default_vlm_config()
    config["device"] = args.device
    try:
        adapter = LocalVLM(config)
    except Exception as error:
        write_json(
            args.output / "experiments" / "vlm-blocker.json",
            {
                "status": "NOT RUN",
                "provenance": split["provenance"],
                "config": config,
                "blocker": f"{type(error).__name__}: {error}",
            },
        )
        raise
    results = []
    for index in range(2):
        candidate = default_vlm_config(index)
        candidate["device"] = args.device
        result = benchmark(
            adapter,
            candidate,
            validation,
            root,
            args.output / "experiments" / f"vlm-validation-{index}.json",
            "val",
            split["provenance"],
        )
        results.append(result)
    selected = max(
        results,
        key=lambda r: (r["metrics"]["f2"], -r["failure_rate"], -r["config"]["prompt_index"]),
    )
    write_json(
        args.output / "artifacts" / "vlm-selection.json",
        {
            "status": "COMPLETE",
            "split_sha256": split["split_sha256"],
            "provenance": split["provenance"],
            "config": selected["config"],
            "selected_on": "val",
            "code_sha256": code_hash(Path(".")),
            "validation_metrics": selected["metrics"],
        },
    )


if __name__ == "__main__":
    main()
