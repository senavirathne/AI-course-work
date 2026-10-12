"""Run the complete scientific study into a fresh output directory."""

import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys

from fabric_inspector.common import now, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    source = Path.cwd().resolve()
    output = args.output.resolve()
    if output == source or (output.exists() and any(output.iterdir())):
        raise ValueError(
            "Choose a new empty reproduction directory; submitted results are immutable"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "docs").mkdir(exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    shutil.copytree(source / "data/manifests", output / "data/manifests")
    (output / "data/raw").symlink_to(source / "data/raw", target_is_directory=True)
    manifest = output / "data/manifests/split.json"
    for name in ["pre-evaluation-hypothesis.md", "study-budget.json", "ai-use-declaration.md"]:
        shutil.copyfile(source / "docs" / name, output / "docs" / name)
    env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0", OPENBLAS_NUM_THREADS="1")
    commands = [
        ("scripts/eda.py", []),
        ("scripts/train.py", ["ann", "--epochs", "4", "--device", "cpu"]),
        ("tools/reproduce_classical_pilots.py", []),
        ("scripts/train.py", ["svm", "--start-index", "24"]),
        ("tools/refine_ann.py", ["--device", "cpu"]),
        ("scripts/select_models.py", []),
        ("scripts/repeat_seeds.py", ["--device", "cpu"]),
        ("scripts/xai_validate.py", []),
        ("scripts/vlm_validate.py", ["--device", "cpu"]),
        ("scripts/make_figures.py", []),
    ]
    for script, extra in commands:
        command = [sys.executable, script, *extra, "--output", str(output)]
        if script != "scripts/make_figures.py":
            command.extend(["--manifest", str(manifest)])
        print("Running", script, extra, flush=True)
        with (output / (Path(script).stem + "-execution.log")).open("w") as log:
            subprocess.run(
                command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    template = read_json(source / "docs/failed-approaches.json")
    failures = copy.deepcopy(template)
    failures["recorded_at"] = now()
    for failure in failures["attempts"]:
        attempt = read_json(output / "experiments" / (failure["run_id"] + ".json"))
        control = read_json(output / "experiments" / (failure["control_run_id"] + ".json"))
        a, c = attempt["validation_metrics"], control["validation_metrics"]
        if a["f2"] >= c["f2"]:
            raise ValueError(
                "Reproduction changed a failure conclusion. Preserve results and diagnose the new evidence before freezing; do not copy the original diagnosis."
            )
        failure["evidence"] = {
            "partition": "val",
            "examples": a["n"],
            "attempt_f2": a["f2"],
            "control_f2": c["f2"],
            "f2_delta": a["f2"] - c["f2"],
            "attempt_average_precision": a["average_precision"],
            "control_average_precision": c["average_precision"],
            "attempt_tp_fp_fn": [a["tp"], a["fp"], a["fn"]],
            "control_tp_fp_fn": [c["tp"], c["fp"], c["fn"]],
        }
        failure["diagnosis"] = (
            f"Reproduced within the recorded budget: F2 {a['f2']:.6f} versus {c['f2']:.6f}, AP {a['average_precision']:.6f} versus {c['average_precision']:.6f}, FP/FN {a['fp']}/{a['fn']} versus {c['fp']}/{c['fn']}. The attempted setting/approach underperforms this comparator on the selection metric. The stated control scope bounds causal interpretation; no test evidence is used. Review the new learning curves and origin limitation rather than reusing historical numerical claims."
        )
    write_json(output / "docs/failed-approaches.json", failures)
    subprocess.run(
        [sys.executable, "tools/summarize_experiments.py", "--output", str(output)],
        cwd=source,
        env=env,
        check=True,
    )
    if not args.validation_only:
        subprocess.run(
            [
                sys.executable,
                "scripts/freeze.py",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
            ],
            cwd=source,
            env=env,
            check=True,
        )
        subprocess.run(
            [sys.executable, "scripts/final_evaluate.py", "--output", str(output)],
            cwd=source,
            env=env,
            check=True,
        )
        subprocess.run(
            [sys.executable, str(source / "tools/finish_evidence.py"), "--post-test"],
            cwd=output,
            env=env,
            check=True,
        )
    print("Reproduction complete:", output, flush=True)


if __name__ == "__main__":
    main()
