"""Reconstruct lost final artifacts with the unchanged, previously frozen protocol."""

from pathlib import Path
import subprocess
import sys

from fabric_inspector.common import code_hash, now, read_json, sha256, write_json


def main():
    audit_path = Path("docs/engineering-runs/final-results-workspace-reset.json")
    audit = read_json(audit_path)
    assert audit["status"] == "RECONSTRUCTING"
    frozen_path = Path("artifacts/final-evaluation-manifest.json")
    assert sha256(frozen_path) == audit["frozen_manifest_sha256"]
    assert code_hash(Path(".")) == audit["predictive_source_sha256"]
    assert not Path("reports/final-evaluation/test-access.json").exists()
    audit["reconstruction_started_at"] = now()
    write_json(audit_path, audit)
    try:
        subprocess.run([sys.executable, "-u", "scripts/final_evaluate.py"], check=True)
        comparison_path = Path("reports/final-evaluation/comparison.json")
        comparison = read_json(comparison_path)
        observed = {r["model"]: r for r in comparison["comparisons"]}
        for name, expected in audit["historical_test_confusions_tp_fp_fn"].items():
            result = (
                observed[name]
                if name in observed
                else read_json(Path("reports/final-evaluation") / (name + ".json"))
            )
            actual = [result["metrics"][key] for key in ["tp", "fp", "fn"]]
            assert actual == expected, (name, actual, expected)
        assert observed["vlm"]["metrics"]["unresolved_count"] == audit["historical_vlm_unresolved"]
        assert code_hash(Path(".")) == audit["predictive_source_sha256"]
        audit.update(
            status="COMPLETE", reconstruction_completed_at=now(),
            historical_confusion_crosschecks="PASS: all six scored outputs and VLM unresolved count match prior observations",
            reconstructed_per_example_files="Actual fresh inference on every frozen test example; no summary or historical value substituted",
        )
        write_json(audit_path, audit)
        comparison["workspace_recovery"] = audit_path.as_posix()
        comparison["physical_test_computation_count"] = 2
        write_json(comparison_path, comparison)
        guard_path = Path("reports/final-evaluation/test-access.json")
        guard = read_json(guard_path)
        guard.update(workspace_recovery=audit_path.as_posix(), physical_test_computation_count=2)
        write_json(guard_path, guard)
        print("COMPLETE: genuine frozen-protocol reconstruction; historical confusion crosschecks passed", flush=True)
    except Exception as error:
        audit.update(status="FAILED_RECONSTRUCTION", error=f"{type(error).__name__}: {error}")
        write_json(audit_path, audit)
        raise


if __name__ == "__main__":
    main()
