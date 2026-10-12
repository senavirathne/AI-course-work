"""Exercise real checkpoint selection with a coexisting search-plan file."""

from pathlib import Path
import sys

from fabric_inspector.common import read_json, write_json
from fabric_inspector.training.runner import default_config, train_run
import select_models


def test_actual_validation_selection_ignores_nonrun_search_plan(monkeypatch, tmp_path):
    project = Path(__file__).resolve().parents[1]
    manifest = project / "data/development-manifests/split.json"
    config = default_config()
    config.update(epochs=1, cpu_threads=1)
    run = train_run(manifest, tmp_path, config, project, "cpu", "ann-search-000")
    write_json(tmp_path / "experiments/ann-search-plan.json", {"configurations": [config]})
    monkeypatch.setattr(
        sys, "argv", ["select_models.py", "--manifest", str(manifest), "--output", str(tmp_path)]
    )
    select_models.main()
    selected = read_json(tmp_path / "artifacts/serving-bundle.json")
    predictions = read_json(tmp_path / "artifacts/validation-predictions.json")
    assert selected["run_id"] == run["run_id"]
    assert selected["checkpoint_sha256"] == run["checkpoint_sha256"]
    assert selected["split_sha256"] == run["split_sha256"]
    assert selected["validation_metrics"]["n"] == len(predictions["predictions"])
    assert predictions["partition"] == "val"
