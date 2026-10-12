"""The assessed final workflow must refuse the bundled synthetic data."""

from pathlib import Path
import sys

import pytest

import freeze


def test_synthetic_manifest_cannot_be_frozen_as_assessed_evidence(monkeypatch, tmp_path):
    manifest = Path(__file__).resolve().parents[1] / "data/development-manifests/split.json"
    monkeypatch.setattr(
        sys, "argv", ["freeze.py", "--manifest", str(manifest), "--output", str(tmp_path)]
    )
    with pytest.raises(ValueError, match="Synthetic or partial"):
        freeze.main()
    assert not (tmp_path / "artifacts/final-evaluation-manifest.json").exists()
