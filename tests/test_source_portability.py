"""A reinstall must not invalidate a scientifically frozen source hash."""

from fabric_inspector.common import code_hash


def test_source_hash_ignores_generated_install_metadata_but_tracks_code(tmp_path):
    source = tmp_path / "src" / "fabric_inspector"
    source.mkdir(parents=True)
    module = source / "model.py"
    module.write_text("MODEL_WIDTH = 8\n")
    original = code_hash(tmp_path)
    metadata = tmp_path / "src" / "fabric_inspector_coursework.egg-info"
    metadata.mkdir()
    record = metadata / "PKG-INFO"
    record.write_text("Generated wheel metadata from installation A\n")
    assert code_hash(tmp_path) == original
    record.write_text("Different optional dependencies and absolute paths at installation B\n")
    assert code_hash(tmp_path) == original
    module.write_text("MODEL_WIDTH = 16\n")
    assert code_hash(tmp_path) != original
