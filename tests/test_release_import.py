"""Test actual schema mapping and patch-level labels with an explicit publisher-format fixture."""

import csv

import numpy as np
from PIL import Image
import pytest

from prepare_data import publisher_rows


def release_fixture(tmp_path):
    rows = []
    for name, folder, defect in [
        ("normal.png", "train/good", False),
        ("defect.png", "test/defect", True),
    ]:
        destination = tmp_path / "MVTec" / folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (256, 256), 128).save(destination)
        if defect:
            mask = np.zeros((256, 256), dtype=np.uint8)
            mask[10:15, 10:15] = 255
            destination = tmp_path / "MVTec" / "ground_truth/defect" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(destination)
        # Both patches share a defective parent; the normal patch stays normal.
        rows.append(
            {
                "actual_patch_field": name,
                "actual_source_field": "same-parent.png",
                "actual_area_field": 25 if defect else 0,
            }
        )
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return metadata


def test_patch_label_is_not_inherited_from_defective_parent(tmp_path):
    metadata = release_fixture(tmp_path)
    rows = publisher_rows(
        tmp_path,
        metadata,
        "actual_patch_field",
        "actual_source_field",
        area_column="actual_area_field",
    )
    assert {r["patch_id"]: r["label"] for r in rows} == {"normal.png": 0, "defect.png": 1}
    assert len({r["source_id"] for r in rows}) == 1


def test_missing_release_image_is_not_silently_dropped(tmp_path):
    metadata = release_fixture(tmp_path)
    (tmp_path / "MVTec/train/good/normal.png").unlink()
    with pytest.raises(ValueError, match="Missing images"):
        publisher_rows(tmp_path, metadata, "actual_patch_field", "actual_source_field")


def test_publisher_columns_are_never_guessed(tmp_path):
    metadata = release_fixture(tmp_path)
    with pytest.raises(ValueError, match="actual schema"):
        publisher_rows(tmp_path, metadata, "invented_filename", "invented_roll")
