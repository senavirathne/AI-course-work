"""Regression checks for integrity issues observed in the actual V1 archive."""

import numpy as np
from PIL import Image
import pytest
from fabric_inspector.data.manifest import read_mask, assert_separation
from fabric_inspector.genai.vlm import parse_decision


def test_intermediate_mask_edges_are_explicitly_interpreted(tmp_path):
    a = np.zeros((256, 256), dtype=np.uint8)
    a[1, :4] = [1, 93, 250, 255]
    path = tmp_path / "mask.png"
    Image.fromarray(a).save(path)
    with pytest.raises(ValueError):
        read_mask(path)
    interpreted = read_mask(path, "exact_white")
    assert np.count_nonzero(interpreted) == 1
    assert interpreted[1, 3] == 255
    assert np.array_equal(np.asarray(Image.open(path)), a)


def test_roll_relationship_must_not_cross_partitions():
    rows = [
        dict(
            patch_id=str(i),
            source_id=str(i),
            group_id=str(i),
            pixel_sha256=str(i),
            verified_extra_group="roll-A",
            split=split,
            label=i % 2,
        )
        for i, split in enumerate(["train", "train", "val", "val", "test", "test"])
    ]
    with pytest.raises(ValueError, match="verified_extra_group"):
        assert_separation(rows)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("defective", 1),
        ("defect_free", 0),
        (" The fabric is defective.", 1),
        ("This fabric patch is defect-free.", 0),
    ],
)
def test_declared_label_normalisation(text, expected):
    assert parse_decision(text, "label") == expected


@pytest.mark.parametrize(
    "text",
    [
        "not defective",
        "defective or defect_free",
        "The fabric is defective. Please inspect it.",
        "There is possibly a defect.",
    ],
)
def test_ambiguous_vlm_prose_stays_unresolved(text):
    with pytest.raises(ValueError):
        parse_decision(text, "label")
