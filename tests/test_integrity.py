"""Guard consequential scientific and input-contract failures."""

from io import BytesIO

import numpy as np
from PIL import Image
import pytest
from sklearn.metrics import fbeta_score

from fabric_inspector.data.images import decode_upload
from fabric_inspector.data.manifest import (
    assert_separation,
    connect_groups,
    image_record,
)
from fabric_inspector.evaluation.metrics import metrics, select_threshold, group_bootstrap
from fabric_inspector.genai.vlm import parse_decision, benchmark, default_vlm_config


def image_bytes(mode="L", size=(256, 256), value=128, format="PNG"):
    output = BytesIO()
    Image.new(mode, size, value).save(output, format=format)
    return output.getvalue()


def test_f2_matches_independent_library_and_manual_counts():
    labels, decisions = [1, 1, 0, 0], [1, 0, 1, 0]
    result = metrics(labels, decisions, [0.9, 0.2, 0.6, 0.1])
    assert result["tp"] == result["tn"] == result["fp"] == result["fn"] == 1
    assert result["f2"] == fbeta_score(labels, decisions, beta=2) == 0.5


def test_unresolved_vlm_keeps_all_examples_and_penalises_both_classes():
    result = metrics([1, 1, 0, 0], [1, -1, 0, -1])
    assert result["n"] == 4 and result["unresolved_count"] == 2
    assert result["f2"] == 0.5 and result["accuracy"] == 0.5


def test_reference_and_zero_division_behaviour():
    assert metrics([0, 1], [0, 0])["precision"] == 0
    assert metrics([0, 1], [0, 0])["f2"] == 0
    assert metrics([0, 0], [0, 0], [0.1, 0.2])["roc_auc"] is None


def test_threshold_uses_validation_and_agrees_with_recomputed_metric():
    labels, scores = [1, 1, 0, 0], [0.8, 0.7, 0.6, 0.1]
    threshold, result = select_threshold(labels, scores)
    assert result["f2"] == 1
    assert metrics(labels, np.asarray(scores) >= threshold)["f2"] == result["f2"]
    with pytest.raises(ValueError):
        select_threshold([0, 0], [0.1, 0.2])


def test_source_and_pixel_duplicates_cannot_cross_partitions():
    rows = [
        {
            "patch_id": str(i),
            "source_id": str(i),
            "group_id": str(i),
            "pixel_sha256": str(i),
            "split": split,
            "label": label,
        }
        for i, (split, label) in enumerate(
            [("train", 0), ("train", 1), ("val", 0), ("val", 1), ("test", 0), ("test", 1)]
        )
    ]
    assert_separation(rows)
    rows[2]["source_id"] = rows[0]["source_id"]
    with pytest.raises(ValueError, match="Leakage"):
        assert_separation(rows)
    rows[2]["source_id"] = "2"
    rows[4]["pixel_sha256"] = rows[0]["pixel_sha256"]
    with pytest.raises(ValueError, match="Leakage"):
        assert_separation(rows)


def test_missing_mask_does_not_mean_normal(tmp_path):
    Image.new("L", (256, 256), 100).save(tmp_path / "image.png")
    row = {
        "patch_id": "a",
        "source_id": "source",
        "image_path": "image.png",
        "mask_path": "",
        "label": 0,
    }
    with pytest.raises(ValueError, match="normal status"):
        image_record(tmp_path, row)
    row.update(label=1)
    with pytest.raises(ValueError, match="Missing defect mask"):
        image_record(tmp_path, row)


def test_mask_label_alignment_and_binary_convention(tmp_path):
    Image.new("L", (256, 256), 100).save(tmp_path / "image.png")
    Image.new("L", (256, 256), 0).save(tmp_path / "mask.png")
    row = {
        "patch_id": "a",
        "source_id": "source",
        "image_path": "image.png",
        "mask_path": "mask.png",
        "label": 1,
    }
    with pytest.raises(ValueError, match="mismatch"):
        image_record(tmp_path, row)
    Image.new("L", (64, 64), 255).save(tmp_path / "mask.png")
    with pytest.raises(ValueError, match="misaligned"):
        image_record(tmp_path, row)


def test_duplicate_content_connects_different_sources(tmp_path):
    rng = np.random.default_rng(8)
    image = Image.fromarray(rng.integers(0, 256, (256, 256), dtype=np.uint8))
    rows = []
    for index in range(2):
        image.save(tmp_path / f"{index}.png")
        rows.append(
            image_record(
                tmp_path,
                {
                    "patch_id": str(index),
                    "source_id": f"source{index}",
                    "image_path": f"{index}.png",
                    "mask_path": "",
                    "label": 0,
                    "normal_confirmed": True,
                },
            )
        )
    joined, audit = connect_groups(rows, tmp_path)
    assert joined[0]["group_id"] == joined[1]["group_id"]
    assert audit["exact_duplicate_count"] == 1


@pytest.mark.parametrize(
    "content",
    [
        b"not an image",
        image_bytes(size=(255, 256)),
        image_bytes(mode="RGBA", value=(10, 20, 30, 255)),
        image_bytes(format="GIF"),
        b"x" * (5 * 1024 * 1024 + 1),
    ],
)
def test_upload_rejects_invalid_content_or_contract(content):
    with pytest.raises(ValueError):
        decode_upload(content)


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_upload_accepts_documented_channels(mode):
    image = decode_upload(image_bytes(mode=mode))
    assert image.mode == "L" and image.size == (256, 256)


@pytest.mark.parametrize(
    "response, decision",
    [
        (' {"decision":"defective"} ', 1),
        ('{"decision":"defect_free"}', 0),
        ('```json\n{"decision":"defective"}\n```', 1),
    ],
)
def test_vlm_structured_parser(response, decision):
    assert parse_decision(response) == decision


@pytest.mark.parametrize(
    "response",
    ["defective", "{}", '{"decision":"unsure"}', '{"decision":"defective","confidence":1}', "[]"],
)
def test_vlm_parser_rejects_unsupported_outputs(response):
    with pytest.raises((ValueError, TypeError)):
        parse_decision(response)


def test_vlm_errors_are_logged_and_not_dropped(tmp_path):
    # Adapter test double only: this is not a foundation-model benchmark.
    class BrokenAdapter:
        def predict_image(self, image, prompt):
            assert isinstance(image, Image.Image)
            assert "ground_truth" not in prompt
            return {"raw_response": "cannot decide", "latency_ms": 2.0}

    Image.new("L", (256, 256), 120).save(tmp_path / "a.png")
    rows = [{"patch_id": "opaque-id", "image_path": "a.png", "sha256": "fixture", "label": 1}]
    result = benchmark(
        BrokenAdapter(),
        default_vlm_config(),
        rows,
        tmp_path,
        tmp_path / "result.json",
        "val",
        "TEST_FIXTURE",
    )
    assert result["metrics"]["n"] == result["metrics"]["unresolved_count"] == 1
    assert result["failure_rate"] == 1
    assert len(result["examples"][0]["attempts"]) == 2


def test_bootstrap_reports_independent_group_limit():
    assert group_bootstrap([1, 0], [1, 0], ["same", "same"])["f2_95_percent_interval"] is None
    result = group_bootstrap([1, 0, 1, 0], [1, 0, 1, 0], ["a", "a", "b", "b"], repeats=30)
    assert result["f2_95_percent_interval"] == [1, 1]
