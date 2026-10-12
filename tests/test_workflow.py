"""Real PyTorch training/loading/HTTP/Grad-CAM integration on development images."""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from fastapi.testclient import TestClient

from app.main import create_app
from fabric_inspector.common import write_json, sha256
from fabric_inspector.data.images import preprocess, preprocessing_spec
from fabric_inspector.data.manifest import load_manifest
from fabric_inspector.explainability.gradcam import gradcam, spatial_metrics, occlusion_check
from fabric_inspector.models.neural import CustomCNN, MobileNetBinary
from fabric_inspector.training.runner import default_config, train_run, load_checkpoint

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    output = tmp_path_factory.mktemp("trained")
    config = default_config()
    config.update(epochs=1, cpu_threads=2)
    manifest = PROJECT / "data/development-manifests/split.json"
    run = train_run(manifest, output, config, PROJECT, "cpu", "test-trained")
    checkpoint = output / run["checkpoint"]
    model, metadata, device = load_checkpoint(checkpoint)
    bundle = {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "split_sha256": metadata["split_sha256"],
        "preprocessing": metadata["preprocessing"],
        "provenance": metadata["provenance"],
        "temperature": 1.0,
        "classification_threshold": run["validation_threshold"],
        "recommendation_thresholds": {"low": 0.2, "high": 0.7},
        "score_kind": "uncalibrated_model_score",
    }
    bundle_path = output / "artifacts" / "test-bundle.json"
    write_json(bundle_path, bundle)
    return output, bundle_path, model, metadata, device


def content(value):
    stream = BytesIO()
    Image.new("L", (256, 256), value).save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_custom_depth_is_actual_convolutional_depth(depth):
    model = CustomCNN(depth=depth).eval()
    assert sum(isinstance(m, torch.nn.Conv2d) for m in model.modules()) == depth
    with torch.inference_mode():
        assert model(torch.zeros(2, 1, 256, 256)).shape == (2,)


def test_mobilenet_accepts_replicated_grayscale_without_download():
    # Random initialisation is only an architecture test, never an application fallback.
    model = MobileNetBinary(pretrained=False).eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 256, 256)).shape == (1,)


def test_request_time_inference_matches_direct_torch_forward(trained):
    _, bundle_path, model, metadata, _ = trained
    with TestClient(create_app(bundle_path, allow_development=True)) as client:
        assert client.get("/health").json()["ready"]
        scores = []
        for value in [60, 180]:
            response = client.post(
                "/predict", files={"file": ("untrusted-name.png", content(value), "image/png")}
            )
            assert response.status_code == 200
            result = response.json()
            with torch.inference_mode():
                expected = float(
                    torch.sigmoid(
                        model(
                            preprocess(
                                Image.new("L", (256, 256), value), metadata["preprocessing"]
                            ).unsqueeze(0)
                        )
                    )[0]
                )
            assert result["defect_score"] == pytest.approx(expected, abs=1e-6)
            assert result["request_processing_ms"] > 0
            assert result["input_checks"]["ood_detection"] == "Not implemented or claimed"
            scores.append(result["defect_score"])
        assert scores[0] != pytest.approx(scores[1], abs=1e-8)


def test_preprocessing_is_shared_and_rgb_replication_is_explicit():
    image = Image.new("L", (256, 256), 128)
    gray = preprocess(image, preprocessing_spec("custom"))
    assert torch.allclose(gray, torch.full_like(gray, (128 / 255 - 0.5) / 0.5), atol=1e-7)
    rgb = preprocess(image, preprocessing_spec("mobilenet"))
    assert rgb.shape == (3, 256, 256)
    assert not torch.allclose(rgb[0], rgb[1])  # Different pretrained channel normalisation.


def test_missing_or_development_model_never_gets_a_fake_fallback(trained, tmp_path):
    _, bundle_path, _, _, _ = trained
    for path in [tmp_path / "missing.json", bundle_path]:
        with TestClient(create_app(path)) as client:
            assert not client.get("/health").json()["ready"]
            assert (
                client.post("/predict", files={"file": ("a.png", content(120))}).status_code == 503
            )


def test_http_invalid_upload_and_opacity_errors(trained):
    _, bundle_path, _, _, _ = trained
    with TestClient(create_app(bundle_path, allow_development=True)) as client:
        assert client.post("/predict", files={"file": ("a.png", b"plain text")}).status_code == 422
        assert (
            client.post(
                "/explain", files={"file": ("a.png", content(120))}, data={"alpha": 2}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/predict", files={"file": ("a.png", b"x" * (5 * 1024 * 1024 + 1))}
            ).status_code
            == 413
        )


def test_explanation_http_and_finite_spatial_alignment(trained):
    _, bundle_path, model, metadata, _ = trained
    image = Image.fromarray(np.random.default_rng(8).integers(0, 256, (256, 256), dtype=np.uint8))
    tensor = preprocess(image, metadata["preprocessing"]).unsqueeze(0)
    cam = gradcam(model, tensor)
    assert cam.shape == (256, 256) and np.isfinite(cam).all()
    assert cam.min() >= 0 and cam.max() <= 1
    check = occlusion_check(model, tensor, cam)
    assert set(check["logit_drops"]) == {"top", "low", "random"}
    assert all(np.isfinite(v) for v in check["logit_drops"].values())
    assert spatial_metrics(cam, np.zeros_like(cam))["mask_status"] == "empty"
    with TestClient(create_app(bundle_path, allow_development=True)) as client:
        response = client.post("/explain", files={"file": ("a.png", content(120))})
        assert response.status_code == 200
        assert response.json()["overlay"].startswith("data:image/png;base64,")


def test_artifact_tampering_is_rejected_before_inference(trained, tmp_path):
    output, _, _, metadata, _ = trained
    checkpoint = output / "artifacts/test-trained.pt"
    changed = tmp_path / "changed.pt"
    changed.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(changed, expected_sha=sha256(checkpoint))


def test_frozen_split_changed_csv_is_rejected(tmp_path):
    import shutil

    original = PROJECT / "data/development-manifests/split.json"
    shutil.copy(original, tmp_path / "split.json")
    shutil.copy(original.with_suffix(".csv"), tmp_path / "split.csv")
    with (tmp_path / "split.csv").open("a") as stream:
        stream.write("altered\n")
    with pytest.raises(ValueError, match="changed"):
        load_manifest(tmp_path / "split.json")
