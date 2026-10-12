"""Load a selected trained checkpoint once; never substitute a fake prediction."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from threading import RLock
import time

import numpy as np
import torch

from fabric_inspector.common import read_json, resolve_under, sha256
from fabric_inspector.data.images import decode_upload, preprocess
from fabric_inspector.data.manifest import REAL
from fabric_inspector.explainability.gradcam import gradcam, overlay
from fabric_inspector.training.runner import load_checkpoint, synchronize


def adjusted_scores(scores, temperature=1.0):
    scores = np.asarray(scores, dtype=float)
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped)) / temperature
    return 1 / (1 + np.exp(-logits))


def png_data(image):
    stream = BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


class Predictor:
    def __init__(self, bundle_path: Path, device="cpu", allow_development=False):
        torch.set_num_threads(int(os.getenv("FABRIC_CPU_THREADS", "4")))
        self.bundle = read_json(bundle_path)
        if self.bundle["provenance"] != REAL and not allow_development:
            raise ValueError("Development weights require FABRIC_ALLOW_DEVELOPMENT=1")
        checkpoint = resolve_under(bundle_path.parent, self.bundle["checkpoint"])
        self.model, self.metadata, self.device = load_checkpoint(
            checkpoint, device, self.bundle["checkpoint_sha256"]
        )
        if self.metadata["split_sha256"] != self.bundle["split_sha256"]:
            raise ValueError("Bundle/checkpoint split mismatch")
        if self.metadata["provenance"] != self.bundle["provenance"]:
            raise ValueError("Bundle/checkpoint provenance mismatch")
        if self.bundle["preprocessing"] != self.metadata["preprocessing"]:
            raise ValueError("Bundle/checkpoint preprocessing mismatch")
        self.version = sha256(checkpoint)[:12]
        self.lock = RLock()  # Hooks and gradients are isolated from concurrent requests.

    def predict(self, content: bytes, explain=False, alpha=0.55) -> dict:
        start = time.perf_counter()
        image = decode_upload(content)
        tensor = preprocess(image, self.metadata["preprocessing"]).unsqueeze(0).to(self.device)
        with self.lock:
            synchronize(self.device)
            model_start = time.perf_counter()
            with torch.inference_mode():
                raw_score = float(torch.sigmoid(self.model(tensor))[0])
            synchronize(self.device)
            model_ms = (time.perf_counter() - model_start) * 1000
            score = float(adjusted_scores([raw_score], self.bundle["temperature"])[0])
            recommendation = self.bundle["recommendation_thresholds"]
            action = (
                "routine_inspection"
                if score < recommendation["low"]
                else "manual_review"
                if score < recommendation["high"]
                else "priority_hold"
            )
            result = {
                "class": "defective"
                if score >= self.bundle["classification_threshold"]
                else "defect_free",
                "defect_score": score,
                "score_kind": self.bundle["score_kind"],
                "recommendation": action,
                "model_version": self.version,
                "run_id": self.metadata["run_id"],
                "provenance": self.bundle["provenance"],
                "model_inference_ms": model_ms,
                "device": str(self.device),
                "classification_threshold": self.bundle["classification_threshold"],
                "recommendation_cutoffs": {
                    "low": recommendation["low"],
                    "high": recommendation["high"],
                },
                "input_checks": {
                    "mean_gray": float(np.asarray(image).mean()),
                    "std_gray": float(np.asarray(image).std()),
                    "ood_detection": "Not implemented or claimed",
                },
                "limitations": "Patch-level decision support only. Unseen fabrics, phone lighting, blur and tiny defects can cause errors. The inspector decides.",
            }
            if explain:
                explanation_start = time.perf_counter()
                cam = gradcam(self.model, tensor)
                result.update(
                    original=png_data(image),
                    overlay=png_data(overlay(image, cam, alpha)),
                    explanation_ms=(time.perf_counter() - explanation_start) * 1000,
                    explanation={
                        "method": "Grad-CAM",
                        "target": "defect logit",
                        "zero_attribution": bool(cam.max() == 0),
                        "meaning": "Regions influencing the model score; not a certified defect boundary or causal proof",
                    },
                )
        result["request_processing_ms"] = (time.perf_counter() - start) * 1000
        return result
