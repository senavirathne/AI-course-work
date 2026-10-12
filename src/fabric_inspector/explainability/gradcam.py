"""Grad-CAM of the defect logit, with matched-area perturbation controls."""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def gradcam(model, tensor: torch.Tensor) -> np.ndarray:
    """Return a finite, image-aligned heatmap; a zero CAM remains zero."""
    if tensor.ndim != 4 or len(tensor) != 1:
        raise ValueError("Grad-CAM expects exactly one image")
    captured = []
    handle = model.cam_layer.register_forward_hook(
        lambda module, inputs, output: captured.append(output)
    )
    model.eval()
    try:
        with torch.enable_grad():
            image = tensor.detach().requires_grad_(True)
            score = model(image)[0]  # target the defect logit, even for a predicted normal patch
            activation = captured[-1]
            gradients = torch.autograd.grad(score, activation)[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
    finally:
        handle.remove()
    if not np.isfinite(cam).all():
        raise ValueError("Nonfinite explanation")
    maximum = float(cam.max())
    return cam / maximum if maximum > 0 else np.zeros_like(cam)


def overlay(image: Image.Image, cam: np.ndarray, alpha=0.55) -> Image.Image:
    if not 0 <= alpha <= 1:
        raise ValueError("Opacity must be between zero and one")
    from matplotlib import colormaps

    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    colors = colormaps["inferno"](cam)[..., :3] * 255
    # No attribution means no painted highlight, rather than an invented hotspot.
    intensity = (alpha * cam)[..., None]
    return Image.fromarray(
        np.clip(original * (1 - intensity) + colors * intensity, 0, 255).astype(np.uint8)
    )


def spatial_metrics(cam: np.ndarray, mask: np.ndarray) -> dict:
    binary = mask > 0
    if cam.shape != binary.shape:
        raise ValueError("Explanation/mask dimensions differ")
    if not binary.any():
        return {"mask_status": "empty", "pointing_hit": None, "attribution_mass_in_mask": None}
    total = float(cam.sum())
    if total == 0:
        return {
            "mask_status": "positive",
            "pointing_hit": None,
            "attribution_mass_in_mask": 0.0,
            "uniform_mass_reference": float(binary.mean()),
            "zero_cam": True,
        }
    point = np.unravel_index(np.argmax(cam), cam.shape)
    return {
        "mask_status": "positive",
        "pointing_hit": bool(binary[point]),
        "attribution_mass_in_mask": float(cam[binary].sum()) / total,
        "uniform_mass_reference": float(binary.mean()),
        "random_point_hit_expectation": float(binary.mean()),
        "zero_cam": False,
    }


def occlusion_check(model, tensor, cam, fraction=0.1, seed=42) -> dict:
    """Compare equal-area top, low and random pixel deletion at a fixed 10% area.

    Replace with the within-image channel mean, and measure defect-logit drop.
    This is a perturbation test, not causal proof; deletion can be unnatural.
    """
    if not 0 < fraction < 1:
        raise ValueError("Occlusion area must be a fraction between zero and one")
    n = cam.size
    k = max(1, int(n * fraction))
    order = np.argsort(cam.ravel(), kind="stable")
    rng = np.random.default_rng(seed)
    positions = {"top": order[-k:], "low": order[:k], "random": rng.choice(n, k, replace=False)}
    model.eval()
    with torch.inference_mode():
        base = float(model(tensor)[0])
        drops = {}
        for name, selected in positions.items():
            changed = tensor.clone()
            flat = changed[0].reshape(changed.shape[1], -1)
            mean = flat.mean(dim=1, keepdim=True)
            flat[:, torch.as_tensor(selected.copy(), device=tensor.device)] = mean
            drops[name] = base - float(model(changed)[0])
    return {
        "area_fraction": k / n,
        "seed": seed,
        "original_defect_logit": base,
        "logit_drops": drops,
        "top_minus_random_drop": drops["top"] - drops["random"],
        "top_minus_low_drop": drops["top"] - drops["low"],
        "degenerate_zero_cam": bool(cam.max() == 0),
        "interpretation": "More positive top-region drop than controls supports sensitivity; no causal claim",
    }
