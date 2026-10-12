"""Understandable PyTorch CNNs, following the lecturer's layer/training examples."""

from __future__ import annotations

import torch
from torch import nn


class CustomCNN(nn.Module):
    """Two, three or four convolutional blocks with a binary logit output."""

    def __init__(self, depth=3, width=16, dropout=0.2):
        super().__init__()
        if depth not in {2, 3, 4} or width not in {8, 16, 32} or not 0 <= dropout < 1:
            raise ValueError("Unsupported custom architecture")
        blocks, channels = [], 1
        for index in range(depth):
            out_channels = width * 2**index
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(2),
                )
            )
            channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels, 1))

    @property
    def cam_layer(self):
        # The last block's ReLU, before pooling, preserves twice the final map resolution.
        return self.features[-1][2]

    def forward(self, images):
        return self.classifier(self.pool(self.features(images))).squeeze(1)


class MobileNetBinary(nn.Module):
    """Compact pretrained candidate; all layers are fine-tuned."""

    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.network = mobilenet_v3_small(weights=weights)
        self.network.classifier[-1] = nn.Linear(self.network.classifier[-1].in_features, 1)

    @property
    def cam_layer(self):
        return self.network.features[-1]

    def forward(self, images):
        return self.network(images).squeeze(1)


def build_model(spec: dict, initialise_pretrained=False) -> nn.Module:
    if spec["name"] == "custom":
        return CustomCNN(spec.get("depth", 3), spec.get("width", 16), spec.get("dropout", 0.2))
    if spec["name"] == "mobilenet":
        return MobileNetBinary(pretrained=initialise_pretrained)
    raise ValueError("Unknown ANN architecture")


def choose_device(requested="auto") -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("Use auto, cpu, cuda or mps")
    return torch.device(requested)


def architecture_table(model: nn.Module, spec: dict) -> dict:
    """Record actual output tensor shapes and parameter counts with hooks."""
    records, handles = [], []
    for name, layer in model.named_modules():
        if not list(layer.children()):

            def hook(module, inputs, output, layer_name=name):
                records.append(
                    {
                        "layer": layer_name,
                        "type": type(module).__name__,
                        "input": list(inputs[0].shape),
                        "output": list(output.shape),
                        "parameters": sum(p.numel() for p in module.parameters(recurse=False)),
                    }
                )

            handles.append(layer.register_forward_hook(hook))
    previous = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 1 if spec["name"] == "custom" else 3, 256, 256))
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous)
    return {
        "layers": records,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
