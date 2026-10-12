"""One preprocessing implementation for training, serving and explanations."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset

from fabric_inspector.common import resolve_under

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def decode_upload(content: bytes) -> Image.Image:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload must contain an image no larger than 5 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise ValueError("Only PNG and JPEG content is supported")
                if image.size != (256, 256):
                    raise ValueError(
                        "Provide a 256 x 256 fabric patch; full images are not resized"
                    )
                if getattr(image, "n_frames", 1) != 1 or image.mode not in {"L", "RGB"}:
                    raise ValueError(
                        "Use a single 8-bit grayscale or RGB image without transparency"
                    )
                image.load()
                return image.convert("L").copy()
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError("Invalid or unsafe image content") from error


def preprocessing_spec(model_name: str) -> dict:
    if model_name not in {"custom", "mobilenet"}:
        raise ValueError("Unknown model")
    return {
        "version": 1,
        "size": [256, 256],
        "grayscale": True,
        "channels": 1 if model_name == "custom" else 3,
        "mean": [0.5] if model_name == "custom" else [0.485, 0.456, 0.406],
        "std": [0.5] if model_name == "custom" else [0.229, 0.224, 0.225],
        "resize": False,
        "center_crop": False,
    }


def preprocess(image: Image.Image, spec: dict, augment=False) -> torch.Tensor:
    if image.size != tuple(spec["size"]):
        raise ValueError("Preprocessing expects the exact trained patch size")
    array = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
    if augment:
        # Fabric orientation is not a class label; flips preserve local weave
        # detail. No crop, blur, severe colour jitter or resize is introduced.
        if torch.rand(()) < 0.5:
            array = np.fliplr(array).copy()
        if torch.rand(()) < 0.5:
            array = np.flipud(array).copy()
    # Keep per-image arithmetic out of the multithreaded torch pool. These
    # float32 operations are bit-identical to the original tensor contract
    # for every 8-bit gray level in both supported preprocessing specifications.
    channels = np.repeat(array[None, :, :], spec["channels"], axis=0)
    mean = np.asarray(spec["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(spec["std"], dtype=np.float32)[:, None, None]
    return torch.from_numpy((channels - mean) / std)


class FabricDataset(Dataset):
    """Return image, label and opaque patch ID; masks never enter the classifier."""

    def __init__(self, rows: list[dict], root: Path, spec: dict, augment=False):
        self.rows, self.root, self.spec, self.augment = rows, root, spec, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(resolve_under(self.root, row["image_path"])) as image:
            tensor = preprocess(image, self.spec, self.augment)
        return tensor, torch.tensor(float(row["label"])), row["patch_id"]
