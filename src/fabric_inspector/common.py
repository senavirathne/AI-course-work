"""Small reproducibility helpers shared by command-line workflows."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically write strict JSON; NaN must never resemble a valid result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def code_hash(project: Path) -> str:
    """Hash executable sources and protocol settings, independently of generated results."""
    files = []
    for directory in ("src", "app", "scripts", "configs"):
        files.extend(
            p
            for p in (project / directory).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and not any(part.endswith(".egg-info") for part in p.parts)
            and p.suffix != ".pyc"
        )
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def environment() -> dict:
    packages = {}
    for package in (
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scikit-learn",
        "scikit-image",
        "fastapi",
        "transformers",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "NOT INSTALLED"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset root: {relative}")
    return path
