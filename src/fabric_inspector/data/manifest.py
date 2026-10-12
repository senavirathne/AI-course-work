"""Auditable manifests with immutable source-group partitions."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

from fabric_inspector.common import read_json, resolve_under, sha256, write_json, now

DEVELOPMENT = "SYNTHETIC_DEVELOPMENT_ONLY"
REAL = "RAW-FABRID_V1"


def read_mask(path: Path, foreground="binary") -> np.ndarray:
    """Interpret publisher masks consistently; preserve the original file bytes.

    The V1 files contain intermediate edge intensities. Its CSV defect areas
    count exactly-white pixels, so the real importer explicitly requests 255.
    """
    with Image.open(path) as mask:
        mask.load()
        a = np.asarray(mask)
        if mask.size != (256, 256) or a.ndim != 2 or a.dtype != np.uint8:
            raise ValueError(f"Invalid or misaligned binary mask: {path}")
        if foreground == "exact_white":
            return (a == 255).astype(np.uint8) * 255
        if foreground != "binary" or not np.isin(a, [0, 255]).all():
            raise ValueError(f"Invalid or misaligned binary mask: {path}")
        return a.copy()


class Components:
    """Union verified source relationships and duplicate image content."""

    def __init__(self, names):
        self.parents = {name: name for name in names}

    def find(self, name):
        parent = self.parents[name]
        if parent != name:
            self.parents[name] = self.find(parent)
        return self.parents[name]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parents[max(a, b)] = min(a, b)


def image_record(root: Path, row: dict) -> dict:
    path = resolve_under(root, row["image_path"])
    with Image.open(path) as image:
        image.load()
        if image.size != (256, 256) or image.mode != "L":
            raise ValueError(f"Expected 256x256 8-bit grayscale patch: {path}")
        pixels = np.asarray(image)
        thumbnail = np.asarray(image.resize((9, 8), Image.Resampling.BILINEAR))
    mask_path = row.get("mask_path", "")
    label = int(row["label"])
    if label not in (0, 1):
        raise ValueError("Invalid binary label")
    defect_pixels = 0
    if mask_path:
        a = read_mask(resolve_under(root, mask_path), row.get("mask_foreground", "binary"))
        defect_pixels = int(np.count_nonzero(a))
        if bool(defect_pixels) != bool(label):
            raise ValueError(f"Label/mask mismatch: {row['patch_id']}")
    elif label == 1:
        raise ValueError(f"Missing defect mask: {row['patch_id']}")
    elif not row.get("normal_confirmed", False):
        raise ValueError("A missing mask does not establish normal status")
    row = dict(row)
    row.update(
        sha256=sha256(path),
        pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
        mask_sha256=sha256(resolve_under(root, mask_path)) if mask_path else "",
        defect_pixels=defect_pixels,
        near_hash=f"{int.from_bytes(np.packbits(thumbnail[:, 1:] > thumbnail[:, :-1]).tobytes()):016x}",
        near_block_means=",".join(
            str(float(v)) for v in pixels.reshape(8, 32, 8, 32).mean(axis=(1, 3)).ravel()
        ),
    )
    return row


def connect_groups(rows: list[dict], root: Path, extra_fields=()) -> tuple[list[dict], dict]:
    """Conservatively join duplicate candidates after a pixel-distance check.

    A 64-bit difference hash screens candidates (Hamming distance <=2); mean
    absolute full-resolution difference <=3/255 confirms near matches. This
    does not detect all crops/rotations. Source mapping remains the main guard.
    """
    components = Components(r["patch_id"] for r in rows)
    seen = {}
    for field in ("source_id", "pixel_sha256", *extra_fields):
        for row in rows:
            key = (field, row.get(field))
            if not key[1]:
                continue
            if key in seen:
                components.union(row["patch_id"], seen[key])
            else:
                seen[key] = row["patch_id"]
    initially_connected = {r["patch_id"]: components.find(r["patch_id"]) for r in rows}
    # Index all hashes in eight-byte buckets. <=2 bit differences necessarily
    # leave at least six bytes equal; this avoids a quadratic full scan.
    buckets, candidates, near_matches = {}, set(), 0
    block_means = {
        r["patch_id"]: np.fromstring(r["near_block_means"], sep=",")
        for r in rows
        if r.get("near_block_means")
    }
    full_checks = 0
    for index, row in enumerate(rows):
        value = int(row["near_hash"], 16)
        possible = set()
        for slot in range(8):
            key = (slot, (value >> (8 * slot)) & 255)
            possible.update(buckets.get(key, []))
            buckets.setdefault(key, []).append(index)
        for other in possible:
            before = rows[other]
            if initially_connected[before["patch_id"]] == initially_connected[row["patch_id"]]:
                continue
            if (value ^ int(before["near_hash"], 16)).bit_count() > 2:
                continue
            pair = (other, index)
            if pair in candidates:
                continue
            candidates.add(pair)
            # Jensen's inequality: the mean difference between equal-area block
            # means is a lower bound on full-pixel MAE. Rejecting >3 is exact.
            if row["patch_id"] in block_means and before["patch_id"] in block_means:
                lower_bound = float(
                    np.abs(block_means[row["patch_id"]] - block_means[before["patch_id"]]).mean()
                )
                if lower_bound > 3:
                    continue
            full_checks += 1
            with Image.open(resolve_under(root, row["image_path"])) as a:
                aa = np.asarray(a, dtype=np.float32)
            with Image.open(resolve_under(root, before["image_path"])) as b:
                distance = float(np.abs(aa - np.asarray(b, dtype=np.float32)).mean())
            if distance <= 3:
                components.union(row["patch_id"], before["patch_id"])
                near_matches += 1
    for row in rows:
        row["group_id"] = components.find(row["patch_id"])
    return rows, {
        "near_candidate_pairs_checked": len(candidates),
        "near_match_pairs_joined": near_matches,
        "near_full_pixel_checks": full_checks,
        "exact_duplicate_count": len(rows) - len({r["pixel_sha256"] for r in rows}),
        "duplicate_limitations": "No guarantee for rotations, nonidentical crops, or unrecorded rolls",
    }


def assert_separation(rows: list[dict]) -> None:
    seen = {}
    for row in rows:
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Invalid partition")
        for field in ("source_id", "group_id", "pixel_sha256", "verified_extra_group"):
            if not row.get(field):
                continue
            key = (field, row[field])
            if key in seen and seen[key] != row["split"]:
                raise ValueError(f"Leakage across partitions: {field}={row[field]}")
            seen[key] = row["split"]
    for split in ("train", "val", "test"):
        if {int(r["label"]) for r in rows if r["split"] == split} != {0, 1}:
            raise ValueError(f"Both classes required in {split}")


def freeze_split(
    rows: list[dict],
    root: Path,
    output: Path,
    provenance: str,
    ratios=(0.7, 0.15, 0.15),
    seed=42,
    audit=None,
) -> dict:
    """Choose split by class/group coverage only, before any model results.

    Try a fixed, declared 128 grouping draws and minimise partition size and
    class-balance deviations. None of these draws trains or scores a model.
    """
    if output.exists() or output.with_suffix(".csv").exists():
        raise FileExistsError("Split already exists; reuse it rather than regenerate it")
    if len({r["patch_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate patch IDs")
    if not np.isclose(sum(ratios), 1) or min(ratios) <= 0:
        raise ValueError("Positive split ratios must sum to one")
    y = np.array([int(r["label"]) for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    if len(np.unique(groups)) < 3:
        raise ValueError("Insufficient independent source/duplicate groups")
    candidates = []
    for draw in range(128):
        train, rest = next(
            GroupShuffleSplit(n_splits=1, train_size=ratios[0], random_state=seed + draw).split(
                y, y, groups
            )
        )
        try:
            val_local, test_local = next(
                GroupShuffleSplit(
                    n_splits=1,
                    train_size=ratios[1] / (ratios[1] + ratios[2]),
                    random_state=seed + draw,
                ).split(y[rest], y[rest], groups[rest])
            )
        except ValueError:
            continue
        partitions = (train, rest[val_local], rest[test_local])
        if any(len(np.unique(y[idx])) < 2 for idx in partitions):
            continue
        objective = sum(
            abs(len(idx) / len(y) - ratio) + abs(y[idx].mean() - y.mean())
            for idx, ratio in zip(partitions, ratios)
        )
        candidates.append((float(objective), draw, partitions))
    if not candidates:
        raise ValueError(
            "Groups cannot support both classes in three partitions; revise protocol before training"
        )
    _, draw, partitions = min(candidates, key=lambda item: (item[0], item[1]))
    for name, indices in zip(("train", "val", "test"), partitions):
        for index in indices:
            rows[int(index)]["split"] = name
    assert_separation(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["patch_id"]))
    record = {
        "created_at": now(),
        "provenance": provenance,
        "data_root": os.path.relpath(root.resolve(), output.parent.resolve()),
        "rows_csv": csv_path.name,
        "split_sha256": sha256(csv_path),
        "seed": seed,
        "integrity_only_split_draws": 128,
        "selected_draw": draw,
        "requested_ratios": list(ratios),
        "audit": audit or {},
        "counts": {
            name: {
                "patches": len(idx),
                "positive": int(y[idx].sum()),
                "source_images": len({rows[int(i)]["source_id"] for i in idx}),
                "independent_groups": len(np.unique(groups[idx])),
                "actual_ratio": len(idx) / len(y),
            }
            for name, idx in zip(("train", "val", "test"), partitions)
        },
    }
    write_json(output, record)
    return record


def load_manifest(path: Path, verify_files=False) -> tuple[dict, list[dict], Path]:
    record = read_json(path)
    csv_path = resolve_under(path.parent, record["rows_csv"])
    if sha256(csv_path) != record["split_sha256"]:
        raise ValueError("Frozen split CSV was changed")
    rows = list(csv.DictReader(csv_path.open()))
    root = (path.parent / record["data_root"]).resolve()
    assert_separation(rows)
    if verify_files:
        for row in rows:
            if sha256(resolve_under(root, row["image_path"])) != row["sha256"]:
                raise ValueError(f"Image changed since split: {row['patch_id']}")
            if (
                row["mask_path"]
                and sha256(resolve_under(root, row["mask_path"])) != row["mask_sha256"]
            ):
                raise ValueError(f"Mask changed since split: {row['patch_id']}")
    return record, rows, root
