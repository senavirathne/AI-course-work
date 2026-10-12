"""Inspect actual publisher schemas, audit patches, and freeze the supervised split."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from fabric_inspector.common import sha256, write_json
from fabric_inspector.data.manifest import (
    DEVELOPMENT,
    REAL,
    connect_groups,
    freeze_split,
    image_record,
)


def inspect_release(root: Path) -> dict:
    """Report observed schemas, without guessing publisher column names."""
    schemas = {}
    for path in sorted(root.rglob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            schemas[str(path.relative_to(root))] = {
                "columns": reader.fieldnames,
                "first_row": next(reader, None),
            }
    return {
        "csv_schemas": schemas,
        "png_files": sum(1 for _ in root.rglob("*.png")),
        "note": "Supply the observed patch and source columns explicitly. Roll IDs are not assumed.",
    }


def publisher_rows(
    root: Path,
    metadata: Path,
    patch_column: str,
    source_column: str,
    group_column: str | None = None,
    area_column: str | None = None,
) -> list[dict]:
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        needed = [
            patch_column,
            source_column,
            *([group_column] if group_column else []),
            *([area_column] if area_column else []),
        ]
        if any(c not in (reader.fieldnames or []) for c in needed):
            raise ValueError(f"Column mapping does not match actual schema: {reader.fieldnames}")
        source = {}
        for row in reader:
            name = Path(row[patch_column]).name
            if not name or name in source or not row[source_column].strip():
                raise ValueError("Missing or ambiguous patch-to-source mapping")
            source[name] = row
    roots = [p for p in root.rglob("MVTec") if p.is_dir()]
    if root.name == "MVTec":
        roots.insert(0, root)
    if len(roots) != 1:
        raise ValueError("Expected exactly one publisher MVTec directory")
    mvtec = roots[0]
    rows = []
    for folder, label in (("train/good", 0), ("test/good", 0), ("test/defect", 1)):
        for path in sorted((mvtec / folder).glob("*.png")):
            if path.name not in source:
                raise ValueError(f"Unmapped publisher patch: {path.name}")
            meta = source[path.name]
            mask = mvtec / "ground_truth/defect" / path.name
            record = {
                "patch_id": path.name,
                "image_path": str(path.relative_to(root)),
                "source_id": meta[source_column],
                "label": label,
                "mask_path": str(mask.relative_to(root)) if mask.exists() else "",
                "normal_confirmed": label == 0,
                "mask_foreground": "exact_white",
                "published_partition": folder,
            }
            if group_column:
                record["verified_extra_group"] = meta[group_column]
            checked = image_record(root, record)
            if area_column and int(float(meta[area_column])) != checked["defect_pixels"]:
                raise ValueError(f"Metadata/mask defect-area mismatch: {path.name}")
            rows.append(checked)
    if {r["patch_id"] for r in rows} != set(source):
        raise ValueError(
            "Missing images or extra metadata records; audit/quarantine before proceeding"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--patch-column")
    parser.add_argument("--source-column")
    parser.add_argument("--group-column", help="Only a documented physical/capture relationship")
    parser.add_argument(
        "--source-metadata",
        type=Path,
        help="Observed high-resolution metadata for a verified parent-to-roll join",
    )
    parser.add_argument("--source-metadata-key", default="filename")
    parser.add_argument("--source-group-column", default="roll_id")
    parser.add_argument("--area-column")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/manifests/split.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.inspect:
        print(inspect_release(args.inspect.resolve()))
        return
    if args.development:
        root = Path("data/development").resolve()
        source = list(csv.DictReader((root / "metadata.csv").open()))
        rows = [
            image_record(
                root,
                {
                    "patch_id": r["patch_id"],
                    "source_id": r["parent_id"],
                    "image_path": r["image_path"],
                    "mask_path": r["mask_path"],
                    "label": int(r["is_defective"]),
                    "normal_confirmed": r["is_defective"] == "0",
                },
            )
            for r in source
        ]
        provenance = DEVELOPMENT
    else:
        if not all([args.release, args.metadata, args.patch_column, args.source_column]):
            parser.error(
                "Use --inspect first, then provide release, metadata, patch-column and source-column"
            )
        root = args.release.resolve()
        rows = publisher_rows(
            root,
            args.metadata.resolve(),
            args.patch_column,
            args.source_column,
            args.group_column,
            args.area_column,
        )
        if args.source_metadata:
            if args.group_column:
                raise ValueError("Use one documented grouping source")
            with args.source_metadata.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not {args.source_metadata_key, args.source_group_column}.issubset(
                    reader.fieldnames or []
                ):
                    raise ValueError("Source metadata column mapping differs from observed schema")
                source_groups = {}
                for item in reader:
                    key, group = item[args.source_metadata_key], item[args.source_group_column]
                    if key in source_groups or not group:
                        raise ValueError("Ambiguous or missing source-to-roll relationship")
                    source_groups[key] = group
            if {r["source_id"] for r in rows} != set(source_groups):
                raise ValueError(
                    "High-resolution source metadata does not exactly cover patch parents"
                )
            for row in rows:
                row["verified_extra_group"] = source_groups[row["source_id"]]
        provenance = REAL
        observed = Counter(r["published_partition"] for r in rows)
        expected = {"train/good": 14196, "test/good": 4969, "test/defect": 687}
        if dict(observed) != expected:
            raise ValueError(
                f"Release differs from published V1 patch counts: {dict(observed)}. Investigate before training."
            )
    rows, audit = connect_groups(
        rows, root, ("verified_extra_group",) if args.group_column or args.source_metadata else ()
    )
    audit.update(
        metadata_sha256=sha256(args.metadata) if args.metadata else sha256(root / "metadata.csv"),
        verified_patch_counts=dict(Counter(int(r["label"]) for r in rows)),
        source_images=len({r["source_id"] for r in rows}),
        label_definition="At least one exact-white (255) annotated defect pixel; normal status explicitly confirmed",
        mask_policy="Intermediate edge intensities excluded; exact 255 agrees with publisher CSV defect areas. Raw mask bytes preserved.",
        physical_roll_information="Not assumed; only use an explicitly documented group column",
    )
    if args.source_metadata:
        audit.update(
            source_metadata_sha256=sha256(args.source_metadata),
            verified_group_field=args.source_group_column,
            verified_source_groups=len(set(source_groups.values())),
            group_origin="Explicit join through publisher HighRes_Metadata.csv; no filename inference",
        )
    # This file is an integrity audit, not test-driven exploratory analysis.
    result = freeze_split(rows, root, args.output, provenance, seed=args.seed, audit=audit)
    write_json(args.output.parent / "data-audit.json", result)
    print(f"Frozen {len(rows)} {provenance} patches; split hash {result['split_sha256']}")


if __name__ == "__main__":
    main()
