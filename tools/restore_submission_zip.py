"""Restore and verify the committed coursework submission ZIP from its numbered parts."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(root, output=None, verify_only=False):
    root = root.resolve()
    manifest = json.loads((root / "submission/archive-manifest.json").read_text())
    if manifest["format"] != 1:
        raise ValueError("Unsupported archive manifest")
    name = manifest["archive_name"]
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Unsafe archive filename")
    output = output or root / "submission" / name
    output = output.resolve()
    for part in manifest["parts"]:
        path = (root / part["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Unsafe part path")
        if path.stat().st_size != part["bytes"] or sha256(path) != part["sha256"]:
            raise ValueError(f"Corrupted archive part: {part['path']}")
    if output.exists():
        if output.stat().st_size != manifest["archive_bytes"] or sha256(output) != manifest["archive_sha256"]:
            raise FileExistsError("Existing output differs; choose another --output path")
    elif verify_only:
        raise FileNotFoundError("Reassembled ZIP is missing; run without --verify-only")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".assembling")
        try:
            digest = hashlib.sha256()
            with temporary.open("wb") as target:
                for part in manifest["parts"]:
                    with (root / part["path"]).open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            target.write(block)
                            digest.update(block)
            if temporary.stat().st_size != manifest["archive_bytes"] or digest.hexdigest() != manifest["archive_sha256"]:
                raise ValueError("Reassembled archive checksum mismatch")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(output) as archive:
        if len(archive.namelist()) != manifest["archive_members"]:
            raise ValueError("Archive member count mismatch")
        if archive.testzip() is not None:
            raise ValueError("Archive CRC validation failed")
    print(f"PASS: {len(manifest['parts'])} parts and exact ZIP SHA-256 verified.")
    print(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    restore(args.root, args.output, args.verify_only)


if __name__ == "__main__":
    main()

