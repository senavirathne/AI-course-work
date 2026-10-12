"""Restore the exact large coursework files stored in numbered Git-compatible parts."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(root, verify_only=False):
    manifest = json.loads((root / "repository-assets" / "manifest.json").read_text())
    for entry in manifest["files"]:
        target = root / entry["path"]
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe target path")
        for part in entry["parts"]:
            path = root / part["path"]
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Unsafe part path")
            if path.stat().st_size != part["bytes"] or digest(path) != part["sha256"]:
                raise ValueError(f"Corrupted repository part: {part['path']}")
        if target.is_file() and target.stat().st_size == entry["bytes"] and digest(target) == entry["sha256"]:
            print(f"PASS {entry['path']} (already restored)", flush=True)
            continue
        if verify_only:
            raise ValueError(f"Missing or corrupt restored file: {entry['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".restore.tmp")
        try:
            h = hashlib.sha256()
            with temp.open("wb") as output:
                for part in entry["parts"]:
                    with (root / part["path"]).open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            output.write(block)
                            h.update(block)
            if temp.stat().st_size != entry["bytes"] or h.hexdigest() != entry["sha256"]:
                raise ValueError(f"Reassembled checksum mismatch: {entry['path']}")
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        print(f"PASS {entry['path']} (restored exact bytes)", flush=True)
    return len(manifest["files"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    count = restore(args.root.resolve(), args.verify_only)
    print(f"Verified {count} large coursework files.")


if __name__ == "__main__":
    main()

