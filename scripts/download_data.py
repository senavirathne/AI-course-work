"""Download a user-supplied publisher file URL and record provenance.

Mendeley's download URL must come from the V1 release page; it is not guessed.
An already downloaded archive can be recorded with --archive. Extraction is
separate so the user can inspect the archive first.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from fabric_inspector.common import now, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Exact Download All/file URL from the approved V1 page")
    parser.add_argument("--archive", type=Path, default=Path("data/raw/RAW-FABRID-V1.zip"))
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if args.url:
        if not args.url.startswith("https://"):
            parser.error("HTTPS publisher download URL required")
        temporary = args.archive.with_suffix(args.archive.suffix + ".part")
        request = urllib.request.Request(
            args.url, headers={"User-Agent": "FabricInspectorCoursework/0.3"}
        )
        try:
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("wb") as stream,
            ):
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
            temporary.replace(args.archive)
        finally:
            temporary.unlink(missing_ok=True)
    if not args.archive.exists():
        parser.error(
            "Download the V1 archive from https://data.mendeley.com/datasets/db6g85xsyg/1 first"
        )
    digest = sha256(args.archive)
    if args.expected_sha256 and digest != args.expected_sha256:
        raise ValueError("Archive checksum mismatch")
    write_json(
        args.archive.parent / "download-provenance.json",
        {
            "dataset": "RAW-FABRID",
            "version": 1,
            "doi": "10.17632/db6g85xsyg.1",
            "source": "https://data.mendeley.com/datasets/db6g85xsyg/1",
            "license": "CC BY 4.0",
            "authors": "Pérez-Llorens, Albero-Albero and Silvestre-Blanes (2026)",
            "recorded_at": now(),
            "archive": args.archive.name,
            "archive_sha256": digest,
            "retrieval": "Downloaded by this script"
            if args.url
            else "User-supplied downloaded archive",
            "archive_bytes": args.archive.stat().st_size,
        },
    )
    print(f"Recorded archive SHA-256: {digest}")


if __name__ == "__main__":
    main()
