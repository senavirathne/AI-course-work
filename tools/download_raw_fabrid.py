"""Acquire the exact public archive in verified ranges; preserve resumable chunks."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import time
import urllib.request

URL = "https://data.mendeley.com/public-files/datasets/db6g85xsyg/files/93e17d48-b5e6-45f7-9064-be84cc5fb91c/file_downloaded"
SIZE = 4133046386
DIGEST = "f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9"
root = Path("data/raw")
root.mkdir(parents=True, exist_ok=True)
target_existing = root / "RAW-FABRID-V1.zip"
if target_existing.exists():
    checked = hashlib.sha256()
    with target_existing.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            checked.update(block)
    if checked.hexdigest() != DIGEST:
        raise ValueError("Existing archive does not match publisher checksum")
    print("Existing official archive checksum verified.")
    raise SystemExit(0)

chunks = root / ".download-chunks"
chunks.mkdir(exist_ok=True)
width = 64 * 1024 * 1024


def fetch(start):
    end = min(SIZE, start + width) - 1
    path = chunks / f"{start:012d}.part"
    if path.exists() and path.stat().st_size == end - start + 1:
        return path
    for attempt in range(4):
        temporary = path.with_suffix(".pending")
        try:
            request = urllib.request.Request(
                URL + f"?range_start={start}&range_end={end}",
                headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"},
            )
            with urllib.request.urlopen(request, timeout=50) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                    raise ValueError("Server returned a different archive range")
                with temporary.open("wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            if temporary.stat().st_size != end - start + 1:
                raise ValueError("Truncated archive range")
            temporary.replace(path)
            return path
        except Exception as error:
            temporary.unlink(missing_ok=True)
            print(f"Retry range {start}: {type(error).__name__}: {str(error)[:100]}", flush=True)
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


started = time.monotonic()
starts = list(range(0, SIZE, width))
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(fetch, start): start for start in starts}
    completed = 0
    for future in as_completed(futures):
        path = future.result()
        completed += path.stat().st_size
        print(f"Dataset {completed / SIZE:.1%}: {completed / 1e6:.0f} MB, {time.monotonic() - started:.0f}s", flush=True)
target = root / "RAW-FABRID-V1.zip"
digest = hashlib.sha256()
with target.with_suffix(".assembling").open("wb") as stream:
    for start in starts:
        with (chunks / f"{start:012d}.part").open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
                stream.write(block)
assert digest.hexdigest() == DIGEST, "Publisher archive checksum mismatch"
target.with_suffix(".assembling").replace(target)
for path in chunks.iterdir():
    path.unlink()
chunks.rmdir()
(root / "publisher-archive.json").write_text(json.dumps({"dataset": "RAW-FABRID V1", "url": URL, "bytes": SIZE, "publisher_sha256": DIGEST, "verified_sha256": digest.hexdigest(), "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
print("Full official archive checksum verified.", flush=True)
