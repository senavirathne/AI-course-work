"""Package code, authentic sample, logs and required trained artifacts without raw GB data."""

import argparse
import hashlib
from pathlib import Path
import zipfile

from fabric_inspector.common import now, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    required_svm = {"svm-search-024.joblib", "svm-search-036.joblib"}
    svm_runs = [read_json(p) for p in Path("experiments").glob("svm-search-*.json")]
    complete = [r for r in svm_runs if r.get("status") == "COMPLETE"]
    if complete:
        best = max(
            complete,
            key=lambda r: (
                r["validation_metrics"]["f2"],
                r["validation_metrics"].get("average_precision") or 0,
            ),
        )
        required_svm.add(Path(best["checkpoint"]).name)
    frozen = Path("artifacts/final-evaluation-manifest.json")
    if frozen.exists():
        required_svm.update(
            Path(r["checkpoint"]).name
            for r in read_json(frozen)["comparisons"].values()
            if r["checkpoint"].endswith(".joblib")
        )
    if not args.progress:
        verification = read_json(Path("verification.json"))
        if verification["technical_status"] != "COMPLETE":
            raise ValueError("A final archive requires completed technical verification")
    records = []
    for p in sorted(Path(".").rglob("*")):
        if not p.is_file():
            continue
        parts = p.parts
        if any(
            x in parts
            for x in [
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                ".git",
                ".venv",
                "node_modules",
            ]
        ):
            continue
        if any(x.endswith(".egg-info") for x in parts):
            continue
        if p.suffix == ".pyc":
            continue
        # Atomic model writes can leave large staging files after interruptions.
        # They are not the named, hashed scientific checkpoints.
        if p.name.endswith(".tmp") or (parts[0] == "artifacts" and p.name.startswith(".")):
            continue
        if parts[0] in {"output", "repository-assets", "submission", "work", "tmp"} or "recording" in parts:
            continue
        if p.name in ["internal-browser-qa.png", "browser-spec.json"]:
            continue
        if parts[:2] == ("data", "raw") and p.name != "publisher-archive.json":
            continue
        if parts[0] == "artifacts" and p.suffix == ".joblib" and p.name not in required_svm:
            continue
        records.append(p)
    omitted = sorted(
        {r["checkpoint"] for r in complete if Path(r["checkpoint"]).name not in required_svm}
    )
    # Prefer intentionally captured project logs; never add a duplicate ZIP member.
    included_names = {p.as_posix() for p in records}
    extra_logs = {
        f"docs/execution-logs/{p.name}": p
        for p in sorted(Path("..").glob("*.log"))
        if f"docs/execution-logs/{p.name}" not in included_names
    }
    index = Path("docs/packaged-artifacts.json")
    index_record = {
        "created_at": now(),
        "progress_checkpoint": args.progress,
        "retained_svm_checkpoints": sorted(required_svm),
        "omitted_svm_checkpoints": omitted,
        "reason": "All measured run records retained. Nonselected SVM support-vector artifacts can be large; the final comparison checkpoint and the substantive failed descriptor/control checkpoints are included. Reproduce other weights with the supplied full-data training scripts.",
        "raw_dataset": "Excluded because archive exceeds 500 MB; checksum-verified download script and authentic sample included.",
        "hashing": "SHA-256 of the exact bytes streamed into each ZIP member; safe for active progress logs",
    }
    checksums = Path("submission-files.sha256")
    records = [p for p in records if p not in {index, checksums}]
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.destination.with_suffix(".assembling")
    prefix = "fabric_defect_detection_coursework3"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        file_hashes = {}

        def add_file(path, name):
            digest = hashlib.sha256()
            with (
                path.open("rb") as source,
                archive.open(f"{prefix}/{name}", "w", force_zip64=True) as target,
            ):
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
                    digest.update(block)
            file_hashes[name] = digest.hexdigest()

        for p in records:
            add_file(p, p.as_posix())
        for name, p in extra_logs.items():
            add_file(p, name)
        checksums.write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(file_hashes.items()))
        )
        add_file(checksums, checksums.as_posix())
        index_record["file_sha256"] = file_hashes
        write_json(index, index_record)
        archive.write(index, f"{prefix}/{index.as_posix()}")
    tmp.replace(args.destination)
    with zipfile.ZipFile(args.destination) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Duplicate archive member")
        if archive.testzip() is not None:
            raise ValueError("Archive CRC check failed")
    print(
        {
            "archive": str(args.destination.resolve()),
            "bytes": args.destination.stat().st_size,
            "files": len(records) + len(extra_logs) + 2,
            "sha256": sha256(args.destination),
            "progress": args.progress,
        }
    )


if __name__ == "__main__":
    main()
