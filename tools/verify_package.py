"""Extract the final archive elsewhere and verify portable paths and fresh inference."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from fabric_inspector.common import now, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/package-verification.json"))
    args = parser.parse_args()
    archive_path = args.archive.resolve()
    with tempfile.TemporaryDirectory(prefix="fabric-portability-") as temporary:
        target = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            assert archive.testzip() is None
            names = archive.namelist()
            for name in names:
                assert (target / name).resolve().is_relative_to(target)
            assert not any("internal-browser-qa.png" in n or "/recording/" in n for n in names)
            assert not any(
                n.endswith(".tmp")
                or ("/artifacts/" in n and Path(n).name.startswith("."))
                for n in names
            ), "Temporary serialization files must be excluded"
            assert not any(
                "data/raw/" in n and not n.endswith("publisher-archive.json") for n in names
            )
            archive.extractall(target)
        project = target / "fabric_defect_detection_coursework3"
        assert (project / "verification.json").exists()
        script = """
import sys,os,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
os.environ['FABRIC_CPU_THREADS']='1'
from fabric_inspector.common import read_json,sha256,code_hash
from fabric_inspector.data.manifest import load_manifest
from fabric_inspector.serving.predictor import Predictor
verified=read_json(Path('verification.json'))
assert verified['technical_status']=='COMPLETE'
assert verified['code_sha256']==code_hash(Path('.'))
sample,rows,root=load_manifest(Path('data/sample-manifests/split.json'),verify_files=True)
assert len(rows)==168
index=read_json(Path('docs/packaged-artifacts.json'))
assert not index['progress_checkpoint']
for path,digest in index['file_sha256'].items():
    assert sha256(Path(path))==digest,path
predictor=Predictor(Path('artifacts/serving-bundle.json'))
cases=read_json(Path('data/demo/cases.json'))['cases']
for case in cases:
    content=Path(case['path']).read_bytes()
    assert sha256(Path(case['path']))==case['image_sha256']
    result=predictor.predict(content)
    assert abs(result['defect_score']-case['expected_score'])<1e-5
    assert result['provenance']=='RAW-FABRID_V1'
print(json.dumps({'status':'PASS','sample_images':len(rows),'fresh_prediction_cases':len(cases),'run_id':predictor.metadata['run_id'],'checked_packaged_file_hashes':len(index['file_sha256']),'source_hash':code_hash(Path('.'))}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=project, check=True, capture_output=True, text=True
        )
        result = json.loads(completed.stdout.strip())
    write_json(
        args.output,
        {
            "status": "PASS",
            "verified_at": now(),
            "archive_sha256": sha256(archive_path),
            "archive_bytes": archive_path.stat().st_size,
            "archive_members": len(names),
            "crc": "PASS",
            "safe_extraction": "PASS",
            "separate_directory_smoke": result,
            "scope": "Every indexed delivered file hash, authentic sample paths/bytes and selected checkpoint fresh inference in a separately extracted directory. No held-out metric rerun or model selection.",
        },
    )
    print(result)


if __name__ == "__main__":
    main()
