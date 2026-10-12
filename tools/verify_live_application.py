"""Verify a real checkpoint through fresh HTTP requests and an actual browser."""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
import numpy as np
from PIL import Image

from fabric_inspector.common import now, write_json
from fabric_inspector.serving.predictor import Predictor


def encode(image, fmt="PNG"):
    stream = BytesIO()
    image.save(stream, format=fmt)
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/serving-bundle.json"))
    parser.add_argument(
        "--cases", type=Path, required=True, help="Actual demonstration cases and labels"
    )
    parser.add_argument("--output", type=Path, default=Path("reports/application-verification"))
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["FABRIC_CPU_THREADS"] = "1"
    predictor = Predictor(args.bundle)
    cases = json.loads(args.cases.read_text())["cases"]
    for item in cases:
        item["path"] = str(Path(item["path"]).resolve())
        fresh_score = predictor.predict(Path(item["path"]).read_bytes())["defect_score"]
        if "expected_score" in item:
            assert abs(item["expected_score"] - fresh_score) < 1e-5, (
                "Fresh application score differs from its immutable reference case",
                item["patch_id"],
            )
        item["expected_score"] = fresh_score
    checks = []
    env = dict(os.environ, FABRIC_BUNDLE=str(args.bundle.resolve()), FABRIC_CPU_THREADS="1")
    url = "http://127.0.0.1:8765"
    with (args.output / "server.log").open("w") as server_log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
            ],
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(timeout=90, trust_env=False) as client:
                for _ in range(120):
                    if server.poll() is not None:
                        raise RuntimeError("Application server exited during startup")
                    try:
                        health = client.get(url + "/health").json()
                        if health["ready"]:
                            break
                    except (httpx.HTTPError, ValueError):
                        pass
                    time.sleep(0.5)
                else:
                    raise TimeoutError("Real model not ready")
                assert health["provenance"] == "RAW-FABRID_V1"
                assert health["model_version"] == predictor.version
                for endpoint in ["/", "/static/app.js", "/static/style.css"]:
                    assert client.get(url + endpoint).status_code == 200
                checks.append({"check": "readiness and static routes", "status": "PASS"})
                content = Path(cases[0]["path"]).read_bytes()
                with Image.open(BytesIO(content)) as image:
                    gray = image.convert("L").copy()
                valid = [
                    ("original grayscale PNG", content),
                    ("RGB PNG", encode(gray.convert("RGB"))),
                    ("grayscale JPEG", encode(gray, "JPEG")),
                ]
                for title, content in valid:
                    direct = predictor.predict(content)
                    response = client.post(
                        url + "/predict",
                        files={"file": ("renamed.png", content, "application/octet-stream")},
                    )
                    assert response.status_code == 200
                    result = response.json()
                    assert abs(result["defect_score"] - direct["defect_score"]) < 1e-6
                    assert result["class"] == direct["class"]
                    checks.append(
                        {"check": title + " direct/HTTP parity", "status": "PASS", "result": result}
                    )
                invalid = [
                    ("empty bytes", b"", 422),
                    ("malformed bytes", b"invalid", 422),
                    ("wrong size", encode(Image.new("L", (128, 256))), 422),
                    ("alpha channels", encode(Image.new("RGBA", (256, 256))), 422),
                    ("paletted input", encode(Image.new("P", (256, 256))), 422),
                    ("oversize upload", b"x" * (5 * 1024 * 1024 + 1), 413),
                ]
                for title, content, code in invalid:
                    response = client.post(
                        url + "/predict", files={"file": ("bad.png", content, "image/png")}
                    )
                    assert response.status_code == code, (title, response.status_code)
                    checks.append(
                        {
                            "check": title,
                            "status": "PASS",
                            "http_status": code,
                            "detail": response.json(),
                        }
                    )
                response = client.post(
                    url + "/explain",
                    files={"file": ("valid.png", Path(cases[0]["path"]).read_bytes(), "image/png")},
                    data={"alpha": "1.1"},
                )
                assert response.status_code == 422
                checks.append(
                    {"check": "invalid explanation opacity", "status": "PASS", "http_status": 422}
                )
                rng = np.random.default_rng(42)
                for title, image in [
                    ("uniform blank", Image.new("L", (256, 256), 128)),
                    (
                        "random noise",
                        Image.fromarray(rng.integers(0, 256, (256, 256), dtype=np.uint8)),
                    ),
                ]:
                    response = client.post(
                        url + "/predict", files={"file": ("ood.png", encode(image), "image/png")}
                    )
                    assert response.status_code == 200
                    result = response.json()
                    assert np.isfinite(result["defect_score"])
                    assert result["input_checks"]["ood_detection"] == "Not implemented or claimed"
                    checks.append(
                        {
                            "check": title,
                            "status": "PASS",
                            "result": result,
                            "meaning": "Finite response and explicit OOD limitation; no claim that this is a meaningful defect label",
                        }
                    )
            spec = {
                "url": url,
                "cases": cases,
                "record": args.record,
                "video_directory": str((args.output / "recording").resolve()),
                "screenshot": str((args.output / "internal-browser-qa.png").resolve()),
                "result_path": str((args.output / "browser-results.json").resolve()),
                "model_version": predictor.version,
            }
            write_json(args.output / "browser-spec.json", spec)
            subprocess.run(
                [
                    os.getenv("CODEX_PRIMARY_RUNTIME_NODE", "node"),
                    "tools/browser_walkthrough.cjs",
                    str(args.output / "browser-spec.json"),
                ],
                check=True,
            )
            browser = json.loads((args.output / "browser-results.json").read_text())
            assert browser["status"] == "PASS"
            write_json(
                args.output / "results.json",
                {
                    "status": "PASS",
                    "timestamp": now(),
                    "model_version": predictor.version,
                    "run_id": predictor.metadata["run_id"],
                    "bundle": str(args.bundle),
                    "http_checks": checks,
                    "browser": browser,
                    "scope": "Fresh real checkpoint inference, input validation, direct/HTTP/browser parity, visible explanations and recorded genuine incorrect prediction when requested",
                },
            )
            print(
                f"Verified {len(checks)} live HTTP checks and {len(browser['checks'])} browser checks",
                flush=True,
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


if __name__ == "__main__":
    main()
