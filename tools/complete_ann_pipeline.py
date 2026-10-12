"""Wait for the running screening study, then complete its pre-test ANN stages."""

import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    deadline = time.monotonic() + 8 * 3600
    last_count = -1
    while time.monotonic() < deadline:
        records = [Path(f"experiments/ann-search-{i:03d}.json") for i in range(20)]
        values = [json.loads(p.read_text()) for p in records if p.exists()]
        if any(v.get("status") == "FAILED_EXECUTION" for v in values):
            raise RuntimeError("Screening execution failure: preserve the record and resolve it")
        count = sum(v.get("status") == "COMPLETE" for v in values)
        if count != last_count:
            print(f"Completed ANN screening configurations: {count}/20", flush=True)
            last_count = count
        if count == 20:
            break
        time.sleep(5)
    else:
        raise TimeoutError("Screening did not complete within eight hours")
    for script, extra in [
        ("tools/refine_ann.py", ["--device", "cpu"]),
        ("scripts/select_models.py", []),
        ("scripts/repeat_seeds.py", ["--device", "cpu"]),
        ("scripts/xai_validate.py", []),
        ("scripts/make_figures.py", []),
    ]:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, script, *extra], check=True)


if __name__ == "__main__":
    main()
