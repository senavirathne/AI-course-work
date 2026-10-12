"""Execute every notebook cell in the pinned environment and retain outputs."""

import json
import argparse
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from fabric_inspector.common import now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["ipython", "kernel"], default="ipython")
    args = parser.parse_args()
    records = []
    with tempfile.TemporaryDirectory(prefix="fabric-kernel-") as folder:
        kernel = Path(folder) / "kernels/fabric-coursework"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Fabric coursework",
                    "language": "python",
                }
            )
        )
        prior = os.environ.get("JUPYTER_PATH", "")
        os.environ["JUPYTER_PATH"] = folder + (os.pathsep + prior if prior else "")
        for path in sorted(Path("notebooks").glob("*.ipynb")):
            started = time.perf_counter()
            if args.engine == "ipython":
                subprocess.run(
                    [sys.executable, "tools/run_notebook_cells.py", str(path)],
                    check=True,
                    timeout=180,
                )
                notebook = nbformat.read(path, as_version=4)
            else:
                notebook = nbformat.read(path, as_version=4)
                NotebookClient(
                    notebook,
                    timeout=180,
                    kernel_name="fabric-coursework",
                    allow_errors=False,
                    resources={"metadata": {"path": str(Path.cwd())}},
                ).execute()
            nbformat.validate(notebook)
            assert not any(
                o.get("output_type") == "error"
                for c in notebook.cells
                if c.cell_type == "code"
                for o in c.get("outputs", [])
            )
            nbformat.write(notebook, path)
            records.append(
                {
                    "notebook": path.as_posix(),
                    "status": "PASS",
                    "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
                    "seconds": time.perf_counter() - started,
                }
            )
            print(path, "PASS", flush=True)
    write_json(
        Path("reports/notebook-verification.json"),
        {
            "status": "PASS",
            "timestamp": now(),
            "notebooks": records,
            "engine": args.engine,
            "execution_method": "Each notebook in an isolated IPython process; actual code and rich outputs captured"
            if args.engine == "ipython"
            else "Standard nbclient Jupyter kernel",
            "environment_note": "Default IPython execution avoids the hosted runtime's unavailable Jupyter ZeroMQ interface-discovery sockets. Standard kernel execution is available with --engine kernel on compatible environments. Neither path substitutes results or reruns the held-out study.",
        },
    )


if __name__ == "__main__":
    main()
