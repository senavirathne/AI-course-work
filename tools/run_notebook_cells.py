"""Execute a notebook's actual cells in an isolated IPython process without network transport."""

import argparse
import base64
from pathlib import Path
import time

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    settings = Config()
    settings.HistoryManager.enabled = False
    shell = InteractiveShell.instance(config=settings)
    started = time.perf_counter()
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        cell.execution_count = shell.execution_count
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True, cell_id=cell.get("id"))
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for rich in captured.outputs:
            # PIL rich display publishes real PNG bytes. Notebook MIME bundles
            # require binary images as base64 strings; preserve their exact bytes.
            data = {
                mime: (
                    value.decode("utf-8")
                    if isinstance(value, (bytes, bytearray)) and mime.startswith("text/")
                    else base64.b64encode(value).decode("ascii")
                    if isinstance(value, (bytes, bytearray))
                    else value
                )
                for mime, value in rich.data.items()
            }
            outputs.append(
                nbformat.v4.new_output("display_data", data=data, metadata=rich.metadata)
            )
        cell.outputs = outputs
        if not result.success:
            error = result.error_in_exec or result.error_before_exec
            cell.outputs.append(
                nbformat.v4.new_output(
                    "error",
                    ename=type(error).__name__,
                    evalue=str(error),
                    traceback=(captured.stdout + captured.stderr).splitlines(),
                )
            )
            target = Path("reports/notebook-failures") / args.notebook.name
            target.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, target)
            result.raise_error()
        print(f"{args.notebook.name}: cell {count} PASS", flush=True)
    notebook.metadata["execution"] = {
        "engine": "Isolated IPython InteractiveShell.run_cell",
        "transport": "No Jupyter messaging transport; actual code cells executed sequentially",
        "code_cells": count,
        "seconds": time.perf_counter() - started,
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, args.notebook)


if __name__ == "__main__":
    main()
