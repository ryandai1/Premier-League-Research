"""Execute the numbered notebooks in order, each in a fresh kernel."""

import argparse
import json
from datetime import datetime, timezone
import nbformat
from nbclient import NotebookClient
from experiments.study import ROOT, RUN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()
    if RUN.exists():
        raise SystemExit(
            "results/current already exists. Preserve or explicitly remove it before a new complete execution."
        )
    completed = []
    for path in sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb")):
        print("EXECUTING", path.name, flush=True)
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        started = datetime.now(timezone.utc).isoformat()
        try:
            NotebookClient(
                notebook,
                timeout=1800,
                kernel_name=args.kernel,
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
        finally:
            nbformat.write(notebook, path)
        nbformat.validate(notebook)
        completed.append(
            {
                "notebook": path.name,
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
                "status": "executed",
            }
        )
        (RUN / "notebook_execution.json").write_text(json.dumps(completed, indent=2))
        print("COMPLETED", path.name, flush=True)
    print("All notebooks executed:", RUN, flush=True)


if __name__ == "__main__":
    main()
