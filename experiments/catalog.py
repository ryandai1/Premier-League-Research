"""Resolve explicitly registered scientific runs; never choose results by directory order."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def entries():
    return json.loads((ROOT / "results/catalog.json").read_text())["runs"]


def run_path(run_id):
    matches = [entry for entry in entries() if entry["id"] == run_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one registered run for {run_id!r}, got {len(matches)}"
        )
    return ROOT / matches[0]["path"]
