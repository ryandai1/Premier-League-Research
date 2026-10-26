"""Read-only verification of the scientific artifacts retained during cleanup."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / "results/preservation_checksums.json").read_text())
    failures = []
    for path, expected in manifest["sha256"].items():
        file = ROOT / path
        if (
            not file.is_file()
            or hashlib.sha256(file.read_bytes()).hexdigest() != expected
        ):
            failures.append(path)
    if failures:
        raise AssertionError(
            "Missing or modified scientific artifacts: " + ", ".join(failures)
        )
    return len(manifest["sha256"])


if __name__ == "__main__":
    print(
        f"PASS: {verify()} retained scientific artifacts match the pre-cleanup snapshot."
    )
