"""Check remote bytes; record blocked/mismatched states without replacing raw files."""

import hashlib, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run():
    results = []
    for s in json.loads(
        (ROOT / "arsenal_predictor_v1/source_manifest.json").read_text()
    ):
        row = {"name": s["name"], "url": s["url"], "expected_sha256": s["sha256"]}
        try:
            with urllib.request.urlopen(s["url"], timeout=20) as response:
                actual = hashlib.sha256(response.read()).hexdigest()
            row.update(
                actual_sha256=actual,
                status="matched" if actual == s["sha256"] else "changed",
            )
        except Exception as e:
            row.update(status="unverified", reason=str(e))
        results.append(row)
    out = ROOT / "results/audit/remote_sources.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
