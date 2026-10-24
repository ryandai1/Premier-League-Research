"""Restore missing frozen CSVs. Refuse changed bytes instead of silently updating."""
from pathlib import Path
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parents[1]
for item in json.loads((ROOT/'source_manifest.json').read_text()):
    dest=ROOT/'raw'/(item['name']+'.csv')
    payload=dest.read_bytes() if dest.exists() else urllib.request.urlopen(item['url'],timeout=60).read()
    if hashlib.sha256(payload).hexdigest()!=item['sha256']:
        raise ValueError(f"Changed source: {item['name']}. Use bundled frozen bytes or explicitly audit a new release.")
    dest.parent.mkdir(exist_ok=True);dest.write_bytes(payload)
    print(item['name'],'verified')
