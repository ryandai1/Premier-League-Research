"""Execute active notebooks without retraining; notebook 02 has an explicit opt-in."""
import argparse
from datetime import datetime, timezone
import json
import nbformat
from nbclient import NotebookClient
from experiments.isolated_arsenal import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', default='python3')
    args = parser.parse_args()
    folder = ROOT / 'results/arsenal_isolated_review'
    folder.mkdir(exist_ok=True)
    records = []
    paths = sorted((ROOT / 'notebooks').glob('[0-9][0-9]_*.ipynb')) + [ROOT / 'notebooks/Frozen_Model_Test_Review.ipynb']
    for path in paths:
        n = nbformat.read(path, as_version=4)
        started = datetime.now(timezone.utc).isoformat()
        print('EXECUTING', path.name, flush=True)
        NotebookClient(n, timeout=300, kernel_name=args.kernel, resources={'metadata': {'path': str(ROOT)}}).execute()
        nbformat.validate(n)
        nbformat.write(n, path)
        records.append({'notebook': path.name, 'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'code_cells': sum(c.cell_type=='code' for c in n.cells), 'status':'executed'})
        (folder / 'notebook_execution.json').write_text(json.dumps(records, indent=2)+'\n')
    print('Executed all five notebooks', flush=True)

if __name__ == '__main__':
    main()
