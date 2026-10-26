# Reproduce the Arsenal-excluded workflow

Install the pinned `requirements.txt` in a Python 3.12 environment and register its kernel. CPU PyTorch can be installed first with `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu`.

## Review existing runs

```bash
python -m experiments.execute_isolated_notebooks --kernel python3
python -m unittest discover -s tests -v
```

This executes all five notebooks, recomputes scores from the saved corrected models, rebuilds the plots and exports new per-fixture test reviews. It does not fit models by default. Derived outputs go to `results/arsenal_isolated_review/`; hash-pinned training evidence remains in `results/arsenal_isolated/`.

## Fresh training

In notebook 02, set `RUN_NEW_EXPERIMENT=True` and choose an unused `NEW_OUTPUT`. Alternatively:

```bash
python -m experiments.isolated_arsenal --output work/arsenal_isolated_rerun
```

The runner refuses to overwrite a directory. To inspect that new run, set `RUN` in each notebook's setup to the new directory **before** reading its manifest; change `record=json.loads((RUN/'manifest.json').read_text())` accordingly if needed. Keep a separate `REVIEW` directory for its derived results. No retraining was needed for this notebook correction: all requested model arms were already complete.

## Notebook sources and historical evidence

The active templates are in `experiments/build_isolated_notebooks.py`. Run `python -m experiments.build_isolated_notebooks` only when rebuilding notebooks, then execute them to restore outputs.

`experiments/build_notebooks.py`, `execute_study.py`, and `study.py` are hash-pinned historical code for the earlier inclusive experiment. Do not use those commands for the active workflow: the old builder would replace the active notebooks with historical templates. Earlier notebooks and results are recoverable from commit `c431b5f0f14414ac7eef528fddd389150950812a`. Historical model/audit artifacts remain intact.

Source-only scaler fitting and validation, temporal splits and frozen-layer rules are documented in [the active protocol](../results/arsenal_isolated/README.md). Original source reuse restrictions remain in [DATA_REUSE_NOTICE.md](../DATA_REUSE_NOTICE.md).
