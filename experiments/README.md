# Code responsibilities

| Module | Responsibility |
|---|---|
| `execute_study.py` | Execute 01 → 02 → 03 in separate kernels; refuse existing current outputs |
| `build_notebooks.py` | Rebuild editable notebook sources; does not execute models, overwrites notebook presentation |
| `study.py` | Enforce stages, hash inputs/code, train fresh arms, freeze selection and evaluate new artifacts |
| `audit.py` | Reconstruct cleaning/features, check sources and splits, test temporal invariance |
| `epoch_training.py` | Same fixed MLP fitting/stopping rules with visible epoch and ancestry records |
| `train.py` | Original canonical configuration, architecture, feature loader and metric definitions; preserved as reference implementation |
| `inference.py` | Reload neural/logistic/prior artifacts, validate feature order and orient Arsenal probabilities |
| `check_sources.py` | Optional remote source-hash inspection; not part of the offline controlled run |

Old experiment-specific orchestration, result catalogs, preservation checkers and plot-only profile code were retired because they depended on deleted result directories. The current pipeline never reads old result paths. Use `python -m experiments.execute_study`, rather than the legacy standalone runner in `train.py`, to reproduce the new report.
