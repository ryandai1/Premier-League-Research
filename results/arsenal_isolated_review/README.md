# Corrected notebook review

Derived from the immutable experiment in `../arsenal_isolated/`; no additional training is represented by these outputs.

- `figures/`: ten source/data plots, learning curves, test metrics, paired uncertainty, calibration/confusion.
- `comparison.json`: paired month-block bootstrap for league minus Arsenal-only MLP test log loss (1,000 replicates, seed 42).
- `calibration_bins.csv`: fixed-bin Arsenal-win calibration with sample counts.
- `test_review/`: all 75 target test fixtures and seven per-model verdict CSVs with actual goals, predicted outcomes, correctness and probabilities.
- `notebook_execution.json`: actual execution records for all five active notebooks.

Source model checkpoints use non-Arsenal validation. Plot/test-review candidate selection is retrospective within the fixed candidate set, using Arsenal validation log loss only; test does not select it. The duplicate deterministic source fit is excluded from candidate selection. The previously inspected test period and single seed limit conclusions. Reproduce with `python -m experiments.execute_isolated_notebooks`.
