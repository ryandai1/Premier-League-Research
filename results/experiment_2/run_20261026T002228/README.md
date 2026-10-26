# Experiment 2 — reproducibility rerun
Test results were already inspected; no independent confirmatory claim is made.

Order: Arsenal-only baseline → independently initialized league challenger → optional Arsenal fine-tuning (enabled: True). All frozen inputs, splits, seed and fitting settings match experiment 1. The unchanged fitting function resets seed 42 for each stage. Experiment 1 files remain unchanged.

Arsenal-only test log loss: 0.959723; league test log loss: 0.899941. Maximum probability difference from experiment 1 across all models, validation and test: 0. Candidate selected on validation: league_logistic.

The order clarifies the baseline/challenger narrative. It provides no new held-out evidence. Inspect experiment_1_comparison.csv for all metric deltas and prediction_reproducibility.csv for per-model checks. Model checkpoints were reloaded and their exported test probabilities verified.
