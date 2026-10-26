> Historical Arsenal-inclusive experiment. For the active Arsenal-excluded workflow, see [active protocol](../results/arsenal_isolated/README.md).

# Current scientific report

The executed notebooks are the complete report: [data and audit](../notebooks/01_data_sources_and_audit.ipynb), [models and training](../notebooks/02_models_and_training.ipynb), [evaluation](../notebooks/03_evaluation_and_results.ipynb). This summary is generated from the fresh output tables.

## Provenance and population

Reza supplies 9,880 unique matches across 26 seasons; Tara contributes 3,800 overlapping checks, and three football-data snapshots check 380 fixtures each. No supplementary fixtures are appended. The source matrix and conflict table are computed anew. The supplied source snapshots are frozen, not refreshed online; underlying original-observer lineage remains incomplete.

The audit reconstructs cleaning/features and all split cells, verifies hashes and chronology, and checks temporal invariance under future-data mutation. Modeling retains 9,439 eligible fixtures with 39 lagged features; 441 remain archived but ineligible. Train/validation/test populations are 7,977/730/732 EPL matches, including 815/74/75 Arsenal fixtures. Missing xG and end-of-season metadata are not default predictors.

## Controlled modeling

Arsenal-only is the control; independent league-wide training is the treatment. The same architecture and fixed seed/settings are used. Specialization is an explicitly enabled extension of the league checkpoint, not initialization for the challenger. Logistic models and a league prior are reference controls. See the [protocol](../EXPERIMENT_PROTOCOL.md) for exact settings and selection rules.

## Fresh evaluation

| Model | Test log loss | Brier sum | Accuracy |
|---|---:|---:|---:|
| arsenal_mlp | 0.959723 | 0.569044 | 60.00% |
| league_mlp | 0.899941 | 0.529087 | 61.33% |
| league_to_arsenal_mlp | 0.900410 | 0.529357 | 61.33% |
| league_logistic | 0.908957 | 0.532644 | 61.33% |
| arsenal_logistic | 0.909300 | 0.534445 | 64.00% |
| league_prior | 1.094381 | 0.664183 | 40.00% |

Primary league-minus-Arsenal loss difference: **-0.059782**. Month-block 95% percentile interval: **[-0.100787, -0.016194]**, from 1,000 resamples of 20 month blocks. It excludes training-seed uncertainty. Validation selected **league_logistic**.

The independent arms have different learned weights and different predictions on the same test rows; maximum absolute probability difference is **0.254230**. Reordering feature rows changes league outputs, confirming that prediction depends on the input data. These implementation checks do not prove model superiority.

## Execution evidence

Run ID: `e3ceec90-4b79-4a9f-a6dc-b2dd43e67546`. Start: `2026-10-26T05:55:08.441857+00:00`; finish: `2026-10-26T05:55:41.443612+00:00`. All three notebooks executed in separate kernels. Neural training completed 100, 100 and 21 epochs, with 400, 3,200 and 84 optimizer updates. Best epochs were 100, 98 and 1. The original inputs and fixed numerical settings were not changed to force different scores.

[All epochs](../results/current/models/EVERY_EPOCH.md) · [Model ancestry](../results/current/models/independence_checks.json) · [Execution timeline](../results/current/events.jsonl) · [Manifest](../results/current/manifest.json)

## Interpretation and limits

The fresh results have the same rounded performance as previous executions under the same fixed seed and settings. This is compatible with deterministic training and does not indicate that the Arsenal and league arms are identical. No archived result/model file is read by this pipeline. The previous three-execution bar chart was removed; the current figure compares model arms within a single execution.

The test period was inspected previously. These results are a reproducibility rerun, not independent confirmatory evidence. The league gets more updates per epoch, so the comparison does not isolate data-volume effects at equal compute. Independent fits reached the epoch cap; convergence is not established. Calibration bins are sparse. The validation-selected model predicts no draws under argmax on this 75-match sample, a limitation visible in the confusion matrix; probability loss is the primary objective.

Eligibility exclusions, a short test horizon, one seed and incomplete source lineage constrain generalization. Other-EPL test results are same-competition transfer, not external validation. No live-fixture forecasting claim is made. A new untouched evaluation period is required for a new confirmatory claim.
