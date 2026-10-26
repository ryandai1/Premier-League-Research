> Historical Arsenal-inclusive experiment. For the active Arsenal-excluded workflow, see [active protocol](../arsenal_isolated/README.md).

# Held-out test review — frozen models

Inference-only companion to [the executed notebook](../../notebooks/Frozen_Model_Test_Review.ipynb). No retraining, tuning, checkpoint changes or new selection. The test period was held out from fitting but previously inspected.

Frozen run: `e3ceec90-4b79-4a9f-a6dc-b2dd43e67546`. Candidate selected on validation: **league_logistic**.

[All 732 eligible test fixtures and actual scores](held_out_test_fixtures.csv) · [75 Arsenal fixtures](arsenal_test_fixtures.csv) · [Metrics](metrics.csv) · [Three neural predictions side by side](neural_model_fixture_comparison.csv)

Original test files: [features](../../arsenal_predictor_v1/data/test/X.csv), [outcomes](../../arsenal_predictor_v1/data/test/y.csv), [fixture index](../../arsenal_predictor_v1/data/test/index.csv). Rows align by position. A/D/H means away win/draw/home win; the review tables use Arsenal Win/Draw/Loss.

- [arsenal_mlp: all 75 Arsenal fixtures](arsenal_mlp_arsenal_test_review.md) · [CSV](arsenal_mlp_arsenal_test_review.csv)
- [league_mlp: all 75 Arsenal fixtures](league_mlp_arsenal_test_review.md) · [CSV](league_mlp_arsenal_test_review.csv)
- [league_to_arsenal_mlp: all 75 Arsenal fixtures](league_to_arsenal_mlp_arsenal_test_review.md) · [CSV](league_to_arsenal_mlp_arsenal_test_review.csv)
- [league_logistic: all 75 Arsenal fixtures](league_logistic_arsenal_test_review.md) · [CSV](league_logistic_arsenal_test_review.csv)
- [arsenal_logistic: all 75 Arsenal fixtures](arsenal_logistic_arsenal_test_review.md) · [CSV](arsenal_logistic_arsenal_test_review.csv)
- [league_prior: all 75 Arsenal fixtures](league_prior_arsenal_test_review.md) · [CSV](league_prior_arsenal_test_review.csv)

Each model also has a complete `*_all_test_predictions.csv` containing all 732 fixtures and A/D/H probabilities. Model files are loaded from [current/models/](../current/models/).

[Execution and preservation checks](execution.json). Tables are recomputed when this notebook runs; original inputs and current-study artifacts are verified unchanged.
