# Arsenal-isolated transfer experiment

This experiment addresses target-club overlap in the earlier league-pretraining study. Earlier results remain historical evidence; use this directory for the Arsenal-excluded experiment.

## Two data populations

- `league/`: remove every fixture involving Arsenal from the cleaned match archive **before** reconstructing rolling statistics, rest intervals and Elo. Arsenal observations therefore cannot influence league inputs indirectly through opponent histories. Reapply the original eligibility rules after reconstruction: 7,142 training, 654 validation and 656 test fixtures.
- `arsenal/`: preserve the original eligible target fixtures and their strictly lagged pre-match features: 815 training, 74 validation and 75 test fixtures. These features describe actual match histories, including opponents' past games. This differs from the deliberately restricted source histories and introduces a source/target feature-distribution difference.

The initial 7,977 training rows minus 815 Arsenal rows would leave 7,162 rows. Rebuilding histories changes eligibility, so the final source set is not merely that row subtraction. `excluded_league_fixtures.csv` lists all rejected source fixtures. The two populations have no shared fixture IDs. Seasons remain training through 2021/22, validation 2022/23–2023/24 and test 2024/25–2025/26. Histories use strictly earlier dates; during chronological evaluation they may include earlier observed fixtures in the evaluation period, as in the original protocol. Model parameters and scalers are never fitted to validation/test rows.

## Fixed model arms

1. Fresh Arsenal-only MLP, with an Arsenal-training scaler and Arsenal validation checkpoint selection.
2. Fresh league-excluding-Arsenal MLP, with a source-training scaler and **non-Arsenal** validation checkpoint selection.
3. Full fine-tuning from arm 2 on Arsenal training; retain source scaler, reset AdamW, select using Arsenal validation.
4. Independently train another source MLP from scratch with the same fixed seed/settings and save its parameters. Matching source weights are expected under deterministic training; this is not an extra uncertainty replicate.
5. Load arm 4, set `requires_grad=False` for both weights and biases in the 39→32 and 32→16 layers, and optimize only the 16→3 layer on Arsenal training. Only trainable parameters enter AdamW. Hidden parameters and absent hidden gradients are checked every epoch, and the chosen checkpoint is checked again. Original dropout behavior is retained during training. Source scaler is retained, optimizer reset, and checkpoint selection uses Arsenal validation.
6. Random Forest fitted once to source training; directly evaluate Arsenal, without fine-tuning. Fixed 500 trees, minimum leaf size 5, square-root feature sampling, seed 42, no class weighting or parameter search.
7. DummyClassifier fitted once to source training with `strategy="prior"`; directly evaluate Arsenal. Probabilities are source-training A/D/H class frequencies, with no target-label access during fitting.

MLP settings retain the 39→32→16→3 architecture, ReLU/dropout 0.1, AdamW lr 0.0001, weight decay 0.01, batch 256, seed 42, maximum 100 epochs and patience 20. Fine-tuning allows the inherited epoch-zero model to win. Both hidden layers stay trainable in arm 3. Test predictions are generated only after every fit is saved. Target validation scores for source models are reported afterward and do not select their weights.

## Reproduce and inspect

Use the repository's pinned dependencies:

```bash
python -m experiments.isolated_arsenal --output work/arsenal_isolated_rerun
python -m unittest discover -s tests -v
```

The runner refuses to overwrite an existing directory. `models/` contains checkpoints, full epoch logs, training summaries and execution audits. `evaluation/` contains raw metrics and per-fixture probabilities. `manifest.json` records configuration, source hashes, data hashes, versions, stage timestamps and output checksums. The separate trainer preserves the hash-pinned historical training code.

Tests cover Arsenal-observation mutation invariance of source features, disjoint fixtures, source-only scalers, frozen hidden parameters, head-only optimizer eligibility, dummy priors, saved inference and recomputed metrics. No other target-team experiments were run. Numerical interpretation is provided in chat. The already inspected test period and single training seed limit confirmatory conclusions.
