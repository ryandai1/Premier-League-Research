# Experiment 2 — Arsenal-first reproducibility rerun

**Test results were already inspected.** This is a separately versioned reproducibility rerun of experiment 1, not an independent confirmatory experiment. The test remains outside checkpoint and candidate selection; a new untouched period is required for a new confirmatory evaluation.

## Reordered workflow

1. Train the Arsenal-only MLP baseline from scratch on 815 Arsenal training rows.
2. Train the league-wide challenger independently from scratch on the same 7,977 league training rows as experiment 1. It receives no Arsenal checkpoint. League data includes Arsenal fixtures, as before.
3. Optionally fine-tune the best league checkpoint on Arsenal training rows, retain its scaler, and reset AdamW. Epoch zero remains eligible. This run enables the stage before scoring test predictions; `experiment_2.run(fine_tune=False)` omits it.

Each fit resets seed 42, so moving independent fits should preserve numerical results. We reuse the exact unchanged `experiments.train.fit`: architecture 39→32→16→3 with ReLU/dropout 0.1; cross-entropy; AdamW lr=1e-4 and weight decay 0.01; batch 256; maximum 100 epochs per fit; patience 20 and min_delta 1e-4. Every new validation minimum is saved and the best checkpoint reloaded. All nine X/y/index file hashes, feature order, and configuration are checked against experiment 1 before training.

Train 2000/01–2021/22, validation 2022/23–2023/24, test 2024/25–2025/26. All neural fits use the same 74 Arsenal validation fixtures; the primary evaluation uses the same 75 Arsenal test fixtures. Logistic and prior controls and validation-only candidate selection are unchanged.

## Observed comparison

The executed notebook is `notebooks/Experiment_2_Arsenal_First_Reproducibility.ipynb`; outputs are in `results/experiment_2/run_20261025T124105/`.

All six models exactly reproduce experiment 1's saved probabilities across validation and test, including all EPL populations. Maximum absolute probability difference: **0**. All reported metric deltas are zero. Arsenal-only and league checkpoint file hashes also match; the fine-tuned checkpoint stores a different initial-checkpoint path, so its serialized file hash differs while its probabilities reproduce exactly.

| Neural model | Arsenal test log loss, exp. 1 | Exp. 2 | Best epoch | Epochs run |
| --- | ---: | ---: | ---: | ---: |
| Arsenal-only | 0.959723 | 0.959723 | 100 | 100 |
| League-wide | 0.899941 | 0.899941 | 98 | 100 |
| League → Arsenal | 0.900410 | 0.900410 | 1 | 21 |

The paired league-minus-Arsenal test log-loss difference remains **−0.059782**, with the same 1,000-resample month-block 95% percentile interval **[−0.100787, −0.016194]**. Fine-tuning offers no practical gain here. League logistic regression remains the candidate selected by lowest Arsenal validation log loss.

The order makes the baseline/challenger interpretation clearer, but does not alter independent deterministic fits. Single seed, a short previously inspected test period, and more league optimizer updates per epoch limit the interpretation. The interval covers test-sample uncertainty, not training randomness. These are historical probabilities, not live fixture forecasts.

## Reproduce and inspect

Install `requirements.txt` (this execution used CPU PyTorch 2.5.1+cpu), then execute the new notebook top to bottom. Every execution creates a new timestamped directory; experiment 1 is read only. The notebook verifies inputs, stage independence, scaler retention, checkpoint reloads, candidate selection, and fixture alignment. It saves full metrics, histories, checkpoints, per-match probabilities, `experiment_1_comparison.csv`, `prediction_reproducibility.csv`, a summary, and figures.

The existing experiment 1 notebook, code, raw inputs, splits, audits, and result files are preserved byte for byte. No source refresh, feature engineering revision, hyperparameter search, or test-based candidate selection was performed.
