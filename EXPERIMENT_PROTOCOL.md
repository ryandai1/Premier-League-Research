> Historical Arsenal-inclusive experiment. For the active Arsenal-excluded workflow, see [results/arsenal_isolated/README.md](results/arsenal_isolated/README.md).

# Controlled fresh-execution protocol

This documents the fixed methods used in the rebuilt notebook sequence. Earlier test results have already been inspected. It is a reproducibility rerun; no new preregistration or independent confirmatory test is claimed.

## Data before modeling

1. Start with an absent `results/current/` directory. Record a unique run ID, UTC timestamps, source-code hashes, runtime versions and frozen input hashes.
2. Inspect the primary Reza archive, Tara overlap and three football-data snapshots. Join by normalized season/home/away identity for verification; do not append duplicate observations. Preserve the raw bytes and attribution.
3. Visualize coverage, missingness, clubs and outcomes before modeling. Disclose that test descriptive statistics are post-inspection and do not drive tuning.
4. Reconstruct cleaning and features, check all X/y/index cells and chronological partitions, and perform temporal mutation checks. Five inconsistent shot-stat sides across three matches become missing; metadata corrections remain separate from the predictors. Export a new audit.
5. Freeze the same 39-feature allowlist. Strictly earlier dates supply histories; default eligibility requires complete lagged inputs. Missing xG is excluded from default features. Study training-only distributions and correlations.

## Control, treatment and extension

This is a computational comparison on observational match data, not a randomized intervention on clubs.

| Arm | Initialization | Training data |
|---|---|---|
| Arsenal-only MLP control | Fresh seed-42 initialization | 815 Arsenal training fixtures |
| League-wide MLP treatment | Fresh seed-42 initialization | 7,977 EPL training fixtures, including Arsenal |
| Optional Arsenal specialization (enabled) | Best current-run league checkpoint | Same 815 Arsenal fixtures, league scaler retained, optimizer reset |
| Reference controls | Fixed logistic models; league class frequencies | Corresponding training populations |

The independent fits do not share learned weights. The same seed intentionally gives the same initial tensors. Only specialization loads a checkpoint. Different sample sizes mean different optimizer updates under the same epoch budget: 4 batches per Arsenal epoch and 32 per league epoch.

## Frozen settings and stage order

- Train 2000/01–2021/22, validation 2022/23–2023/24, test 2024/25–2025/26. All neural selection uses the same 74 Arsenal validation fixtures; primary test uses 75 Arsenal fixtures.
- 39→32→16→3 MLP, ReLU, dropout 0.1, unweighted cross-entropy, A/D/H class order.
- AdamW lr=1e-4, weight decay 0.01, batch 256; per-fit seed 42; deterministic CPU algorithms; maximum 100 epochs, patience 20, min_delta 1e-4.
- Save every true validation minimum; patience resets only for an improvement exceeding min_delta relative to the previous best. Reload the best checkpoint. Specialization permits its inherited epoch-zero checkpoint to win.
- Run Arsenal control first, then league challenger, save the frozen validation comparison, then run the enabled specialization. Fit fixed logistic controls (C=1, max_iter=2000) and class prior. Select the candidate on Arsenal validation log loss only.
- Record every epoch's online training loss, validation loss/accuracy, best epoch, checkpoint save, patience, optimizer steps and timing. Record initial/best weight hashes and transfer ancestry.
- Evaluate test only after models and candidate selection are frozen. Evaluation reloads the newly created artifacts through the same inference interface used for saved-model checks. Input values use the frozen float32 loader; CSV checks restore original probability dtype before comparison.

## Outcomes and interpretation

Primary contrast: league minus Arsenal-only MLP log loss on identical Arsenal test fixtures; negative favors league. Secondary: multiclass Brier sum (0–2), accuracy and calibration. The paired month-block bootstrap uses 1,000 resamples and reports the 95% percentile interval. It excludes training-seed uncertainty.

Report one current result set. Corresponding deterministic reruns are not additional treatment arms. Compare independent-arm weights and probabilities, and check response to changed input rows; these are implementation checks, not tests of statistical superiority. Do not alter seeds or settings to force different numbers.

Single seed, previously inspected test outcomes, unequal update budgets, eligibility bias and unresolved original-observer lineage remain limitations. Known-competition transfer is not external-league validation. A new untouched period is needed for a new confirmatory claim.
