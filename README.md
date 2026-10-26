# Premier League Research: Arsenal W/D/L probabilities

Historical Arsenal win/draw/loss probabilities from frozen Premier League data, with executed notebooks, model checkpoints, and every training epoch available for inspection.

**[Experiment 2: executed notebook](notebooks/Experiment_2_Epoch_By_Epoch.ipynb)** · **[Experiment 1: fresh verification](notebooks/Experiment_1_Fresh_Verification.ipynb)** · [Original experiment 1](notebooks/Arsenal_League_vs_Arsenal.ipynb)

## Published results

**Reproducibility reruns after test inspection.** All three executions use the same data, splits, seed 42 and training settings. Experiment 2 changes the workflow order; its corresponding models reproduce the original weights, epoch losses and probabilities exactly. These reruns add no independent confirmatory evidence.

![Arsenal test log loss for all six models across the original experiment 1 and both reruns; corresponding scores are identical](results/profile/test_results.png)

On the same **75 Arsenal test fixtures in 2024/25–2025/26**, all three executions give:

| Model | Test log loss ↓ | Multiclass Brier ↓ | Accuracy |
|---|---:|---:|---:|
| Arsenal-only MLP | 0.959723 | 0.569044 | 60.00% |
| League-wide MLP | 0.899941 | 0.529087 | 61.33% |
| League → Arsenal fine-tuned MLP | 0.900410 | 0.529357 | 61.33% |
| League logistic | 0.908957 | 0.532644 | 61.33% |
| Arsenal logistic | 0.909300 | 0.534445 | 64.00% |
| League class prior | 1.094381 | 0.664183 | 40.00% |

**Finding:** league-wide neural training has lower test log loss than Arsenal-only training in this fixed setup. The paired league-minus-Arsenal difference is **−0.059782**, with a month-block 95% bootstrap interval **[−0.100787, −0.016194]**. The interval describes test-sample uncertainty, not training-seed uncertainty. Fine-tuning gives no test improvement here; the two logistic models are nearly tied.

**Validation-selected candidate: league logistic regression.** Model selection uses Arsenal validation loss, so the neural model's lower test loss does not change the selected candidate.

[Exact chart data](results/profile/test_results.csv) · [Metrics](results/experiment_2/run_20261026T002413/metrics.csv) · [Bootstrap comparison](results/experiment_2/run_20261026T002413/comparison.json)

## Training workflow and graphics

**Arsenal baseline from scratch → independent league challenger from scratch → frozen validation comparison → optional Arsenal specialization → final test evaluation.** Only specialization loads trained weights: the best league checkpoint from the same execution. Both independent fits start fresh.

![Experiment 2 learning curves in Arsenal-first order, with training and Arsenal validation loss and the selected checkpoint epoch](results/profile/learning_curves.png)

| Stage | Training rows | Epochs run | Best epoch | Optimizer updates |
|---|---:|---:|---:|---:|
| Arsenal-only baseline | 815 | 100 | 100 | 400 |
| Independent league challenger | 7,977 | 100 | 98 | 3,200 |
| League → Arsenal specialization | 815 | 21 | 1 | 84 |

All panels use the same **74 Arsenal validation fixtures**. Training curves use different populations and include dropout; validation uses evaluation mode. Both initial fits reached the cap, so convergence is not established. League training receives more optimizer updates under the same epoch budget.

**Fixed settings:** 39→32→16→3 MLP; ReLU and 0.1 dropout; unweighted cross-entropy; AdamW learning rate 1e-4, weight decay 0.01; batch 256; seed 42; maximum 100 epochs per fit; patience 20, min_delta 1e-4. Train through 2021/22; validation 2022/23–2023/24; test 2024/25–2025/26.

**[Read all 221 experiment 2 epochs](results/experiment_2/run_20261026T002413/EVERY_EPOCH.md)** · [Download epoch CSV](results/experiment_2/run_20261026T002413/all_epochs.csv) · [Frozen comparison before specialization](results/experiment_2/run_20261026T002413/frozen_validation_comparison.json)

## Why the results repeat exactly

Experiment 1 runs league → Arsenal → specialization; experiment 2 runs Arsenal → league → specialization. Seed 42 resets inside each independent fit, repeating initialization, batch order and dropout draws. Reordering those fits therefore does not change the numerical result. The fresh experiment 1 execution verified **exact equality of all model weights, all epoch losses and all validation/test probabilities** against the corresponding models in both saved experiments.

[Experiment 1: every epoch and verification](results/run_20261026T003158/RERUN_REPORT.md) · [Weight/epoch checks](results/run_20261026T003158/weights_and_epochs_comparison.csv) · [Probability checks](results/run_20261026T003158/prediction_comparison.csv) · [Workflow details](EXPERIMENT_2.md)

Original experiment 1 data, code, notebook and artifacts remain preserved. One seed, a short previously inspected test period, and unequal update budgets limit interpretation. A new untouched period is needed for a new confirmatory evaluation. These are historical probabilities, not live fixture forecasts.

## Reproduce

Python 3.12 was used. For economical CPU setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python experiments/audit.py
python -m unittest discover -s tests -v
# Optional: create a new run, with 100 epochs maximum per fit and patience 20.
python experiments/train.py
```

In Colab, the notebook installs its requirements. `RUN_TRAINING=False` displays the saved run and verifies its hashes; `True` trains and saves a new timestamped run. Download its output archive before ending a temporary runtime. Hosted Colab execution has not been independently run; the notebook was executed top-to-bottom in the cloud Python environment.

## Files and evidence

- `arsenal_predictor_v1/`: complete original ZIP contents, including frozen raw data, cleaned CSVs, splits, schema, source manifest and original scripts/reports. The ZIP itself is unchanged. Its original checksum manifest still verifies every original member.
- `experiments/audit.py`: authoritative current audit; validates all raw/clean/feature/split values, hashes, overlap and temporal invariance. Original v1 verifier reports are historical and should not substitute for this audit.
- `results/audit/`: source record links, rolling-history links, 57-cell correction ledger, corrected season metadata, clean Parquet, live-source check status and current verification report. Corrections are outside raw evidence and do not change the 39 model inputs.
- `experiments/train.py`: fixed AdamW comparison and prespecified fine-tuning; checkpoints include optimizer state, scaler, feature/class order and training configuration.
- `experiments/inference.py`: reload a saved candidate and score already prepared pre-match features. It rejects reordered or missing inputs and converts H/D/A to Arsenal W/D/L.
- `results/run_*/`: best checkpoints, logistic coefficient arrays, histories, metrics, manifests, uncertainty estimates and per-match probabilities. No serialized arbitrary estimator is needed for logistic reload.
- `results/figures/`: five exported figures, also embedded in the notebook.
- [Experiment protocol](EXPERIMENT_PROTOCOL.md) and [data attribution/reuse notice](DATA_REUSE_NOTICE.md).

Use `experiments/check_sources.py` to compare remote hashes without overwriting snapshots. Both pinned GitHub files matched. The three football-data downloads remain unverified remotely because this cloud's proxy returned 403; their bundled contents/hashes and cross-source agreement pass. A domain addition was saved in the environment draft; it requires application in environment settings before retrying here. Missing original-provider lineage and upstream permissions cannot be repaired by code. Reza states CC BY-NC 4.0; no commercial clearance is implied.

This project is not a live forecasting service. It saves historical out-of-time probabilities and an inference interface for correctly prepared future features. Personal workflow preferences are intentionally stored outside this repository.
