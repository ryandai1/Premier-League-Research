# One current execution

All previous generated result folders were removed from the current tree at the user's request. The original rebuilt study is in `current/`, created from an empty directory by the three numbered notebooks. Older results can be recovered from Git commit `aa740ef`; they are not read by the new pipeline.

| Current path | Purpose |
|---|---|
| `manifest.json` | Unique run ID, settings, input/source hashes, runtime versions, stage and timestamps |
| `notebook_execution.json` | Start/finish times and execution status of the three notebooks |
| `events.jsonl` | Ordered audit/training/freeze/evaluation events |
| `data/` | Source inventory, overlap, coverage, missingness, eligibility, split and training-feature tables |
| `audit/` | Independently reconstructed features, source checks, corrections and lineage |
| `models/` | New checkpoints, scalers/coefficients, every epoch, selection, execution ancestry |
| `evaluation/` | Fresh metrics, full/per-Arsenal predictions, bootstrap, calibration and diagnostics |
| `figures/` | Fourteen freshly rendered data, training and results figures |

The full inputs are preserved in the original source bundle. Its original audit files are historical bundle metadata; the fresh verification is `current/audit/verification.json`.

## Frozen-model test review

[test_review/](test_review/README.md) contains a separate inference-only review: all 732 held-out fixtures and actual scores, full 75-fixture Arsenal verdict tables for each model, recomputed metrics and preservation hashes. It loads the existing `current/models/` artifacts and leaves the original study unchanged. Reproduce it by running [the companion notebook](../notebooks/Frozen_Model_Test_Review.ipynb) directly. This is the previously inspected test period, not a new untouched dataset.
