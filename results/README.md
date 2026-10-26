# One current execution

All previous generated result folders were removed from the current tree at the user's request. This folder now contains only `current/`, created from an empty directory by the three numbered notebooks. Older results can be recovered from Git commit `aa740ef`; they are not read by the new pipeline.

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
