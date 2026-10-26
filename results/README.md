# Results and evidence catalog

Start with the two **featured** executions below. Historical runs are retained to document what happened, not presented as independent experimental replications. All reruns occur after test inspection.

| Run ID | Status | Purpose and evidence | Exact source commit |
|---|---|---|---|
| `experiment_1_original` | Original | [Initial league-first comparison](run_20261024T140407/) | `a319744` |
| `experiment_2_initial` | Historical | [First Arsenal-first rerun](experiment_2/run_20261025T124105/) | `9a7b500` |
| `experiment_2_instrumented` | Historical | [First visible-epoch execution](experiment_2/run_20261026T002228/) | `49e9106` |
| `experiment_2_current` | **Featured** | [Arsenal-first, explicit frozen comparison, every epoch](experiment_2/run_20261026T002413/EVERY_EPOCH.md) | `7eac1f2` |
| `experiment_1_verification` | **Featured** | [Unchanged original runner rerun and equality verification](run_20261026T003158/RERUN_REPORT.md) | `96d9221` |

Machine-readable identities and full commits: [catalog.json](catalog.json). Plotting and tests resolve named runs from this catalog; a newly created timestamped folder does not silently replace the featured results.

## Folder roles

- `audit/`: source integrity, transformation checks, row lineage and metadata corrections.
- `profile/`: publication graphics and their exact chart data; regenerate with `python -m experiments.profile_figures`.
- `figures/`: original experiment 1 notebook graphics; preserved historical evidence.
- `run_20261024T140407/`, `run_20261026T003158/`: original experiment 1 and its verification rerun.
- `experiment_2/run_*/`: separate experiment 2 executions, labeled above.

Paths are retained because notebooks, checkpoints and manifests reference them. They are not renamed solely for appearance.

## File roles inside a run

| Files | Purpose |
|---|---|
| `run_manifest.json` | Frozen configuration, input hashes, source hash and runtime versions |
| `*_best.pt`, `*.npz`, `*.npy` | Reloadable neural checkpoints, logistic coefficients and class prior |
| `*_history.csv`, `all_epochs.csv`, `observed_epochs.csv`, `epoch_log.txt` | Epoch-level execution evidence; availability varies by execution |
| `*_execution.json` | Initialization/transfer hashes, timing and optimizer updates for instrumented runs |
| `training_summary.csv`, `selected_model.json` | Best epochs, actual training duration and validation-only model selection |
| `metrics.csv`, `comparison.json` | Metrics and paired month-block uncertainty |
| `*_predictions.csv` | All eligible fixtures with A/D/H probabilities and outcomes |
| `*_arsenal_probabilities.csv` | Arsenal subset reoriented to win/draw/loss for inspection |
| Comparison CSVs and verification JSON | Explicit cross-run equality checks |
| PNG, Markdown and `validation.txt` | Run-specific presentation and historical validation record |

The Arsenal subset exports intentionally remain: they encode a useful home/away orientation and are checked against full predictions. Checkpoints remain even when weights agree, because each carries run-specific provenance/optimizer state. Four HTML previews were removed as regenerable presentations; exact old bytes remain in Git history. Historical validation statements about HTML describe the execution at that time.
