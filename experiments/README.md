# Executable code

Run modules from the repository root. Training creates a new timestamped output directory; existing runs are never overwritten.

| Module | Single responsibility |
|---|---|
| `train.py` | Frozen experiment 1 reference implementation, including canonical settings, feature loading, model and metrics. Byte-for-byte preserved because archived manifests pin its hash. |
| `experiment_2.py` | Orchestrate Arsenal-first independent fits, frozen validation comparison, optional specialization and scoring. Imports shared settings/helpers from the frozen reference instead of duplicating them. |
| `epoch_training.py` | Instrumented neural fitting loop with per-epoch logging and initialization ancestry. Update/stopping behavior is unchanged. |
| `rerun_experiment_1.py` | Execute the original runner with a read-only Python trace observer; compare weights, losses and predictions. |
| `inference.py` | Reload selected frozen models; validate feature order and orient Arsenal probabilities. |
| `catalog.py` | Resolve explicit run IDs in `results/catalog.json`. |
| `profile_figures.py` | Render overview figures from registered saved results; does not train models. |
| `audit.py` | Verify frozen data, preprocessing, lineage and chronological split integrity. Writes audit outputs when explicitly run. |
| `check_sources.py` | Optional remote-source hash comparison; does not replace frozen data. |

Format maintained code with Black (`requirements-dev.txt`); `train.py` and the original bundle are explicitly excluded. Historical source bytes are available through each run's source commit, even where current maintained code has been formatted or deduplicated.
