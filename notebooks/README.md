# Read and execute in order

1. [01_data_sources_and_audit.ipynb](01_data_sources_and_audit.ipynb): source roles and overlap, ten exploratory figures, independent feature reconstruction, exclusions and fixed partitions.
2. [02_models_and_training.ipynb](02_models_and_training.ipynb): architecture, control/treatment definition, independent initialization, every epoch, validation-only comparison/selection and checkpoint ancestry.
3. [03_evaluation_and_results.ipynb](03_evaluation_and_results.ipynb): fresh checkpoint reload, test metrics, paired uncertainty, calibration, confusion matrix, transfer and interpretation.

Each notebook executes in a fresh kernel. Stage guards prevent starting training before the data audit or evaluating before model freezing. Notebook 01 refuses an existing `results/current/`; move/preserve it deliberately before a new run. Never run the notebook builder merely to view results: it regenerates unexecuted notebook sources. Use the saved executed notebooks for inspection.

[Reproduce](../docs/reproduction.md) · [Protocol](../EXPERIMENT_PROTOCOL.md) · [Execution record](../results/current/notebook_execution.json)
