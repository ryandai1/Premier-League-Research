# Read and execute in order

1. [01_data_sources_and_audit.ipynb](01_data_sources_and_audit.ipynb): source roles and overlap, ten exploratory figures, independent feature reconstruction, exclusions and fixed partitions.
2. [02_models_and_training.ipynb](02_models_and_training.ipynb): architecture, control/treatment definition, independent initialization, every epoch, validation-only comparison/selection and checkpoint ancestry.
3. [03_evaluation_and_results.ipynb](03_evaluation_and_results.ipynb): fresh checkpoint reload, test metrics, paired uncertainty, calibration, confusion matrix, transfer and interpretation.

Each notebook executes in a fresh kernel. Stage guards prevent starting training before the data audit or evaluating before model freezing. Notebook 01 refuses an existing `results/current/`; move/preserve it deliberately before a new run. Never run the notebook builder merely to view results: it regenerates unexecuted notebook sources. Use the saved executed notebooks for inspection.

[Reproduce](../docs/reproduction.md) · [Protocol](../EXPERIMENT_PROTOCOL.md) · [Execution record](../results/current/notebook_execution.json)

## Inference-only companion

[Inspect the held-out test with frozen models](Frozen_Model_Test_Review.ipynb) runs independently of the numbered sequence. It loads the existing checkpoints, scores the original 732 test fixtures (75 Arsenal), and exports actual outcomes and probabilities to [results/test_review/](../results/test_review/README.md). It performs no fitting or selection and verifies the original data/models/study outputs remain unchanged. Open it directly in Jupyter and run all cells; do not invoke the training executor for this task.

## Corrected Arsenal-excluded experiment

[04_arsenal_isolated_experiments.ipynb](04_arsenal_isolated_experiments.ipynb) is an executed review of the corrected runs. It displays all epoch records and reloads saved models to verify predictions and metrics. It does not retrain; the notebook includes the separate training command. Earlier notebooks retain the historical Arsenal-inclusive experiment.
