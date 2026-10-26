# Premier League Research — Arsenal-excluded transfer

The active workflow removes every Arsenal fixture before constructing league histories and Elo, fitting source scalers, training source models, and selecting source checkpoints. Arsenal adaptation and evaluation use their separate chronological partitions.

| Population | Train through 2021/22 | Validation 2022/23–2023/24 | Test 2024/25–2025/26 |
|---|---:|---:|---:|
| League excluding Arsenal | 7,142 | 654 | 656 |
| Arsenal | 815 | 74 | 75 |

Read the executed notebooks in order:

1. [Data sources and corrected isolation audit](notebooks/01_data_sources_and_audit.ipynb)
2. [Models, optional fresh training, learning curves and every epoch](notebooks/02_models_and_training.ipynb)
3. [Recomputed evaluation, uncertainty, calibration and confusion](notebooks/03_evaluation_and_results.ipynb)
4. [Experiment overview with graphics](notebooks/04_arsenal_isolated_experiments.ipynb)
5. [Frozen-model test review with every Arsenal fixture](notebooks/Frozen_Model_Test_Review.ipynb)

Completed arms: Arsenal-only MLP; league-excluding-Arsenal MLP; full fine-tuning; a separate source fit followed by output-layer-only fine-tuning; source-trained Random Forest and dummy. The two deterministic source fits are not independent uncertainty replicates. No other target-team experiments have been run.

[Current protocol and training artifacts](results/arsenal_isolated/README.md) · [Notebook outputs and verdict tables](results/arsenal_isolated_review/README.md) · [Reproduction](docs/reproduction.md)

Default notebook execution verifies completed experiments without retraining. Notebook 02 includes an explicit fresh-training switch and a new output path. Source features exclude Arsenal history, whereas target features retain real strictly earlier histories; this creates a documented source/target feature-distribution difference. The test period was previously inspected, and one seed does not establish statistical superiority.

Historical Arsenal-inclusive outputs remain in `results/current/`, `results/test_review/` and `results/arsenal_benchmarks/`. They are not the active model results. Their notebooks are recoverable from Git history and the [pre-isolation backup snapshot](https://github.com/ryandai1/Premier-League-Research/archive/c431b5f0f14414ac7eef528fddd389150950812a.zip). Raw source attribution is in [DATA_REUSE_NOTICE.md](DATA_REUSE_NOTICE.md).
