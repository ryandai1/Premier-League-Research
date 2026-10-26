# Reproduce the controlled notebook sequence

## Environment

Python 3.12 and CPU PyTorch 2.5.1+cpu were used. Install the pinned packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name python3
```

Hardware, BLAS and serialization precision can affect low-order numerical agreement. The manifest records the actual Python, Torch and NumPy versions. Same-seed reruns are reproducibility checks, not independent performance replications.

## Inspect without changing outputs

Open the three numbered notebooks in order; all outputs are saved. Check the evidence with:

```bash
python -m unittest discover -s tests -v
```

## Fresh full execution

The executor refuses to overwrite `results/current`. Preserve the current execution outside the results tree before rerunning:

```bash
mkdir -p work/previous-executions
mv results/current work/previous-executions/current-before-rerun
python -m experiments.execute_study
```

Choose a new archive name if that destination already exists. Do not reuse or delete old artifacts silently. This execution command clears notebook outputs and reruns 01, 02 and 03 in separate fresh kernels. The only inputs are the frozen bundle and code; no earlier model is loaded. `--kernel NAME` selects another installed Python kernel.

To edit notebook source templates, modify `experiments/build_notebooks.py`, then run `python -m experiments.build_notebooks` before the complete execution. The builder overwrites notebook outputs with an unexecuted scaffold. It is not needed to inspect or rerun existing notebooks.

## Outputs and provenance

`results/current/manifest.json` records code/input hashes and run identity; `notebook_execution.json` records execution times; `events.jsonl` records stage order. Data/audit/model/evaluation/figure subdirectories have distinct roles. Models retain optimizer state, settings, feature order and scaler parameters. Only optional specialization loads the new league checkpoint from the same directory.

The previous repository state is recoverable at commit `aa740ef3d188655d4f3048597a5dd6a413a4a20f`. The reset inventory records removed result and notebook file hashes. Original raw snapshots, supplied metadata and frozen splits remain intact; they were independently re-audited in notebook 01.

GitHub renders the executed notebooks directly. Optional HTML previews can be generated into ignored `work/` using nbconvert from `requirements-dev.txt`; full browser-layout review was not available in this session. All fourteen exported figures were inspected directly.

## Interpretation and reuse

Test results had already been inspected, and descriptive plots of that period are disclosed as such. No new seed, split, architecture or tuning rule was selected to force changed performance. Retain [the source reuse notice](../DATA_REUSE_NOTICE.md) for redistribution. Repository visibility is unchanged.

## Test existing frozen models without training

Open [Frozen_Model_Test_Review.ipynb](../notebooks/Frozen_Model_Test_Review.ipynb) in Jupyter and run all cells. Or, with the optional development dependencies installed:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/Frozen_Model_Test_Review.ipynb
```

This standalone companion reads the frozen test features and model artifacts, writes only `results/test_review/` and its own notebook outputs, and verifies original inputs/study artifacts are unchanged. It does not call the training executor. The complete held-out fixture lists and model-specific verdict tables are linked from [the review index](../results/test_review/README.md).
