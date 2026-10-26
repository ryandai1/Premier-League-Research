# Reproducing and validating the study

## Install

Python 3.12 and CPU PyTorch 2.5.1+cpu match the recorded executions. Dependency versions are pinned in `requirements.txt` and the original data bundle's requirements.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Hardware/BLAS/platform differences may affect bitwise reproduction. Saved manifests state actual Python, Torch and NumPy versions. Do not infer independent confirmation from reproducing this inspected test period.

## Inspect without retraining

```bash
python -m unittest discover -s tests -v
python -m experiments.verify_repository
```

Tests cover artifact reloads, feature validation, probability orientation, epoch/optimizer state, checkpoint ancestry, data hashes, registered source commits and notebook structure. `verify_repository` checks the cleanup preservation manifest without rewriting it. It does not download sources or train models.

## Execute current workflows

```bash
python -m experiments.train                 # Original league-first runner
python -m experiments.experiment_2          # Arsenal-first with optional fine-tuning enabled
python -m experiments.rerun_experiment_1     # Original runner plus epoch observation/comparison
python -m experiments.profile_figures       # Figures only; uses registered saved results
```

Disable optional specialization explicitly with `python -c "from experiments.experiment_2 import run; print(run(fine_tune=False))"`. New runs are not automatically promoted into `results/catalog.json` or the publication figures. Record any proposed new experiment's protocol before changing its settings.

To rebuild the complete data audit, run `python -m experiments.audit` in a disposable checkout: it writes `results/audit/`. It is not necessary just to inspect frozen results. Remote freshness checks are separate and optional.

## Exact historical sources

Each catalog record provides the full source commit. For example, a clean checkout of the published experiment 2 execution can be created without disturbing the current checkout:

```bash
git worktree add --detach /tmp/arsenal-exp2-reproduction 7eac1f2888f110b4e1480e28d8804ea09c2d1a27
cd /tmp/arsenal-exp2-reproduction
# Install the requirements above, then execute the notebook or runner.
```

Run manifest source hashes refer to that historical source, not to reformatted current files. Input hashes and all retained historical output bytes remain unchanged. The cleanup integrity manifest is `results/preservation_checksums.json`.

## Notebook execution and optional preview

Install `requirements-dev.txt` for notebook export and formatting tools, then:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/Experiment_2_Epoch_By_Epoch.ipynb
mkdir -p work/previews
python -m jupyter nbconvert --to html --output-dir work/previews notebooks/Experiment_2_Epoch_By_Epoch.ipynb
```

Executing a notebook retrains its models and replaces that local notebook's outputs; use a disposable checkout for historical reconstruction. Exporting HTML without `--execute` renders the saved outputs. HTML and local scratch are ignored by Git. The exact four removed HTML files can also be recovered from commit `d3838bb` using `git show COMMIT:path/to/notebook.html`.

## Public release scope

Repository visibility has not changed. Keep [the data reuse notice](../DATA_REUSE_NOTICE.md) attached to any future release: the bundle includes third-party attribution and noncommercial restrictions, and original-provider lineage is incomplete. No new license or commercial clearance is asserted by this cleanup.
