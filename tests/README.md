# Validation checks

Run `python -m unittest discover -s tests -v` from the repository root.

- `test_artifacts.py`: explicitly registered experiment 1 verification run; checkpoint reloads, input rejection, Arsenal home/away orientation, model selection and epoch caps.
- `test_experiment2_epochs.py`: explicitly registered featured experiment 2; patience replay, optimizer steps, initialization and transfer hashes, frozen comparison artifact consistency. It checks the recorded artifacts; it does not independently prove real-world timing or performance.
- `test_repository_integrity.py`: pre-cleanup artifact hashes, all five runs' source commits/input hashes, all 30 saved model reloads, shared definitions, and valid executed notebooks, and side-effect-free figure-module imports.

No test selects the newest directory implicitly. Saved outputs must be registered deliberately. The suite does not train models or claim an independent replication.
