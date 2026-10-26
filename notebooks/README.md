# Active Arsenal-excluded notebooks

Run 01 → 02 → 03 → 04, then the standalone frozen-model review. All five use `results/arsenal_isolated/`. They write derived figures and verdict tables only to `results/arsenal_isolated_review/`.

- **01:** original source evidence, corrected partitions, exact reconstruction of source feature rows, class balance and training distributions.
- **02:** explicit optional fresh training; completed-run evidence, learning curves, every epoch and frozen-parameter verification.
- **03:** saved-model inference, probability scores, paired month-block uncertainty, calibration and confusion.
- **04:** overview including the same corrected graphics.
- **Frozen_Model_Test_Review:** all 75 Arsenal test fixtures and all seven saved model verdict tables.

Build with `python -m experiments.build_isolated_notebooks`; execute with `python -m experiments.execute_isolated_notebooks --kernel python3`. Building clears saved notebook outputs; always execute afterward. Default execution does not retrain. See [reproduction](../docs/reproduction.md).
