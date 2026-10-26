"""Behavioral checks of frozen model reloads and prediction interfaces."""

import unittest, json
from pathlib import Path
import pandas as pd
import numpy as np
from experiments.inference import predict, arsenal_probabilities
from experiments.catalog import run_path

ROOT = Path(__file__).resolve().parents[1]


class FrozenArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dir = run_path("experiment_1_verification")
        cls.x = pd.read_csv(ROOT / "arsenal_predictor_v1/data/test/X.csv")
        cls.meta = pd.read_csv(ROOT / "arsenal_predictor_v1/data/test/index.csv")

    def test_all_models_reproduce_saved_predictions(self):
        models = json.loads((self.run_dir / "selected_model.json").read_text())[
            "validation_scores"
        ]
        for model in models:
            with self.subTest(model=model):
                p = predict(self.run_dir, self.x, model)
                expected = pd.read_csv(self.run_dir / f"{model}_test_predictions.csv")
                np.testing.assert_allclose(
                    p, expected[["p_A", "p_D", "p_H"]], rtol=1e-5, atol=1e-7
                )
                ap = arsenal_probabilities(p, self.meta)
                saved = pd.read_csv(
                    self.run_dir / f"{model}_test_arsenal_probabilities.csv"
                )
                np.testing.assert_allclose(
                    ap[["p_win", "p_draw", "p_loss"]],
                    saved[["p_win", "p_draw", "p_loss"]],
                    rtol=1e-5,
                    atol=1e-7,
                )

    def test_rejects_reordered_or_missing_features(self):
        with self.assertRaises(ValueError):
            predict(self.run_dir, self.x[self.x.columns[::-1]])
        damaged = self.x.copy()
        damaged.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            predict(self.run_dir, damaged)

    def test_away_orientation(self):
        p = pd.DataFrame({"A": [0.7, 0.1], "D": [0.2, 0.2], "H": [0.1, 0.7]})
        m = pd.DataFrame(
            {"home_team": ["Chelsea", "Arsenal"], "away_team": ["Arsenal", "Chelsea"]}
        )
        np.testing.assert_allclose(arsenal_probabilities(p, m).p_win, [0.7, 0.7])

    def test_selection_and_caps(self):
        selected = json.loads((self.run_dir / "selected_model.json").read_text())
        self.assertEqual(
            selected["selected_model"],
            min(selected["validation_scores"], key=selected["validation_scores"].get),
        )
        summary = pd.read_csv(self.run_dir / "training_summary.csv")
        self.assertTrue(summary.epochs_run.le(100).all())
        for row in summary.itertuples():
            history = pd.read_csv(self.run_dir / f"{row.model}_history.csv")
            if row.best_epoch > 0:
                self.assertEqual(
                    row.best_epoch,
                    int(
                        history.loc[
                            history.validation_arsenal_log_loss.idxmin(), "epoch"
                        ]
                    ),
                )


if __name__ == "__main__":
    unittest.main()
