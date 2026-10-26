"""Check persisted benchmark evidence against frozen data and independent scores."""

import json
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from experiments import train
from experiments.arsenal_benchmarks import DEFAULT_OUTPUT, make_models


class ArsenalBenchmarks(unittest.TestCase):
    def test_provenance_and_train_only_fits(self):
        record = json.loads((DEFAULT_OUTPUT / "manifest.json").read_text())
        self.assertEqual(
            record["code_sha256"],
            train.digest(train.ROOT / "experiments/arsenal_benchmarks.py"),
        )
        for path, digest in record["input_sha256"].items():
            self.assertEqual(train.digest(train.DATA / path), digest)
        for path, digest in record["artifacts_sha256"].items():
            self.assertEqual(train.digest(DEFAULT_OUTPUT / path), digest)
        schema, splits = train.load()
        self.assertEqual(record["features"], schema["features"])
        x, y, meta = splits["train"]
        mask = train.arsenal(meta)
        for name, fresh in make_models().items():
            fresh.fit(x[mask], y[mask])
            saved = joblib.load(DEFAULT_OUTPUT / f"{name}.joblib")
            for split in ["validation", "test"]:
                xe, _, me = splits[split]
                np.testing.assert_array_equal(
                    saved.predict_proba(xe[train.arsenal(me)]),
                    fresh.predict_proba(xe[train.arsenal(me)]),
                )
        dummy = joblib.load(DEFAULT_OUTPUT / "arsenal_dummy.joblib")
        np.testing.assert_array_equal(
            dummy.class_prior_, np.bincount(y[mask], minlength=3) / mask.sum()
        )

    def test_fixture_alignment_probabilities_and_metrics(self):
        _, splits = train.load()
        metrics = pd.read_csv(DEFAULT_OUTPUT / "metrics.csv")
        for row in metrics.itertuples():
            _, y, meta = splits[row.split]
            mask = train.arsenal(meta)
            y, meta = y[mask], meta.loc[mask].reset_index(drop=True)
            saved = pd.read_csv(
                DEFAULT_OUTPUT / f"{row.model}_{row.split}_predictions.csv"
            )
            self.assertEqual(saved.match_id.tolist(), meta.match_id.tolist())
            self.assertEqual(
                saved.target_result.tolist(), np.array(train.CLASSES)[y].tolist()
            )
            p = saved[["p_A", "p_D", "p_H"]].to_numpy()
            self.assertTrue(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all())
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-14)
            self.assertAlmostEqual(
                log_loss(y, p, labels=[0, 1, 2]), row.log_loss, places=12
            )
            self.assertAlmostEqual(
                np.mean(((p - np.eye(3)[y]) ** 2).sum(1)),
                row.brier_multiclass,
                places=12,
            )
            self.assertAlmostEqual(accuracy_score(y, p.argmax(1)), row.accuracy)
            self.assertEqual(len(y), row.n)
            ap = pd.read_csv(
                DEFAULT_OUTPUT / f"{row.model}_{row.split}_arsenal_probabilities.csv"
            )
            home = meta.home_team.eq("Arsenal").to_numpy()
            np.testing.assert_allclose(ap.p_win, np.where(home, p[:, 2], p[:, 0]))
            np.testing.assert_allclose(ap.p_loss, np.where(home, p[:, 0], p[:, 2]))
            np.testing.assert_allclose(ap.p_draw, p[:, 1])
            # Exact same evaluation fixtures as every previously saved MLP arm.
            for name in ["arsenal_mlp", "league_mlp", "league_to_arsenal_mlp"]:
                old = pd.read_csv(
                    train.ROOT
                    / "results/current/evaluation"
                    / f"{name}_{row.split}_predictions.csv"
                )
                self.assertEqual(
                    old.loc[train.arsenal(old), "match_id"].tolist(),
                    meta.match_id.tolist(),
                )


if __name__ == "__main__":
    unittest.main()
