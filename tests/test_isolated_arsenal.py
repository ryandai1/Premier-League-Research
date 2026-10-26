"""Isolation, frozen-weight, provenance, and evaluation regression checks."""

import json
import unittest

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from experiments import train
from experiments.isolated_arsenal import OUTPUT, league_features


class IsolatedArsenal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((OUTPUT / "manifest.json").read_text())

    def test_provenance_and_partitions(self):
        for path, digest in self.record["code_sha256"].items():
            self.assertEqual(train.digest(train.ROOT / path), digest)
        for path, digest in self.record["input_sha256"].items():
            self.assertEqual(train.digest(train.DATA / path), digest)
        for path, digest in self.record["artifact_sha256"].items():
            self.assertEqual(train.digest(OUTPUT / path), digest)
        all_ids = []
        for population in ["league", "arsenal"]:
            for split in ["train", "validation", "test"]:
                meta = pd.read_csv(OUTPUT / population / split / "index.csv")
                self.assertEqual(len(meta), self.record["rows"][population][split])
                self.assertTrue(meta.match_id.is_unique)
                self.assertTrue(
                    train.arsenal(meta).all()
                    if population == "arsenal"
                    else not train.arsenal(meta).any()
                )
                all_ids.append(set(meta.match_id))
        for i, ids in enumerate(all_ids):
            for other in all_ids[:i]:
                self.assertFalse(ids & other)

    def test_arsenal_removal_precedes_features(self):
        matches = pd.read_csv(
            train.DATA / "data/matches_clean.csv", parse_dates=["date"]
        )
        # Corrupt every Arsenal observation; source features must remain identical.
        altered = matches.copy()
        columns = [
            c
            for c in matches
            if c.endswith(("_goals", "_shots", "_sot", "_corners", "_xg"))
        ]
        altered.loc[train.arsenal(altered), columns] = 999
        changed = league_features(altered)
        expected = league_features(matches.loc[~train.arsenal(matches)])
        pd.testing.assert_frame_equal(changed, expected)
        ready = expected[self.record["features"]].notna().all(axis=1)
        for split, mask in [
            ("train", expected.season_start <= 2021),
            ("validation", expected.season_start.between(2022, 2023)),
            ("test", expected.season_start >= 2024),
        ]:
            rows = expected.loc[ready & mask]
            meta = pd.read_csv(OUTPUT / "league" / split / "index.csv")
            self.assertEqual(rows.match_id.tolist(), meta.match_id.tolist())
            saved = pd.read_csv(OUTPUT / "league" / split / "X.csv").to_numpy(
                dtype=np.float32
            )
            np.testing.assert_array_equal(
                rows[self.record["features"]].to_numpy(dtype=np.float32), saved
            )

    def test_transfer_freezing_and_scalers(self):
        folder = OUTPUT / "models"

        def ck(name):
            return torch.load(folder / f"{name}_best.pt", weights_only=False)

        source = ck("league_frozen_source_mlp")
        target = ck("head_finetune_mlp")
        for key in ["0.weight", "0.bias", "3.weight", "3.bias"]:
            self.assertTrue(
                torch.equal(source["model_state"][key], target["model_state"][key])
            )
        audit = json.loads((folder / "head_finetune_mlp_execution.json").read_text())
        self.assertEqual(audit["trainable_parameters"], ["5.weight", "5.bias"])
        self.assertTrue(audit["frozen_hidden_layers_unchanged"])
        self.assertEqual(
            audit["initial_checkpoint_sha256"],
            train.digest(folder / "league_frozen_source_mlp_best.pt"),
        )
        if target["epoch"] > 0:
            self.assertTrue(
                any(
                    not torch.equal(source["model_state"][k], target["model_state"][k])
                    for k in ["5.weight", "5.bias"]
                )
            )
        for name, population in [
            ("league_mlp", "league"),
            ("league_frozen_source_mlp", "league"),
            ("arsenal_mlp", "arsenal"),
        ]:
            model = ck(name)
            x = pd.read_csv(OUTPUT / population / "train/X.csv").to_numpy(
                dtype=np.float32
            )
            scaler = StandardScaler().fit(x)
            np.testing.assert_array_equal(model["scaler_mean"], scaler.mean_)
            # Reduction order can differ for CSV-loaded array layouts at float64 roundoff.
            np.testing.assert_allclose(
                model["scaler_scale"], scaler.scale_, rtol=1e-13, atol=0
            )
            self.assertIsNone(model["initial_checkpoint"])
        # Verify source checkpoint loss uses only the isolated league validation rows.
        xv = pd.read_csv(OUTPUT / "league/validation/X.csv").to_numpy(dtype=np.float32)
        yv = (
            pd.read_csv(OUTPUT / "league/validation/y.csv")
            .target_result.map({c: i for i, c in enumerate(train.CLASSES)})
            .to_numpy()
        )
        for name in ["league_mlp", "league_frozen_source_mlp"]:
            saved = ck(name)
            model = train.network(39)
            model.load_state_dict(saved["model_state"])
            model.eval()
            scaled = (xv - np.array(saved["scaler_mean"])) / np.array(
                saved["scaler_scale"]
            )
            with torch.no_grad():
                loss = torch.nn.functional.cross_entropy(
                    model(torch.tensor(scaled, dtype=torch.float32)), torch.tensor(yv)
                ).item()
            self.assertAlmostEqual(loss, saved["validation_log_loss"], places=6)
        for name, parent in [
            ("full_finetune_mlp", "league_mlp"),
            ("head_finetune_mlp", "league_frozen_source_mlp"),
        ]:
            self.assertEqual(ck(name)["scaler_mean"], ck(parent)["scaler_mean"])
            self.assertEqual(ck(name)["scaler_scale"], ck(parent)["scaler_scale"])
        y = (
            pd.read_csv(OUTPUT / "league/train/y.csv")
            .target_result.map({c: i for i, c in enumerate(train.CLASSES)})
            .to_numpy()
        )
        dummy = joblib.load(folder / "league_dummy.joblib")
        np.testing.assert_array_equal(
            dummy.class_prior_, np.bincount(y, minlength=3) / len(y)
        )
        forest = joblib.load(folder / "league_random_forest.joblib")
        self.assertEqual(forest.n_features_in_, 39)
        for tree in forest.estimators_:
            self.assertEqual(tree.tree_.weighted_n_node_samples[0], len(y))

    def test_saved_predictions_and_metrics(self):
        schema, original = train.load()
        metrics = pd.read_csv(OUTPUT / "evaluation/metrics.csv")
        for row in metrics.itertuples():
            x, y, meta = original[row.split]
            mask = train.arsenal(meta)
            x, y, meta = x[mask], y[mask], meta.loc[mask].reset_index(drop=True)
            frame = pd.DataFrame(x, columns=schema["features"])
            if row.model.endswith("mlp"):
                p = train.predict_checkpoint(
                    OUTPUT / "models" / f"{row.model}_best.pt", frame
                ).to_numpy()
            else:
                p = joblib.load(
                    OUTPUT / "models" / f"{row.model}.joblib"
                ).predict_proba(x)
            saved = pd.read_csv(
                OUTPUT / "evaluation" / f"{row.model}_{row.split}_predictions.csv"
            )
            self.assertEqual(saved.match_id.tolist(), meta.match_id.tolist())
            self.assertEqual(
                saved.target_result.tolist(), np.array(train.CLASSES)[y].tolist()
            )
            np.testing.assert_allclose(
                saved[["p_A", "p_D", "p_H"]].to_numpy().astype(p.dtype),
                p,
                rtol=0,
                atol=1e-15,
            )
            scores = train.score(y, p)
            for metric in ["log_loss", "brier_multiclass", "accuracy", "n"]:
                self.assertAlmostEqual(scores[metric], getattr(row, metric), places=12)


if __name__ == "__main__":
    unittest.main()
