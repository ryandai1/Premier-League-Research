"""Behavioral checks for the newly executed, isolated study."""

import json
import math
from pathlib import Path
import unittest
import nbformat
import numpy as np
import pandas as pd
import torch
from experiments import study, train
from experiments.inference import predict


class CurrentStudy(unittest.TestCase):
    def test_inputs_code_and_audit(self):
        study.check_inputs()
        record = study.manifest()
        self.assertEqual(record["stage"], "complete")
        self.assertFalse(record["previous_outputs_loaded"])
        self.assertEqual(record["config"], train.CONFIG)
        for path, digest in record["code_sha256"].items():
            self.assertEqual(train.digest(study.ROOT / path), digest, path)
        self.assertTrue(
            json.loads((study.RUN / "audit/verification.json").read_text())["passed"]
        )

    def test_partitions_and_source_overlap(self):
        _, splits = train.load()
        ids = []
        for _, (_, y, meta) in splits.items():
            self.assertTrue(meta.match_id.is_unique)
            ids.append(set(meta.match_id))
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        sources = pd.read_csv(study.RUN / "data/source_summary.csv")
        self.assertEqual(int(sources.additional_fixtures.sum()), 0)
        self.assertEqual(
            int(sources.loc[sources.source.eq("reza"), "rows"].iloc[0]), 9880
        )

    def test_sequence_and_model_independence(self):
        events = [
            json.loads(x) for x in (study.RUN / "events.jsonl").read_text().splitlines()
        ]
        starts = [e for e in events if e["stage"] == "fit_started"]
        self.assertEqual([e["model"] for e in starts], study.NEURAL)
        self.assertIsNone(starts[0]["initial_checkpoint"])
        self.assertIsNone(starts[1]["initial_checkpoint"])
        stages = [e["stage"] for e in events]
        self.assertLess(stages.index("data_verified"), stages.index("training_started"))
        self.assertLess(
            stages.index("models_frozen"), stages.index("evaluation_started")
        )
        frozen = next(
            i
            for i, e in enumerate(events)
            if e["stage"] == "independent_comparison_frozen"
        )
        specialization = next(
            i
            for i, e in enumerate(events)
            if e["stage"] == "fit_started" and e["model"] == study.NEURAL[2]
        )
        self.assertLess(frozen, specialization)
        self.assertTrue(
            all(
                json.loads(
                    (study.MODELS / "independence_checks.json").read_text()
                ).values()
            )
        )
        checks = json.loads(
            (study.RUN / "evaluation/data_sensitivity_checks.json").read_text()
        )
        self.assertTrue(checks["independent_models_are_not_identical"])
        self.assertGreater(
            checks["max_probability_change_when_feature_rows_reordered"], 0
        )

    def test_every_epoch_and_saved_checkpoint(self):
        summary = pd.read_csv(study.MODELS / "training_summary.csv")
        for row in summary.itertuples():
            h = pd.read_csv(study.MODELS / f"{row.model}_history.csv")
            ck = torch.load(study.MODELS / f"{row.model}_best.pt", weights_only=False)
            self.assertEqual(ck["config"], train.CONFIG)
            self.assertEqual(h.epoch.tolist(), list(range(1, len(h) + 1)))
            self.assertLessEqual(len(h), 100)
            self.assertEqual(
                int(h.optimizer_steps.iloc[-1]),
                math.ceil(row.training_rows / 256) * len(h),
            )
            self.assertEqual(ck["epoch"], row.best_epoch)
            self.assertEqual(
                train.digest(study.MODELS / f"{row.model}_best.pt"),
                row.checkpoint_sha256,
            )
            best = float("inf")
            wait = 0
            if ck["initial_checkpoint"]:
                self.assertEqual(
                    "league_mlp_best.pt", Path(ck["initial_checkpoint"]).name
                )
                execution = json.loads(
                    (study.MODELS / f"{row.model}_execution.json").read_text()
                )
                self.assertEqual(
                    execution["initial_checkpoint_sha256"],
                    train.digest(study.MODELS / "league_mlp_best.pt"),
                )
                best = torch.load(
                    study.MODELS / "league_mlp_best.pt", weights_only=False
                )["validation_log_loss"]
            for entry in h.itertuples():
                material = (
                    entry.validation_arsenal_log_loss < best - train.CONFIG["min_delta"]
                )
                best = min(best, entry.validation_arsenal_log_loss)
                wait = 0 if material else wait + 1
                self.assertEqual(wait, entry.patience_used)
            self.assertTrue(len(h) == 100 or wait == 20)
            self.assertAlmostEqual(best, ck["validation_log_loss"])

    def test_reload_metrics_and_validation_selection(self):
        selection = json.loads((study.MODELS / "selected_model.json").read_text())
        self.assertEqual(
            selection["selected_model"],
            min(selection["validation_scores"], key=selection["validation_scores"].get),
        )
        schema, splits = train.load()
        metrics = pd.read_csv(study.RUN / "evaluation/metrics.csv")
        for split in ["validation", "test"]:
            x, y, meta = splits[split]
            am = train.arsenal(meta)
            frame = pd.DataFrame(x, columns=schema["features"])
            for name in selection["validation_scores"]:
                probs = predict(study.MODELS, frame, name)
                saved = pd.read_csv(
                    study.RUN / "evaluation" / f"{name}_{split}_predictions.csv"
                )
                # CSV stores shortest float32 representations; compare after restoring the original dtype.
                expected = (
                    saved[["p_A", "p_D", "p_H"]]
                    .to_numpy()
                    .astype(probs.to_numpy().dtype)
                )
                np.testing.assert_allclose(probs, expected, rtol=0, atol=1e-15)
                metric = train.score(y[am], probs.to_numpy()[am])
                stored = metrics.query(
                    'model==@name and split==@split and population=="Arsenal"'
                ).iloc[0]
                self.assertAlmostEqual(metric["log_loss"], stored.log_loss, places=12)
        with self.assertRaises(ValueError):
            predict(study.MODELS, frame[frame.columns[::-1]])

    def test_notebooks_executed_in_order(self):
        execution = json.loads((study.RUN / "notebook_execution.json").read_text())
        self.assertEqual([x["notebook"][:2] for x in execution], ["01", "02", "03"])
        for record in execution:
            notebook = nbformat.read(
                study.ROOT / "notebooks" / record["notebook"], as_version=4
            )
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    self.assertIsNotNone(cell.execution_count)
                    self.assertFalse(
                        any(o.output_type == "error" for o in cell.outputs)
                    )
        self.assertEqual(len(list((study.RUN / "figures").glob("*.png"))), 14)


if __name__ == "__main__":
    unittest.main()
