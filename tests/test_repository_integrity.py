"""Scientific preservation, historical provenance, and all-run model reload checks."""

import hashlib
import json
import subprocess
import unittest
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from experiments import experiment_2, train
from experiments.catalog import entries, ROOT
from experiments.inference import predict
from experiments.verify_repository import verify


class RepositoryIntegrity(unittest.TestCase):
    def test_precleanup_scientific_artifacts_preserved(self):
        self.assertGreater(verify(), 250)

    def test_shared_experiment_definitions(self):
        for name in ["CONFIG", "network", "load", "score", "predict_checkpoint"]:
            self.assertIs(getattr(experiment_2, name), getattr(train, name))

    def test_historical_sources_and_inputs(self):
        for entry in entries():
            with self.subTest(run=entry["id"]):
                manifest = json.loads(
                    (ROOT / entry["path"] / "run_manifest.json").read_text()
                )
                blob = subprocess.check_output(
                    [
                        "git",
                        "show",
                        f"{entry['source_commit']}:{entry['runner_source']}",
                    ],
                    cwd=ROOT,
                )
                self.assertEqual(
                    hashlib.sha256(blob).hexdigest(), manifest["code_sha256"]
                )
                for name, expected in manifest["input_sha256"].items():
                    self.assertEqual(train.digest(train.DATA / "data" / name), expected)

    def test_every_registered_model_reloads(self):
        features = pd.read_csv(train.DATA / "data/test/X.csv")
        for entry in entries():
            folder = ROOT / entry["path"]
            selection = json.loads((folder / "selected_model.json").read_text())
            for model in selection["validation_scores"]:
                with self.subTest(run=entry["id"], model=model):
                    actual = predict(folder, features, model)
                    saved = pd.read_csv(folder / f"{model}_test_predictions.csv")
                    np.testing.assert_allclose(
                        actual, saved[["p_A", "p_D", "p_H"]], rtol=1e-5, atol=1e-7
                    )

    def test_all_notebooks_valid_and_executed(self):
        for path in (ROOT / "notebooks").glob("*.ipynb"):
            with self.subTest(notebook=path.name):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                for cell in notebook.cells:
                    if cell.cell_type == "code" and cell.source.strip():
                        self.assertIsNotNone(cell.execution_count)
                        self.assertFalse(
                            any(
                                output.output_type == "error" for output in cell.outputs
                            )
                        )

    def test_figure_module_import_does_not_render(self):
        import importlib
        from unittest.mock import patch

        with patch(
            "matplotlib.pyplot.subplots",
            side_effect=AssertionError("Import must not render figures"),
        ):
            importlib.import_module("experiments.profile_figures")
