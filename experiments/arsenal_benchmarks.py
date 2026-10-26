"""Fixed Arsenal-only benchmarks on the existing study's frozen partitions."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from experiments import train
from experiments.inference import arsenal_probabilities

DEFAULT_OUTPUT = train.ROOT / "results/arsenal_benchmarks"


def make_models():
    # Fixed before execution: no search, calibration, or validation/test refit.
    return {
        "arsenal_random_forest": RandomForestClassifier(
            n_estimators=500,
            criterion="gini",
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            bootstrap=True,
            class_weight=None,
            random_state=42,
            n_jobs=1,
        ),
        "arsenal_dummy": DummyClassifier(strategy="prior", random_state=42),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(output=DEFAULT_OUTPUT):
    output = Path(output)
    # Verify exactly the input bytes used by the saved MLP study.
    reference = train.ROOT / "results/current/manifest.json"
    prior = json.loads(reference.read_text())
    for path, expected in prior["input_sha256"].items():
        if train.digest(train.DATA / path) != expected:
            raise ValueError(f"Frozen input differs from MLP study: {path}")
    schema, splits = train.load()
    subsets = {}
    for split, (x, y, meta) in splits.items():
        mask = train.arsenal(meta)
        subsets[split] = (x[mask], y[mask], meta.loc[mask].reset_index(drop=True))
        if not np.isfinite(x[mask]).all():
            raise ValueError(f"Non-finite features in {split}")
    ids = [set(meta.match_id) for _, _, meta in subsets.values()]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
        raise ValueError("Overlapping partitions")
    if not all(meta.match_id.is_unique for _, _, meta in subsets.values()):
        raise ValueError("Duplicate fixtures")
    if not (
        subsets["train"][2].date.max() < subsets["validation"][2].date.min()
        and subsets["validation"][2].date.max() < subsets["test"][2].date.min()
    ):
        raise ValueError("Partitions are not chronological")
    output.mkdir(parents=True, exist_ok=False)
    models = make_models()
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=train.ROOT, text=True
        ).strip(),
        "code_sha256": train.digest(Path(__file__)),
        "reference_manifest_sha256": train.digest(reference),
        "input_sha256": prior["input_sha256"],
        "features": schema["features"],
        "classes": train.CLASSES,
        "training_population": "Arsenal fixtures only",
        "rows": {s: len(y) for s, (_, y, _) in subsets.items()},
        "parameters": {name: model.get_params() for name, model in models.items()},
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "tuning": "None; fixed settings; fit on training only",
        "dummy_semantics": "Constant training frequencies in away/draw/home order; argmax predicts the most frequent match-result class, not necessarily an Arsenal win.",
        "limitation": "Previously inspected test period; exploratory comparison, not independent confirmatory evidence.",
    }
    write_json(output / "manifest.json", record)
    xt, yt, _ = subsets["train"]
    # Finish and persist both fits before scoring either evaluation partition.
    for name, model in models.items():
        model.fit(xt, yt)
        if model.classes_.tolist() != [0, 1, 2]:
            raise ValueError("Unexpected class order")
        joblib.dump(model, output / f"{name}.joblib", compress=3)
    metrics = []
    for name, model in models.items():
        restored = joblib.load(output / f"{name}.joblib")
        for split in ["validation", "test"]:
            x, y, meta = subsets[split]
            p = restored.predict_proba(x)
            np.testing.assert_array_equal(p, model.predict_proba(x))
            metrics.append(
                {
                    "model": name,
                    "split": split,
                    "population": "Arsenal",
                    **train.score(y, p),
                }
            )
            table = meta.copy()
            table["target_result"] = np.array(train.CLASSES)[y]
            table[["p_A", "p_D", "p_H"]] = p
            table["predicted_result"] = np.array(train.CLASSES)[p.argmax(axis=1)]
            table.to_csv(output / f"{name}_{split}_predictions.csv", index=False)
            arsenal_probabilities(pd.DataFrame(p, columns=train.CLASSES), meta).to_csv(
                output / f"{name}_{split}_arsenal_probabilities.csv", index=False
            )
    result = pd.DataFrame(metrics)
    result.to_csv(output / "metrics.csv", index=False)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["artifacts_sha256"] = {
        p.name: train.digest(p)
        for p in sorted(output.iterdir())
        if p.name != "manifest.json"
    }
    record["reload_predictions_identical"] = True
    write_json(output / "manifest.json", record)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output)
