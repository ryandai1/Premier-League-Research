"""Strict Arsenal-excluded league pretraining and controlled Arsenal adaptation."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from experiments import train
from experiments.isolated_training import fit
from experiments.inference import arsenal_probabilities

ROOT = train.ROOT
OUTPUT = ROOT / "results/arsenal_isolated"
BUILDER = train.DATA / "scripts/build_dataset.py"
spec = importlib.util.spec_from_file_location("frozen_builder", BUILDER)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def league_features(matches):
    # Remove before any rolling-history or Elo update, in every time partition.
    clean = matches.loc[~train.arsenal(matches)].copy()
    return builder.make_features(clean)


def build_data(output):
    schema, original = train.load()
    matches = pd.read_csv(train.DATA / "data/matches_clean.csv", parse_dates=["date"])
    league = league_features(matches)
    league["split"] = np.select(
        [league.season_start <= 2021, league.season_start <= 2023],
        ["train", "validation"],
        default="test",
    )
    ready = league[schema["features"]].notna().all(axis=1)
    data = {"league": {}, "arsenal": {}}
    excluded = league.loc[~ready, schema["metadata"]].copy()
    excluded["reason"] = "Incomplete pre-match features after Arsenal history exclusion"
    excluded.to_csv(output / "excluded_league_fixtures.csv", index=False)
    for split, (x, y, meta) in original.items():
        mask = train.arsenal(meta)
        data["arsenal"][split] = (
            x[mask],
            y[mask],
            meta.loc[mask].reset_index(drop=True),
        )
        rows = league.loc[ready & league.split.eq(split)].reset_index(drop=True)
        data["league"][split] = (
            rows[schema["features"]].to_numpy(dtype=np.float32),
            rows.target_result.map(
                {c: i for i, c in enumerate(train.CLASSES)}
            ).to_numpy(),
            rows[schema["metadata"]].copy(),
        )
    for population, splits in data.items():
        for split, (x, y, meta) in splits.items():
            assert np.isfinite(x).all() and meta.match_id.is_unique
            assert (
                train.arsenal(meta).all()
                if population == "arsenal"
                else not train.arsenal(meta).any()
            )
            folder = output / population / split
            folder.mkdir(parents=True)
            pd.DataFrame(x, columns=schema["features"]).to_csv(
                folder / "X.csv", index=False
            )
            pd.DataFrame({"target_result": np.array(train.CLASSES)[y]}).to_csv(
                folder / "y.csv", index=False
            )
            meta.to_csv(folder / "index.csv", index=False)
        assert splits["train"][2].date.max() < splits["validation"][2].date.min()
        assert splits["validation"][2].date.max() < splits["test"][2].date.min()
    assert not set(pd.concat([v[2] for v in data["league"].values()]).match_id) & set(
        pd.concat([v[2] for v in data["arsenal"].values()]).match_id
    )
    return schema, data


def run(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reference = json.loads((ROOT / "results/current/manifest.json").read_text())
    for path, digest in reference["input_sha256"].items():
        assert train.digest(train.DATA / path) == digest, path
    schema, data = build_data(output)
    models = output / "models"
    models.mkdir()
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
        ).strip(),
        "config": train.CONFIG,
        "features": schema["features"],
        "classes": train.CLASSES,
        "rows": {
            p: {s: len(v[1]) for s, v in splits.items()} for p, splits in data.items()
        },
        "code_sha256": {
            str(p.relative_to(ROOT)): train.digest(p)
            for p in [
                Path(__file__),
                ROOT / "experiments/isolated_training.py",
                ROOT / "experiments/train.py",
                BUILDER,
            ]
        },
        "input_sha256": reference["input_sha256"],
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "league_isolation": "All Arsenal fixtures removed before histories/Elo; league scaler, gradients, and checkpoint selection use non-Arsenal data only.",
        "arsenal_features": "Original strictly lagged full-history features retained for the fixed Arsenal target task; source league features omit Arsenal history, so feature distributions differ.",
        "selection": "Source checkpoints use non-Arsenal validation; Arsenal-only and adaptations use Arsenal validation. Test never selects weights or settings.",
        "interpretation": "Exploratory comparison on previously inspected test period; single seed. Frozen-layer dropout retains original training behavior; weights and biases remain unchanged.",
    }
    write(output / "manifest.json", record)
    summaries = []
    arms = [
        ("arsenal_mlp", "arsenal", None, False),
        ("league_mlp", "league", None, False),
        ("full_finetune_mlp", "arsenal", "league_mlp", False),
        ("league_frozen_source_mlp", "league", None, False),
        ("head_finetune_mlp", "arsenal", "league_frozen_source_mlp", True),
    ]
    for name, population, parent, freeze in arms:
        _, info = fit(
            name,
            data[population]["train"][:2],
            data[population]["validation"][:2],
            models,
            schema["features"],
            models / f"{parent}_best.pt" if parent else None,
            freeze_hidden=freeze,
        )
        info.update(training_population=population, validation_population=population)
        summaries.append(info)
    baselines = {
        "league_random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=1,
        ),
        "league_dummy": DummyClassifier(strategy="prior", random_state=42),
    }
    x, y, _ = data["league"]["train"]
    record["baseline_parameters"] = {n: m.get_params() for n, m in baselines.items()}
    for name, model in baselines.items():
        model.fit(x, y)
        assert model.classes_.tolist() == [0, 1, 2]
        joblib.dump(model, models / f"{name}.joblib", compress=3)
    # Persist completion of all fits before any target test evaluation.
    pd.DataFrame(summaries).to_csv(models / "training_summary.csv", index=False)
    record["models_frozen_utc"] = datetime.now(timezone.utc).isoformat()
    write(output / "manifest.json", record)
    evaluation = output / "evaluation"
    evaluation.mkdir()
    metrics = []
    for split in ["validation", "test"]:
        x, y, meta = data["arsenal"][split]
        frame = pd.DataFrame(x, columns=schema["features"])
        for name in [a[0] for a in arms] + list(baselines):
            if name in baselines:
                p = joblib.load(models / f"{name}.joblib").predict_proba(x)
            else:
                p = train.predict_checkpoint(
                    models / f"{name}_best.pt", frame
                ).to_numpy()
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
            table.to_csv(evaluation / f"{name}_{split}_predictions.csv", index=False)
            arsenal_probabilities(pd.DataFrame(p, columns=train.CLASSES), meta).to_csv(
                evaluation / f"{name}_{split}_arsenal_probabilities.csv", index=False
            )
    pd.DataFrame(metrics).to_csv(evaluation / "metrics.csv", index=False)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["artifact_sha256"] = {
        str(p.relative_to(output)): train.digest(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    write(output / "manifest.json", record)
    print(json.dumps(record["rows"], indent=2))
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)
