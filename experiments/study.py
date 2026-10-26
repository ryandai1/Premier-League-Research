"""One fresh, sequential study. No archived results are read by this pipeline."""

from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import uuid
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from experiments import train
from experiments.epoch_training import fit
from experiments.inference import predict, arsenal_probabilities

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "arsenal_predictor_v1"
RUN = ROOT / "results/current"
MODELS = RUN / "models"
NEURAL = ["arsenal_mlp", "league_mlp", "league_to_arsenal_mlp"]


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def manifest():
    return json.loads((RUN / "manifest.json").read_text())


def event(stage, **details):
    with (RUN / "events.jsonl").open("a") as file:
        file.write(json.dumps({"utc": now(), "stage": stage, **details}) + "\n")


def begin():
    RUN.mkdir(parents=True, exist_ok=False)
    for name in ["data", "figures", "audit", "models", "evaluation"]:
        (RUN / name).mkdir()
    input_hashes = {
        str(p.relative_to(DATA)): train.digest(p)
        for p in DATA.rglob("*")
        if p.is_file() and p.suffix in [".csv", ".json"]
    }
    record = {
        "run_id": str(uuid.uuid4()),
        "started_utc": now(),
        "label": "fresh reproducibility rerun after test inspection",
        "config": train.CONFIG,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "input_sha256": input_hashes,
        "code_sha256": {
            str(p.relative_to(ROOT)): train.digest(p)
            for p in (ROOT / "experiments").glob("*.py")
        },
        "stage": "data_review",
        "fine_tuning_enabled": True,
        "previous_outputs_loaded": False,
    }
    write(RUN / "manifest.json", record)
    event("begin", run_id=record["run_id"])
    return record


def check_inputs():
    for path, expected in manifest()["input_sha256"].items():
        assert train.digest(DATA / path) == expected, path


def data_complete(audit):
    assert audit["passed"]
    check_inputs()
    m = manifest()
    m["stage"] = "data_verified"
    write(RUN / "manifest.json", m)
    event("data_verified")


def initialize_training():
    assert manifest()["stage"] == "data_verified"
    check_inputs()
    schema, splits = train.load()
    assert not list(MODELS.iterdir()), "Model directory must start empty."
    write(
        MODELS / "run_manifest.json",
        {
            "features": schema["features"],
            "classes": train.CLASSES,
            "config": train.CONFIG,
            "run_id": manifest()["run_id"],
        },
    )
    m = manifest()
    m["stage"] = "training"
    write(RUN / "manifest.json", m)
    event("training_started")
    return pd.DataFrame(
        [
            {
                "split": s,
                "EPL_rows": len(y),
                "Arsenal_rows": int(train.arsenal(meta).sum()),
            }
            for s, (x, y, meta) in splits.items()
        ]
    )


def fit_stage(name):
    assert manifest()["stage"] == "training"
    index = NEURAL.index(name)
    assert all((MODELS / f"{n}_execution.json").exists() for n in NEURAL[:index])
    assert not (
        MODELS / f"{name}_best.pt"
    ).exists(), "Refusing to reuse an existing fit."
    schema, splits = train.load()
    xt, yt, mt = splits["train"]
    xv, yv, mv = splits["validation"]
    am = train.arsenal(mt)
    av = train.arsenal(mv)
    mask = np.ones(len(yt), dtype=bool) if name == "league_mlp" else am
    initial = MODELS / "league_mlp_best.pt" if name == "league_to_arsenal_mlp" else None
    event(
        "fit_started",
        model=name,
        initial_checkpoint=str(initial.relative_to(RUN)) if initial else None,
    )
    _, info = fit(
        name,
        (xt[mask], yt[mask]),
        (xv[av], yv[av]),
        MODELS,
        schema["features"],
        initial,
    )
    write(MODELS / f"{name}_summary.json", info)
    event("fit_finished", **info)
    if name == "league_mlp":
        frame = pd.DataFrame(xv[av], columns=schema["features"])
        comparison = {
            n: train.score(yv[av], predict(MODELS, frame, n).to_numpy())
            for n in NEURAL[:2]
        }
        write(MODELS / "frozen_validation_comparison.json", comparison)
        event("independent_comparison_frozen", scores=comparison)
    return info


def controls_and_freeze():
    assert manifest()["stage"] == "training"
    assert all((MODELS / f"{name}_summary.json").exists() for name in NEURAL)
    schema, splits = train.load()
    xt, yt, mt = splits["train"]
    xv, yv, mv = splits["validation"]
    am = train.arsenal(mt)
    av = train.arsenal(mv)
    for name, mask in [
        ("league_logistic", np.ones(len(yt), dtype=bool)),
        ("arsenal_logistic", am),
    ]:
        scaler = StandardScaler().fit(xt[mask])
        model = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(
            scaler.transform(xt[mask]), yt[mask]
        )
        np.savez(
            MODELS / f"{name}.npz",
            mean=scaler.mean_,
            scale=scaler.scale_,
            coef=model.coef_,
            intercept=model.intercept_,
            classes=model.classes_,
        )
    np.save(MODELS / "league_prior.npy", np.bincount(yt, minlength=3) / len(yt))
    names = NEURAL + ["league_logistic", "arsenal_logistic", "league_prior"]
    frame = pd.DataFrame(xv[av], columns=schema["features"])
    scores = {
        name: train.score(yv[av], predict(MODELS, frame, name).to_numpy())["log_loss"]
        for name in names
    }
    selection = {
        "selected_model": min(scores, key=scores.get),
        "criterion": "Arsenal validation log loss only",
        "validation_scores": scores,
    }
    write(MODELS / "selected_model.json", selection)
    summary = pd.DataFrame(
        [json.loads((MODELS / f"{n}_summary.json").read_text()) for n in NEURAL]
    )
    summary.to_csv(MODELS / "training_summary.csv", index=False)
    audits = {
        n: json.loads((MODELS / f"{n}_execution.json").read_text()) for n in NEURAL
    }
    checks = {
        "independent_models_start_fresh": all(
            audits[n]["initial_checkpoint"] is None for n in NEURAL[:2]
        ),
        "same_seed_same_initial_weights": audits[NEURAL[0]]["initial_weights_sha256"]
        == audits[NEURAL[1]]["initial_weights_sha256"],
        "trained_models_have_different_weights": audits[NEURAL[0]][
            "best_weights_sha256"
        ]
        != audits[NEURAL[1]]["best_weights_sha256"],
        "fine_tuning_loads_current_league_weights": audits[NEURAL[2]][
            "initial_weights_sha256"
        ]
        == audits[NEURAL[1]]["best_weights_sha256"],
        "all_fits_changed_weights": all(
            a["initial_weights_sha256"] != a["best_weights_sha256"]
            for a in audits.values()
        ),
    }
    assert all(checks.values())
    write(MODELS / "independence_checks.json", checks)
    m = manifest()
    m["stage"] = "models_frozen"
    write(RUN / "manifest.json", m)
    event("models_frozen", selected_model=selection["selected_model"])
    return summary, selection, checks


def evaluate():
    assert manifest()["stage"] == "models_frozen"
    check_inputs()
    schema, splits = train.load()
    selection = json.loads((MODELS / "selected_model.json").read_text())
    records = []
    predictions = {}
    event("evaluation_started")
    for split in ["validation", "test"]:
        x, y, meta = splits[split]
        am = train.arsenal(meta)
        frame = pd.DataFrame(x, columns=schema["features"])
        for name in selection["validation_scores"]:
            probs = predict(MODELS, frame, name)
            p = probs.to_numpy()
            predictions[split, name] = p
            for population, mask in [
                ("Arsenal", am),
                ("Other EPL", ~am),
                ("All EPL", np.ones(len(y), dtype=bool)),
            ]:
                records.append(
                    {
                        "model": name,
                        "split": split,
                        "population": population,
                        **train.score(y[mask], p[mask]),
                    }
                )
            table = meta.copy()
            table["target_class"] = np.array(train.CLASSES)[y]
            for i, c in enumerate(train.CLASSES):
                table["p_" + c] = p[:, i]
            table.to_csv(
                RUN / "evaluation" / f"{name}_{split}_predictions.csv", index=False
            )
            oriented = arsenal_probabilities(probs, meta)
            oriented["target_class"] = table.loc[am, "target_class"].to_numpy()
            oriented.to_csv(
                RUN / "evaluation" / f"{name}_{split}_arsenal_probabilities.csv",
                index=False,
            )
    metrics = pd.DataFrame(records)
    metrics.to_csv(RUN / "evaluation/metrics.csv", index=False)
    x, y, meta = splits["test"]
    am = train.arsenal(meta)
    target = y[am]
    months = meta.loc[am, "date"].str[:7].to_numpy()
    lp = predictions["test", "league_mlp"][am]
    ap = predictions["test", "arsenal_mlp"][am]
    delta = -np.log(np.clip(lp[np.arange(len(target)), target], 1e-12, 1)) + np.log(
        np.clip(ap[np.arange(len(target)), target], 1e-12, 1)
    )
    rng = np.random.default_rng(42)
    unique = np.unique(months)
    boots = []
    for _ in range(1000):
        ids = np.concatenate(
            [
                np.flatnonzero(months == m)
                for m in rng.choice(unique, len(unique), replace=True)
            ]
        )
        boots.append(float(delta[ids].mean()))
    comparison = {
        "contrast": "League minus Arsenal-only MLP test log loss",
        "difference": float(delta.mean()),
        "interval_95_percentile": np.quantile(boots, [0.025, 0.975]).tolist(),
        "blocks": len(unique),
        "resamples": 1000,
        "limitation": "Same inspected test period; interval excludes training-seed uncertainty.",
    }
    write(RUN / "evaluation/comparison.json", comparison)
    checks = {
        "max_difference_between_independent_models": float(np.abs(lp - ap).max()),
        "independent_models_are_not_identical": not np.array_equal(lp, ap),
    }
    frame = pd.DataFrame(x, columns=schema["features"])
    changed = predict(
        MODELS, frame.iloc[::-1].reset_index(drop=True), "league_mlp"
    ).to_numpy()
    checks["max_probability_change_when_feature_rows_reordered"] = float(
        np.abs(changed - predictions["test", "league_mlp"]).max()
    )
    assert (
        checks["independent_models_are_not_identical"]
        and checks["max_probability_change_when_feature_rows_reordered"] > 0
    )
    write(RUN / "evaluation/data_sensitivity_checks.json", checks)
    m = manifest()
    m["stage"] = "complete"
    m["finished_utc"] = now()
    write(RUN / "manifest.json", m)
    event("complete")
    return metrics, comparison, checks
