"""Fail-closed audit of frozen observations, transformations, and row alignment."""

import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "arsenal_predictor_v1"
sys.path.insert(0, str(DATA / "scripts"))
import build_dataset as b


def run(output=None):
    out = Path(output) if output is not None else ROOT / "results" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "source_manifest.json").read_text())
    for s in manifest:
        assert (
            hashlib.sha256((DATA / "raw" / f"{s['name']}.csv").read_bytes()).hexdigest()
            == s["sha256"]
        )
    raw = pd.read_csv(DATA / "raw/reza.csv", dtype={"Season": str})
    r = b.canonical(raw, "reza")
    tara = b.canonical(pd.read_csv(DATA / "raw/tara.csv"), "tara")
    overlap = r.merge(tara, on=b.KEY, suffixes=("_r", "_t"), validate="one_to_one")
    assert len(overlap) == 3800
    disagreements = []
    for col in ["date", "result", *b.MAP_T.values()]:
        bad = overlap[col + "_r"].ne(overlap[col + "_t"])
        for _, v in overlap[bad].iterrows():
            disagreements.append((col, v[col + "_r"], v[col + "_t"]))
    assert disagreements == [("home_shots", 15.0, 5)]
    for code in ["1516", "2425", "2526"]:
        fd = b.canonical(
            pd.read_csv(DATA / f"raw/football_data_{code}.csv"),
            "fd",
            2000 + int(code[:2]),
        )
        joined = r.merge(fd, on=b.KEY, suffixes=("_r", "_f"), validate="one_to_one")
        assert len(joined) == 380
        for col in ["date", "result", *b.MAP_F.values()]:
            assert joined[col + "_r"].eq(joined[col + "_f"]).all()
    r["raw_csv_record"] = np.arange(len(r)) + 2  # header is record 1
    r = r.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    r["match_id"] = [
        hashlib.sha256(
            f"{int(v.season_start)}|{v.home_team}|{v.away_team}".encode()
        ).hexdigest()[:16]
        for v in r.itertuples()
    ]
    assert r.match_id.is_unique
    lineage = r[["match_id", "raw_csv_record", *b.KEY, "date"]].copy()
    lineage["source_name"] = "reza"
    lineage["source_sha256"] = manifest[0]["sha256"]
    lineage.to_csv(out / "row_lineage.csv", index=False)
    for side in ["home", "away"]:
        bad = r[f"{side}_sot"] > r[f"{side}_shots"]
        r.loc[bad, [f"{side}_sot", f"{side}_shots"]] = np.nan
    clean = pd.read_csv(DATA / "data/matches_clean.csv", parse_dates=["date"])
    cols = [c for c in r if c != "raw_csv_record"]
    pd.testing.assert_frame_equal(r[cols], clean[cols], check_dtype=False)
    for _, g in r.groupby("season_start"):
        counts = pd.concat([g.home_team, g.away_team]).value_counts()
        assert len(g) == 380 and len(counts) == 20 and counts.eq(38).all()
    recomputed = b.make_features(clean)
    full = pd.read_csv(
        DATA / "data/prematch_all.csv",
        parse_dates=["date", "home_history_latest_date", "away_history_latest_date"],
    )
    pd.testing.assert_frame_equal(
        recomputed, full[recomputed.columns], check_dtype=False, rtol=1e-12, atol=1e-10
    )
    assert full.core_ready.equals(full[b.FEATURES].notna().all(axis=1))
    assert full.xg_ready.equals(full[b.XG_FEATURES].notna().all(axis=1))
    expected_split = np.select(
        [full.season_start <= 2021, full.season_start <= 2023],
        ["train", "validation"],
        default="test",
    )
    assert (full.split == expected_split).all()
    schema = json.loads((DATA / "feature_schema.json").read_text())
    for split in ["train", "validation", "test"]:
        expected = full[full.core_ready & full.split.eq(split)].reset_index(drop=True)
        for filename, fields in [
            ("X", b.FEATURES),
            ("y", schema["targets"]),
            ("index", schema["metadata"]),
        ]:
            actual = pd.read_csv(DATA / f"data/{split}/{filename}.csv")
            exp = expected[fields].copy()
            if "date" in exp:
                exp["date"] = exp.date.dt.strftime("%Y-%m-%d")
            pd.testing.assert_frame_equal(
                actual, exp, check_dtype=False, rtol=1e-12, atol=1e-10
            )
    # Deterministic mutation checks at multiple eras and across season transitions.
    for cut in ["2001-08-01", "2015-08-01", "2024-08-01"]:
        sample = clean[
            clean.date.between(
                pd.Timestamp(cut) - pd.Timedelta(days=400),
                pd.Timestamp(cut) + pd.Timedelta(days=30),
            )
        ].copy()
        before = b.make_features(sample)
        changed = sample.copy()
        future = changed.date.ge(cut)
        changed.loc[future, "home_goals"] += 10
        changed.loc[future, "home_shots"] += 10
        after = b.make_features(changed)
        mask = before.date.le(cut)
        pd.testing.assert_frame_equal(
            before.loc[mask, b.FEATURES], after.loc[mask, b.FEATURES]
        )
    # Materialize each side's prior fixtures; individual metric missingness remains in clean.
    history = {}
    edges = []
    for row in clean.itertuples():
        for side in ["home", "away"]:
            team = getattr(row, side + "_team")
            previous = [
                (d, i)
                for d, i in history.get(team, [])
                if 0 < (row.date - d).days <= 365
            ][-10:]
            edges.append(
                {
                    "match_id": row.match_id,
                    "side": side,
                    "prior_match_ids": json.dumps([i for d, i in previous]),
                }
            )
            history.setdefault(team, []).append((row.date, row.match_id))
    pd.DataFrame(edges).to_parquet(out / "rolling_lineage.parquet", index=False)
    # Preserve original source bytes; provide a separate corrected metadata table.
    corrected = raw.copy()
    corrections = []
    for side in ["Home", "Away"]:
        fields = ["SeasonGF" + side, "SeasonGA" + side, "SeasonGD" + side]
        bad = raw[fields[2]] != raw[fields[0]] - raw[fields[1]]
        for i in raw.index[bad]:
            value = raw.loc[i, fields[0]] - raw.loc[i, fields[1]]
            corrections.append(
                {
                    "raw_csv_record": int(i + 2),
                    "column": fields[2],
                    "old": raw.loc[i, fields[2]],
                    "new": value,
                    "basis": "GF minus GA; algebraic repair only",
                }
            )
            corrected.loc[i, fields[2]] = value
    # Resolve inconsistent season W/D/L by recomputing from all match outcomes.
    for side in ["Home", "Away"]:
        bad = raw["SeasonPlayed" + side] != sum(
            raw["Season" + x + side] for x in ["Win", "Draw", "Loss"]
        )
        for i in raw.index[bad]:
            team = raw.loc[i, side + "Team"]
            season = raw.loc[i, "Season"]
            g = raw[raw.Season.eq(season)]
            wins = int(
                (
                    (g.HomeTeam.eq(team) & g.Result.eq("H"))
                    | (g.AwayTeam.eq(team) & g.Result.eq("A"))
                ).sum()
            )
            draws = int(
                ((g.HomeTeam.eq(team) | g.AwayTeam.eq(team)) & g.Result.eq("D")).sum()
            )
            losses = int(
                (
                    (g.HomeTeam.eq(team) & g.Result.eq("A"))
                    | (g.AwayTeam.eq(team) & g.Result.eq("H"))
                ).sum()
            )
            for metric, value in zip(["Win", "Draw", "Loss"], [wins, draws, losses]):
                col = "Season" + metric + side
                if corrected.loc[i, col] != value:
                    corrections.append(
                        {
                            "raw_csv_record": int(i + 2),
                            "column": col,
                            "old": raw.loc[i, col],
                            "new": value,
                            "basis": "Recomputed from frozen season match results",
                        }
                    )
                    corrected.loc[i, col] = value
    pd.DataFrame(corrections).to_csv(out / "metadata_corrections.csv", index=False)
    corrected[
        ["Season", "HomeTeam", "AwayTeam"]
        + [c for c in corrected if c.startswith("Season") and c != "Season"]
    ].to_parquet(out / "corrected_season_metadata.parquet", index=False)
    clean.to_parquet(out / "matches_clean.parquet", index=False)
    result = {
        "passed": True,
        "matches": len(clean),
        "reproduced_features": len(full),
        "correction_cells": len(corrections),
        "raw_preserved": True,
        "remote_football_data_verified": False,
        "original_provider_per_cell_verified": False,
        "checks": [
            "raw SHA256",
            "raw-to-clean equality",
            "season counts",
            "full feature reconstruction",
            "all split cells and alignment",
            "readiness and split definitions",
            "multi-era temporal mutation",
        ],
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
