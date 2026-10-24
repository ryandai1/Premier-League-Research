# Arsenal-focused EPL predictor — audited training dataset v1

Prepared 24 October 2026. This is a reproducible historical training release, not a live forecasting service or an optimized model.

## What you can build

Train a league-wide predictor of home win / draw / away win, then report Arsenal forecasts. The same rows include home and away goal-count targets for separate regression/count models. Arsenal-only training remains available as a comparison, but its smaller sample is not the default. These match-level sources do not support direct player recruitment recommendations, inferred coaching intentions, or causal tactical claims.

## Verified contents

| Item | Count |
|---|---:|
| Unique original matches | 9,880 |
| Seasons | 26 (2000/01–2025/26) |
| Distinct clubs across all seasons | 46 |
| Arsenal original matches | 988 |
| Default predictor inputs | 39 |
| Eligible matches | 9,439 |
| Retained but excluded from default training | 441 |
| Eligible Arsenal matches | 964 |
| Eligible matches with optional xG histories | 2,551 |

Latest match date: **24 May 2026**. There are no 2026/27 fixtures or results in this release. Each season contains 380 fixtures, 20 teams, and 38 appearances per team. One training row is one match; home and away copies have not been counted as independent examples.

## The two datasets are not additive

The RezaGooner snapshot contains 9,880 rows and **85 columns**, although its README says 65. Tara Nguyen contains 3,800 rows. After normalizing team names, all 3,800 Tara fixtures are already present in RezaGooner, with identical dates and full-time goals. Tara therefore adds a comparison source, not 3,800 new training examples. These sources can share upstream data, so agreement is not fully independent validation.

The chosen base is RezaGooner. Tara is used for reconciliation. Football-data.co.uk archives for 2015/16, 2024/25 and 2025/26 were additionally compared: all 1,140 fixtures agree with RezaGooner on dates, full-time results and 14 shared goal/shot/corner/foul/card fields. No fixtures from those archives were appended.

## Data issues and handling

1. **One disagreement between the selected datasets:** Aston Villa–West Ham on 26 December 2015. RezaGooner gives 15 home shots; Tara gives 5. Football-data confirms 15. Reza's value is retained; the discrepancy is logged.
2. **Three contradictory shot-stat matches (five team sides):** Coventry–Tottenham, 14 October 2000; Bradford–Charlton, 13 April 2001; Newcastle–West Ham, 15 August 2021. Shots on target exceed total shots on the affected side. Neither measurement is guessed: both are set to missing on that side. Goals, the fixture and the original raw bytes remain available. Histories using those measurements must pass the minimum-valid-observations rule.
3. **xG absent in at least one side for 7,023 / 9,880 matches (71.1%).** It is excluded from the 39 default inputs. Optional lagged xG features are provided in `prematch_all.csv`; missing xG is never treated as zero. Provider/method continuity has not been established, so those columns are experimental.
4. **Future-information fields:** final league position, season totals, qualification, same-season trophies, full-time/half-time results, same-match shots and other post-match statistics cannot enter a pre-match predictor. Current-match goals/results are targets only. The supplied form strings, manager fields, geography, referee, kickoff time and other metadata are not used in the default feature set; their historical availability or definition was not sufficiently audited. Form is recomputed from earlier results.
5. **441 fixtures lack sufficient usable histories.** They remain in the canonical archive and `prematch_all.csv`, with an exclusion reason. The eligible set is therefore conditional on enough prior EPL data; evaluation does not cover all season-opening/promoted-club situations. No synthetic matches, duplicate augmentation or label imputation was used.

A source-level audit cannot establish that every original observation is correct. The checks above establish internal consistency, selected cross-source agreement and safe chronological construction of the exported features. Historical data revisions may mean a modern snapshot differs from exactly what was published at the time.

## Features and timing

For each side, compute means over the last 5 and last 10 EPL matches dated strictly before the fixture, limited to the past 365 days:

- goals scored and conceded;
- shots taken and conceded;
- shots on target taken and conceded;
- corners won;
- points earned.

This produces 32 inputs. Seven more are the two recent-history counts, two intervals since the previous EPL match, two Elo ratings, and the Elo difference.

Every rolling mean requires at least five nonmissing measurements. A 10-match window can contain 5–10 observations; the number of recent fixtures is explicitly available. No global imputation or normalization has been fitted. Fit any scaler or imputer using training rows only.

**Days since the previous EPL match is not actual rest:** European competitions and domestic cups are absent. Do not call this feature fatigue or all-competition rest in an article.

Elo is an explicit untuned baseline feature: initialize at 1500, K=20, home advantage 60, retain 75% of rating deviation at each season transition, and reset a club after more than 365 days without an observed EPL game. All matches on a date receive features before any results from that date update the state. No kickoff-time accuracy is assumed.

Earlier matches within validation/test periods may supply histories for later fixtures: this simulates sequential forecasting as results become known. Model parameters remain fixed during evaluation. It is not a forecast of an entire future season made on one initial date.

## Fixed chronological split

| Split | Seasons | Eligible matches | Arsenal matches |
|---|---|---:|---:|
| Train | 2000/01–2021/22 | 7,977 | 815 |
| Validation | 2022/23–2023/24 | 730 | 74 |
| Test | 2024/25–2025/26 | 732 | 75 |

These are initial experimental boundaries, not optimized choices. The training partition is long: use validation to compare all history with a recent training window (for example 2014/15 onward). Avoid choosing a window using test performance. Keep multiple earlier rolling evaluation periods for robustness work.

No model predictions or performance metrics were generated for the test partition. Its schemas, labels and counts were inspected for data quality. Changing split boundaries requires a new documented release.

## Load into a predictor

```python
from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

root = Path('arsenal_predictor_v1')  # adjust to the extracted folder
X_train = pd.read_csv(root / 'data/train/X.csv')
y_train = pd.read_csv(root / 'data/train/y.csv')['target_result']
X_val = pd.read_csv(root / 'data/validation/X.csv')
val_index = pd.read_csv(root / 'data/validation/index.csv')

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
model.fit(X_train, y_train)
probabilities = pd.DataFrame(model.predict_proba(X_val), columns=model.classes_)
arsenal = val_index.home_team.eq('Arsenal') | val_index.away_team.eq('Arsenal')
arsenal_forecasts = pd.concat([val_index, probabilities], axis=1).loc[arsenal]
```

The class labels are **H = home win, D = draw, A = away win**. For Arsenal away fixtures, Arsenal's win probability is column A, not H. Always use `model.classes_` rather than assume a probability-column ordering.

Each split directory has aligned `X.csv`, `y.csv` and `index.csv` files. `X.csv` contains only numeric predictors; `y.csv` contains three targets, so select one explicitly. Row i in each file describes the same match. Keep the metadata available when subsetting or shuffling so you preserve alignment.

`data/arsenal_view.csv` is a convenience view including metadata, targets and optional features. **Do not pass its full contents to a model.** Select only the `features` allowlist in `feature_schema.json`; use `arsenal_result` only as an alternative target. Club identities are metadata rather than default features, making cross-club experiments easier to interpret.

For an Arsenal-only comparison, filter each split's metadata and apply the same row mask to X and y. For strict unseen-club evaluation, remove all training matches involving the held-out club on either side. This changes the experiment and may also require rebuilding Elo/history features depending on what information is permitted at deployment; the supplied split tests future matches, not unseen clubs.

## What is included

- `data/matches_clean.csv`: all canonical fixtures and observed post-match values; not direct model input.
- `data/prematch_all.csv`: derived features, targets, metadata and readiness flags for all matches.
- `data/{train,validation,test}/X.csv`, `y.csv`, `index.csv`: model-ready aligned arrays.
- `data/arsenal_view.csv`: Arsenal-specific view.
- `feature_schema.json` and `data_dictionary.csv`: permitted inputs and definitions.
- `audit/`: coverage, conflicts, excluded fixtures, column decisions, build summary and validation results.
- `scripts/build_dataset.py`: deterministic cleaning and chronological feature construction.
- `scripts/verify_dataset.py`: integrity, leakage-invariance and baseline compatibility checks.
- `scripts/download_sources.py`: restores missing CSV snapshots and verifies checksums.
- `audit_walkthrough.ipynb`: inspectable notebook companion; run locally or in Colab after extracting the folder.
- `raw/`: exact downloaded CSV snapshots and the two repository README files.
- `source_manifest.json`: source URLs, pinned Git commits where available and SHA-256 checksums.

## Rebuild and verify

```bash
pip install -r requirements.txt
python scripts/build_dataset.py
python scripts/verify_dataset.py
```

The bundled raw files allow an offline rebuild. Downloads are only needed to restore missing raw files. GitHub downloads are commit-pinned; football-data archives can be revised, so the restore script refuses bytes that differ from the recorded checksum.

The verification script checks that changing a current/future outcome cannot alter any earlier or same-day input, and that features built from a historical prefix equal those produced using the full snapshot. It also fits one fixed logistic baseline and compares it with a class-prior baseline on validation only. Validation log loss was 0.9492 versus 1.0525 (lower is better). This is an execution/compatibility check, not hyperparameter tuning, a league-strength baseline comparison, a calibrated public forecast or evidence of an optimal model. No trained model is packaged in this data release.

The script currently builds historical training rows. Live scoring requires a fixture-input interface, updated histories and a separately validated saved model; do not invent goals/results for future fixtures to use the historical builder.

## Provenance and reuse

Sources:

- https://github.com/RezaGooner/english-premier-league-match-dataset — pinned commit in manifest; README states CC BY-NC 4.0.
- https://github.com/tara-nguyen/english-premier-league-datasets-for-10-seasons — pinned commit in manifest; author requests attribution and identifies football-data.co.uk as the source.
- https://www.football-data.co.uk/englandm.php — supplemental archive checks for three seasons.

The README snapshots and source manifest are included. This bundle does not grant broader rights than the upstream sources. Before a paid account uses or redistributes the data, resolve the noncommercial and upstream reuse terms. Reporting your methodology and attribution should accompany published analysis.
