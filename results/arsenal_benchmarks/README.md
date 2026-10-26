# Arsenal benchmark execution

Reproduce with the existing pinned repository dependencies:

```bash
python -m experiments.arsenal_benchmarks --output work/arsenal_benchmarks_rerun
python -m unittest discover -s tests -v
```

The runner refuses to overwrite an existing output directory. It verifies the frozen input hashes against the original MLP execution, uses its float32 loader and 39-feature order, and fits only the 815 Arsenal training fixtures. Validation uses 74 Arsenal fixtures from 2022/23–2023/24; test uses 75 from 2024/25–2025/26. Neither evaluation partition is used to fit or tune these benchmarks. Existing study outputs and model selection are preserved.

Random Forest uses 500 trees, Gini impurity, square-root feature sampling, minimum leaf size 5, unlimited depth, bootstrap sampling, no class weighting, seed 42 and one worker. These are fixed settings, not a search result. Trees use the raw frozen features without scaling. DummyClassifier uses `strategy="prior"`: constant training class frequencies in A/D/H (away win/draw/home win) order. Its hard prediction is the most frequent match-result class; this is not an always-Arsenal-win baseline.

Both fitted models are saved as compressed joblib artifacts and reloaded before scoring. Use trusted repository artifacts only; joblib loading executes Python serialization. The manifest records full estimator parameters, runtime versions, source/input hashes and artifact checksums. Fixture-level CSVs include A/D/H probabilities and separate Arsenal win/draw/loss exports. Metrics use the original study's log loss, multiclass Brier sum (0–2), and accuracy definitions.

This is an exploratory extension on a previously inspected test period. The MLP comparison and interpretation are provided in chat, not duplicated here.
