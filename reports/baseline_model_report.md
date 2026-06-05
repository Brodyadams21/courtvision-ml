# Baseline Model Report — Logistic Regression Shot Make Probability

**Project:** CourtVision ML  
**Phase:** 5 — Baseline Modeling  
**Season:** 2024-25  
**Training script:** `src/courtvision/models/train_baseline.py`  
**Report date:** 2026-06-05
**MLflow run:** `9ee09aac52914c9aa79d158d2932880c`

---

## Summary

This report documents the first interpretable baseline for predicting whether an NBA shot is made (`shot_made_flag = 1`). The model is a scikit-learn pipeline: median imputation → standard scaling → logistic regression. It is trained on a time-based split exported from `gold_shot_features` and evaluated on held-out future games.

The baseline is intended as a reference point for later tree-based and deep learning models, and as the first calibrated make-probability layer for expected shot value work.

---

## Data and split

| Split | Games | Shots | Target mean (make rate) |
|-------|------:|------:|------------------------:|
| Train | 984 | 175,708 | 0.4658 |
| Test  | 246 | 43,819  | 0.4729 |

**Source files:**

- `data/processed/features/train_shot_features_2024-25.parquet`
- `data/processed/features/test_shot_features_2024-25.parquet`

**Split rule:** earliest ~80% of games by date for train; latest ~20% for test. No game appears in both splits.

**Target:** `shot_made_flag` (binary: 1 = made, 0 = missed)

---

## Features

**30 numeric/boolean modeling columns** are used. Identifiers, dates, and categorical zone strings are excluded from this baseline.

### Included

| Group | Features |
|-------|----------|
| Shot geometry | `shot_value`, `shot_distance`, `loc_x`, `loc_y`, `abs_loc_x`, `shot_angle`, `is_corner_three` |
| Game context | `is_home`, `period`, `seconds_remaining_period`, `seconds_remaining_game`, `score_margin` |
| Player rolling (5-game) | `player_recent_fg_pct_5`, `player_recent_fg3_pct_5`, `player_recent_fga_5`, `player_recent_fg3a_5`, `player_recent_minutes_5`, `player_recent_points_5` |
| Team rolling (5-game) | `team_recent_off_eff_proxy_5`, `team_recent_pace_proxy_5`, `team_recent_fg_pct_5`, `team_recent_three_point_rate_5`, `team_recent_fga_5`, `team_recent_points_5`, `team_recent_turnovers_5` |
| Opponent rolling (5-game) | `opp_recent_points_allowed_5`, `opp_recent_fg_pct_allowed_5`, `opp_recent_three_point_rate_allowed_5`, `opp_recent_pace_proxy_5`, `opp_recent_fga_allowed_5` |

### Excluded (for now)

- **IDs:** `shot_id`, `game_id`, `game_event_id`, `player_id`, `team_id`, `opponent_team_id`
- **Metadata:** `season`, `game_date`
- **Categorical zones:** `shot_zone_basic`, `shot_zone_area`, `shot_zone_range`

Zone columns and player/team IDs may be added in later models with explicit encoding or embeddings.

---

## Model pipeline

```
SimpleImputer(strategy="median")
  → StandardScaler()
  → LogisticRegression(max_iter=1000)
```

| Step | Purpose |
|------|---------|
| Median imputation | Fill missing rolling features and `score_margin` using train-set medians |
| Standard scaling | Normalize features for stable logistic regression training |
| Logistic regression | Linear, interpretable baseline for make probability |

**Decision threshold for accuracy:** 0.5 (`predicted_label = predicted_probability >= 0.5`)

---

## Missing values (train / test)

Base shot and game-state columns (except `score_margin`) are complete. Rolling features have modest null rates at the start of the season or for players/teams with limited history. `score_margin` has the largest gap because it depends on play-by-play score coverage.

| Feature group | Train nulls | Test nulls |
|---------------|------------:|-----------:|
| Core geometry & game state | 0 | 0 |
| Player rolling | 3,281–11,670 | 65–1,983 |
| Team / opponent rolling | ~2,689–2,690 | 0 |
| `score_margin` | 93,865 (~53%) | 23,096 (~53%) |

Median imputation handles these at training time without dropping rows.

---

## Test-set results (2024-25)

| Metric | Value |
|--------|------:|
| **AUC** | 0.6409 |
| **Log loss** | 0.6606 |
| **Brier score** | 0.2341 |
| **Accuracy** (threshold 0.5) | 0.6054 |
| **Predicted probability mean** | 0.4690 |
| **Predicted probability range** | 0.0094 – 0.7299 |

### Interpretation

- **AUC ~0.64** — clearly above random (0.50); the model separates makes from misses better than chance.
- **Log loss / Brier score** — reasonable for a ~47% positive rate; useful for comparing future models on the same test split.
- **Accuracy ~0.61** — above always predicting the majority class (~53% if always predicting miss).
- **Mean predicted probability ~0.47** — close to the test make rate (~0.47), suggesting the model is not systematically over- or under-shooting overall probability mass.

Predicted probabilities are moderately spread (not collapsed near 0 or 1), which is expected for a linear baseline without strong regularization toward extremes.

---

## Calibration

Calibration is assessed on the test set using a **quantile binning** strategy (`strategy="quantile"`, 10 bins). Quantile bins put a similar number of shots in each bin, which tends to produce more stable calibration curves for basketball shot data than uniform probability-width bins.

**Figures:**

| Plot | Path |
|------|------|
| Calibration curve (predicted vs. actual make rate) | `reports/figures/baseline_calibration_curve.png` |
| Predicted probability distribution | `reports/figures/baseline_probability_distribution.png` |

Calibration matters for expected shot value: if predicted make probabilities are systematically too high or too low in certain regions, expected points will be biased.

---

## MLflow experiment tracking

Runs are logged to the MLflow tracking server (not directly to PostgreSQL from the training script):

```
train_baseline.py → http://127.0.0.1:5000 → courtvision_mlflow (PostgreSQL)
```

**Experiment:** `courtvision-baseline`  
**Run name pattern:** `baseline-logistic-{season}`

### Logged parameters

`model_type`, `season`, `feature_count`, `train_rows`, `test_rows`, `imputer_strategy`, `scaler`, `classifier`, `max_iter`

### Logged metrics

`auc`, `log_loss`, `brier_score`, `accuracy`, `train_target_mean`, `test_target_mean`, `predicted_probability_mean`

### Logged artifacts

- `baseline_calibration_curve.png`
- `baseline_probability_distribution.png`
- Trained sklearn pipeline (`model/`)
- `feature_columns.json` (target + feature list)

**Artifacts storage:** `mlartifacts/` (local, via MLflow server `file:///` URI)

---

## How to reproduce

### 1. Start PostgreSQL and MLflow server

```powershell
.\scripts\start_mlflow.ps1
```

### 2. Train and log the baseline

In a second terminal:

```powershell
$env:PYTHONPATH = "src"
python -m courtvision.models.train_baseline
```

### 3. View runs

Open http://127.0.0.1:5000 (started by `start_mlflow.ps1`).

---

## Limitations and next steps

1. **Linear model** — logistic regression cannot capture strong spatial non-linearities (e.g., rim vs. deep three interactions) without manual feature engineering.
2. **No categorical zones** — `shot_zone_*` string columns are excluded; one-hot or target encoding could help the baseline.
3. **No player/team identity** — No player/team identity features — IDs are excluded by design in this baseline. Later models may add player/team context through explicit encodings, embeddings, or aggregated historical features.
4. **`score_margin` coverage** — ~53% missing; median imputation is a simple placeholder; richer game-state features or explicit missingness flags may help.
5. **Calibration** — review the calibration curve by shot type and zone in Phase 12 monitoring.

**Planned comparisons (Phase 6+):** XGBoost / LightGBM candidate, feature importance report, and registration of a production candidate model.

---

## References

- Feature export: `src/courtvision/data/build_features.py`
- Gold schema: `sql/schema.sql` (`gold_shot_features`)
- Training script: `src/courtvision/models/train_baseline.py`
- MLflow startup: `scripts/start_mlflow.ps1`
- MLflow database setup: `sql/create_mlflow_database.sql`
