# LightGBM Candidate Report — Shot Make Probability

**Project:** CourtVision ML  
**Phase:** 6 — Tree-Based Production Candidate  
**Season:** 2024-25  
**Training script:** `src/courtvision/models/train_lgbm.py`  
**Leakage audit:** `src/courtvision/models/audit_feature_leakage.py`  
**Report date:** 2026-06-09  
**MLflow best run:** `lightgbm-best-2024-25` (`bdcf494e1e474c3a9df9118c215f20d5`)  
**Registered model:** `courtvision-shot-make-model` (version 1, alias `Candidate`)

---

## Summary

This report documents the first tree-based production candidate for predicting whether an NBA shot is made (`shot_made_flag = 1`). The model is a LightGBM binary classifier trained on the same time-based train/test Parquet export used by the Phase 5 logistic regression baseline.

A small hyperparameter search (10 hand-picked configs) selects the best model by **validation log loss** on an inner time-based split of the training data. The winning configuration is retrained on the full training split and evaluated on held-out future games. The final model is logged to MLflow, registered in the model registry, and promoted to the `Candidate` alias.

Compared with the baseline on the same 2024-25 test split, LightGBM improves every reported classification metric: higher AUC and accuracy, lower log loss and Brier score. Shot geometry and court location dominate gain-based feature importance.

This candidate is the result of a deliberate leakage investigation: an early LightGBM run produced implausibly perfect metrics, which triggered a single-feature audit, a `score_margin` engineering fix, feature rebuild, and retraining. The corrected pipeline passes the audit with no suspicious single-feature AUCs.

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

### Features (31 modeling columns)

LightGBM uses **31 numeric/boolean columns** from `FEATURE_COLUMNS` in `src/courtvision/models/common.py`. Identifiers, dates, and categorical zone strings are excluded.

| Group | Features |
|-------|----------|
| Shot geometry | `shot_value`, `shot_distance`, `loc_x`, `loc_y`, `abs_loc_x`, `shot_angle`, `is_corner_three` |
| Game context | `is_home`, `period`, `seconds_remaining_period`, `seconds_remaining_game`, `score_margin`, `score_margin_missing` |
| Player rolling (5-game) | `player_recent_fg_pct_5`, `player_recent_fg3_pct_5`, `player_recent_fga_5`, `player_recent_fg3a_5`, `player_recent_minutes_5`, `player_recent_points_5` |
| Team rolling (5-game) | `team_recent_off_eff_proxy_5`, `team_recent_pace_proxy_5`, `team_recent_fg_pct_5`, `team_recent_three_point_rate_5`, `team_recent_fga_5`, `team_recent_points_5`, `team_recent_turnovers_5` |
| Opponent rolling (5-game) | `opp_recent_points_allowed_5`, `opp_recent_fg_pct_allowed_5`, `opp_recent_three_point_rate_allowed_5`, `opp_recent_pace_proxy_5`, `opp_recent_fga_allowed_5` |

**Modeling difference from baseline:** LightGBM consumes raw feature values (including nulls) with **no imputation or scaling**. The baseline pipeline median-imputes missing values and standardizes features before logistic regression. LightGBM also adds `score_margin_missing`, an explicit boolean flag for shots without prior play-by-play score alignment.

### Missing values

Rolling features have modest null rates at season start or for players/teams with limited history. `score_margin` remains the largest gap (~53% missing in train and test) because it depends on play-by-play alignment. Unmatched shots keep `score_margin` null and `score_margin_missing = true`.

---

## Leakage audit and score_margin fix

### What happened

An initial LightGBM run produced perfect metrics, which triggered a leakage audit. The issue was traced to `score_margin` being computed using `shot_made_flag`. The feature engineering logic was corrected to use prior play-by-play score snapshots only, and a unit test was added to ensure made and missed shots at the same event receive the same pre-shot margin. After rebuilding features, the leakage audit showed no single feature exceeded the suspicious AUC threshold.

That sequence — suspicious results → targeted audit → root-cause fix → regression test → feature rebuild → clean re-audit — is the quality gate applied before registering this candidate.

### Leakage audit method

The audit (`audit_feature_leakage.py`) trains a one-feature LightGBM model per column and evaluates on the held-out test split. Features with test AUC ≥ 0.90 are flagged as suspicious.

**Threshold:** AUC ≥ 0.90  
**Features audited:** 31 (all modeling columns)

### Post-fix audit results (2024-25 test set)

| Rank | Feature | Test AUC | Flagged |
|-----:|---------|----------:|:-------:|
| 1 | `shot_distance` | 0.6409 | |
| 2 | `loc_y` | 0.6182 | |
| 3 | `abs_loc_x` | 0.6171 | |
| 4 | `loc_x` | 0.6168 | |
| 5 | `shot_value` | 0.5900 | |
| … | … | … | |
| 26 | `score_margin_missing` | 0.5000 | |
| 27 | `score_margin` | 0.4995 | |

**No single feature exceeded the suspicious AUC threshold.**

`score_margin` and `score_margin_missing` both perform at chance (~0.50 AUC) in isolation, which is the expected behavior after the leakage fix. High AUC on `score_margin_missing` alone would have indicated label-correlated missingness; that did not occur.

### score_margin fix (prior-event logic)

The original bug tied margin to the shot outcome via `shot_made_flag`. The corrected pipeline in `build_features.py` attaches score margin as follows:

1. Match each shot to play-by-play on `game_id` + `game_event_id` = `action_number`.
2. Take scores from the **previous PBP event** in the same game (shift-by-one within game).
3. Compute margin from the **shooting team's perspective** (home: home − away; away: away − home).
4. Never use `shot_made_flag`, `shot_value`, or the shot's own scoring outcome.
5. Unmatched shots keep `score_margin` null; `score_margin_missing` is set accordingly.

Unit tests in `tests/test_score_margin.py` verify that made and missed shots at the same event receive identical pre-shot margins, away-team perspective is correct, unmatched shots stay null, and the first event in a game uses a zero prior score.

---

## Hyperparameter search setup

| Setting | Value |
|---------|-------|
| Mode | `--mode search` |
| Search space | 10 hand-picked configs (not full Cartesian grid) |
| Inner split | Time-based, earliest 80% of **train** games → inner train; latest 20% → validation |
| Inner train | 787 games, 140,120 shots |
| Inner validation | 197 games, 35,588 shots |
| Selection metric | **Validation log loss** (lower is better) |
| Final training | Best config retrained on full train split (984 games, 175,708 shots) |
| Evaluation | Held-out test split (246 games, 43,819 shots) — not used for selection |
| MLflow experiment | `courtvision-lightgbm` |
| Trial run pattern | `lightgbm-search-{season}-{config_index:02d}` |
| Best run pattern | `lightgbm-best-{season}` |

**Tuned hyperparameters:** `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `reg_lambda`  
**Fixed across configs:** `subsample=0.8`, `colsample_bytree=0.8`, `objective=binary`, `random_state=42`

### Search trial summary (validation log loss)

| Config | Leaves | LR | Trees | Min child | Lambda | Val log loss | Val AUC | Test log loss* |
|-------:|-------:|---:|------:|----------:|-------:|-------------:|--------:|---------------:|
| **0** | **15** | **0.05** | **300** | **50** | **0.0** | **0.6492** | **0.6494** | 0.6496 |
| 3 | 31 | 0.03 | 300 | 50 | 0.0 | 0.6492 | 0.6496 | 0.6497 |
| 1 | 31 | 0.05 | 300 | 50 | 0.0 | 0.6493 | 0.6487 | 0.6500 |
| 7 | 31 | 0.05 | 300 | 100 | 0.0 | 0.6495 | 0.6485 | 0.6501 |
| 4 | 31 | 0.03 | 600 | 50 | 0.0 | 0.6497 | 0.6479 | 0.6502 |
| 8 | 63 | 0.05 | 300 | 100 | 1.0 | 0.6501 | 0.6468 | 0.6518 |
| 2 | 63 | 0.05 | 300 | 50 | 0.0 | 0.6505 | 0.6447 | 0.6510 |
| 6 | 63 | 0.03 | 600 | 50 | 0.0 | 0.6504 | 0.6456 | 0.6513 |
| 5 | 31 | 0.05 | 600 | 50 | 0.0 | 0.6507 | 0.6451 | 0.6513 |
| 9 | 31 | 0.05 | 600 | 100 | 1.0 | 0.6510 | 0.6444 | 0.6516 |

\*Test metrics during search are logged for comparison only; config selection uses validation log loss. Final test metrics below come from retraining the winner on full train.

Config **0** wins on validation log loss (tied with config 3 at 0.6492; index 0 selected as the recorded best). Shallower trees (`num_leaves=15`) generalize better than the larger configurations in this grid.

---

## Best hyperparameters

Selected config (index 0), retrained on full train:

| Parameter | Value |
|-----------|------:|
| `num_leaves` | 15 |
| `learning_rate` | 0.05 |
| `n_estimators` | 300 |
| `min_child_samples` | 50 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_lambda` | 0.0 |
| `objective` | binary |
| `random_state` | 42 |

---

## Baseline comparison

Both models are evaluated on the same 2024-25 held-out test split (43,819 shots) and the same **31-feature** Parquet export (including `score_margin` and `score_margin_missing`). Baseline figures below are from a **baseline rerun** on that feature set — not the original 30-feature Phase 5 report in `reports/baseline_model_report.md` (AUC 0.6409 on the earlier export).

| Metric | Baseline (logistic, 31 features) | LightGBM candidate | Δ (LGBM − baseline) |
|--------|---------------------------------:|-------------------:|--------------------:|
| **AUC** | 0.6397 | **0.6479** | +0.0082 |
| **Log loss** | 0.6610 | **0.6495** | −0.0115 |
| **Brier score** | 0.2343 | **0.2292** | −0.0051 |
| **Accuracy** (threshold 0.5) | 0.6062 | **0.6213** | +0.0151 |

LightGBM improves separation (AUC), probabilistic fit (log loss, Brier), and hard-classification accuracy. Gains are modest but consistent across metrics, which is reasonable for a same-feature-family comparison on a noisy basketball target.

---

## Final test metrics

Retrained best model on full train, evaluated on held-out test (2024-25):

| Metric | Value |
|--------|------:|
| **AUC** | 0.6479 |
| **Log loss** | 0.6495 |
| **Brier score** | 0.2292 |
| **Accuracy** (threshold 0.5) | 0.6213 |

Inner validation metrics for the selected config: log loss 0.6492, AUC 0.6494.

### Figures

| Plot | Path |
|------|------|
| Feature importance (gain) | `reports/figures/lightgbm_feature_importance_gain.png` |
| Feature importance table (gain) | `reports/tables/lightgbm_feature_importance_gain.csv` |
| Calibration curve (test) | MLflow artifact `lightgbm_calibration_curve.png` |
| Predicted probability distribution (test) | MLflow artifact `lightgbm_probability_distribution.png` |

---

## Feature importance interpretation

**Primary artifacts (gain importance):**

| Artifact | Path |
|----------|------|
| Plot | `reports/figures/lightgbm_feature_importance_gain.png` |
| Table | `reports/tables/lightgbm_feature_importance_gain.csv` |

Gain importance measures total loss reduction when a feature is used in splits. It is the main basketball-facing interpretation. Split-count importance (`lightgbm_feature_importance_split.csv`, MLflow only) is kept as a secondary technical reference.

### Main interpretation

Shot geometry dominates the candidate model. `shot_distance` is the strongest feature by a wide margin, followed by `loc_y` and `abs_loc_x`. This indicates that the model mainly learns make probability from where the shot was taken. Game-state features, recent player form, team context, and opponent context add secondary signal but do not dominate.

### Top features by gain

| Rank | Feature | Gain importance |
|-----:|---------|----------------:|
| 1 | `shot_distance` | 100,889 |
| 2 | `loc_y` | 21,327 |
| 3 | `abs_loc_x` | 20,824 |
| 4 | `seconds_remaining_period` | 5,577 |
| 5 | `shot_value` | 5,477 |
| 6 | `shot_angle` | 4,248 |
| 7 | `player_recent_fg_pct_5` | 2,540 |
| 8 | `loc_x` | 2,538 |
| 9 | `seconds_remaining_game` | 1,935 |
| 10 | `player_recent_points_5` | 1,752 |
| 11 | `score_margin` | 1,448 |

Full rankings are in `reports/tables/lightgbm_feature_importance_gain.csv`.

### Supporting detail

- **Geometry vs. context** — the top three gain features are all spatial; clock and shot-type features (`seconds_remaining_period`, `shot_value`, `shot_angle`) form a second tier well below distance.
- **Player and team signal** — rolling player FG% and scoring volume appear in the top ten but with gain scores an order of magnitude smaller than `shot_distance`.
- **`score_margin` after the fix** — rank 11 with no leakage signal in the audit; the tree uses game state when available without relying on missingness structure.
- **Split count differs** — e.g., `loc_y` leads on split count while `shot_distance` leads on gain, which is why gain is the primary report artifact.

---

## MLflow tracking and registry

Runs are logged to the MLflow tracking server:

```
train_lgbm.py → MLFLOW_TRACKING_URI (from .env) → courtvision-lightgbm experiment
```

**Best run tags:** `best_model=true`, `selected_by=validation_log_loss`, `candidate=true`  
**Best run name:** `lightgbm-best-2024-25`

### Logged parameters

`model_type`, `season`, `feature_count`, `train_rows`, `validation_rows`, `test_rows`, `best_config_index`, `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_lambda`

### Logged metrics (best run)

`test_auc`, `test_log_loss`, `test_brier_score`, `test_accuracy`

### Logged artifacts

- `lightgbm_calibration_curve.png`
- `lightgbm_probability_distribution.png`
- `lightgbm_feature_importance_gain.png`
- `lightgbm_feature_importance_gain.csv`
- `lightgbm_feature_importance_split.csv`
- `feature_columns.json`
- Trained sklearn model (`model/`)

### Model registry

| Field | Value |
|-------|-------|
| Registered name | `courtvision-shot-make-model` |
| Version | 1 |
| Alias | `Candidate` |
| Source run | `bdcf494e1e474c3a9df9118c215f20d5` |
| Registration helper | `src/courtvision/models/registry.py` |

### How to reproduce

```powershell
.\scripts\start_mlflow.ps1
```

In a second terminal:

```powershell
$env:PYTHONPATH = "src"
python -m courtvision.models.train_lgbm --mode search --register-candidate
```

Optional leakage audit before or after training:

```powershell
python -m courtvision.models.audit_feature_leakage
```

View runs and the registered model at the MLflow UI (default `http://127.0.0.1:5000` when using `start_mlflow.ps1`).

---

## Limitations

1. **Modest lift over linear baseline** — improvements are real but small; basketball shot outcomes remain inherently noisy.
2. **No categorical zones** — `shot_zone_*` strings are still excluded; encoding or embeddings may help.
3. **No player/team identity** — IDs are excluded by design; only rolling aggregates represent player/team context.
4. **`score_margin` coverage** — ~47% of shots have a margin; the rest rely on `score_margin_missing` and other features.
5. **Hand-picked search grid** — 10 configs explore key knobs but are not exhaustive; further tuning may yield small gains.
6. **No probability calibration pass** — unlike a dedicated calibrator, raw LightGBM probabilities should be reviewed before expected-value work (Phase 8).
7. **Single season** — 2024-25 only; multi-season training and drift monitoring are future work.

---

## Next steps

1. **Phase 7 — Deep learning comparison** — PyTorch spatial model vs. this tree candidate on the same split.
2. **Phase 8 — Expected shot value** — load the `Candidate` alias, score evaluation shots, and build player above-expectation summaries.
3. **Phase 10 — Inference API** — serve `courtvision-shot-make-model` via FastAPI with schema validation.
4. **Calibration review** — bin calibration by shot type, zone, and distance decile before production use.
5. **Monitoring** — track AUC, log loss, and feature drift in Phase 12; re-run search when new seasons are added.
6. **Promotion path** — define criteria to move from `Candidate` to a production alias (e.g., `Champion`) after API and monitoring are in place.

---

## References

- Baseline report: `reports/baseline_model_report.md`
- Feature export: `src/courtvision/data/build_features.py`
- Shared modeling utilities: `src/courtvision/models/common.py`
- LightGBM training: `src/courtvision/models/train_lgbm.py`
- Leakage audit: `src/courtvision/models/audit_feature_leakage.py`
- Model registry: `src/courtvision/models/registry.py`
- Score margin tests: `tests/test_score_margin.py`
- MLflow startup: `scripts/start_mlflow.ps1`
