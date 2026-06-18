# Deep Learning Report - Shot Make Probability

**Project:** CourtVision ML

**Phase:** 7 - Deep Learning and Spatial Modeling

**Season:** 2024-25

**Report updated:** 2026-06-18

**Training scripts:** `train_mlp.py`, `train_gru.py`

**Sequence builder:** `sequence_features.py`

**MLflow experiments:** `courtvision-mlp`, `courtvision-gru`

---

## Summary

Phase 7 adds PyTorch models alongside the Phase 6 LightGBM Candidate. The model ladder now contains four neural checkpoints:

1. **MLP v1 (tabular)** - the same 31 features used by LightGBM.
2. **MLP v2 (spatial)** - 31 base features plus 18 court-location encodings.
3. **GRU v2 (spatial + sequence)** - 49 tabular features plus the previous 5 play-by-play events with 26 features per event.
4. **GRU v3 (spatial + sequence + pressure)** - 62 tabular features plus the previous 5 events with 29 features per event.

GRU v3 is the best evaluated model. It reaches **0.6468 log loss**, compared with **0.6495** for the registered LightGBM Candidate. The published expected-shot-value analysis uses the v3 run.

**GRU v3 MLflow run ID:** `40fe1b8851f7423f831a77fce30b770d`

LightGBM remains the registered Candidate because it has a simpler serving contract. GRU v3 is described as **Challenger+** in project reporting; the training code records the MLflow role tag as `Challenger`.

---

## Data and split

All models use the same time-ordered 2024-25 feature export:

| Split | Shots | Purpose |
|-------|------:|---------|
| Inner train | 140,120 | Earliest 80% of exported training games |
| Inner validation | 35,588 | Latest 20% of exported training games; early stopping |
| Full exported train | 175,708 | 984 games |
| Held-out test | 43,819 | 246 later games |

Source files:

- `data/processed/features/train_shot_features_2024-25.parquet`
- `data/processed/features/test_shot_features_2024-25.parquet`

The target is `shot_made_flag`. Preprocessors are fitted on inner-train data and then applied to validation and test data.

---

## MLP checkpoints

| Item | MLP v1 | MLP v2 |
|------|--------|--------|
| Command | `python -m courtvision.models.train_mlp --mode default --feature-set tabular` | `python -m courtvision.models.train_mlp --mode default --feature-set spatial` |
| Inputs | 31 tabular features | 31 tabular + 18 spatial features |
| Preprocessing | Median imputation, then standard scaling | Same |
| Architecture | `128 -> 64 -> 32`, dropout 0.2 | Same |
| Optimization | Adam + `BCEWithLogitsLoss` | Same |
| Selection | Validation log loss with patience-based early stopping | Same |

The spatial encodings include scaled court coordinates, polynomial terms, angle sin/cos, side indicators, and distances to basketball-relevant landmarks. Spatial MLP improves slightly over tabular MLP but does not beat LightGBM.

---

## GRU v3 setup

| Item | Detail |
|------|--------|
| Command | `python -m courtvision.models.train_gru --mode default` |
| Feature set | `spatial_sequence` |
| Tabular branch | 49 spatial features + 13 pressure/sequence-summary features = **62** |
| Sequence branch | Previous **5** play-by-play events x **29** numeric features |
| Sequence alignment | Explicit `shot_id -> sequence` mapping |
| Preprocessing | Separate train-fitted tabular and sequence preprocessors |
| Architecture | Tabular embed `62 -> 64` + `GRU(29 -> 64)` -> head `64 -> 32 -> 1` |
| Optimization | Adam, weight decay, BCE loss, early stopping on validation log loss |
| Default training | Batch size 1,024; up to 50 epochs; patience 5; seed 42 |

### v3 sequence additions

The v3 event representation expands v2 from 26 to 29 fields with:

- `event_seconds_before_shot`
- `event_likely_possession_change`
- `event_same_possession_as_shot`

The tabular branch adds 13 values derived from the prior-event window:

- Shot-period clock and a possession-age late-clock proxy
- Timeout, offensive-rebound, and turnover presence flags
- Prior-5 score-change, turnover, steal, offensive-rebound, defensive-rebound, foul, same-team-event, and opponent-event counts

These additions provide compact game-flow context without including the current shot result.

---

## Leakage and alignment controls

Sequence construction enforces strict temporal causality:

1. For a shot at `game_event_id = N`, only events with `action_number < N` are eligible.
2. The shot's own play-by-play row is excluded because it contains the make/miss result.
3. Future events are excluded.
4. `score_margin_before_event` uses a prior-event score snapshot.
5. Sequence coverage is checked independently for inner train, validation, and test shot IDs.
6. Row order is not trusted; sequences are realigned through the shot-ID map.
7. A final check requires every observed prior action number to be lower than the shot event ID.

The safeguards are covered by `test_sequence_features.py`, `test_torch_sequence_data.py`, `test_pressure_features.py`, and GRU inference tests.

---

## MLflow and inference bundle

Each successful GRU training run logs:

| Artifact | Purpose |
|----------|---------|
| `gru_state_dict.pt` | PyTorch weights |
| `model_config.json` | Architecture and training settings |
| `feature_columns.json` | Ordered 62-column tabular contract |
| `event_feature_columns.json` | Ordered 29-column sequence contract and sequence length |
| `tabular_preprocessor.joblib` | Fitted tabular imputer/scaler |
| `sequence_preprocessor.joblib` | Fitted sequence scaler |
| `gru_training_curve.png` | Training and validation history |
| `gru_calibration_curve.png` | Test calibration plot |
| `gru_probability_distribution.png` | Test probability distribution |

`src/courtvision/evaluation/predict_gru.py` downloads or reuses the bundle, validates its metadata, rebuilds the architecture, loads weights with `weights_only=True`, aligns sequences, and scores shots in batches.

---

## Model comparison

All values below use the same 43,819-shot test export.

| Model | Features | AUC | Log loss | Brier | Accuracy | Role |
|-------|----------|----:|---------:|------:|---------:|------|
| Logistic regression | 31 tabular | 0.6397 | 0.6610 | 0.2343 | 0.6062 | Baseline |
| LightGBM | 31 tabular | 0.6479 | 0.6495 | 0.2292 | 0.6213 | Registered Candidate |
| MLP tabular | 31 tabular | 0.6437 | 0.6532 | 0.2307 | 0.6203 | Neural baseline |
| MLP spatial | 49 tabular | 0.6443 | 0.6530 | 0.2306 | 0.6209 | Spatial neural |
| GRU v2 | 49 + 5 x 26 sequence | 0.6517 | 0.6470 | 0.2282 | 0.6233 | Challenger |
| **GRU v3** | **62 + 5 x 29 sequence** | **0.6516** | **0.6468** | **0.2282** | **0.6235** | **Challenger+** |

Deep learning adds value only after sequence context is included. The MLP checkpoints trail LightGBM, while both GRU versions improve log loss and Brier score. V3 trades a negligible AUC decrease from v2 for better log loss and accuracy.

---

## Evaluation caveat

Epoch selection uses the inner validation split, but multiple architecture and feature iterations have now been compared on the same 2024-25 test export. The test metrics remain useful for this portfolio-stage comparison, but they should not be treated as a permanently untouched production estimate.

Before a Champion decision, evaluate the frozen pipeline on a fresh later season or a new final holdout that was not used during v2/v3 development.

---

## Why GRU v3 is not the production Candidate

1. **Serving complexity:** inference needs a live or reconstructed prior-event sequence, two preprocessors, and PyTorch.
2. **Calibration:** no separate post-hoc calibration model has been promoted.
3. **Data scope:** all reported training and evaluation data comes from one NBA season.
4. **Operational gaps:** FastAPI serving, monitoring, drift thresholds, and retraining triggers are not implemented.
5. **Registry policy:** the registered Candidate remains the simpler LightGBM model until explicit promotion gates exist.

---

## Next steps

- Evaluate GRU and LightGBM calibration by shot type, distance, zone, and game segment.
- Freeze the current model-selection process and reserve a fresh season for final comparison.
- Register the GRU under a separate model name and Challenger alias.
- Add a small synthetic end-to-end test around each PyTorch training loop.
- Implement the serving bundle, latency checks, monitoring, and Champion criteria.
- Extend training to multiple seasons and add lineup or tracking-derived context when available.

---

## References

- `reports/model_card.md`
- `reports/lightgbm_candidate_report.md`
- `src/courtvision/models/train_mlp.py`
- `src/courtvision/models/train_gru.py`
- `src/courtvision/models/spatial_features.py`
- `src/courtvision/models/sequence_features.py`
- `src/courtvision/models/pressure_features.py`
- `src/courtvision/models/torch_sequence_data.py`
- `src/courtvision/evaluation/predict_gru.py`
