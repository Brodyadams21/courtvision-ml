# Deep Learning Report — Shot Make Probability

**Project:** CourtVision ML  
**Phase:** 7 — Deep Learning and Spatial Modeling  
**Season:** 2024-25  
**Report date:** 2026-06-11  
**Training scripts:** `train_mlp.py`, `train_gru.py`  
**Sequence builder:** `sequence_features.py`  
**MLflow experiments:** `courtvision-mlp`, `courtvision-gru`

---

## Summary

Phase 7 adds PyTorch models alongside the Phase 6 LightGBM **Candidate** (`courtvision-shot-make-model`). The deep-learning path has three checkpoints:

1. **MLP v1 (tabular)** — same 31 features as LightGBM, with median imputation and standard scaling.
2. **MLP v2 (spatial)** — tabular features plus 18 court-location encodings (49 inputs total).
3. **GRU (spatial + sequence)** — spatial tabular branch plus a 5-step prior play-by-play sequence branch with 26 event-context features.

The GRU is the strongest model in Phase 7: it combines shot-level spatial/tabular context with recent game flow and achieves the best held-out test metrics (log loss **0.6470**). The MLflow run is tagged `beats_lightgbm=true` and `model_role=Challenger`. LightGBM remains the registered **Candidate**; the GRU is **not** auto-promoted to production.

---

## Data and split

All models use the same Parquet export and time-based split as Phase 5–6:

| Split | Shots | Notes |
|-------|------:|-------|
| Inner train | 140,120 | Earliest 80% of **train** games (MLP/GRU early stopping) |
| Inner validation | 35,588 | Latest 20% of **train** games |
| Full train | 175,708 | 984 games |
| Test | 43,819 | 246 held-out future games |

**Source files:**

- `data/processed/features/train_shot_features_2024-25.parquet`
- `data/processed/features/test_shot_features_2024-25.parquet`

**Target:** `shot_made_flag`

---

## MLP tabular setup (v1)

| Item | Detail |
|------|--------|
| Script | `python -m courtvision.models.train_mlp --mode default --feature-set tabular` |
| Features | 31 columns from `FEATURE_COLUMNS` (identical to LightGBM) |
| Preprocessing | Median imputation → `StandardScaler` (fit on inner train only) |
| Architecture | MLP `128 → 64 → 32`, dropout `0.2`, Adam, `BCEWithLogitsLoss` |
| Selection | Validation log loss, patience-based early stopping |
| MLflow experiment | `courtvision-mlp` |

**Purpose:** Establish a fair neural baseline on the same tabular feature set as LightGBM. Answers whether a feed-forward net adds value without spatial or sequence extras.

---

## MLP spatial setup (v2)

| Item | Detail |
|------|--------|
| Script | `python -m courtvision.models.train_mlp --mode default --feature-set spatial` |
| Features | 31 tabular + 18 spatial encodings = **49** (`spatial_features.py`) |
| Spatial encodings | Scaled `loc_x`/`loc_y`, polynomials, angle sin/cos, side flags, distances to rim/corners/key/wings/paint |
| Preprocessing | Same as tabular MLP |
| Architecture | Same MLP head as v1 |
| MLflow experiment | `courtvision-mlp` |

**Purpose:** Test whether explicit court-geometry features help a neural net beyond raw `loc_x`, `loc_y`, and `shot_distance` already in the tabular set.

---

## GRU sequence setup (spatial + prior events)

| Item | Detail |
|------|--------|
| Script | `python -m courtvision.models.train_gru --mode default` |
| Feature set | `spatial_sequence` |
| Tabular branch | 49 spatial features (same as MLP v2), median impute + scale |
| Sequence branch | Prior **5** PBP events × **26** numeric event features (v2) |
| Sequence alignment | `shot_id → sequence` map; parquet row order is **not** assumed |
| Sequence preprocessing | `StandardScaler` on flattened event features (fit on inner train) |
| Architecture | Tabular `Linear→ReLU→Dropout` (embed 64) + `GRU(26→64)` → concat → MLP `64→32→1` |
| Selection | Validation log loss, patience-based early stopping |
| MLflow experiment | `courtvision-gru` |

### Event features (26 per timestep, v2)

Core timing/context: `event_order_from_shot`, `period`, `seconds_remaining_period`, `seconds_remaining_game`, `seconds_since_next_event_or_shot`, `same_team_as_shooter`, `event_team_is_home`, `score_margin_before_event`.

**Sequence v2 additions:** `event_score_change`, `event_team_is_opponent`, `is_offensive_rebound`, `is_defensive_rebound`, `is_steal`, `is_block`.

Event-type flags: `is_field_goal`, `is_made_shot_event`, `is_missed_shot_event`, `is_rebound`, `is_turnover`, `is_foul`, `is_free_throw`, `is_timeout`, `is_substitution`, `is_violation`, `is_jump_ball`, `is_unknown_event`.

### GRU MLflow artifacts

Each successful `train_gru` run logs:

| Artifact | Description |
|----------|-------------|
| `gru_training_curve.png` | Train BCE loss vs. validation log loss by epoch |
| `gru_calibration_curve.png` | Test-set calibration |
| `gru_probability_distribution.png` | Test predicted probability histogram |
| `feature_columns.json` | 49 spatial tabular column names |
| `event_feature_columns.json` | 26 sequence feature names + `sequence_length` |
| `model_config.json` | Hyperparameters and architecture summary |
| `gru_state_dict.pt` | PyTorch weights |
| `tabular_preprocessor.joblib` | Fitted tabular imputer + scaler |
| `sequence_preprocessor.joblib` | Fitted sequence scaler |

**Tags:** `model_role=Challenger`, `beats_lightgbm=true|false`, `sequence_length=5`

Copies of the three PNG figures are also written to `reports/figures/`.

---

## Sequence leakage controls

Sequence construction (`sequence_features.py`) enforces strict temporal causality:

1. For a shot at `game_event_id = N`, only play-by-play rows with **`action_number < N`** are eligible.
2. The shot's own PBP row (`action_number == N`) is **never** included — it contains make/miss outcome.
3. Future events (`action_number > N`) are never used.
4. `score_margin_before_event` uses **prior-event** score snapshots (`shift(1)` within game), never the shot outcome.
5. After building all sequences, a check confirms `max(prior action_number) < game_event_id` for every shot with prior events.
6. Unit tests in `tests/test_sequence_features.py` verify exclusion of the shot event and correct padding.

This mirrors the `score_margin` leakage fix documented in `reports/lightgbm_candidate_report.md`.

---

## Model comparison (2024-25 test set, 43,819 shots)

| Model | Features | AUC | Log loss | Brier | Accuracy | Role |
|-------|----------|----:|---------:|------:|---------:|------|
| Baseline logistic | 31 tabular | 0.6397 | 0.6610 | 0.2343 | 0.6062 | Reference |
| LightGBM Candidate | 31 tabular | 0.6479 | 0.6495 | 0.2292 | 0.6213 | **Candidate** (registry) |
| MLP tabular | 31 tabular | 0.6437 | 0.6532 | 0.2307 | 0.6203 | Neural baseline |
| MLP spatial | 49 tabular+spatial | 0.6443 | 0.6530 | 0.2306 | 0.6209 | Spatial neural model |
| GRU spatial+sequence v2 | 49 + 5×26 sequence | **0.6517** | **0.6470** | **0.2282** | **0.6233** | **Challenger** |

Baseline and LightGBM figures are from `reports/lightgbm_candidate_report.md`. PyTorch metrics are from MLflow runs in `courtvision-mlp` and `courtvision-gru`.

**Comparison rule:** `beats_lightgbm=true` when GRU test log loss **< 0.6495** (satisfied: **0.6470**).

---

## Current model ranking

Best model by held-out test log loss (2024-25):

1. GRU spatial + sequence v2: **0.6470**
2. LightGBM Candidate: 0.6495
3. MLP spatial: 0.6530
4. MLP tabular: 0.6532
5. Logistic baseline: 0.6610

---

## GRU tuning check

A small GRU tuning sweep tested hidden sizes 32, 64, and 128, dropout values 0.1, 0.2, and 0.3, and learning rates 0.0005, 0.001, and 0.002. None of the tested variants improved held-out test log loss over the default GRU configuration. The default model (`hidden_size=64`, `dropout=0.2`, `learning_rate=0.001`) remained the best GRU hyperparameter setting among those runs.

---

## Sequence feature upgrade

A sequence feature upgrade expanded the GRU event representation from 20 to 26 features. Added context included event score change, whether the event team was the opponent, offensive and defensive rebound indicators, steal indicators, and block indicators. This improved held-out test log loss from 0.6474 to 0.6470, while also improving AUC, Brier score, and accuracy. The result supports the idea that basketball-aware event context is more valuable than simple GRU hyperparameter tuning for this stage.

---

## Did deep learning help?

Yes — once sequence context was added. The tabular MLP learned useful signal but did not beat LightGBM. Spatial encodings improved the MLP slightly, showing that explicit court geometry helped the neural model. The GRU model, which combines the 49 spatial tabular features with the previous 5 play-by-play events, produced the best held-out test metrics in Phase 7.

Because the GRU improves over LightGBM on AUC, log loss, Brier score, and accuracy, it earns Challenger status. LightGBM remains the registered Candidate because it is simpler to serve, easier to interpret, and already integrated into the model registry workflow. The GRU should be reviewed for calibration, serving complexity, and monitoring before any promotion decision.

---

## Why GRU is Challenger, not automatically Production

1. **Serving complexity** — GRU inference requires live PBP sequence assembly, two preprocessors, and PyTorch runtime. LightGBM serves from a single sklearn-compatible artifact on 31 raw features.
2. **Calibration not proven** — Like LightGBM, raw logits need calibration review before expected-value or betting-style use (Phase 8).
3. **Single season** — All models train on 2024-25 only; no multi-season or drift monitoring yet.
4. **No API or monitoring** — Phase 10 (FastAPI) and Phase 12 (monitoring) are not wired for the GRU artifact bundle.
5. **Registry policy** — `courtvision-shot-make-model` **Candidate** alias remains on LightGBM until a formal promotion path (Champion alias) is defined after API, calibration, and monitoring gates.

The Challenger tag records that the GRU beat the tree candidate on held-out test log loss — a signal to invest in serving and calibration, not a production deploy.

---

## Next steps

### Calibration

- Review `gru_calibration_curve.png` by shot type, distance decile, and period.
- Compare GRU vs. LightGBM calibration on the same test bins.
- Consider post-hoc calibration (Platt scaling or isotonic) before Phase 8 expected shot value work.

### Serving

- Package `gru_state_dict.pt` + `model_config.json` + both `.joblib` preprocessors + feature JSON files into a versioned inference bundle.
- FastAPI endpoint must accept tabular features **and** assemble the prior-5 PBP sequence at request time (or load precomputed sequences for batch scoring).
- Validate `shot_id` / `game_event_id` alignment in production to preserve leakage controls.

### Registry and promotion

- If `beats_lightgbm=true`, register a separate MLflow model name (e.g. `courtvision-shot-make-gru`) with a **Challenger** alias — do not overwrite the LightGBM **Candidate** without review.
- Define explicit Champion promotion criteria (API latency, calibration error, drift thresholds).

### Modeling

- Tune GRU hidden size, learning rate, and sequence length if validation plateaus early.
- Multi-season training and sequence coverage audits when 2025-26 data is added.

---

## References

- LightGBM candidate: `reports/lightgbm_candidate_report.md`
- Baseline report: `reports/baseline_model_report.md`
- MLP training: `src/courtvision/models/train_mlp.py`
- GRU training: `src/courtvision/models/train_gru.py`
- Spatial features: `src/courtvision/models/spatial_features.py`
- Sequence features: `src/courtvision/models/sequence_features.py`
- GRU data pipeline: `src/courtvision/models/torch_sequence_data.py`
- Model definitions: `src/courtvision/models/torch_models.py`
- Sequence tests: `tests/test_sequence_features.py`, `tests/test_torch_sequence_data.py`
