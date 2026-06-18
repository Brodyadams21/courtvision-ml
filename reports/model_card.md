# Model Card - GRU Spatial Sequence v3

Last updated: 2026-06-18

## Model details

| Field | Value |
|-------|-------|
| Project | CourtVision ML |
| Model | PyTorch GRU spatial_sequence v3 |
| Project role | Challenger+ |
| MLflow role tag | `Challenger` |
| MLflow run ID | `40fe1b8851f7423f831a77fce30b770d` |
| Training season | 2024-25 NBA season |
| Target | Probability that a field-goal attempt is made |
| Framework | PyTorch |
| Registered production candidate | LightGBM `courtvision-shot-make-model`, alias `Candidate` |

GRU v3 is the best evaluated model in the project, but it is not registered or deployed as the production Candidate. The LightGBM model retains that role because its serving contract is considerably simpler.

## Intended use

- Retrospective shot-quality analysis
- Expected shot value and points-above-expected summaries
- Player, team, zone, and monthly evaluation reports
- Research and portfolio demonstration of an end-to-end sports ML workflow

## Out-of-scope use

- Automated personnel, contract, betting, medical, or disciplinary decisions
- Real-time production inference without the required prior-event sequence
- Causal claims about player skill or coaching impact
- Cross-season ranking without retraining and drift review
- Use as a substitute for scouting, tracking data, or contextual basketball judgment

## Data

Public NBA Stats data for 2024-25 supplies shot charts, player and team game logs, and play-by-play events. The exported feature data contains 175,708 training shots from 984 earlier games and 43,819 evaluation shots from 246 later games.

The exported training set is split again by game date:

- Inner train: 140,120 shots
- Inner validation: 35,588 shots
- Test/evaluation: 43,819 shots

The evaluation period is the latest approximately 20% of games, not the full season.

## Inputs

The model combines:

- **62 tabular features:** 31 base shot/game/rolling features, 18 spatial encodings, and 13 pressure or prior-event summary features
- **Sequence input:** the 5 play-by-play events immediately before the shot, represented by 29 numeric features each

Rolling features use shifted historical games. Sequence construction only accepts events with `action_number < game_event_id`; the shot event and future events are excluded.

## Architecture and preprocessing

- Median imputation and standard scaling for the tabular branch
- Separate standard scaling for flattened sequence features
- Tabular embedding: 62 inputs to 64 dimensions
- GRU: 29 event inputs to 64 hidden dimensions across 5 timesteps
- Prediction head: 64 then 32 hidden units to one logit
- BCE loss, Adam optimizer, early stopping on inner-validation log loss
- Default seed 42, batch size 1,024, maximum 50 epochs, patience 5

All preprocessing is fitted on inner-train data and serialized with the model bundle.

## Evaluation

| Metric | GRU v3 | LightGBM Candidate |
|--------|-------:|-------------------:|
| AUC | 0.6516 | 0.6479 |
| Log loss | 0.6468 | 0.6495 |
| Brier score | 0.2282 | 0.2292 |
| Accuracy | 0.6235 | 0.6213 |

GRU v3 improves all four reported metrics over LightGBM, although the gain is modest. Calibration is visualized but has not been promoted into a separate calibrated production model.

## Evaluation caveat

The inner validation split controls early stopping, but several model and feature iterations have been reviewed on the same 2024-25 test export. Consequently, these results should be treated as portfolio-stage comparative evidence rather than a permanently untouched estimate.

A frozen pipeline should be evaluated on a fresh later season or newly reserved holdout before promotion to Champion.

## Limitations

- Single-season training and evaluation
- No defender distance, player tracking, lineup, injury, or matchup features
- Missing score-margin values are explicitly flagged but still reflect source coverage limitations
- The prior-five-event window is only a compact representation of possession context
- Raw probabilities do not have a separate post-hoc calibration layer
- Real-time use requires reliable live play-by-play alignment
- No production API, latency benchmark, drift monitoring, or retraining trigger exists yet

## Fairness and responsible interpretation

The data represents NBA players who appeared in the source season; it does not support conclusions about other leagues or populations. Playing role, attempt mix, teammate quality, opponent quality, and sample size can materially affect player-level results. Points above expected should be paired with attempt volume, uncertainty, and scouting context rather than treated as a standalone player ranking.

## Artifacts and reproducibility

The MLflow run stores:

- `gru_state_dict.pt`
- `model_config.json`
- `feature_columns.json`
- `event_feature_columns.json`
- `tabular_preprocessor.joblib`
- `sequence_preprocessor.joblib`
- Training, calibration, and probability-distribution figures

The repository tracks code and selected reports, not model weights or MLflow storage. Reproduction requires the processed feature export, PostgreSQL play-by-play data for sequence construction, and the dependencies documented in `docs/training.md`.

## Promotion requirements

- Evaluate on a fresh holdout season
- Quantify calibration overall and by meaningful segments
- Define a stable request/feature contract for sequence inference
- Add API and end-to-end serving tests
- Measure latency and resource usage
- Implement drift and performance monitoring
- Register the GRU separately with explicit Challenger-to-Champion criteria
