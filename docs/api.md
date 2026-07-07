# CourtVision FastAPI Inference Service

The CourtVision inference API serves shot-make probabilities and expected shot value from the registered LightGBM **Candidate** model in MLflow.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/predict/shot` | Score a single shot |
| `POST` | `/predict/shots` | Score up to 500 shots in one request |

Interactive OpenAPI docs are available at `/docs` when the server is running.

## Local run

From the repository root:

```powershell
$env:PYTHONPATH = "src"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
uvicorn courtvision.api.main:app --reload
```

Start MLflow first if you need to load the registered model (see `scripts/start_mlflow.ps1`).

## Model loading modes

The API supports two model-loading behaviors, controlled by environment variables read at import time via `create_app_from_env()`.

### Lazy mode (default)

The app starts without calling MLflow. Prediction endpoints return **503** until a model is loaded elsewhere. This is the default and is useful for schema checks, OpenAPI docs, and lightweight local boot testing.

```powershell
$env:PYTHONPATH = "src"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
uvicorn courtvision.api.main:app --reload
```

### Startup mode (fail-fast)

Set `COURTVISION_API_LOAD_MODEL_ON_STARTUP=true` to load the registered MLflow model during app creation. If loading fails (MLflow unreachable, alias missing, etc.), the process raises `RuntimeError` and exits instead of serving **503** on every prediction.

```powershell
$env:PYTHONPATH = "src"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:COURTVISION_API_LOAD_MODEL_ON_STARTUP = "true"
$env:COURTVISION_API_MODEL_ALIAS = "Candidate"
uvicorn courtvision.api.main:app --reload
```

Optional: override the MLflow alias with `COURTVISION_API_MODEL_ALIAS` (default `Candidate`).

| Variable | Default | Description |
|----------|---------|-------------|
| `COURTVISION_API_LOAD_MODEL_ON_STARTUP` | `false` | Load the registered model when the app is created |
| `COURTVISION_API_MODEL_ALIAS` | `Candidate` | MLflow model alias to load in startup mode |

Tests inject a fake `ShotModelService` via `create_app(model_service=...)` and never call MLflow, regardless of startup flags.

## Example single-shot request

Send every column in `FEATURE_COLUMNS` (`src/courtvision/models/common.py`) in `features`. The top-level `shot_value` must be `2` or `3` (it is also included in the feature dict as `shot_value`).

```http
POST /predict/shot
Content-Type: application/json

{
  "features": {
    "shot_value": 3.0,
    "shot_distance": 24.0,
    "loc_x": 120.0,
    "loc_y": 250.0,
    "abs_loc_x": 120.0,
    "shot_angle": 0.15,
    "is_corner_three": 0.0,
    "is_home": 1.0,
    "period": 2.0,
    "seconds_remaining_period": 360.0,
    "seconds_remaining_game": 2160.0,
    "score_margin": 0.0,
    "score_margin_missing": 0.0,
    "player_recent_fg_pct_5": 0.45,
    "player_recent_fg3_pct_5": 0.36,
    "player_recent_fga_5": 12.0,
    "player_recent_fg3a_5": 5.0,
    "player_recent_minutes_5": 28.0,
    "player_recent_points_5": 18.0,
    "team_recent_off_eff_proxy_5": 1.05,
    "team_recent_pace_proxy_5": 98.0,
    "team_recent_fg_pct_5": 0.46,
    "team_recent_three_point_rate_5": 0.38,
    "team_recent_fga_5": 88.0,
    "team_recent_points_5": 112.0,
    "team_recent_turnovers_5": 14.0,
    "opp_recent_points_allowed_5": 108.0,
    "opp_recent_fg_pct_allowed_5": 0.44,
    "opp_recent_three_point_rate_allowed_5": 0.37,
    "opp_recent_pace_proxy_5": 97.0,
    "opp_recent_fga_allowed_5": 86.0
  },
  "shot_value": 3
}
```

Example response:

```json
{
  "predicted_make_probability": 0.38,
  "expected_shot_value": 1.14,
  "model_name": "courtvision-shot-make-model"
}
```

## Example batch request

```http
POST /predict/shots
Content-Type: application/json

{
  "shots": [
    {
      "features": { "...": 0.0 },
      "shot_value": 2
    },
    {
      "features": { "...": 0.0 },
      "shot_value": 3
    }
  ]
}
```

Each item in `shots` uses the same shape as the single-shot request. The API returns one prediction per input shot, in the same order. Batch size is limited to **500** shots per request.

Example response:

```json
{
  "predictions": [
    {
      "predicted_make_probability": 0.42,
      "expected_shot_value": 0.84,
      "model_name": "courtvision-shot-make-model"
    },
    {
      "predicted_make_probability": 0.38,
      "expected_shot_value": 1.14,
      "model_name": "courtvision-shot-make-model"
    }
  ]
}
```

## Model requirement

By default, `ShotModelService` must load the MLflow registered model `courtvision-shot-make-model` with alias **Candidate** before predictions succeed. If the model is not loaded, prediction endpoints return **503 Service Unavailable**.

Unit tests inject a fake `ShotModelService` via `create_app(model_service=...)` and do not require MLflow.

## Troubleshooting

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| `503` — model not loaded | No Candidate model in MLflow or lazy mode without manual load | Set `COURTVISION_API_LOAD_MODEL_ON_STARTUP=true` and confirm MLflow has the alias, or inject a loaded service in tests |
| `422` — missing feature columns | Request `features` omit required model columns | Send the full feature set defined in `FEATURE_COLUMNS` (see `src/courtvision/models/common.py`) |
| `422` — validation error on batch | Empty `shots` list or more than 500 items | Send between 1 and 500 shot requests |
| Connection refused on port 5000 | MLflow tracking server not running | Run `scripts/start_mlflow.ps1` or set `MLFLOW_TRACKING_URI` to your tracking backend |
