# CourtVision ML - Project Layout

Last updated: 2026-06-16

This document describes the repository-facing project layout. It intentionally
excludes local/private/generated folders such as `.env`, `.venv/`,
`.pytest_cache/`, `.ruff_cache/`, `__pycache__/`, `data/raw/`,
`data/processed/`, `mlruns/`, `mlartifacts/`, and `model_artifacts/`.

Tracked files at the time of this update: 110. This layout document is separate
until it is added to git.

---

## Top-Level Structure

| Path | Purpose |
|------|---------|
| `.github/workflows/` | CI workflow for linting and tests. |
| `configs/` | Local, AWS, and model configuration files. |
| `dashboard/` | Streamlit dashboard entry point. |
| `data/` | Tracked data documentation and metadata only. Raw/processed data is ignored. |
| `pipelines/` | Local and SageMaker-oriented pipeline scripts. |
| `reports/` | Project reports, figures, and selected publishable result tables. |
| `scripts/` | Helper scripts for local services. |
| `sql/` | Database schema and inspection/evaluation queries. |
| `src/courtvision/` | Main application and ML package. |
| `tests/` | Unit and smoke tests. |

---

## File Count Summary

| Location | Tracked Files |
|----------|--------------:|
| `.env.example` | 1 |
| `.github` | 1 |
| `.gitignore` | 1 |
| `configs` | 4 |
| `dashboard` | 1 |
| `data` | 2 |
| `docker-compose.yml` | 1 |
| `LICENSE` | 1 |
| `pipelines` | 2 |
| `pyproject.toml` | 1 |
| `README.md` | 1 |
| `reports` | 20 |
| `requirements.txt` | 1 |
| `scripts` | 1 |
| `sql` | 6 |
| `src` | 50 |
| `tests` | 16 |
| **Total tracked** | **110** |

---

## Directory Tree

```text
courtvision-ml/
|-- .env.example
|-- .gitignore
|-- LICENSE
|-- PROJECT_LAYOUT.md
|-- README.md
|-- docker-compose.yml
|-- pyproject.toml
|-- requirements.txt
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- configs/
|   |-- aws.yaml
|   |-- config.yaml
|   |-- local.yaml
|   `-- model_config.yaml
|-- dashboard/
|   `-- app.py
|-- data/
|   |-- README.md
|   `-- metadata/
|       `-- data_collection_metadata.json
|-- pipelines/
|   |-- run_local_pipeline.py
|   `-- sagemaker_pipeline.py
|-- reports/
|   |-- baseline_model_report.md
|   |-- basketball_insights.md
|   |-- cloud_architecture.md
|   |-- deep_learning_report.md
|   |-- development_plan.md
|   |-- lightgbm_candidate_report.md
|   |-- model_card.md
|   |-- figures/
|   |   |-- baseline_calibration_curve.png
|   |   |-- baseline_probability_distribution.png
|   |   |-- gru_calibration_curve.png
|   |   |-- gru_probability_distribution.png
|   |   |-- gru_training_curve.png
|   |   `-- lightgbm_feature_importance_gain.png
|   `-- tables/
|       |-- lightgbm_feature_importance_gain.csv
|       |-- player_evaluation_2024-25.csv
|       |-- player_trends_2024-25.csv
|       |-- shot_predictions_2024-25.csv
|       |-- shot_predictions_2024-25_gru.csv
|       |-- team_evaluation_2024-25.csv
|       `-- zone_evaluation_2024-25.csv
|-- scripts/
|   `-- start_mlflow.ps1
|-- sql/
|   |-- create_mlflow_database.sql
|   |-- evaluation_queries.sql
|   |-- feature_inspection_queries.sql
|   |-- feature_queries.sql
|   |-- inspection_queries.sql
|   `-- schema.sql
|-- src/
|   |-- __init__.py
|   |-- api/
|   |   `-- main.py
|   |-- courtvision/
|   |   |-- __init__.py
|   |   |-- api/
|   |   |   |-- __init__.py
|   |   |   |-- main.py
|   |   |   `-- schemas.py
|   |   |-- data/
|   |   |   |-- __init__.py
|   |   |   |-- build_features.py
|   |   |   |-- clean.py
|   |   |   |-- collect.py
|   |   |   |-- load_data.py
|   |   |   |-- schemas.py
|   |   |   `-- validate.py
|   |   |-- evaluation/
|   |   |   |-- __init__.py
|   |   |   |-- expected_value.py
|   |   |   |-- predict_gru.py
|   |   |   |-- predict_shots.py
|   |   |   |-- summaries.py
|   |   |   `-- write_insights.py
|   |   |-- models/
|   |   |   |-- __init__.py
|   |   |   |-- audit_feature_leakage.py
|   |   |   |-- common.py
|   |   |   |-- evaluate.py
|   |   |   |-- predict.py
|   |   |   |-- pressure_features.py
|   |   |   |-- registry.py
|   |   |   |-- sequence_features.py
|   |   |   |-- spatial_features.py
|   |   |   |-- torch_data.py
|   |   |   |-- torch_models.py
|   |   |   |-- torch_sequence_data.py
|   |   |   |-- train.py
|   |   |   |-- train_baseline.py
|   |   |   |-- train_gru.py
|   |   |   |-- train_lgbm.py
|   |   |   `-- train_mlp.py
|   |   |-- monitoring/
|   |   |   |-- __init__.py
|   |   |   |-- calibration.py
|   |   |   |-- drift.py
|   |   |   `-- performance_report.py
|   |   `-- utils/
|   |       |-- __init__.py
|   |       |-- config.py
|   |       `-- logging.py
|   |-- data/
|   |   |-- clean.py
|   |   |-- collect.py
|   |   `-- validate.py
|   |-- features/
|   |   `-- build_features.py
|   `-- models/
|       |-- evaluate.py
|       |-- predict.py
|       `-- train.py
`-- tests/
    |-- test_audit_feature_leakage.py
    |-- test_evaluation_summaries.py
    |-- test_expected_value.py
    |-- test_features.py
    |-- test_insert_chunksize.py
    |-- test_predict_gru.py
    |-- test_predict_shots.py
    |-- test_pressure_features.py
    |-- test_registry.py
    |-- test_score_margin.py
    |-- test_sequence_features.py
    |-- test_smoke.py
    |-- test_spatial_features.py
    |-- test_torch_sequence_data.py
    |-- test_train_lgbm.py
    `-- test_write_insights.py
```

---

## Main Package Layout

The active package is `src/courtvision/`.

| Package Area | Responsibility |
|--------------|----------------|
| `courtvision.api` | FastAPI app and request/response schemas. |
| `courtvision.data` | Collection, cleaning, validation, feature building, and loading. |
| `courtvision.evaluation` | Expected-value scoring, prediction exports, summaries, and insight writing. |
| `courtvision.models` | Baseline, LightGBM, GRU/torch model training, prediction, evaluation, and registry helpers. |
| `courtvision.monitoring` | Calibration, drift, and performance-report utilities. |
| `courtvision.utils` | Shared config and logging helpers. |

Legacy placeholder paths also exist under `src/api/`, `src/data/`,
`src/features/`, and `src/models/`. The real implementation lives under
`src/courtvision/`.

---

## Tracked Reports And Outputs

The repository tracks selected outputs that support the README and phase reports:

| Path | Notes |
|------|-------|
| `reports/baseline_model_report.md` | Baseline model write-up. |
| `reports/lightgbm_candidate_report.md` | LightGBM candidate model write-up. |
| `reports/deep_learning_report.md` | GRU/deep-learning model write-up. |
| `reports/basketball_insights.md` | Player/team/zone basketball analysis. |
| `reports/model_card.md` | Model card. |
| `reports/cloud_architecture.md` | Cloud architecture notes. |
| `reports/development_plan.md` | Development plan snapshot. |
| `reports/figures/*.png` | Calibration, probability, training, and feature-importance figures. |
| `reports/tables/*.csv` | Publishable report tables and prediction exports. |

Large raw datasets, processed feature tables, MLflow runs, and trained model
artifacts are intentionally ignored and should be regenerated or stored outside
git.

---

## Local Files Intentionally Excluded

These paths may exist on a developer machine but should not be listed as tracked
project files:

```text
.env
.venv/
.pytest_cache/
.ruff_cache/
__pycache__/
data/raw/
data/processed/
data/interim/
data/external/
mlruns/
mlartifacts/
model_artifacts/
logs/
*.db
*.sqlite
*.sqlite3
```

---

## Verification Commands

```powershell
ruff check .
pytest
```
