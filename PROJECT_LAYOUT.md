# CourtVision ML - Project Layout

Last updated: 2026-06-18

This document describes the tracked repository. Local environments, caches, raw data, processed data, MLflow stores, and trained model artifacts are intentionally excluded by `.gitignore`.

## Repository summary

| Location | Tracked files | Purpose |
|----------|--------------:|---------|
| Root | 13 | Project metadata, containers, dependencies, and local services |
| `.github/` | 1 | GitHub Actions CI |
| `configs/` | 5 | Local, Docker, AWS, and model configuration |
| `dashboard/` | 1 | Planned Streamlit entry point |
| `data/` | 2 | Data documentation and collection metadata |
| `docs/` | 1 | Training and dependency guide |
| `pipelines/` | 2 | Local runner and planned SageMaker entry point |
| `reports/` | 20 | Model reports, figures, and selected result tables |
| `scripts/` | 1 | Local MLflow launcher |
| `sql/` | 6 | PostgreSQL schema and analysis queries |
| `src/` | 50 | Main Python package plus legacy placeholders |
| `tests/` | 19 | Unit, inference, leakage, and pipeline tests |
| **Total** | **121** | |

## Top-level layout

```text
courtvision-ml/
|-- .github/workflows/ci.yml
|-- configs/
|   |-- aws.yaml
|   |-- config.yaml
|   |-- docker.yaml
|   |-- local.yaml
|   `-- model_config.yaml
|-- dashboard/app.py
|-- data/
|   |-- README.md
|   `-- metadata/data_collection_metadata.json
|-- docs/training.md
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
|   `-- tables/
|-- scripts/start_mlflow.ps1
|-- sql/
|-- src/courtvision/
|-- tests/
|-- .dockerignore
|-- .env.example
|-- .gitignore
|-- Dockerfile.lock
|-- Dockerfile.train
|-- LICENSE
|-- PROJECT_LAYOUT.md
|-- README.md
|-- docker-compose.yml
|-- pyproject.toml
|-- requirements.in
|-- requirements.txt
`-- requirements-linux.txt
```

## Active Python package

The active import root is `src/courtvision/`.

| Package | Status | Responsibility |
|---------|--------|----------------|
| `courtvision.data` | Implemented | NBA API collection, schemas, validation, PostgreSQL loading, feature building, and Parquet export |
| `courtvision.models` | Implemented | Baseline, LightGBM, MLP, GRU, feature transforms, leakage audit, registry, and shared evaluation helpers |
| `courtvision.evaluation` | Implemented | Candidate/GRU scoring, expected shot value, aggregate summaries, and insight generation |
| `courtvision.utils.config` | Implemented | YAML loading and project configuration resolution |
| `courtvision.api` | Placeholder | Planned FastAPI app and Pydantic schemas |
| `courtvision.monitoring` | Placeholder | Planned calibration, drift, and performance reporting |

The zero-byte modules under `src/api/`, `src/data/`, `src/features/`, and `src/models/` are legacy scaffolding. They are not used by the active package and can be removed in a dedicated cleanup after confirming no external tooling depends on those paths.

## Training and pipelines

| Path | Status | Notes |
|------|--------|-------|
| `src/courtvision/models/train.py` | Implemented | Config-driven LightGBM entry point |
| `pipelines/run_local_pipeline.py` | Implemented | Plans or executes feature export and local training commands |
| `Dockerfile.train` | Implemented | Python 3.11 Linux training image |
| `Dockerfile.lock` | Implemented | Rebuilds the Linux dependency lock |
| `configs/local.yaml` | Implemented | Local paths and MLflow URI |
| `configs/docker.yaml` | Implemented | Docker-to-host MLflow URI |
| `configs/aws.yaml` | Scaffolding | Target AWS/S3 settings only |
| `pipelines/sagemaker_pipeline.py` | Placeholder | No SageMaker job submission yet |

## Reports and outputs

Tracked reports document the baseline, LightGBM Candidate, GRU v3, expected-shot-value analysis, model card, cloud plan, and phase progress.

Tracked tables include the published GRU shot predictions and player, team, zone, and monthly summaries. Raw data, processed feature Parquet files, MLflow runs, and model bundles are deliberately ignored and must be regenerated or stored externally.

The generic `reports/tables/shot_predictions_2024-25.csv` is an earlier LightGBM export. Current code generates model-qualified names such as `shot_predictions_2024-25_candidate.csv` and `shot_predictions_2024-25_gru.csv`.

## Tests and CI

The test suite covers configuration, feature engineering, leakage controls, model registry behavior, LightGBM selection, sequence construction, GRU artifact loading/inference, expected value, summary generation, and local pipeline orchestration.

`.github/workflows/ci.yml` runs Ruff and pytest on Python 3.11 using `requirements-linux.txt` for pushes and pull requests targeting `main`.

## Intentionally untracked local paths

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

## Verification

```powershell
ruff check .
pytest
git status --short
```
