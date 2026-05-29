# CourtVision ML

CourtVision ML is a cloud-assisted basketball machine learning platform designed to evaluate shot quality, player performance, and decision-making insights for basketball operations.

The project is built to mirror a professional machine learning workflow, including data ingestion, SQL-backed storage, feature engineering, model training, model evaluation, API-based inference, dashboard delivery, experiment tracking, model versioning, monitoring, and cloud-assisted training.

## Project Goal

The goal is to build an end-to-end basketball analytics system that answers:

- Which players generate high-quality shots?
- Which players outperform or underperform expected shot value?
- How can shot-level and player-level data support player evaluation, game strategy, and player development?
- How can machine learning outputs be delivered as reliable tools for basketball decision makers?

## Planned System Architecture

1. Data ingestion from public basketball datasets
2. SQL database for cleaned shot, player, team, and game data
3. Feature engineering pipeline
4. Shot quality machine learning model
5. Player evaluation metrics
6. Cloud-assisted training and experiment tracking
7. Model registry and versioning
8. FastAPI inference service
9. Streamlit dashboard
10. Monitoring and retraining workflow

## Core ML Tasks

- Shot make probability prediction
- Expected shot value modeling
- Player shot-making above expectation
- Shot quality profile by player, zone, and game context
- Player development trend analysis

## Tech Stack

- Python
- SQL
- scikit-learn
- XGBoost or LightGBM
- PyTorch
- FastAPI
- Streamlit
- MLflow
- PostgreSQL or SQLite
- Docker
- GitHub Actions
- Cloud-assisted ML training through AWS, Google Cloud, or Colab/Kaggle notebooks

## Current Status

Initial repository setup in progress.