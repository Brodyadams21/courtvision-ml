# CourtVision ML

CourtVision ML is a cloud-assisted basketball machine learning platform for shot quality modeling, player evaluation, and basketball operations decision support.

The project is designed to mirror a professional Basketball Operations machine learning workflow, including data ingestion, SQL-backed storage, feature engineering, model training, experiment tracking, model registry, API inference, dashboard delivery, monitoring, testing, documentation, and cloud-assisted retraining.

## Project Goal

The goal is to build an end-to-end basketball analytics system that answers:

- Which players generate high-quality shots?
- Which players outperform or underperform expected shot value?
- How can shot-level, player-level, and game-context data support player evaluation, game strategy, and player development?
- How can machine learning outputs be delivered as reliable tools for basketball decision makers?

## Role Alignment

CourtVision ML is built to align with a Data Scientist, Machine Learning role in Basketball Operations.

| Role Expectation | CourtVision ML Response |
|---|---|
| Build and productionize machine learning models | Shot quality model, expected shot value model, player evaluation metrics, FastAPI inference service |
| Design scalable data pipelines | Raw data ingestion, cleaned SQL tables, feature generation, validation, cloud-ready storage |
| Support basketball decision-making | Expected shot value, points above expected, zone profiles, player trend reports |
| Use modern ML workflows | MLflow tracking, model registry, repeatable training scripts, model artifacts, CI tests |
| Demonstrate testing and observability | pytest, Ruff, validation checks, model cards, monitoring reports |
| Communicate insights clearly | Streamlit dashboard and basketball-facing reports for coaches, analysts, and decision-makers |

## Planned System Architecture

```mermaid
flowchart TD
    A[Public Basketball Data] --> B[Raw Data Layer]
    B --> C[Cleaning and Validation]
    C --> D[SQL Feature Tables]
    D --> E[Model Training]
    E --> F[MLflow Tracking and Registry]
    F --> G[FastAPI Inference Service]
    F --> H[Batch Predictions]
    G --> I[Streamlit Dashboard]
    H --> I
    I --> J[Basketball Operations Insights]
    H --> K[Monitoring Reports]
    K --> E
```

## Core ML Tasks

- Shot make probability prediction
- Expected shot value modeling
- Player shot-making above expectation
- Shot quality profile by player, zone, and game context
- Player development trend analysis
- Model monitoring and retraining readiness

## Planned Features

### Data Pipeline

- Collect public basketball shot, player, team, and game data
- Store immutable raw data files
- Clean and normalize shot-level and player-level data
- Load structured data into SQL tables
- Create model-ready feature tables
- Validate data quality before training

### Machine Learning

- Train a baseline logistic regression model
- Train a tree-based candidate model using XGBoost or LightGBM
- Add a PyTorch model for deep learning comparison
- Evaluate models using AUC, log loss, Brier score, calibration, and accuracy
- Track experiments, parameters, metrics, and artifacts with MLflow
- Register the best model for API and dashboard use

### Basketball Evaluation Layer

- Calculate expected shot value
- Calculate points above expected
- Compare actual shooting performance against model expectations
- Build player-level summaries
- Analyze shot quality by player, team, shot zone, and game context
- Generate basketball-facing insights for player development and strategy

### API and Dashboard

- Build a FastAPI inference service
- Add endpoints for health checks, model information, single-shot prediction, batch prediction, and player evaluation
- Build a Streamlit dashboard for basketball decision support
- Show model performance, player trends, shot quality, and expected value insights

### MLOps and Cloud-Assisted Workflow

- Use MLflow for experiment tracking and model versioning
- Prepare cloud-ready configuration files
- Structure the project so training can run locally or in a cloud environment
- Add monitoring reports for data drift, prediction drift, and calibration
- Add GitHub Actions for automated linting and testing
- Keep data, models, secrets, and artifacts out of GitHub

## Tech Stack

- Python 3.11
- SQL
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- PyTorch, planned
- Matplotlib
- Plotly
- FastAPI
- Streamlit
- MLflow
- pytest
- Ruff
- Pydantic
- SQLAlchemy
- PostgreSQL or SQLite
- nba-api
- pyarrow
- Docker (local PostgreSQL via Compose)
- GitHub Actions
- AWS-style cloud-assisted ML workflow, planned

## Repository Structure

```txt
courtvision-ml/
├── README.md
├── pyproject.toml
├── requirements.txt
├── docker-compose.yml
├── .gitignore
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── local.yaml
│   ├── aws.yaml
│   └── model_config.yaml
├── data/
│   ├── metadata/
│   │   └── data_collection_metadata.json
│   └── raw/
│       ├── shots/
│       ├── player_game_logs/
│       ├── team_game_logs/
│       └── play_by_play/
├── notebooks/
├── sql/
│   ├── schema.sql
│   ├── inspection_queries.sql
│   ├── feature_queries.sql
│   └── evaluation_queries.sql
├── src/
│   └── courtvision/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── collect.py
│       │   ├── clean.py
│       │   ├── validate.py
│       │   └── build_features.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   ├── predict.py
│       │   └── registry.py
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── drift.py
│       │   ├── calibration.py
│       │   └── performance_report.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── schemas.py
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           └── logging.py
├── dashboard/
│   ├── app.py
│   └── pages/
├── pipelines/
│   ├── run_local_pipeline.py
│   └── sagemaker_pipeline.py
├── tests/
│   └── test_smoke.py
└── reports/
    ├── model_card.md
    ├── basketball_insights.md
    └── cloud_architecture.md
```

## Local Setup

Create a Python 3.11 virtual environment:

```powershell
py -3.11 -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Set `PYTHONPATH` so the package can be imported from the project root:

```powershell
$env:PYTHONPATH = "src"
```

## Local PostgreSQL

Start a PostgreSQL 16 instance for local development:

```powershell
docker compose up -d postgres
```

| Setting | Value |
|---|---|
| Service | `postgres` |
| Image | `postgres:16` |
| Database | `courtvision_ml` |
| User | `courtvision_user` |
| Password | `courtvision_local_dev` |
| Host port | `5433` (maps to 5432 in the container; avoids conflict with a local PostgreSQL on 5432) |
| Data volume | `courtvision_pgdata` (Docker named volume; persists across restarts) |

Check that the database is ready:

```powershell
docker compose ps
```

Stop the database (data is kept in the volume):

```powershell
docker compose down
```

To remove the database volume as well:

```powershell
docker compose down -v
```

Copy `.env.example` to `.env` so `DATABASE_URL` points at this instance (see [Environment Variables](#environment-variables)).

### Apply schema (once)

Create tables the first time, or after you change `sql/schema.sql`:

```powershell
Get-Content sql/schema.sql | docker compose exec -T postgres psql -U courtvision_user -d courtvision_ml
```

### Load cleaned data (repeatable)

`load_data.py` clears existing basketball rows, then inserts cleaned Parquet data. You do **not** need to rerun `schema.sql` on every reload during development.

```powershell
$env:PYTHONPATH = "src"
python -m courtvision.data.load_data --season 2024-25
```

The loader reads `DATABASE_URL` from `.env` (no credentials in code). Use `--append` only if you intentionally want to add rows without clearing tables first.

### Inspect loaded data

After loading, run `sql/inspection_queries.sql` to check row counts, missing key columns, duplicate natural keys, and date coverage. Compare table counts to the loader log output.

```powershell
Get-Content sql/inspection_queries.sql | docker compose exec -T postgres psql -U courtvision_user -d courtvision_ml
```

## Data Collection

Raw NBA data is downloaded from the public [NBA Stats API](https://stats.nba.com) using [`nba-api`](https://github.com/swar/nba_api) via `src/courtvision/data/collect.py`.

Each dataset is saved locally as immutable CSV and Parquet files. Collection metadata is written to `data/metadata/data_collection_metadata.json` with the source endpoint, season, download timestamp, row count, and output paths.

Raw data files under `data/raw/` are gitignored and should be generated locally.

### Available datasets

| Dataset | CLI value | API endpoint | Notes |
|---|---|---|---|
| Shot charts | `shot_chart` | `shotchartdetail` | One request per team (~30 calls) |
| Player game logs | `player_game_logs` | `playergamelogs` | Single bulk request |
| Team game logs | `team_game_logs` | `teamgamelogs` | Single bulk request |
| Play-by-play | `play_by_play` | `playbyplayv3` | One request per game (~1,230 calls) |

Default season: `2024-25`

### Output layout

```txt
data/
├── metadata/
│   └── data_collection_metadata.json
└── raw/
    ├── shots/
    │   ├── 2024-25_shot_chart_raw.csv
    │   └── 2024-25_shot_chart_raw.parquet
    ├── player_game_logs/
    │   ├── 2024-25_player_game_logs_raw.csv
    │   └── 2024-25_player_game_logs_raw.parquet
    ├── team_game_logs/
    │   ├── 2024-25_team_game_logs_raw.csv
    │   └── 2024-25_team_game_logs_raw.parquet
    └── play_by_play/
        ├── 2024-25_play_by_play_raw.csv
        └── 2024-25_play_by_play_raw.parquet
```

### Run all datasets

From the project root with your virtual environment activated:

```powershell
$env:PYTHONPATH = "src"
python -m courtvision.data.collect
```

This downloads all four datasets for the default season and updates the metadata file after each dataset completes.

### Run one dataset

```powershell
$env:PYTHONPATH = "src"
python -m courtvision.data.collect --dataset shot_chart
python -m courtvision.data.collect --dataset player_game_logs
python -m courtvision.data.collect --dataset team_game_logs
python -m courtvision.data.collect --dataset play_by_play
```

### CLI options

```powershell
python -m courtvision.data.collect --help
```

| Flag | Default | Description |
|---|---|---|
| `--season` | `2024-25` | NBA season string |
| `--data-dir` | `data` | Root directory for raw files and metadata |
| `--dataset` | `all` | `all`, `shot_chart`, `player_game_logs`, `team_game_logs`, or `play_by_play` |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |

Example for a different season:

```powershell
python -m courtvision.data.collect --season 2023-24 --dataset all --log-level DEBUG
```

### Expected runtime

Collection time depends on NBA API response times and rate limiting built into the collector.

| Dataset | Approximate runtime |
|---|---|
| Player game logs | Under 1 minute |
| Team game logs | Under 1 minute |
| Shot charts | A few minutes |
| Play-by-play | 20–40 minutes |
| All datasets | 25–45 minutes |

Play-by-play is the slowest dataset because it requires one API call per regular-season game.

### Programmatic use

```python
from courtvision.data.collect import collect_and_save

entries = collect_and_save(season="2024-25", data_dir="data")
for entry in entries:
    print(entry["dataset"], entry["row_count"])
```

To collect a subset programmatically:

```python
from courtvision.data.collect import collect_and_save, PLAY_BY_PLAY_DATASET

collect_and_save(season="2024-25", datasets={PLAY_BY_PLAY_DATASET})
```

## Environment Variables

Create a local `.env` file based on `.env.example`.

Example:

```env
ENVIRONMENT=development
DATABASE_URL=postgresql+psycopg2://courtvision_user:courtvision_local_dev@localhost:5432/courtvision_ml
MLFLOW_TRACKING_URI=./mlruns
MODEL_REGISTRY_PATH=models/
AWS_REGION=us-east-1
S3_BUCKET=
```

For local work without Docker, you can use SQLite instead:

```env
DATABASE_URL=sqlite:///courtvision.db
```

Do not commit real secrets, keys, credentials, or cloud account information. The Compose password above is for local development only.

## Quality Checks

Run Ruff linting:

```powershell
ruff check .
```

Run tests:

```powershell
pytest
```

## GitHub Actions

This project uses GitHub Actions to run automated checks on every push and pull request to `main`.

The CI workflow will:

1. Check out the repository
2. Set up Python 3.11
3. Install dependencies
4. Run Ruff linting
5. Run pytest

## Development Phases

### Phase 0: Project Setup and Professional Repo Foundation

- Create GitHub repository
- Create Python 3.11 virtual environment
- Install core dependencies
- Create professional folder structure
- Add README
- Add `.gitignore`
- Add Ruff configuration
- Add GitHub Actions workflow
- Add smoke test

### Phase 1: Data Acquisition and Storage

- Choose public basketball data sources
- Collect shot, game, player, and team data
- Save raw data locally
- Add metadata files for source, season, download date, and row counts
- Convert reusable files to efficient formats such as Parquet

Phase 1 data collection is implemented for the 2024-25 season via `src/courtvision/data/collect.py`.

### Phase 2: SQL Schema and Cleaned Data Tables

- Create database schema
- Load cleaned data into SQL via `src/courtvision/data/load_data.py`
- Create tables for players, teams, games, shots, and game logs
- Add row count and data quality queries in `sql/inspection_queries.sql`

### Phase 3: Data Validation

- Validate required IDs, dates, shot values, and coordinate ranges
- Check missing values and duplicate rows
- Fail the pipeline if critical checks do not pass

### Phase 4: Feature Engineering

- Build shot geometry features
- Build player rolling features
- Build team and opponent context features
- Build game-state features
- Export model-ready feature tables

### Phase 5: Baseline Modeling

- Train logistic regression baseline
- Use a time-based train/test split
- Evaluate AUC, log loss, Brier score, calibration, and accuracy
- Log results to MLflow

### Phase 6: Tree-Based Production Candidate

- Train XGBoost or LightGBM model
- Compare against baseline
- Create feature importance report
- Register candidate model

### Phase 7: Deep Learning and Spatial Modeling

- Train PyTorch model
- Add spatial features from shot location
- Compare against tree-based model
- Document whether deep learning improves results

### Phase 8: Expected Shot Value and Player Evaluation

- Generate predictions for all evaluation shots
- Calculate expected shot value
- Calculate points above expected
- Build player evaluation summaries
- Write basketball-facing insights

### Phase 9: Cloud-Assisted Training Path

- Add AWS-style configuration
- Prepare training code to run locally or in cloud
- Store artifacts in a cloud-ready structure
- Log cloud or cloud-ready training runs to MLflow

### Phase 10: Inference API

- Build FastAPI app
- Add model info and prediction endpoints
- Validate inputs with Pydantic
- Add API tests

### Phase 11: Streamlit Dashboard

- Build Overview page
- Build Shot Quality Explorer
- Build Player Evaluation page
- Build Model Performance page
- Display basketball insights clearly

### Phase 12: Monitoring and Retraining

- Create drift report
- Track prediction distribution
- Track calibration by shot type and zone
- Define retraining triggers

### Phase 13: Final Documentation and Polish

- Complete model card
- Complete basketball insights report
- Complete cloud architecture report
- Add screenshots or demo video
- Finalize resume bullets and interview talking points

## Current Status

Phase 0 is complete. Phase 1 raw data collection is available for shot charts, player game logs, team game logs, and play-by-play via `python -m courtvision.data.collect`.

## Final Project Outcome

The finished project should demonstrate the ability to build a production-style basketball machine learning platform that moves beyond a simple notebook. The goal is to show end-to-end ownership across data pipelines, machine learning, model evaluation, model serving, dashboard delivery, monitoring, documentation, and basketball decision support.