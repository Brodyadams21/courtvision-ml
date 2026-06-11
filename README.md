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
| Use modern ML workflows | MLflow tracking, model registry (`Candidate` alias), leakage audits, repeatable training scripts, CI tests |
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
- Create model-ready feature tables (`gold_shot_features`, train/test Parquet export)
- Validate data quality before training (Phase 3 cleaned tables; Phase 4 gold features)

### Machine Learning

- Train a baseline logistic regression model (Phase 5 — complete)
- Train a tree-based candidate model using LightGBM (Phase 6 — complete)
- Run single-feature leakage audits before trusting game-context features
- Train PyTorch MLP and GRU models with spatial and sequence features (Phase 7 — complete)
- Evaluate models using AUC, log loss, Brier score, calibration, and accuracy
- Track experiments, parameters, metrics, and artifacts with MLflow
- Register the best model in the MLflow model registry (`courtvision-shot-make-model`, `Candidate` alias)

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
- PyTorch
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
- Pandera
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
│   ├── processed/
│   │   └── features/          # train/test Parquet (gitignored, generated locally)
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
│   ├── feature_inspection_queries.sql
│   └── evaluation_queries.sql
├── src/
│   └── courtvision/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── collect.py
│       │   ├── load_data.py
│       │   ├── schemas.py
│       │   ├── validate.py
│       │   └── build_features.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── common.py              # shared features, metrics, plots
│       │   ├── train_baseline.py      # Phase 5 logistic regression
│       │   ├── train_lgbm.py          # Phase 6 LightGBM search + registry
│       │   ├── audit_feature_leakage.py
│       │   ├── registry.py            # MLflow Candidate alias helpers
│       │   ├── train.py               # planned unified entrypoint
│       │   ├── evaluate.py
│       │   └── predict.py
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
├── scripts/
│   └── start_mlflow.ps1
├── pipelines/
│   ├── run_local_pipeline.py
│   └── sagemaker_pipeline.py
├── tests/
│   ├── test_smoke.py
│   ├── test_score_margin.py
│   ├── test_audit_feature_leakage.py
│   ├── test_train_lgbm.py
│   └── test_registry.py
└── reports/
    ├── baseline_model_report.md
    ├── lightgbm_candidate_report.md
    ├── figures/                       # calibration, importance plots
    ├── tables/                        # feature importance CSVs
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

### Validate before load

`load_data.py` runs `validate_all_cleaned_datasets()` before any PostgreSQL insert:

1. **Pandera** (`src/courtvision/data/schemas.py`) — column types, nulls, ranges, allowed values
2. **Custom checks** (`src/courtvision/data/validate.py`) — season row counts, duplicate keys, FK sanity, play-by-play warnings

Critical issues stop the pipeline; warnings (mostly play-by-play gaps) are logged only.

### Inspect loaded data

After loading, run `sql/inspection_queries.sql` to check row counts, missing key columns, duplicate natural keys, and date coverage. Compare table counts to the loader log output.

```powershell
Get-Content sql/inspection_queries.sql | docker compose exec -T postgres psql -U courtvision_user -d courtvision_ml
```

## Feature Engineering (Phase 4)

Model-ready shot features are built from PostgreSQL with `src/courtvision/data/build_features.py`. Output lands in the **`gold_shot_features`** table, then optional train/test Parquet exports under `data/processed/features/` (gitignored).

**Prerequisites:** Docker PostgreSQL is running, `sql/schema.sql` has been applied, and cleaned data is loaded via `load_data.py`.

### Pipeline overview

1. **Base shot features** — join `shots`, `games`, and `teams` (geometry, zones, clock, `is_home`, labels)
2. **Player rolling** — previous 5 games from `player_game_logs` (shift-then-roll; excludes current game)
3. **Team rolling** — previous 5 games from `team_game_logs`
4. **Opponent rolling** — previous 5 games allowed (defensive view of `team_game_logs`)
5. **Score margin** — optional join to `play_by_play` on `game_event_id` = `action_number`, using **prior-event** PBP score snapshots only (never `shot_made_flag`); `score_margin_missing` flags unmatched shots (non-blocking)
6. **Phase 4 validation** — Pandera + custom checks via `validate_all_gold_shot_features()` (runs before gold insert)
7. **Load** — insert into `gold_shot_features` (`feature_set_version` default: `base_v1`)
8. **Export** — time-based 80/20 train/test split by game date

Rolling features use **shift then roll** within player/team so the current game is never included. Null rolling values early in the season are expected (no prior games).

### Full build, load, inspect, and export

```powershell
$env:PYTHONPATH = "src"
python -m courtvision.data.build_features --season 2024-25 --load --inspect --export
```

### Common commands

```powershell
# Build features in memory only (no PostgreSQL write)
python -m courtvision.data.build_features --season 2024-25

# Load gold + run post-load SQL checks
python -m courtvision.data.build_features --season 2024-25 --load --inspect

# Export train/test Parquet from existing gold (no rebuild)
python -m courtvision.data.build_features --season 2024-25 --export-only

# Skip optional score margin join
python -m courtvision.data.build_features --season 2024-25 --load --skip-score-margin
```

### CLI options

```powershell
python -m courtvision.data.build_features --help
```

| Flag | Default | Description |
|---|---|---|
| `--season` | `2024-25` | Season label |
| `--load` | off | Insert into `gold_shot_features` |
| `--inspect` | off | Post-load SQL checks (implies `--load`) |
| `--export` | off | Write train/test Parquet after gold is ready |
| `--export-only` | off | Export from existing gold only |
| `--feature-set-version` | `base_v1` | Version label stored in gold |
| `--train-fraction` | `0.8` | Earliest fraction of games for train split |
| `--processed-dir` | `data/processed/features` | Train/test Parquet output directory |
| `--skip-score-margin` | off | Leave `score_margin` null |
| `--append` | off | Append gold rows without clearing season/version |
| `--output` | — | Optional single Parquet path for full feature frame |

### Export layout

```txt
data/processed/features/
├── train_shot_features_2024-25.parquet   # earliest ~80% of games by date
└── test_shot_features_2024-25.parquet    # latest ~20% of games by date
```

Example 2024-25 split: **984** train games / **175,708** shots; **246** test games / **43,819** shots. No game appears in both files.

### Gold table feature groups

| Group | Examples |
|---|---|
| Keys & labels | `shot_id`, `game_id`, `player_id`, `team_id`, `opponent_team_id`, `shot_made_flag`, `shot_value` (2 or 3) |
| Geometry & zones | `shot_distance`, `loc_x`, `loc_y`, `abs_loc_x`, `shot_angle`, `is_corner_three`, zone columns |
| Game state | `period`, `seconds_remaining_period`, `seconds_remaining_game`, `score_margin`, `score_margin_missing` |
| Player rolling | `player_recent_fg_pct_5`, `player_recent_fga_5`, … |
| Team rolling | `team_recent_fg_pct_5`, `team_recent_pace_proxy_5`, … |
| Opponent rolling | `opp_recent_fg_pct_allowed_5`, `opp_recent_points_allowed_5`, … |

Schema: `sql/schema.sql` (`gold_shot_features`). Pandera schema: `GOLD_TABLE_SCHEMAS` in `schemas.py`.

### Phase 4 validation

Before gold insert, `build_features.py` calls `validate_all_gold_shot_features()` in `validate.py`. Critical checks:

- `shot_id` non-null and unique
- `shot_made_flag`, `game_date`, `team_id`, `player_id`, `opponent_team_id`, `is_home` non-null
- `shot_value` in `{2, 3}`

Rolling nulls and missing `score_margin` do **not** fail validation.

### Inspect features in SQL

```powershell
Get-Content sql/feature_queries.sql | docker compose exec -T postgres psql -U courtvision_user -d courtvision_ml
Get-Content sql/feature_inspection_queries.sql | docker compose exec -T postgres psql -U courtvision_user -d courtvision_ml
```

`feature_inspection_queries.sql` covers row counts, duplicate `shot_id`, target nulls, `shot_value` domain, date range, train/test split sanity, and rolling null counts (including by month).

## Modeling (Phases 5–6)

Shot-make models train on exported Parquet in `data/processed/features/`. Default season: **2024-25**. Shared feature list and metrics live in `src/courtvision/models/common.py` (**31** modeling columns, including `score_margin_missing`).

### Start MLflow

Experiment tracking uses a local MLflow server backed by PostgreSQL (`courtvision_mlflow` database):

```powershell
.\scripts\start_mlflow.ps1
```

Open http://127.0.0.1:5000. Set `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` in `.env` (see `.env.example`).

### Phase 5 — Baseline logistic regression

Pipeline: median imputation → standard scaling → logistic regression.

```powershell
$env:PYTHONPATH = "src"
python -m courtvision.models.train_baseline
```

**Experiment:** `courtvision-baseline`  
**Report:** `reports/baseline_model_report.md` (original 30-feature Phase 5 run)  
**Test metrics (2024-25, 31 features):** AUC 0.6397, log loss 0.6610, Brier 0.2343, accuracy 0.6062

These are from a baseline rerun on the current Parquet export (`score_margin` + `score_margin_missing`). The Phase 5 report still documents the earlier 30-feature run (AUC 0.6409).

### Phase 6 — LightGBM production candidate

LightGBM trains on raw features (no imputation or scaling). Hyperparameter search uses an inner time-based validation split (80% of train games); the best config is retrained on full train and evaluated on held-out test.

```powershell
$env:PYTHONPATH = "src"
python -m courtvision.models.train_lgbm --mode search --register-candidate
```

| Flag | Description |
|---|---|
| `--mode default` | Single train with default hyperparameters |
| `--mode search` | 10-config search; logs best run as `lightgbm-best-{season}` |
| `--register-candidate` | Search only: register `courtvision-shot-make-model` and set `Candidate` alias |
| `--no-mlflow` | Skip MLflow logging |

**Experiment:** `courtvision-lightgbm`  
**Registered model:** `courtvision-shot-make-model` (alias `Candidate`)  
**Report:** `reports/lightgbm_candidate_report.md`  
**Test metrics (2024-25):** AUC 0.6479, log loss 0.6495, Brier 0.2292, accuracy 0.6213

**vs. 31-feature baseline:** AUC +0.0082, log loss −0.0115, Brier −0.0051, accuracy +0.0151

**Artifacts (best run):** calibration curve, probability distribution, gain/split feature importance, `feature_columns.json`, sklearn model

**Report figures/tables:**

- `reports/figures/lightgbm_feature_importance_gain.png`
- `reports/tables/lightgbm_feature_importance_gain.csv`

### Leakage audit

An early LightGBM run produced implausibly perfect metrics. The root cause was `score_margin` derived from `shot_made_flag`. Features were rebuilt with prior-PBP logic and covered by `tests/test_score_margin.py`.

Run a single-feature audit before trusting new columns:

```powershell
python -m courtvision.models.audit_feature_leakage
```

Flags any feature with test AUC ≥ 0.90. After the fix, no feature exceeded the threshold on 2024-25.

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
DATABASE_URL=postgresql+psycopg2://courtvision_user:courtvision_local_dev@127.0.0.1:5433/courtvision_ml
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://courtvision_user:courtvision_local_dev@127.0.0.1:5433/courtvision_mlflow
MLFLOW_ARTIFACT_ROOT=./mlartifacts
MODEL_REGISTRY_PATH=model_artifacts/
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

- Create database schema (`sql/schema.sql`) including `gold_shot_features` and future ML tables
- Load cleaned data into SQL via `src/courtvision/data/load_data.py`
- Create tables for players, teams, games, shots, game logs, and play-by-play
- Add row count and data quality queries in `sql/inspection_queries.sql`

Phase 2 load is implemented for the 2024-25 season.

### Phase 3: Data Validation

- Pandera schemas in `src/courtvision/data/schemas.py` plus custom checks in `validate.py`
- Critical: schema violations, duplicates, row-count floors, date ranges, FK sanity
- Warning: optional play-by-play and shot metadata gaps
- Fail the pipeline if critical checks do not pass

Phase 3 validation runs automatically in `load_data.py` before PostgreSQL insert.

### Phase 4: Feature Engineering

- Build base shot features from `shots`, `games`, and `teams`
- Build player rolling features (previous 5 games, shifted) from `player_game_logs`
- Build team rolling features from `team_game_logs`
- Build opponent rolling features (points/FGA allowed) from `team_game_logs`
- Optional game-state `score_margin` from prior PBP snapshots + `score_margin_missing` (non-blocking)
- Load `gold_shot_features` via `src/courtvision/data/build_features.py`
- Phase 4 validation (`validate_all_gold_shot_features`) before gold insert
- Export time-based train/test Parquet files (`data/processed/features/`)
- SQL inspection in `sql/feature_queries.sql` and `sql/feature_inspection_queries.sql`

Phase 4 is implemented for the 2024-25 season. See [Feature Engineering (Phase 4)](#feature-engineering-phase-4).

### Phase 5: Baseline Modeling

- Train logistic regression baseline (`train_baseline.py`)
- Use time-based train/test split (exported Parquet in `data/processed/features/`)
- Evaluate AUC, log loss, Brier score, calibration, and accuracy
- Log results to MLflow (`courtvision-baseline`)
- Write `reports/baseline_model_report.md`

Phase 5 is implemented for the 2024-25 season.

### Phase 6: Tree-Based Production Candidate

- Train LightGBM with inner validation hyperparameter search (`train_lgbm.py`)
- Compare against 31-feature baseline rerun on the same test split
- Run single-feature leakage audit (`audit_feature_leakage.py`)
- Fix `score_margin` prior-PBP logic; add `score_margin_missing`
- Create feature importance report (gain primary, split secondary)
- Register candidate model (`courtvision-shot-make-model`, `Candidate` alias via `registry.py`)
- Write `reports/lightgbm_candidate_report.md`

Phase 6 is implemented for the 2024-25 season.

### Phase 7: Deep Learning and Spatial Modeling

**Complete.** PyTorch MLP and GRU models were trained and compared against the LightGBM Candidate. The GRU spatial+sequence model achieved AUC 0.6516, log loss 0.6474, Brier 0.2283, and accuracy 0.6229, earning Challenger status while LightGBM remains the registered Candidate.

- Train MLP tabular and spatial models (`train_mlp.py`)
- Add spatial features from shot location (`spatial_features.py`)
- Build prior play-by-play sequences (`sequence_features.py`)
- Train tabular + sequence GRU (`train_gru.py`)
- Compare against tree-based model; write `reports/deep_learning_report.md`

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

| Phase | Status |
|---|---|
| Phase 0 — Repo foundation | Complete |
| Phase 1 — Raw data collection | Complete (`collect.py`, 2024-25) |
| Phase 2 — SQL schema & load | Complete (`load_data.py`, `schema.sql`) |
| Phase 3 — Cleaned data validation | Complete (`schemas.py`, `validate.py`) |
| Phase 4 — Feature engineering | Complete (`build_features.py`, `gold_shot_features`, train/test export) |
| Phase 5 — Baseline modeling | Complete (`train_baseline.py`, MLflow, baseline report) |
| Phase 6 — LightGBM candidate | Complete (`train_lgbm.py`, leakage audit, registry, candidate report) |
| Phase 7 — Deep learning & spatial modeling | Complete (`train_mlp.py`, `train_gru.py`, deep learning report) |
| Phase 8+ — EV, serving, monitoring | Planned |

**Typical local workflow (2024-25):**

```powershell
docker compose up -d postgres
Get-Content sql/schema.sql | docker compose exec -T postgres psql -U courtvision_user -d courtvision_ml
$env:PYTHONPATH = "src"
python -m courtvision.data.collect
python -m courtvision.data.load_data --season 2024-25
python -m courtvision.data.build_features --season 2024-25 --load --inspect --export
.\scripts\start_mlflow.ps1
python -m courtvision.models.train_baseline
python -m courtvision.models.audit_feature_leakage
python -m courtvision.models.train_lgbm --mode search --register-candidate
```

**Reports:**

| Report | Path |
|---|---|
| Baseline model (Phase 5) | `reports/baseline_model_report.md` |
| LightGBM candidate (Phase 6) | `reports/lightgbm_candidate_report.md` |
| Deep learning (Phase 7) | `reports/deep_learning_report.md` |

## Final Project Outcome

The finished project should demonstrate the ability to build a production-style basketball machine learning platform that moves beyond a simple notebook. The goal is to show end-to-end ownership across data pipelines, machine learning, model evaluation, model serving, dashboard delivery, monitoring, documentation, and basketball decision support.