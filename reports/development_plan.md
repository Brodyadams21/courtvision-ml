# CourtVision ML - Development Plan Status

Last reviewed: 2026-07-06

This is the repository-facing status snapshot for the 13-phase CourtVision ML development plan. It records what is implemented on `main`; it does not mark placeholder files as completed work.

## Current position

The project is **through Phase 8**, has **partial Phase 9 cloud infrastructure**, has a **basic Phase 10 local FastAPI inference service**, and has a **complete local Phase 11 Streamlit analytics dashboard**. **Phase 12 monitoring** remains not started. **Phase 13 documentation and portfolio polish** are in progress.

Phase 9 cloud execution is still **blocked/incomplete**: managed SageMaker training has not successfully run (quota/capacity limits), and reproducible pipeline/dry-run documentation plus cloud registry verification remain open. S3 bucket setup and processed feature upload are already complete.

| Phase | Goal | Status | Evidence / remaining work |
|------:|------|--------|---------------------------|
| 0 | Repository foundation | Complete | Package layout, configuration, environment template, dependencies, CI, and documentation exist |
| 1 | Data acquisition and raw storage | Complete | NBA shot chart, player/team logs, and play-by-play collectors with tracked collection metadata |
| 2 | SQL schema and cleaned tables | Complete | PostgreSQL schema, validated loaders, dimension/fact tables, and chunk-safe inserts |
| 3 | Validation | Complete | Pandera schemas, custom checks, failure policy, and validation code |
| 4 | Feature engineering | Complete | Time-aware features, rolling statistics, leakage-safe score margin, gold feature table, and Parquet export |
| 5 | Logistic-regression baseline | Complete | Reproducible baseline training, MLflow logging, figures, metrics, and report |
| 6 | LightGBM candidate and registry | Complete | Time-based search, leakage audit, feature importance, MLflow model logging, and Candidate alias workflow |
| 7 | PyTorch and spatial/sequence modeling | Complete, exceeds plan | Tabular/spatial MLPs plus GRU v3 with prior-event sequences, pressure features, inference bundle, tests, and report |
| 8 | Expected shot value and player evaluation | Complete | GRU scoring, ESV/actual/above-expected calculations, player/team/zone/trend summaries, and basketball insights |
| 9 | Cloud-assisted training | In progress / blocked | Config-driven local runner, Docker image, Linux lock, MLflow wiring, S3 bucket setup, and processed feature upload complete; managed SageMaker training, reproducible pipeline/dry-run documentation, and cloud registry verification remain |
| 10 | FastAPI inference service | Partial / local complete | `courtvision.api` has `/health`, `/predict/shot`, and `/predict/shots`, Pydantic schemas, `ShotModelService`, configurable lazy/startup model loading, request ID and prediction metadata logging, `Dockerfile.api`, and Docker Compose API service; production deployment docs remain |
| 11 | Streamlit dashboard | Complete local dashboard | Six-tab Streamlit dashboard, MLflow Candidate scoring, model diagnostics, Shot Edge Explorer, Edge Backtest, CSV exports, and `docs/dashboard.md` |
| 12 | Monitoring and retraining | Not started | Monitoring modules are empty; drift, segment calibration, reports, triggers, and dashboard integration remain |
| 13 | Final documentation and portfolio polish | In progress | README, `docs/dashboard.md`, `reports/dashboard_summary.md`, model reports, model card, architecture, training guide, and layout are present; demo media and final presentation/resume material remain |

## Out-of-order work completed

Phase 10 and Phase 11 were advanced locally while Phase 9 cloud execution remained blocked by SageMaker quota/capacity limits. This kept the project moving without claiming cloud production readiness. The FastAPI service and Streamlit dashboard both load the registered LightGBM **Candidate** from local MLflow; neither depends on a successful managed cloud training run.

## Phase 9 completion criteria

Phase 9 should not be marked complete until all of the following are true:

- Processed features or a representative training subset have been uploaded, or the upload path is documented and reproducible.
- `pipelines/sagemaker_pipeline.py` creates or submits a reproducible managed training workflow.
- At least one cloud training job succeeds, or a fully documented dry run proves the generated job specification without claiming execution.
- Training metrics and artifacts reach a cloud-accessible MLflow backend or an explicitly documented equivalent.
- The cloud path is covered by focused tests that mock external services rather than requiring AWS credentials in CI.

## Phase 10 completion criteria

**Local (done):**

- `/health`, `/predict/shot`, and `/predict/shots` endpoints with Pydantic request/response schemas
- Single-shot and batch prediction endpoints
- Configurable lazy/startup model loading behavior
- Request ID, latency, and prediction metadata logging without feature payloads
- `Dockerfile.api` and Docker Compose API service for local containerized serving
- `ShotModelService` wrapping the registered Candidate model
- Unit tests for health and prediction paths (`tests/test_api_main.py`)

**Hardening (open):**

- Production deployment documentation

## Phase 11 completion criteria

**Local (done):**

- Dashboard loads train/test Parquet splits
- Model Performance tab reads training artifacts (`training_summary.json`, feature importance, calibration figures)
- Prediction tabs use MLflow `courtvision-shot-make-model` @ **Candidate**
- Shot Edge Explorer and Edge Backtest support CSV exports
- Dashboard documentation exists (`docs/dashboard.md`, `reports/dashboard_summary.md`)

**Polish (open):**

- Player/team evaluation pages
- Saved screenshots and richer model comparison (LightGBM vs. GRU)
- Optional deployed demo

## Recommended order

1. Keep Phase 9 cloud work paused until quota/capacity allows a credible managed training run or documented dry run.
2. Harden Phase 10 FastAPI serving around the LightGBM Candidate.
3. Add Phase 12 monitoring and promotion gates.
4. Add dashboard polish: player/team pages, screenshots, model comparison.
5. Finish Phase 13 with demo media, resume bullets, and a clean reproduction walkthrough.

## Quality gates

- Keep all temporal splits game-based and ordered by date.
- Fit preprocessing only on training data.
- Preserve strict `action_number < game_event_id` sequence causality.
- Select hyperparameters on validation data, not the final evaluation split.
- Use a fresh later season or untouched final holdout before a Champion promotion claim.
- Keep secrets, raw data, processed data, MLflow stores, and trained weights out of git.
