# CourtVision ML - Development Plan Status

Last reviewed: 2026-07-07

This is the repository-facing status snapshot for the 13-phase CourtVision ML development plan. It records what is implemented on `main`; it does not mark placeholder files as completed work.

## Current position

The project is **through Phase 8**, has **partial Phase 9 cloud infrastructure**, has **complete Phase 10 local/containerized FastAPI inference service**, and has a **complete local Phase 11 Streamlit analytics dashboard**. **Phase 12 monitoring** remains not started. **Phase 13 documentation and portfolio polish** are in progress.

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
| 10 | FastAPI inference service | Complete local/containerized API serving; managed production deployment not performed | `courtvision.api` endpoints, `ShotModelService`, lazy/startup loading, request logging, `Dockerfile.api`, Docker Compose, `docs/api.md`, `docs/api_deployment.md`, and tests |
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

**Local / containerized (done):**

- `/health`, `/predict/shot`, and `/predict/shots` endpoints with Pydantic request/response schemas
- Single-shot and batch prediction endpoints
- Configurable lazy/startup model loading behavior
- Request ID, latency, and prediction metadata logging without feature payloads
- `Dockerfile.api` and Docker Compose API service for local containerized serving
- Production-oriented deployment guide documenting container runtime, model loading, artifact access, logging, security, and readiness gaps (`docs/api_deployment.md`)
- `ShotModelService` wrapping the registered Candidate model
- Unit tests for health, prediction, settings, Docker artifacts, and API docs

**Managed production (not done):**

- No deployment to ECS, Cloud Run, Kubernetes, or similar has been executed
- Remote MLflow artifact access from a live container not verified in cloud
- Platform secrets, HTTPS, auth, and rate limiting not implemented

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
2. Add Phase 12 monitoring and promotion gates.
3. Add dashboard polish: player/team pages, screenshots, model comparison.
4. When cloud path unblocks, deploy API to managed hosting with S3-backed MLflow artifacts (see `docs/api_deployment.md`).
5. Finish Phase 13 with demo media, resume bullets, and a clean reproduction walkthrough.

## Quality gates

- Keep all temporal splits game-based and ordered by date.
- Fit preprocessing only on training data.
- Preserve strict `action_number < game_event_id` sequence causality.
- Select hyperparameters on validation data, not the final evaluation split.
- Use a fresh later season or untouched final holdout before a Champion promotion claim.
- Keep secrets, raw data, processed data, MLflow stores, and trained weights out of git.
