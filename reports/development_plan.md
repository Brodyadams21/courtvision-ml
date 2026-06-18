# CourtVision ML - Development Plan Status

Last reviewed: 2026-06-18

This is the repository-facing status snapshot for the 13-phase CourtVision ML development plan. It records what is implemented on `main`; it does not mark placeholder files as completed work.

## Current position

The project is **through Phase 8 and actively in Phase 9**. The local and Docker portions of Phase 9 are complete. Managed AWS execution remains unfinished. Phases 10-12 have placeholder paths but no implementation, and Phase 13 is partially complete.

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
| 9 | Cloud-assisted training | In progress | Config-driven local runner, Docker image, Linux lock, and MLflow wiring complete; S3 transfer, SageMaker pipeline/job, and cloud registry verification remain |
| 10 | FastAPI inference service | Not started | `courtvision.api` files are empty; endpoints, schemas, request logging, tests, and API image remain |
| 11 | Streamlit dashboard | Not started | `dashboard/app.py` is empty; planned views remain |
| 12 | Monitoring and retraining | Not started | Monitoring modules are empty; drift, segment calibration, reports, triggers, and dashboard integration remain |
| 13 | Final documentation and portfolio polish | In progress | README, model reports, model card, architecture, training guide, and layout are present; demo media and final presentation/resume material remain |

## Phase 9 completion criteria

Phase 9 should not be marked complete until all of the following are true:

- Processed features or a representative training subset can be uploaded to the configured S3 layout.
- `pipelines/sagemaker_pipeline.py` creates or submits a reproducible managed training workflow.
- At least one cloud training job succeeds, or a fully documented dry run proves the generated job specification without claiming execution.
- Training metrics and artifacts reach a cloud-accessible MLflow backend or an explicitly documented equivalent.
- The cloud path is covered by focused tests that mock external services rather than requiring AWS credentials in CI.

## Recommended order

1. Finish Phase 9 with the smallest credible AWS training path.
2. Build Phase 10 around the registered LightGBM Candidate first, then add the GRU bundle when sequence assembly is operationally defined.
3. Build the Phase 11 dashboard on the published report tables and API contract.
4. Add Phase 12 monitoring and promotion gates before calling either model production-ready.
5. Finish Phase 13 with screenshots/demo material and a clean reproduction walkthrough.

## Quality gates

- Keep all temporal splits game-based and ordered by date.
- Fit preprocessing only on training data.
- Preserve strict `action_number < game_event_id` sequence causality.
- Select hyperparameters on validation data, not the final evaluation split.
- Use a fresh later season or untouched final holdout before a Champion promotion claim.
- Keep secrets, raw data, processed data, MLflow stores, and trained weights out of git.
