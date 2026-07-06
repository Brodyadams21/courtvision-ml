# CourtVision Analytics Dashboard

The CourtVision Streamlit dashboard is a local analytics app for exploring held-out test shots, reviewing LightGBM Candidate model performance, and scoring individual shots or sampled batches against a historical similar-shot baseline.

**Entry point:** `src/courtvision/dashboard/app.py` (or `dashboard/app.py` from the repo root).

---

## What the dashboard shows

The dashboard reads the same time-based train/test Parquet exports used for model training (`data/processed/features/`). It does **not** retrain models or query PostgreSQL at runtime.

| Capability | Source |
|------------|--------|
| Dataset overview | Train/test Parquet splits |
| Shot quality filters and distance buckets | Held-out test split |
| LightGBM test metrics, feature importance, calibration plots | `model_artifacts/training_summary.json`, `reports/tables/`, `reports/figures/` |
| Live shot predictions | MLflow registered model `courtvision-shot-make-model` @ **Candidate** |
| Edge ranking and backtest buckets | Candidate model + similar-shot baseline on test rows |

Prediction tabs (Playground, Shot Edge Explorer, Edge Backtest) require a registered Candidate model in a running MLflow tracking server.

---

## Prerequisites

- Python 3.11 with project dependencies installed (`pip install -r requirements.txt`)
- Virtual environment activated
- Processed feature files present:
  - `data/processed/features/train_shot_features_2024-25.parquet`
  - `data/processed/features/test_shot_features_2024-25.parquet`
- Docker Desktop (for PostgreSQL + MLflow backend via `start_mlflow.ps1`)
- For prediction features: a trained LightGBM run registered as **Candidate** in MLflow

Set `PYTHONPATH` so Python can import `courtvision`:

```powershell
$env:PYTHONPATH = "src"
```

---

## Start MLflow

From the project root, in a dedicated terminal:

```powershell
cd C:\Users\brody\Desktop\ML-Magic\courtvision-ml
.\.venv\Scripts\Activate.ps1

$env:PYTHONPATH = "src"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

.\scripts\start_mlflow.ps1
```

This script:

1. Starts Docker PostgreSQL (`docker compose up -d postgres`)
2. Creates the `courtvision_mlflow` database if needed
3. Starts the MLflow UI at **http://127.0.0.1:5000**

Leave this terminal running while you train or use the dashboard.

---

## Register the Candidate model

The dashboard loads predictions through `ShotModelService`, which resolves:

**`courtvision-shot-make-model` @ `Candidate`**

If you have not registered a model yet, train LightGBM with search mode and promote the run in a **second** terminal (while MLflow is running):

```powershell
cd C:\Users\brody\Desktop\ML-Magic\courtvision-ml
.\.venv\Scripts\Activate.ps1

$env:PYTHONPATH = "src"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

python -m courtvision.models.train_lgbm --mode search --register-candidate
```

Confirm in the MLflow UI (**Models** → `courtvision-shot-make-model`) that the **Candidate** alias points at a version.

You only need to re-register when you want the dashboard to use a newer trained model.

---

## Run Streamlit

In another terminal (MLflow still running):

```powershell
cd C:\Users\brody\Desktop\ML-Magic\courtvision-ml
.\.venv\Scripts\Activate.ps1

$env:PYTHONPATH = "src"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

streamlit run src/courtvision/dashboard/app.py
```

Alternative launcher (same app, bootstraps `src` automatically):

```powershell
streamlit run dashboard/app.py
```

Streamlit opens a local browser session (default **http://localhost:8501**).

---

## Dashboard tabs

### Overview

**Question:** How large is the evaluation dataset, and what are the train vs. test make rates?

Shows shot counts, make rates, and feature count for the time-based train/test split. Use this as a sanity check before exploring filters or scoring shots.

### Shot Quality Explorer

**Question:** How do make rate, shot value, and a simple historical baseline vary by shot type, distance, and game period on held-out test shots?

Filter by 2PT/3PT, period, and distance range. View summary metrics and a distance-bucket table. No MLflow model required.

### Model Performance

**Question:** How well did the registered LightGBM Candidate perform on the held-out test set?

Displays metrics from `model_artifacts/training_summary.json` (AUC, log loss, Brier, accuracy), validation metrics when available, top feature importance from `reports/tables/lightgbm_feature_importance_gain.csv`, and calibration / probability distribution figures when present. No live scoring.

### Prediction Playground

**Question:** For one specific held-out shot, what does the model predict, and how does that compare to similar historical shots?

Pick a random or specific test row, score it with the Candidate model, and compare predicted make probability and expected shot value to a similar-shot baseline. Requires MLflow Candidate model.

### Shot Edge Explorer

**Question:** Which sampled held-out shots does the model rate most above or below the similar-shot baseline?

Sample 25, 50, or 100 test shots (optional 2PT/3PT filter), batch-score with the Candidate model, and rank by EV edge or probability edge. Includes CSV export. Requires MLflow Candidate model.

### Edge Backtest

**Question:** When shots are grouped by model EV edge vs. baseline, do higher-edge buckets show better actual outcomes in this sample?

Sample 100, 250, or 500 test shots, score them, bucket by `ev_edge_vs_baseline`, and compare average model EV, baseline EV, and actual points per bucket. Includes summary and per-shot CSV exports. This is a **sampled diagnostic**, not proof of future performance. Requires MLflow Candidate model.

---

## Exported CSVs

Download buttons appear after you score a sample or run a backtest.

| Tab | Button | Filename | Contents |
|-----|--------|----------|----------|
| Shot Edge Explorer | Download scored edge table CSV | `courtvision_shot_edge_sample.csv` | Scored sample with model probability, EV, baseline, and edge columns |
| Edge Backtest | Download backtest summary CSV | `courtvision_edge_backtest_summary.csv` | Per-bucket aggregates (shot count, avg predicted make rate, avg model/baseline EV, avg actual points) |
| Edge Backtest | Download scored backtest shots CSV | `courtvision_edge_backtest_shots.csv` | Full scored shot table used to build the bucket summary |

---

## Troubleshooting

### "Prediction model is not available locally yet"

- MLflow is not running, or `MLFLOW_TRACKING_URI` is not set to `http://127.0.0.1:5000`
- No model is registered, or the **Candidate** alias is missing on `courtvision-shot-make-model`
- Fix: start MLflow, run `train_lgbm --mode search --register-candidate`, restart or reload the Streamlit app

### Overview or Shot Quality Explorer works, but Playground / Edge tabs do not

Non-prediction tabs only need Parquet files. Prediction tabs need MLflow + Candidate registration (see above).

### "No training summary found" on Model Performance

Run LightGBM training so `model_artifacts/training_summary.json` is created:

```powershell
python -m courtvision.models.train_lgbm --mode search --register-candidate
```

### Feature importance or calibration images missing

Regenerate reports by training LightGBM (writes `reports/tables/lightgbm_feature_importance_gain.csv` and `reports/figures/*.png`). The dashboard degrades gracefully with a warning when files are absent.

### Controls changed since the last scored sample

Shot Edge Explorer and Edge Backtest store results in Streamlit session state. If you change sample size or filters after scoring, click **Score sample** or **Run backtest** again to refresh.

### `ModuleNotFoundError: No module named 'courtvision'`

Set `$env:PYTHONPATH = "src"` before launching Streamlit, or use `streamlit run dashboard/app.py` from the repo root.

### Docker / PostgreSQL errors when starting MLflow

Ensure Docker Desktop is running, then retry `.\scripts\start_mlflow.ps1`. PostgreSQL listens on host port **5433** (see `docker-compose.yml`).

---

## Related docs

- Training and MLflow setup: [`docs/training.md`](training.md)
- Portfolio-oriented dashboard summary: [`reports/dashboard_summary.md`](../reports/dashboard_summary.md)
