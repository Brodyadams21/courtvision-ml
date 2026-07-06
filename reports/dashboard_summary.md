# CourtVision Analytics Dashboard — Portfolio Summary

A short reference for resumes, LinkedIn, and interview talking points. For run instructions, see [`docs/dashboard.md`](../docs/dashboard.md).

---

## What was built

A **local Streamlit analytics dashboard** (`src/courtvision/dashboard/`) that sits on top of the existing CourtVision ML pipeline:

- Reads time-based train/test Parquet exports (no runtime database dependency)
- Surfaces LightGBM Candidate training metrics, feature importance, and diagnostic plots
- Loads live predictions from the MLflow model registry (`courtvision-shot-make-model` @ **Candidate**)
- Scores individual shots and batched samples with shared, testable helpers (`data.py`, `prediction.py`)
- Exports scored edge tables and backtest summaries to CSV

The UI has **six tabs** covering dataset overview, exploratory shot quality, model performance review, interactive prediction, edge ranking, and a sampled edge backtest.

---

## Why it matters

Sports analytics and ML engineering roles care about more than offline AUC. Stakeholders need to **see** model behavior on real shots, compare predictions to interpretable baselines, and sanity-check whether "edge" lines up with outcomes in held-out data.

This dashboard demonstrates:

1. **End-to-end ownership** — from feature Parquet through registry-backed inference to analyst-facing UI
2. **Separation of concerns** — Streamlit-free data and prediction modules with unit tests; thin UI layer in `app.py`
3. **Production-minded patterns** — MLflow alias loading, graceful degradation when artifacts are missing, batch scoring with per-row error collection, session-state staleness warnings
4. **Basketball translation** — expected shot value and edge vs. similar-shot baseline, not raw logits

---

## What each tab demonstrates technically

| Tab | Technical demonstration |
|-----|-------------------------|
| **Overview** | Parquet I/O, train/test split stats, leakage-safe evaluation framing |
| **Shot Quality Explorer** | Pandas filtering, distance bucketing, descriptive baselines without model calls |
| **Model Performance** | JSON artifact parsing, optional validation metrics, static report integration (CSV + PNG) |
| **Prediction Playground** | MLflow model service, feature preparation matching API schema, baseline comparison for one row |
| **Shot Edge Explorer** | Batch preparation + batch prediction, edge table construction, CSV export |
| **Edge Backtest** | Bucket aggregation by EV edge, grouped diagnostics, dual CSV export |

---

## How it supports an ML/analytics portfolio

**Resume / LinkedIn bullets (adapt as needed):**

- Built a Streamlit analytics dashboard on top of an NBA shot-quality ML pipeline, with MLflow registry-backed inference and CSV export for edge analysis.
- Implemented testable batch scoring helpers separating dataframe logic from model calls, with 60+ dashboard unit tests.
- Connected held-out evaluation data, LightGBM Candidate metrics, and similar-shot baselines into an interactive tool for shot-level model review.

**Interview angles:**

- Why LightGBM remains Candidate while GRU is Challenger+ (serving simplicity vs. accuracy)
- How similar-shot baselines are computed and why edge is defined as model EV minus baseline EV
- How you would extend to player/team pages, monitoring, or a deployed demo without rewriting core helpers

---

## Known limitations

- **Local only** — no cloud deployment, auth, or multi-user hosting; Streamlit session state is per browser session
- **Single season** — 2024-25 Parquet exports; no multi-season selector in the UI
- **LightGBM Candidate only** — dashboard predictions use the registered sklearn/LightGBM artifact, not the GRU challenger
- **Sampled edge backtest** — Edge Backtest uses random samples (100–500 shots); it is exploratory, not a full-corpus or betting-grade validation
- **No player/team evaluation pages yet** — those summaries live in `reports/basketball_insights.md` and CSV tables, not in the dashboard
- **FastAPI is separate** — a basic `/predict/shot` endpoint exists for local inference tests; it is not wired as a remote backend for the dashboard
- **Monitoring and drift** — not implemented; no live calibration or data-drift tracking in the UI

---

## Suggested next steps (future work)

- Player and team evaluation pages reusing evaluation-layer summaries
- Richer model comparison (LightGBM vs. GRU metrics side by side)
- Optional deployed demo (containerized Streamlit + MLflow or baked model artifact)
- Saved screenshot / report export for portfolio artifacts
