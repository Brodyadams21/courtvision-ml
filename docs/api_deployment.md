# CourtVision API Deployment Guide

This guide describes how the CourtVision FastAPI inference service would be deployed to a managed environment. It documents the containerized deployable unit, runtime configuration, and operational expectations. **No managed production deployment has been performed yet.**

For local API usage, request examples, and Docker quick start, see [`docs/api.md`](api.md).

## Deployment status

The FastAPI service is **containerized and ready for local Docker-based serving tests**. It has **not been deployed** to a managed production environment (for example AWS ECS, Cloud Run, Azure Container Apps, or Kubernetes).

What exists today:

- A production-style API Docker image (`Dockerfile.api`)
- Docker Compose wiring for local containerized serving
- Configurable lazy vs. startup model loading
- Structured request logging with `X-Request-ID`
- Unit and artifact tests for endpoints, settings, Docker packaging, and documentation

What has **not** been done:

- Selection and provisioning of a managed hosting platform
- Remote artifact store wiring verified from a deployed container
- Platform secrets, HTTPS termination, authentication, or rate limiting
- Load testing or production smoke tests against a live endpoint

Do not describe this service as **production ready** or **deployed** until those gaps are closed.

## Deployable unit

The deployable unit is the Docker image built from `Dockerfile.api`:

| Component | Role |
|-----------|------|
| `Dockerfile.api` | Builds the API runtime image |
| `docker-compose.yml` → `api` service | Local reference for ports, env vars, and build context |
| `courtvision.api.main:app` | Uvicorn/FastAPI application entrypoint |
| `requirements-api.txt` | API-focused Python dependencies |

**Endpoints exposed by the service:**

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness check |
| `POST` | `/predict/shot` | Single-shot make probability and expected value |
| `POST` | `/predict/shots` | Batch scoring (1–500 shots per request) |

**Intentionally excluded from the image** (via `.dockerignore` and build context):

- Raw, interim, and processed data (`data/`)
- MLflow tracking stores (`mlruns/`, `mlartifacts/`)
- Local model artifact directories (`model_artifacts/`, etc.)
- Report figures and tables
- Secrets (`.env`, credentials)

The container expects to load the registered MLflow model at runtime (startup mode) or defer loading until explicitly configured (lazy mode). Model weights are not baked into the image in the current workflow.

## Required runtime configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PYTHONPATH` | Yes | — | Must be `/app/src` inside the container |
| `MLFLOW_TRACKING_URI` | Yes (for predictions) | — | MLflow tracking server URL reachable from the container |
| `COURTVISION_API_LOAD_MODEL_ON_STARTUP` | No | `false` | Load the registered model when the app starts |
| `COURTVISION_API_MODEL_ALIAS` | No | `Candidate` | MLflow model alias to load |

Truthy values for `COURTVISION_API_LOAD_MODEL_ON_STARTUP`: `1`, `true`, `yes`, `y` (case-insensitive).

### Lazy vs. startup mode

**Lazy mode** (`COURTVISION_API_LOAD_MODEL_ON_STARTUP=false`):

- The container starts without calling MLflow.
- `GET /health` succeeds immediately.
- Prediction endpoints return **503** until a model is loaded.
- Useful for boot checks, OpenAPI docs, and verifying the image starts.

**Startup mode** (`COURTVISION_API_LOAD_MODEL_ON_STARTUP=true`):

- The app loads `courtvision-shot-make-model` at the configured alias during creation.
- If loading fails, the process raises `RuntimeError` and exits (fail-fast).
- **Preferred for real serving** so a broken registry or unreachable artifact store is detected at deploy time, not on the first client request.

## Model loading strategy

1. At startup (when enabled), `ShotModelService.load_from_mlflow()` resolves `models:/courtvision-shot-make-model@{alias}` via MLflow.
2. The loaded sklearn/LightGBM model serves `predict_proba` for tabular feature rows validated against `FEATURE_COLUMNS`.
3. Single and batch endpoints share the same validation and error mapping (`503` for unloaded model, `422` for bad features).

**Local development:** tests inject a fake `ShotModelService` via `create_app(model_service=...)` and do not require MLflow.

**Containerized local serving:** point `MLFLOW_TRACKING_URI` at a host-accessible tracking server (for example `http://host.docker.internal:5000` on Docker Desktop for Windows/macOS).

## MLflow and model artifact access

The API depends on two separate MLflow concerns:

1. **Tracking server** — metadata and model registry (alias → version URI)
2. **Artifact store** — the serialized model files MLflow downloads when loading

### Important: artifact store accessibility

If MLflow uses a **local file artifact store** on the host machine (the default for local `mlruns/` development), a container may **not** be able to load the model artifact even if it can reach the MLflow tracking server over HTTP. The tracking API can return a `models:/...` URI whose underlying `file://` path exists only on the host filesystem.

A production deployment should use an **artifact store accessible from the container**, such as:

- **Amazon S3** (or another object store MLflow is configured to use)
- A **network-mounted volume** explicitly mounted into the container at the expected artifact path
- A **controlled build step** that exports the Candidate model artifact into the image (only when artifact provenance and update process are governed)

Until artifact access is verified from the target runtime, treat prediction serving in containers as **unproven**, even if `/health` returns OK.

### Registry expectation

- Registered model name: `courtvision-shot-make-model`
- Default alias for serving: `Candidate`
- Model type: sklearn pipeline / LightGBM loaded via `mlflow.sklearn.load_model`

## Container build and run

Build from the repository root:

```powershell
docker build -f Dockerfile.api -t courtvision-api:local .
```

**Lazy mode** (health only, no MLflow load at startup):

```powershell
docker run --rm -p 8000:8000 `
  -e PYTHONPATH=/app/src `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
  -e COURTVISION_API_LOAD_MODEL_ON_STARTUP=false `
  courtvision-api:local
```

**Startup mode** (fail-fast if Candidate is missing or unreachable):

```powershell
docker run --rm -p 8000:8000 `
  -e PYTHONPATH=/app/src `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
  -e COURTVISION_API_LOAD_MODEL_ON_STARTUP=true `
  -e COURTVISION_API_MODEL_ALIAS=Candidate `
  courtvision-api:local
```

**Docker Compose** (lazy mode by default):

```powershell
docker compose up --build api
```

## Health checks

| Check | Endpoint / mechanism | Expected |
|-------|----------------------|----------|
| Docker `HEALTHCHECK` | `GET http://127.0.0.1:8000/health` inside the container | HTTP 200, body `{"status":"ok"}` |
| Platform liveness | Same path on the service port (8000) | 200 OK |
| Readiness (recommended) | Extend with a `/ready` or startup probe that confirms `ShotModelService.is_loaded` when using startup mode | Not implemented yet |

The built-in Docker health check validates process and HTTP stack only. It does **not** confirm the model is loaded. In startup mode, a failed model load prevents the process from starting; in lazy mode, `/health` can pass while predictions return 503.

Wire the platform health check to `/health` for liveness. For readiness in startup mode, consider an additional probe once a dedicated ready endpoint exists.

## Logging and observability

Structured logs use the `courtvision.api` logger:

| Event | When | Key fields |
|-------|------|------------|
| `request_completed` | Every HTTP request | `request_id`, `path`, `method`, `status_code`, `latency_ms` |
| `single_prediction` | `/predict/shot` success | `request_id`, `model_name`, `batch_size` |
| `batch_prediction` | `/predict/shots` success | `request_id`, `model_name`, `batch_size` |
| `prediction_error` | Mapped 422/503 errors | `request_id`, `error_type` |
| `request_failed` | Unhandled exception | `request_id`, `path`, `method`, `latency_ms`, `error_type` |

Every response includes **`X-Request-ID`** for correlation with client reports.

**Not logged:** full feature dictionaries (large and potentially sensitive).

**Production follow-ups:**

- Route stdout/stderr to the platform log aggregator (CloudWatch, Datadog, etc.)
- Add JSON log formatting if the platform requires it
- Define alerts on error rate, latency, and 503 rate
- Optional: export metrics (request count, batch size histogram, model alias)

## Security and secrets

| Topic | Current state | Production expectation |
|-------|---------------|----------------------|
| Authentication | None on API routes | API key, OAuth proxy, or private network only |
| HTTPS | Not terminated in container | Platform load balancer or ingress handles TLS |
| Secrets | Not in image; `.env` excluded | Inject via platform secret manager (SSM, Secrets Manager, K8s secrets) |
| `MLFLOW_TRACKING_URI` | Plain env var | Use TLS URL; credentials via secrets if tracking server requires auth |
| Rate limiting | None | Gateway or middleware before public exposure |
| Input validation | Pydantic schemas, batch size cap 500 | Review limits under expected load |

Never bake `.env`, database passwords, or cloud credentials into `Dockerfile.api`.

## Production readiness checklist

Use this before calling the service production-ready or exposing it publicly:

- [ ] Managed hosting target selected (ECS, Cloud Run, AKS, etc.)
- [ ] Remote artifact store configured and verified from a running container
- [ ] API image built in CI and pushed to a registry
- [ ] Secrets injected through platform secret manager (not env files in git)
- [ ] Startup mode verified against reachable **Candidate** model
- [ ] Health check connected to platform liveness probe
- [ ] Logs routed to platform logging / observability stack
- [ ] HTTPS / auth / rate limiting decision made and implemented
- [ ] Load test or production smoke test completed
- [ ] Runbook for model alias promotion (Candidate → Champion) documented

## Known limitations

- **No managed deployment** has been executed; this document is planning and local verification only.
- **GRU / sequence models** are not exposed through the API; only the LightGBM Candidate registry path is wired.
- **Lazy mode** allows `/health` to pass without a loaded model; not suitable for production traffic without startup mode or a readiness probe.
- **Local file MLflow artifacts** may break containerized model loading even when the tracking server is reachable.
- **No authentication** on prediction endpoints.
- **Single-process uvicorn** in the default `CMD`; horizontal scaling and worker tuning are platform decisions not yet documented with benchmarks.
- **Batch endpoint** loops per shot (correct but not optimized for high throughput).

## Suggested deployment targets

These are reasonable next platforms; none is configured in this repository today:

| Platform | Fit | Notes |
|----------|-----|-------|
| **AWS ECS Fargate** | Strong | Pairs with existing S3 bucket work; ALB + CloudWatch; MLflow on RDS/tracking server + S3 artifacts |
| **Google Cloud Run** | Strong | Minimal ops for HTTP services; scale to zero; needs remote MLflow and artifact store |
| **Azure Container Apps** | Good | Similar to Cloud Run; integrate with Azure ML tracking if adopted |
| **Kubernetes (EKS/GKE/AKS)** | Flexible | More control; use for multi-service or GPU later; higher ops burden |
| **Fly.io / Render** | Portfolio demo | Fast path to a public HTTPS URL for demos; verify artifact access carefully |

For CourtVision’s current AWS partial setup, **ECS Fargate + S3-backed MLflow artifacts** is the most natural path when cloud execution unblocks, but that remains future work.
