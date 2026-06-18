# CourtVision Training Guide

This guide explains how to run CourtVision model training locally and in Docker.

**Run all commands from the project root** (`courtvision-ml/`).

## Prerequisites

- Python 3.11
- Docker Desktop
- Project dependencies installed from `requirements.txt`
- Processed feature files available under `data/processed/features`
- For local Python commands, set `$env:PYTHONPATH = "src"` in the current PowerShell window.

## 1. Local training without MLflow

Use this for quick smoke tests.

```powershell
$env:PYTHONPATH = "src"
python -m courtvision.models.train --config configs/local.yaml --model lightgbm --mode default --no-mlflow
```

Expected output includes train/test shapes and LightGBM test metrics.

## 2. Local training with MLflow

Start MLflow:

```powershell
.\scripts\start_mlflow.ps1
```

In another PowerShell window:

```powershell
$env:PYTHONPATH = "src"
python -m courtvision.models.train --config configs/local.yaml --model lightgbm --mode default
```

Open MLflow at:

```
http://127.0.0.1:5000
```

## 3. Local pipeline runner

```powershell
$env:PYTHONPATH = "src"
```

Dry run:

```powershell
python pipelines/run_local_pipeline.py --config configs/local.yaml --season 2024-25 --model lightgbm --mode default --dry-run
```

Run training without MLflow:

```powershell
python pipelines/run_local_pipeline.py --config configs/local.yaml --season 2024-25 --model lightgbm --mode default --no-mlflow
```

Run training with MLflow:

```powershell
python pipelines/run_local_pipeline.py --config configs/local.yaml --season 2024-25 --model lightgbm --mode default
```

## 4. Docker training

Build the training image:

```powershell
docker build -f Dockerfile.train -t courtvision-train:local .
```

Run Docker training without MLflow:

```powershell
docker run --rm -v "${PWD}\data\processed\features:/app/data/processed/features" courtvision-train:local python -m courtvision.models.train --config configs/local.yaml --model lightgbm --mode default --no-mlflow
```

## 5. Docker training with MLflow

Start MLflow on the host:

```powershell
.\scripts\start_mlflow.ps1
```

**Use `configs/docker.yaml`, not `configs/local.yaml`.** Inside a container, `127.0.0.1` points at the container itself, not your Windows host. `configs/docker.yaml` sets `mlflow_tracking_uri` to `http://host.docker.internal:5000` so training can reach the MLflow server on the host.

Run Docker training:

```powershell
docker run --rm -v "${PWD}\data\processed\features:/app/data/processed/features" -v "${PWD}\reports:/app/reports" courtvision-train:local python -m courtvision.models.train --config configs/docker.yaml --model lightgbm --mode default
```

The Docker tracking URI in `configs/docker.yaml` is:

```
http://host.docker.internal:5000
```

Mount `reports/` if you want feature-importance plots and tables written on the host (not only inside the container).

## Expected LightGBM smoke-test metrics

A successful default LightGBM run on the 2024-25 processed features should produce metrics close to:

```
auc: 0.6470
log_loss: 0.6498
brier_score: 0.2293
accuracy: 0.6218
```

Small differences can happen across dependency versions, but large differences should be investigated.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Docker training cannot reach MLflow (`Connection refused`, requests to `127.0.0.1:5000`) | Use `--config configs/docker.yaml`, not `configs/local.yaml`. |
| MLflow rejects requests from Docker (`Invalid Host header`) | Restart MLflow with `.\scripts\start_mlflow.ps1`. The script binds to `0.0.0.0` and passes `--allowed-hosts` for `host.docker.internal`. |
| LightGBM MLflow model logging fails on skops / untrusted types | Training code logs sklearn models with `serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE` in `train_lgbm.py`. |
| Docker logs show `Failed to import Git...` | Rebuild the image after `Dockerfile.train` installs `git` (`docker build -f Dockerfile.train -t courtvision-train:local .`). |

## Notes

- Do not commit raw data, processed data, MLflow runs, or model artifacts.
- Use `--no-mlflow` for quick training checks.
- Use Docker training to verify the project works outside the local virtual environment.
- Use Candidate registration only with search mode.

## Dependency management

Direct dependencies are listed in `requirements.in`.

The pinned install file is `requirements.txt` and is generated with:

```powershell
python -m piptools compile --resolver=backtracking --output-file requirements.txt requirements.in
```

Install dependencies with:

```powershell
pip install -r requirements.txt
```
