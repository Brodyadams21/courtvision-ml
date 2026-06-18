# CourtVision Training Guide

This guide explains how to run CourtVision model training locally and in Docker.

## Prerequisites

- Python 3.11
- Docker Desktop
- Project dependencies installed from `requirements.txt`
- Processed feature files available under `data/processed/features`

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

http://127.0.0.1:5000

## 3. Local pipeline runner

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

Run Docker training using the Docker config:

```powershell
docker run --rm -v "${PWD}\data\processed\features:/app/data/processed/features" -v "${PWD}\reports:/app/reports" courtvision-train:local python -m courtvision.models.train --config configs/docker.yaml --model lightgbm --mode default
```

Inside Docker, `configs/docker.yaml` uses:

http://host.docker.internal:5000

so the container can log to the MLflow server running on the host machine.

## Expected LightGBM smoke-test metrics

A successful default LightGBM run on the 2024-25 processed features should produce metrics close to:

```
auc: 0.6470
log_loss: 0.6498
brier_score: 0.2293
accuracy: 0.6218
```

Small differences can happen across dependency versions, but large differences should be investigated.

## Notes

- Do not commit raw data, processed data, MLflow runs, or model artifacts.
- Use `--no-mlflow` for quick training checks.
- Use Docker training to verify the project works outside the local virtual environment.
- Use Candidate registration only with search mode.
