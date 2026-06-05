# Start local MLflow tracking server backed by PostgreSQL (Docker).
#
# Prerequisites:
#   - Docker Desktop running
#   - Python venv with mlflow + psycopg2-binary (pip install -r requirements.txt)
#
# Usage (from anywhere):
#   .\scripts\start_mlflow.ps1
#
# Then open: http://127.0.0.1:5000
# Point training runs at: MLFLOW_TRACKING_URI=http://127.0.0.1:5000

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$PostgresUser = "courtvision_user"
$PostgresPassword = "courtvision_local_dev"
$PostgresHost = "127.0.0.1"
$PostgresPort = "5433"
$PostgresService = "postgres"
$MlflowDatabase = "courtvision_mlflow"
$MlflowHost = "127.0.0.1"
$MlflowPort = "5000"
$ArtifactRootPath = Join-Path $ProjectRoot "mlartifacts"
$CreateDbSql = Join-Path $ProjectRoot "sql/create_mlflow_database.sql"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

Write-Step "Starting Docker PostgreSQL ($PostgresService)"
docker compose up -d $PostgresService --wait

Write-Step "Creating MLflow database if missing ($MlflowDatabase)"
if (-not (Test-Path $CreateDbSql)) {
    throw "Missing SQL script: $CreateDbSql"
}

Get-Content $CreateDbSql |
    docker compose exec -T $PostgresService psql -U $PostgresUser -d postgres -v ON_ERROR_STOP=1

Write-Step "Starting MLflow server (PostgreSQL backend)"
New-Item -ItemType Directory -Force -Path $ArtifactRootPath | Out-Null
$ArtifactRootUri = ([System.Uri]((Resolve-Path $ArtifactRootPath).Path)).AbsoluteUri

$BackendStoreUri = "postgresql://${PostgresUser}:${PostgresPassword}@${PostgresHost}:${PostgresPort}/${MlflowDatabase}"
$TrackingUri = "http://${MlflowHost}:${MlflowPort}"

Write-Host "  Tracking URI:  $TrackingUri"
Write-Host "  Backend store: postgresql://${PostgresUser}:***@${PostgresHost}:${PostgresPort}/${MlflowDatabase}"
Write-Host "  Artifacts:     $ArtifactRootPath"
Write-Host "  Artifact URI:  $ArtifactRootUri"
Write-Host ""
Write-Host "Set before training:"
Write-Host "  `$env:MLFLOW_TRACKING_URI = `"$TrackingUri`""

$MlflowExe = Join-Path $ProjectRoot ".venv/Scripts/mlflow.exe"
if (Test-Path $MlflowExe) {
    & $MlflowExe server `
        --backend-store-uri $BackendStoreUri `
        --default-artifact-root $ArtifactRootUri `
        --host $MlflowHost `
        --port $MlflowPort
} else {
    mlflow server `
        --backend-store-uri $BackendStoreUri `
        --default-artifact-root $ArtifactRootUri `
        --host $MlflowHost `
        --port $MlflowPort
}
