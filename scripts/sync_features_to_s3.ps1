param(
    [string]$BucketName = $env:COURTVISION_S3_BUCKET,
    [string]$Season = "2024-25",
    [string]$Prefix = "processed/features",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not $BucketName) {
    throw "BucketName is required. Pass -BucketName or set `$env:COURTVISION_S3_BUCKET."
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$FeatureDir = Join-Path $ProjectRoot "data/processed/features"

$TrainFile = Join-Path $FeatureDir "train_shot_features_$Season.parquet"
$TestFile = Join-Path $FeatureDir "test_shot_features_$Season.parquet"

if (-not (Test-Path $TrainFile)) {
    throw "Missing train feature file: $TrainFile"
}
if (-not (Test-Path $TestFile)) {
    throw "Missing test feature file: $TestFile"
}

$Destination = "s3://$BucketName/$Prefix"

Write-Host "Syncing CourtVision processed features"
Write-Host "  Season:      $Season"
Write-Host "  Source:      $FeatureDir"
Write-Host "  Destination: $Destination"

$DryRunArgs = @()
if ($DryRun) {
    $DryRunArgs += "--dryrun"
}

aws s3 cp $TrainFile "$Destination/train_shot_features_$Season.parquet" @DryRunArgs
aws s3 cp $TestFile "$Destination/test_shot_features_$Season.parquet" @DryRunArgs

Write-Host ""
Write-Host "S3 contents:"
aws s3 ls "$Destination/" --human-readable --summarize
