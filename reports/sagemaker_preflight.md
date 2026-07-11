# SageMaker LightGBM training preflight

## Current status

Phase 9 cloud-assisted training has had one managed run that reached container execution but failed on the processed-data path. The SageMaker launcher (`pipelines/sagemaker_train.py`) now passes `--processed-dir /opt/ml/input/data/processed/features` so training reads the SageMaker input channel. Use dry run first; submit only after reviewing the printed job configuration.

## Approved instance type

`ml.c5.xlarge`

## Launcher defaults

| Setting | Value |
| --- | --- |
| Default instance type | `ml.c5.xlarge` |
| Training module (container command) | `courtvision.models.train_lgbm` |
| Container args | `--processed-dir /opt/ml/input/data/processed/features --model-output-dir /opt/ml/model --mode default --no-mlflow` |
| Default behavior | Dry run (print job JSON only) |

The container command is:

```text
python -m courtvision.models.train_lgbm --processed-dir /opt/ml/input/data/processed/features --model-output-dir /opt/ml/model --mode default --no-mlflow
```

MLflow is disabled in the default cloud run until cloud MLflow tracking and artifact storage are fully configured.

## First failure note

The first managed training attempt reached container execution but failed because the training script used the local `/app` processed-features default instead of the SageMaker input channel at `/opt/ml/input/data/processed/features`.

## Commands

Set required environment variables locally before running (do not commit real values):

- `COURTVISION_AWS_ACCOUNT_ID` — AWS account ID
- `COURTVISION_S3_BUCKET` — S3 bucket for processed data and outputs (or pass `--bucket`)

Optional overrides: `COURTVISION_ECR_IMAGE_URI`, `COURTVISION_SAGEMAKER_ROLE_ARN`, `AWS_DEFAULT_REGION`.

### Dry run (review configuration only)

```bash
python pipelines/sagemaker_train.py --config configs/aws.yaml
```

Prints the full `CreateTrainingJob` payload without calling AWS.

### Execute (submit job later)

```bash
python pipelines/sagemaker_train.py --config configs/aws.yaml --execute
```

Add `--instance-type ml.c5.xlarge` only if overriding the default. Add `--bucket`, `--account-id`, `--image-uri`, or `--role-arn` as needed without embedding secrets in the repo.

## Safety

Do **not** commit account IDs, credentials, private bucket names, or AWS CLI/log output that contains private identifiers. Use environment variables or local-only config for sensitive values.
