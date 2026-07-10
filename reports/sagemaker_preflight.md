# SageMaker LightGBM training preflight

## Current status

Phase 9 cloud-assisted training is **not yet executed**. The SageMaker launcher (`pipelines/sagemaker_train.py`) is aligned with the current local LightGBM entrypoint and the approved instance type. Use dry run first; submit only after reviewing the printed job configuration.

## Approved instance type

`ml.c5.xlarge`

## Launcher defaults

| Setting | Value |
| --- | --- |
| Default instance type | `ml.c5.xlarge` |
| Training module (container command) | `courtvision.models.train_lgbm` |
| Container args | `--mode default --no-mlflow` |
| Default behavior | Dry run (print job JSON only) |

The container command is:

```text
python -m courtvision.models.train_lgbm --mode default --no-mlflow
```

MLflow is disabled in the default cloud run until cloud MLflow tracking and artifact storage are fully configured.

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
