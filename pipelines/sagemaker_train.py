"""Safe SageMaker training job launcher for CourtVision LightGBM default runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from courtvision.utils.config import ProjectConfig, load_project_config  # noqa: E402
from courtvision.utils.storage import join_s3_uri, s3_uri  # noqa: E402

INPUT_CHANNEL_NAME = "processed"
DEFAULT_ECR_REPO = "courtvision-train"
DEFAULT_ECR_TAG = "phase9-sagemaker-v1"
DEFAULT_ROLE_NAME = "CourtVisionSageMakerExecutionRole"
DEFAULT_INSTANCE_TYPE = "ml.c5.xlarge"
DEFAULT_MAX_RUNTIME_SECONDS = 1800
DEFAULT_VOLUME_SIZE_GB = 30
JOB_NAME_PREFIX = "courtvision-lightgbm-default"
SubmitTrainingJobFn = Callable[[dict[str, Any], Any], str]

CONTAINER_COMMAND: list[str] = [
    "python",
    "-m",
    "courtvision.models.train_lgbm",
    "--processed-dir",
    "/opt/ml/input/data/processed/features",
    "--mode",
    "default",
    "--no-mlflow",
]


def default_job_name(*, timestamp: datetime | None = None) -> str:
    """Return a SageMaker-safe training job name."""
    ts = (timestamp or datetime.now(tz=UTC)).strftime("%Y%m%d-%H%M%S")
    name = f"{JOB_NAME_PREFIX}-{ts}"
    return name[:63].rstrip("-")


def build_ecr_image_uri(
    *,
    account_id: str,
    region: str,
    repo: str = DEFAULT_ECR_REPO,
    tag: str = DEFAULT_ECR_TAG,
) -> str:
    """Build an ECR image URI from account, region, repository, and tag."""
    cleaned_account = account_id.strip()
    cleaned_region = region.strip()
    cleaned_repo = repo.strip().strip("/")
    cleaned_tag = tag.strip()
    if not cleaned_account:
        raise ValueError("AWS account ID cannot be empty")
    if not cleaned_region:
        raise ValueError("AWS region cannot be empty")
    if not cleaned_repo:
        raise ValueError("ECR repository name cannot be empty")
    if not cleaned_tag:
        raise ValueError("ECR image tag cannot be empty")
    return f"{cleaned_account}.dkr.ecr.{cleaned_region}.amazonaws.com/{cleaned_repo}:{cleaned_tag}"


def build_role_arn(
    *,
    account_id: str,
    role_name: str = DEFAULT_ROLE_NAME,
) -> str:
    """Build a SageMaker execution role ARN from account ID and role name."""
    cleaned_account = account_id.strip()
    cleaned_role = role_name.strip()
    if not cleaned_account:
        raise ValueError("AWS account ID cannot be empty")
    if not cleaned_role:
        raise ValueError("SageMaker role name cannot be empty")
    return f"arn:aws:iam::{cleaned_account}:role/{cleaned_role}"


def as_s3_prefix(uri: str) -> str:
    """Ensure an S3 URI is formatted as a prefix."""
    return uri if uri.endswith("/") else f"{uri}/"


def resolve_bucket(config: ProjectConfig, override: str | None) -> str:
    """Resolve the target S3 bucket from CLI, config, or environment."""
    bucket = (override or config.s3_bucket or os.environ.get("COURTVISION_S3_BUCKET", "")).strip()
    if not bucket:
        raise SystemExit(
            "S3 bucket is required. Pass --bucket or set s3_bucket in config / "
            "COURTVISION_S3_BUCKET."
        )
    return bucket


def resolve_account_id(override: str | None) -> str:
    """Resolve the AWS account ID from CLI or environment."""
    account_id = (override or os.environ.get("COURTVISION_AWS_ACCOUNT_ID", "")).strip()
    if not account_id:
        raise SystemExit(
            "AWS account ID is required. Pass --account-id or set COURTVISION_AWS_ACCOUNT_ID."
        )
    return account_id


def resolve_image_uri(
    *,
    account_id: str,
    region: str,
    override: str | None = None,
    repo: str = DEFAULT_ECR_REPO,
    tag: str = DEFAULT_ECR_TAG,
) -> str:
    """Resolve the training image URI from override or ECR components."""
    image_uri = (override or os.environ.get("COURTVISION_ECR_IMAGE_URI", "")).strip()
    if image_uri:
        return image_uri
    return build_ecr_image_uri(
        account_id=account_id,
        region=region,
        repo=repo,
        tag=tag,
    )


def resolve_role_arn(
    *,
    account_id: str,
    override: str | None = None,
    role_name: str = DEFAULT_ROLE_NAME,
) -> str:
    """Resolve the SageMaker execution role ARN from override or role name."""
    role_arn = (override or os.environ.get("COURTVISION_SAGEMAKER_ROLE_ARN", "")).strip()
    if role_arn:
        return role_arn
    return build_role_arn(account_id=account_id, role_name=role_name)


def build_training_job_request(
    *,
    job_name: str,
    role_arn: str,
    image_uri: str,
    input_s3_uri: str,
    output_s3_uri: str,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    max_runtime_seconds: int = DEFAULT_MAX_RUNTIME_SECONDS,
) -> dict[str, Any]:
    """Build a ``CreateTrainingJob`` request payload."""
    return {
        "TrainingJobName": job_name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": list(CONTAINER_COMMAND),
        },
        "InputDataConfig": [
            {
                "ChannelName": INPUT_CHANNEL_NAME,
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": input_s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-parquet",
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": output_s3_uri,
        },
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": DEFAULT_VOLUME_SIZE_GB,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": max_runtime_seconds,
        },
        "Tags": [
            {"Key": "project", "Value": "courtvision-ml"},
            {"Key": "model", "Value": "lightgbm"},
            {"Key": "mode", "Value": "default"},
        ],
    }


def build_job_from_config(
    config: ProjectConfig,
    *,
    bucket: str | None = None,
    account_id: str | None = None,
    image_uri: str | None = None,
    role_arn: str | None = None,
    job_name: str | None = None,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    max_runtime_seconds: int = DEFAULT_MAX_RUNTIME_SECONDS,
    ecr_repo: str = DEFAULT_ECR_REPO,
    ecr_tag: str = DEFAULT_ECR_TAG,
    role_name: str = DEFAULT_ROLE_NAME,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Assemble a training job request from project config and runtime settings."""
    prefixes = config.s3_prefixes or {}
    processed_prefix = prefixes.get("processed", "processed/")
    output_prefix = prefixes.get("sagemaker_output", "sagemaker-output/")

    region = config.aws_region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    resolved_bucket = resolve_bucket(config, bucket)
    resolved_account_id = resolve_account_id(account_id)
    resolved_image_uri = resolve_image_uri(
        account_id=resolved_account_id,
        region=region,
        override=image_uri,
        repo=ecr_repo,
        tag=ecr_tag,
    )
    resolved_role_arn = resolve_role_arn(
        account_id=resolved_account_id,
        override=role_arn,
        role_name=role_name,
    )
    resolved_job_name = job_name or default_job_name(timestamp=timestamp)

    input_s3_uri = as_s3_prefix(join_s3_uri(s3_uri(resolved_bucket), processed_prefix))
    output_s3_uri = as_s3_prefix(join_s3_uri(s3_uri(resolved_bucket), output_prefix))

    return build_training_job_request(
        job_name=resolved_job_name,
        role_arn=resolved_role_arn,
        image_uri=resolved_image_uri,
        input_s3_uri=input_s3_uri,
        output_s3_uri=output_s3_uri,
        instance_type=instance_type,
        max_runtime_seconds=max_runtime_seconds,
    )


def print_dry_run_summary(request: dict[str, Any]) -> None:
    """Print the exact SageMaker job configuration for review."""
    print("SageMaker training job configuration (dry run)")
    print(json.dumps(request, indent=2))


def create_sagemaker_client(*, region_name: str) -> Any:
    """Create a boto3 SageMaker client."""
    try:
        import boto3
    except ImportError as exc:
        raise SystemExit(
            "boto3 is required to submit SageMaker jobs. Install it with: pip install boto3"
        ) from exc
    return boto3.client("sagemaker", region_name=region_name)


def submit_training_job(
    request: dict[str, Any],
    *,
    client: Any | None = None,
    region_name: str = "us-east-1",
) -> str:
    """Submit a SageMaker training job and return the job name."""
    sm_client = client or create_sagemaker_client(region_name=region_name)
    sm_client.create_training_job(**request)
    return str(request["TrainingJobName"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or submit a SageMaker LightGBM default training job. "
            "Defaults to dry run (print configuration only)."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/aws.yaml",
        help="Project config YAML with S3 prefixes (default: configs/aws.yaml)",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="S3 bucket override (default: config s3_bucket or COURTVISION_S3_BUCKET)",
    )
    parser.add_argument(
        "--account-id",
        default=None,
        help="AWS account ID override (default: COURTVISION_AWS_ACCOUNT_ID)",
    )
    parser.add_argument(
        "--image-uri",
        default=None,
        help="Training image URI override (default: built from account/region/repo/tag)",
    )
    parser.add_argument(
        "--role-arn",
        default=None,
        help="SageMaker execution role ARN override (default: CourtVisionSageMakerExecutionRole)",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Optional SageMaker training job name (default: courtvision-lightgbm-default-<ts>)",
    )
    parser.add_argument(
        "--instance-type",
        default=DEFAULT_INSTANCE_TYPE,
        help=f"SageMaker instance type (default: {DEFAULT_INSTANCE_TYPE})",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=DEFAULT_MAX_RUNTIME_SECONDS,
        help=f"Maximum job runtime in seconds (default: {DEFAULT_MAX_RUNTIME_SECONDS})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit the SageMaker training job (default: print configuration only)",
    )
    return parser.parse_args(argv)


def run_from_args(
    args: argparse.Namespace,
    *,
    submit_fn: SubmitTrainingJobFn | None = None,
    client: Any | None = None,
) -> None:
    """Build the SageMaker job request and either print or submit it."""
    config = load_project_config(args.config)
    request = build_job_from_config(
        config,
        bucket=args.bucket,
        account_id=args.account_id,
        image_uri=args.image_uri,
        role_arn=args.role_arn,
        job_name=args.job_name,
        instance_type=args.instance_type,
        max_runtime_seconds=args.max_runtime_seconds,
    )

    if not args.execute:
        print_dry_run_summary(request)
        return

    region_name = config.aws_region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    submit = submit_fn or (
        lambda payload, sm_client: submit_training_job(
            payload,
            client=sm_client,
            region_name=region_name,
        )
    )
    job_name = submit(request, client)
    print(f"Submitted SageMaker training job: {job_name}")


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
