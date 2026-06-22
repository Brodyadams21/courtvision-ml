"""SageMaker training job submission for CourtVision LightGBM runs."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from courtvision.data.collect import DEFAULT_SEASON
from courtvision.utils.config import ProjectConfig, load_project_config
from courtvision.utils.storage import join_s3_uri, s3_uri

INPUT_CHANNEL_NAME = "processed"
SAGEMAKER_CONFIG_PATH = "configs/sagemaker.yaml"
DEFAULT_INSTANCE_TYPE = "ml.m5.xlarge"
DEFAULT_MAX_RUNTIME_SECONDS = 30 * 60
DEFAULT_VOLUME_SIZE_GB = 30
SubmitTrainingJobFn = Callable[[dict[str, Any], Any], str]


def expected_feature_paths(season: str) -> list[str]:
    """Return the feature Parquet paths expected inside the training container."""
    base = "/opt/ml/input/data/processed/features"
    return [
        f"{base}/train_shot_features_{season}.parquet",
        f"{base}/test_shot_features_{season}.parquet",
    ]


def build_container_command(
    *,
    season: str,
    mode: str = "default",
    no_mlflow: bool = True,
) -> list[str]:
    """Build the training container command for SageMaker."""
    command = [
        "python",
        "-m",
        "courtvision.models.train",
        "--config",
        SAGEMAKER_CONFIG_PATH,
        "--model",
        "lightgbm",
        "--mode",
        mode,
        "--season",
        season,
    ]
    if no_mlflow:
        command.append("--no-mlflow")
    return command


def default_job_name(*, season: str, timestamp: datetime | None = None) -> str:
    """Return a SageMaker-safe training job name."""
    ts = (timestamp or datetime.now(tz=UTC)).strftime("%Y%m%d-%H%M%S")
    safe_season = re.sub(r"[^a-zA-Z0-9-]", "-", season)
    name = f"courtvision-lgbm-{safe_season}-{ts}"
    return name[:63].rstrip("-")


def resolve_bucket(config: ProjectConfig, override: str | None) -> str:
    """Resolve the target S3 bucket from CLI, config, or environment."""
    bucket = (override or config.s3_bucket or os.environ.get("COURTVISION_S3_BUCKET", "")).strip()
    if not bucket:
        raise SystemExit(
            "S3 bucket is required. Pass --bucket or set s3_bucket in config / "
            "COURTVISION_S3_BUCKET."
        )
    return bucket


def resolve_image_uri(override: str | None) -> str:
    """Resolve the training image URI from CLI or environment."""
    image_uri = (override or os.environ.get("COURTVISION_ECR_IMAGE_URI", "")).strip()
    if not image_uri:
        raise SystemExit(
            "ECR image URI is required. Pass --image-uri or set COURTVISION_ECR_IMAGE_URI."
        )
    return image_uri


def resolve_role_arn(override: str | None) -> str:
    """Resolve the SageMaker execution role ARN from CLI or environment."""
    role_arn = (override or os.environ.get("COURTVISION_SAGEMAKER_ROLE_ARN", "")).strip()
    if not role_arn:
        raise SystemExit(
            "SageMaker execution role ARN is required. Pass --role-arn or set "
            "COURTVISION_SAGEMAKER_ROLE_ARN."
        )
    return role_arn


def as_s3_prefix(uri: str) -> str:
    """Ensure an S3 URI is formatted as a prefix."""
    return uri if uri.endswith("/") else f"{uri}/"


def build_training_job_request(
    *,
    job_name: str,
    role_arn: str,
    image_uri: str,
    input_s3_uri: str,
    output_s3_uri: str,
    season: str,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    max_runtime_seconds: int = DEFAULT_MAX_RUNTIME_SECONDS,
    training_mode: str = "default",
    no_mlflow: bool = True,
) -> dict[str, Any]:
    """Build a ``CreateTrainingJob`` request payload."""
    return {
        "TrainingJobName": job_name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": build_container_command(
                season=season,
                mode=training_mode,
                no_mlflow=no_mlflow,
            ),
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
            {"Key": "season", "Value": season},
        ],
    }


def build_job_from_config(
    config: ProjectConfig,
    *,
    season: str,
    bucket: str | None = None,
    image_uri: str | None = None,
    role_arn: str | None = None,
    job_name: str | None = None,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    max_runtime_seconds: int = DEFAULT_MAX_RUNTIME_SECONDS,
    training_mode: str = "default",
    no_mlflow: bool = True,
) -> dict[str, Any]:
    """Assemble a training job request from project config and runtime settings."""
    prefixes = config.s3_prefixes or {}
    processed_prefix = prefixes.get("processed", "processed/")
    output_prefix = prefixes.get("sagemaker_output", "sagemaker-output/")

    resolved_bucket = resolve_bucket(config, bucket)
    resolved_image_uri = resolve_image_uri(image_uri)
    resolved_role_arn = resolve_role_arn(role_arn)
    resolved_job_name = job_name or default_job_name(season=season)

    input_s3_uri = as_s3_prefix(join_s3_uri(s3_uri(resolved_bucket), processed_prefix))
    output_s3_uri = as_s3_prefix(join_s3_uri(s3_uri(resolved_bucket), output_prefix))

    return build_training_job_request(
        job_name=resolved_job_name,
        role_arn=resolved_role_arn,
        image_uri=resolved_image_uri,
        input_s3_uri=input_s3_uri,
        output_s3_uri=output_s3_uri,
        season=season,
        instance_type=instance_type,
        max_runtime_seconds=max_runtime_seconds,
        training_mode=training_mode,
        no_mlflow=no_mlflow,
    )


def print_dry_run_summary(request: dict[str, Any], *, season: str) -> None:
    """Print the exact SageMaker job configuration for review."""
    print("SageMaker training job configuration (dry run)")
    print(json.dumps(request, indent=2))
    print("\nExpected feature paths inside container:")
    for path in expected_feature_paths(season):
        print(f"  {path}")


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
            "Prepare or submit a SageMaker LightGBM training job. "
            "Defaults to dry run (print configuration only)."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/aws.yaml",
        help="Project config YAML with S3 prefixes (default: configs/aws.yaml)",
    )
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        help="Season label (e.g. 2024-25)",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="S3 bucket override (default: config s3_bucket or COURTVISION_S3_BUCKET)",
    )
    parser.add_argument(
        "--image-uri",
        default=None,
        help="Training image URI override (default: COURTVISION_ECR_IMAGE_URI)",
    )
    parser.add_argument(
        "--role-arn",
        default=None,
        help="SageMaker execution role ARN override (default: COURTVISION_SAGEMAKER_ROLE_ARN)",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Optional SageMaker training job name (default: generated timestamped name)",
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
        "--training-mode",
        choices=("default", "search"),
        default="default",
        help="Training mode passed to the container entrypoint",
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
        season=args.season,
        bucket=args.bucket,
        image_uri=args.image_uri,
        role_arn=args.role_arn,
        job_name=args.job_name,
        instance_type=args.instance_type,
        max_runtime_seconds=args.max_runtime_seconds,
        training_mode=args.training_mode,
    )

    if not args.execute:
        print_dry_run_summary(request, season=args.season)
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
