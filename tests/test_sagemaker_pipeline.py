"""Tests for the SageMaker training job submitter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from courtvision.utils.config import ProjectConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_PATH = _REPO_ROOT / "pipelines" / "sagemaker_pipeline.py"


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location("sagemaker_pipeline", _PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pipeline module from {_PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pipeline = _load_pipeline_module()


@pytest.fixture
def aws_project_config() -> ProjectConfig:
    return ProjectConfig(
        environment="aws",
        aws_region="us-east-1",
        s3_bucket="courtvision-bucket",
        s3_prefixes={
            "processed": "processed/",
            "sagemaker_output": "sagemaker-output/",
        },
    )


def _base_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "config": "configs/aws.yaml",
        "season": "2024-25",
        "bucket": "courtvision-bucket",
        "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/courtvision-train:latest",
        "role_arn": "arn:aws:iam::123456789012:role/courtvision-sagemaker",
        "job_name": "courtvision-lgbm-2024-25-test",
        "instance_type": "ml.m5.xlarge",
        "max_runtime_seconds": 1800,
        "training_mode": "default",
        "execute": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_training_job_request_uses_processed_input_channel(
    aws_project_config: ProjectConfig,
) -> None:
    request = pipeline.build_job_from_config(
        aws_project_config,
        season="2024-25",
        image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/courtvision-train:latest",
        role_arn="arn:aws:iam::123456789012:role/courtvision-sagemaker",
        job_name="courtvision-lgbm-2024-25-test",
    )

    channel = request["InputDataConfig"][0]
    assert channel["ChannelName"] == "processed"
    assert channel["DataSource"]["S3DataSource"]["S3Uri"] == "s3://courtvision-bucket/processed/"
    assert request["OutputDataConfig"]["S3OutputPath"] == "s3://courtvision-bucket/sagemaker-output/"
    assert request["ResourceConfig"]["InstanceType"] == "ml.m5.xlarge"
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] == 1800


def test_build_training_job_request_uses_sagemaker_container_command() -> None:
    request = pipeline.build_training_job_request(
        job_name="courtvision-lgbm-2024-25-test",
        role_arn="arn:aws:iam::123456789012:role/courtvision-sagemaker",
        image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/courtvision-train:latest",
        input_s3_uri="s3://courtvision-bucket/processed/",
        output_s3_uri="s3://courtvision-bucket/sagemaker-output/",
        season="2024-25",
    )

    entrypoint = request["AlgorithmSpecification"]["ContainerEntrypoint"]
    assert entrypoint == [
        "python",
        "-m",
        "courtvision.models.train",
        "--config",
        "configs/sagemaker.yaml",
        "--model",
        "lightgbm",
        "--mode",
        "default",
        "--season",
        "2024-25",
        "--no-mlflow",
    ]


def test_expected_feature_paths_match_sagemaker_channel_layout() -> None:
    assert pipeline.expected_feature_paths("2024-25") == [
        "/opt/ml/input/data/processed/features/train_shot_features_2024-25.parquet",
        "/opt/ml/input/data/processed/features/test_shot_features_2024-25.parquet",
    ]


def test_default_job_name_is_sagemaker_safe() -> None:
    timestamp = datetime(2026, 6, 21, 14, 30, tzinfo=UTC)
    job_name = pipeline.default_job_name(season="2024-25", timestamp=timestamp)

    assert job_name == "courtvision-lgbm-2024-25-20260621-143000"
    assert len(job_name) <= 63


def test_run_from_args_dry_run_prints_configuration(
    aws_project_config: ProjectConfig,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submit_calls: list[dict[str, Any]] = []

    def fake_submit(request: dict[str, Any], _client: Any) -> str:
        submit_calls.append(request)
        return str(request["TrainingJobName"])

    monkeypatch.setattr(
        pipeline,
        "load_project_config",
        lambda _path: aws_project_config,
    )

    pipeline.run_from_args(_base_args(execute=False), submit_fn=fake_submit)

    captured = capsys.readouterr().out
    assert "SageMaker training job configuration (dry run)" in captured
    assert "Expected feature paths inside container:" in captured
    assert "/opt/ml/input/data/processed/features/train_shot_features_2024-25.parquet" in captured
    assert submit_calls == []

    payload = json.loads(captured.split("\n", 1)[1].split("\n\nExpected", 1)[0])
    assert payload["InputDataConfig"][0]["ChannelName"] == "processed"


def test_run_from_args_execute_submits_job(
    aws_project_config: ProjectConfig,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submit_calls: list[dict[str, Any]] = []

    def fake_submit(request: dict[str, Any], _client: Any) -> str:
        submit_calls.append(request)
        return str(request["TrainingJobName"])

    monkeypatch.setattr(
        pipeline,
        "load_project_config",
        lambda _path: aws_project_config,
    )

    pipeline.run_from_args(_base_args(execute=True), submit_fn=fake_submit, client=object())

    captured = capsys.readouterr().out
    assert captured.strip() == "Submitted SageMaker training job: courtvision-lgbm-2024-25-test"
    assert len(submit_calls) == 1
    assert submit_calls[0]["TrainingJobName"] == "courtvision-lgbm-2024-25-test"


def test_resolve_bucket_requires_value(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ProjectConfig(environment="aws", s3_bucket="")
    monkeypatch.delenv("COURTVISION_S3_BUCKET", raising=False)

    with pytest.raises(SystemExit, match="S3 bucket is required"):
        pipeline.resolve_bucket(config, None)


def test_resolve_image_uri_requires_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COURTVISION_ECR_IMAGE_URI", raising=False)

    with pytest.raises(SystemExit, match="ECR image URI is required"):
        pipeline.resolve_image_uri(None)
