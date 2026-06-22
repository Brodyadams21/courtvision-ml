"""Tests for the SageMaker training job launcher."""

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
_TRAIN_PATH = _REPO_ROOT / "pipelines" / "sagemaker_train.py"


def _load_train_module():
    spec = importlib.util.spec_from_file_location("sagemaker_train", _TRAIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training launcher from {_TRAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train = _load_train_module()


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
        "bucket": "courtvision-bucket",
        "account_id": "123456789012",
        "image_uri": None,
        "role_arn": None,
        "job_name": None,
        "instance_type": "ml.m5.large",
        "max_runtime_seconds": 1800,
        "execute": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_default_job_name_starts_with_expected_prefix() -> None:
    timestamp = datetime(2026, 6, 21, 14, 30, tzinfo=UTC)
    job_name = train.default_job_name(timestamp=timestamp)

    assert job_name.startswith("courtvision-lightgbm-default-")
    assert job_name == "courtvision-lightgbm-default-20260621-143000"


def test_build_ecr_image_uri_uses_region_account_repo_and_tag() -> None:
    image_uri = train.build_ecr_image_uri(
        account_id="123456789012",
        region="us-east-1",
        repo="courtvision-train",
        tag="phase9-sagemaker-v1",
    )

    assert (
        image_uri
        == "123456789012.dkr.ecr.us-east-1.amazonaws.com/courtvision-train:phase9-sagemaker-v1"
    )


def test_build_job_from_config_uses_processed_input_channel(
    aws_project_config: ProjectConfig,
) -> None:
    timestamp = datetime(2026, 6, 21, 14, 30, tzinfo=UTC)
    request = train.build_job_from_config(
        aws_project_config,
        bucket="courtvision-bucket",
        account_id="123456789012",
        timestamp=timestamp,
    )

    channel = request["InputDataConfig"][0]
    assert channel["ChannelName"] == "processed"
    assert channel["DataSource"]["S3DataSource"]["S3Uri"].endswith("/processed/")
    assert request["OutputDataConfig"]["S3OutputPath"].endswith("/sagemaker-output/")
    assert request["ResourceConfig"]["InstanceType"] == "ml.m5.large"
    assert request["ResourceConfig"]["InstanceCount"] == 1
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] == 1800


def test_build_job_from_config_uses_sagemaker_container_command(
    aws_project_config: ProjectConfig,
) -> None:
    request = train.build_job_from_config(
        aws_project_config,
        bucket="courtvision-bucket",
        account_id="123456789012",
        timestamp=datetime(2026, 6, 21, 14, 30, tzinfo=UTC),
    )

    entrypoint = request["AlgorithmSpecification"]["ContainerEntrypoint"]
    assert entrypoint == train.CONTAINER_COMMAND
    assert "configs/sagemaker.yaml" in entrypoint
    assert "configs/local.yaml" not in entrypoint


def test_build_job_from_config_builds_role_arn_from_account(
    aws_project_config: ProjectConfig,
) -> None:
    request = train.build_job_from_config(
        aws_project_config,
        bucket="courtvision-bucket",
        account_id="123456789012",
        timestamp=datetime(2026, 6, 21, 14, 30, tzinfo=UTC),
    )

    assert request["RoleArn"] == "arn:aws:iam::123456789012:role/CourtVisionSageMakerExecutionRole"


def test_run_from_args_dry_run_does_not_submit(
    aws_project_config: ProjectConfig,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submit_calls: list[dict[str, Any]] = []

    def fake_submit(request: dict[str, Any], _client: Any) -> str:
        submit_calls.append(request)
        return str(request["TrainingJobName"])

    monkeypatch.setattr(
        train,
        "load_project_config",
        lambda _path: aws_project_config,
    )

    train.run_from_args(_base_args(execute=False), submit_fn=fake_submit)

    captured = capsys.readouterr().out
    assert "SageMaker training job configuration (dry run)" in captured
    assert submit_calls == []

    payload = json.loads(captured.split("\n", 1)[1])
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
        train,
        "load_project_config",
        lambda _path: aws_project_config,
    )

    train.run_from_args(
        _base_args(execute=True, job_name="courtvision-lightgbm-default-test"),
        submit_fn=fake_submit,
        client=object(),
    )

    captured = capsys.readouterr().out
    assert captured.strip() == "Submitted SageMaker training job: courtvision-lightgbm-default-test"
    assert len(submit_calls) == 1


def test_resolve_account_id_requires_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COURTVISION_AWS_ACCOUNT_ID", raising=False)

    with pytest.raises(SystemExit, match="AWS account ID is required"):
        train.resolve_account_id(None)
