"""Tests for project configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from courtvision.utils.config import (
    PROJECT_ROOT,
    ProjectConfig,
    load_project_config,
    load_yaml_config,
)


def test_load_yaml_config_from_project_root_relative_path() -> None:
    data = load_yaml_config("configs/local.yaml")
    assert data["environment"] == "local"


def test_load_yaml_config_from_outside_project_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data = load_yaml_config("configs/local.yaml")
    assert data["environment"] == "local"


def test_load_yaml_config_prefers_cwd_relative_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_config = tmp_path / "configs" / "local.yaml"
    local_config.parent.mkdir(parents=True)
    local_config.write_text("environment: tmp\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    data = load_yaml_config("configs/local.yaml")
    assert data["environment"] == "tmp"


def test_load_project_config_resolves_local_path_fields() -> None:
    config = load_project_config("configs/local.yaml")
    assert config.data_dir == PROJECT_ROOT / "data"
    assert config.model_dir == PROJECT_ROOT / "model_artifacts"
    assert config.data_dir.is_absolute()
    assert config.model_dir.is_absolute()


def test_load_project_config_keeps_s3_prefixes_as_strings() -> None:
    config = load_project_config("configs/aws.yaml")
    assert config.s3_prefixes == {
        "raw": "raw/",
        "processed": "processed/",
        "features": "processed/features/",
        "models": "model_artifacts/",
        "reports": "reports/",
        "sagemaker_output": "sagemaker-output/",
        "mlflow_artifacts": "mlflow-artifacts/",
    }


def test_project_config_requires_environment() -> None:
    with pytest.raises(ValueError, match="Missing required config key\\(s\\): environment"):
        ProjectConfig.from_dict({})


def test_load_yaml_config_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="also checked"):
        load_yaml_config("configs/missing.yaml")
