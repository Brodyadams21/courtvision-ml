"""Tests for the local pipeline orchestrator."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from courtvision.utils.config import ProjectConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_PATH = _REPO_ROOT / "pipelines" / "run_local_pipeline.py"


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location("run_local_pipeline", _PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pipeline module from {_PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pipeline = _load_pipeline_module()


@pytest.fixture
def local_project_config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        environment="local",
        data_dir=tmp_path / "data",
    )


def _base_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "config": "configs/local.yaml",
        "season": "2024-25",
        "model": "lightgbm",
        "mode": "default",
        "no_mlflow": False,
        "register_candidate": False,
        "rebuild_features": False,
        "dry_run": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_dry_run_default_prints_train_command_only(
    monkeypatch: pytest.MonkeyPatch,
    local_project_config: ProjectConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        pipeline,
        "load_project_config",
        lambda _path: local_project_config,
    )

    pipeline.run_from_args(_base_args(dry_run=True))

    output = capsys.readouterr().out
    assert output.strip() == "\n".join(
        [
            "Would run:",
            (
                "python -m courtvision.models.train --config configs/local.yaml "
                "--model lightgbm --mode default --season 2024-25"
            ),
        ]
    )


def test_dry_run_rebuild_features_includes_export_command(
    monkeypatch: pytest.MonkeyPatch,
    local_project_config: ProjectConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        pipeline,
        "load_project_config",
        lambda _path: local_project_config,
    )

    pipeline.run_from_args(_base_args(dry_run=True, rebuild_features=True))

    output = capsys.readouterr().out
    processed_dir = local_project_config.data_dir / "processed" / "features"
    assert output.strip() == "\n".join(
        [
            "Would run:",
            (
                "python -m courtvision.data.build_features --season 2024-25 "
                f"--export-only --processed-dir {processed_dir}"
            ),
            (
                "python -m courtvision.models.train --config configs/local.yaml "
                "--model lightgbm --mode default --season 2024-25"
            ),
        ]
    )


def test_dry_run_search_with_register_candidate(
    monkeypatch: pytest.MonkeyPatch,
    local_project_config: ProjectConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        pipeline,
        "load_project_config",
        lambda _path: local_project_config,
    )

    pipeline.run_from_args(
        _base_args(
            dry_run=True,
            mode="search",
            register_candidate=True,
        )
    )

    output = capsys.readouterr().out
    assert (
        "python -m courtvision.models.train --config configs/local.yaml "
        "--model lightgbm --mode search --season 2024-25 --register-candidate"
    ) in output


def test_run_executes_train_command(
    monkeypatch: pytest.MonkeyPatch,
    local_project_config: ProjectConfig,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        pipeline,
        "load_project_config",
        lambda _path: local_project_config,
    )

    pipeline.run_from_args(_base_args(), run_command_fn=fake_run)

    assert len(calls) == 1
    assert calls[0][1:3] == ["-m", "courtvision.models.train"]
    assert "--config" in calls[0]
    assert "--model" in calls[0]
    assert "--mode" in calls[0]
    assert "--season" in calls[0]


def test_run_executes_feature_export_then_train_when_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    local_project_config: ProjectConfig,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        pipeline,
        "load_project_config",
        lambda _path: local_project_config,
    )

    pipeline.run_from_args(_base_args(rebuild_features=True), run_command_fn=fake_run)

    assert len(calls) == 2
    assert calls[0][1:3] == ["-m", "courtvision.data.build_features"]
    assert "--export-only" in calls[0]
    assert calls[1][1:3] == ["-m", "courtvision.models.train"]


def test_failed_subprocess_raises_system_exit(
    monkeypatch: pytest.MonkeyPatch,
    local_project_config: ProjectConfig,
) -> None:
    def fake_run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(
        pipeline,
        "load_project_config",
        lambda _path: local_project_config,
    )

    with pytest.raises(SystemExit) as exc_info:
        pipeline.run_from_args(_base_args(), run_command_fn=fake_run)

    assert exc_info.value.code == 1


def test_register_candidate_requires_search_mode() -> None:
    with pytest.raises(SystemExit, match="requires --mode search"):
        pipeline.run_from_args(_base_args(mode="default", register_candidate=True))


def test_register_candidate_requires_mlflow_logging() -> None:
    with pytest.raises(SystemExit, match="cannot be used with --no-mlflow"):
        pipeline.run_from_args(
            _base_args(mode="search", register_candidate=True, no_mlflow=True),
        )
