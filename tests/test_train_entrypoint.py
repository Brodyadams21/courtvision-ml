"""Tests for the unified training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from courtvision.models import train as train_entrypoint
from courtvision.models.train_lgbm import INNER_TRAIN_FRACTION
from courtvision.utils.config import ProjectConfig


@pytest.fixture
def local_project_config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        environment="local",
        data_dir=tmp_path / "data",
    )


def _base_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "config": "configs/local.yaml",
        "model": "lightgbm",
        "mode": "default",
        "season": "2024-25",
        "inner_train_fraction": INNER_TRAIN_FRACTION,
        "no_mlflow": False,
        "register_candidate": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_default_mode_calls_run_default(
    monkeypatch: pytest.MonkeyPatch,
    local_project_config: ProjectConfig,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_default(
        season: str,
        *,
        processed_dir: Path | None = None,
        log_mlflow: bool = True,
    ) -> dict[str, float]:
        calls.append(
            {
                "season": season,
                "processed_dir": processed_dir,
                "log_mlflow": log_mlflow,
            }
        )
        return {"auc": 0.5}

    monkeypatch.setattr(
        train_entrypoint,
        "load_project_config",
        lambda _path: local_project_config,
    )
    monkeypatch.setattr(train_entrypoint.train_lgbm, "run_default", fake_run_default)

    train_entrypoint.run_from_args(_base_args(mode="default", no_mlflow=True))

    assert len(calls) == 1
    assert calls[0]["season"] == "2024-25"
    assert calls[0]["processed_dir"] == local_project_config.data_dir / "processed" / "features"
    assert calls[0]["log_mlflow"] is False


def test_search_mode_calls_run_search(
    monkeypatch: pytest.MonkeyPatch,
    local_project_config: ProjectConfig,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_search(
        season: str,
        *,
        processed_dir: Path | None = None,
        inner_train_fraction: float = INNER_TRAIN_FRACTION,
        log_mlflow: bool = True,
        register_candidate: bool = False,
    ) -> dict[str, object]:
        calls.append(
            {
                "season": season,
                "processed_dir": processed_dir,
                "inner_train_fraction": inner_train_fraction,
                "log_mlflow": log_mlflow,
                "register_candidate": register_candidate,
            }
        )
        return {"config_index": 0}

    monkeypatch.setattr(
        train_entrypoint,
        "load_project_config",
        lambda _path: local_project_config,
    )
    monkeypatch.setattr(train_entrypoint.train_lgbm, "run_search", fake_run_search)

    train_entrypoint.run_from_args(
        _base_args(
            mode="search",
            inner_train_fraction=0.7,
            register_candidate=True,
        )
    )

    assert len(calls) == 1
    assert calls[0]["season"] == "2024-25"
    assert calls[0]["processed_dir"] == local_project_config.data_dir / "processed" / "features"
    assert calls[0]["inner_train_fraction"] == 0.7
    assert calls[0]["log_mlflow"] is True
    assert calls[0]["register_candidate"] is True


def test_unsupported_model_fails_cleanly() -> None:
    with pytest.raises(SystemExit, match="Unsupported model: xgboost"):
        train_entrypoint.run_from_args(_base_args(model="xgboost"))


def test_register_candidate_requires_search_mode() -> None:
    with pytest.raises(SystemExit, match="requires --mode search"):
        train_entrypoint.run_from_args(
            _base_args(mode="default", register_candidate=True),
        )


def test_register_candidate_requires_mlflow_logging() -> None:
    with pytest.raises(SystemExit, match="cannot be used with --no-mlflow"):
        train_entrypoint.run_from_args(
            _base_args(mode="search", register_candidate=True, no_mlflow=True),
        )
