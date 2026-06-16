"""Project configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]

REQUIRED_PROJECT_CONFIG_KEYS: tuple[str, ...] = ("environment",)


def _resolve_config_path(path: str | Path) -> Path:
    """Resolve a config file path from the cwd or ``PROJECT_ROOT``."""
    config_path = Path(path)
    if config_path.is_file():
        return config_path.resolve()

    if not config_path.is_absolute():
        project_path = PROJECT_ROOT / config_path
        if project_path.is_file():
            return project_path.resolve()
        raise FileNotFoundError(
            f"Config file not found: {config_path} (also checked {project_path})"
        )

    raise FileNotFoundError(f"Config file not found: {config_path}")


def _resolve_local_path(value: str | Path | None) -> Path | None:
    """Resolve a repo-relative local path against ``PROJECT_ROOT``."""
    if value is None:
        return None

    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _validate_required_keys(
    data: dict[str, Any],
    required: tuple[str, ...] = REQUIRED_PROJECT_CONFIG_KEYS,
) -> None:
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing required config key(s): {', '.join(missing)}")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    config_path = _resolve_config_path(path)

    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected top-level mapping in {config_path}, got {type(loaded).__name__}"
        )
    return loaded


@dataclass(frozen=True)
class ProjectConfig:
    environment: str
    database_url: str = ""
    mlflow_tracking_uri: str = ""
    mlflow_artifact_root: str | None = None
    data_dir: Path | None = None
    model_dir: Path | None = None
    aws_region: str | None = None
    s3_bucket: str | None = None
    s3_prefixes: dict[str, str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectConfig:
        _validate_required_keys(data)

        field_names = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in data.items() if key in field_names}
        kwargs["data_dir"] = _resolve_local_path(kwargs.get("data_dir"))
        kwargs["model_dir"] = _resolve_local_path(kwargs.get("model_dir"))
        return cls(**kwargs)


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load a project YAML config file and return a ``ProjectConfig`` instance."""
    return ProjectConfig.from_dict(load_yaml_config(path))
