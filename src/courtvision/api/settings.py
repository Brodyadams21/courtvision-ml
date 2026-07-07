"""Environment-driven settings for the CourtVision inference API."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiSettings:
    load_model_on_startup: bool = False
    model_alias: str = "Candidate"


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def load_api_settings() -> ApiSettings:
    return ApiSettings(
        load_model_on_startup=_env_bool("COURTVISION_API_LOAD_MODEL_ON_STARTUP"),
        model_alias=os.getenv("COURTVISION_API_MODEL_ALIAS", "Candidate"),
    )
