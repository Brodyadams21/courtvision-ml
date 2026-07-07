"""Tests for CourtVision API environment settings."""

from __future__ import annotations

import pytest

from courtvision.api.settings import ApiSettings, load_api_settings


def test_load_api_settings_defaults_to_lazy_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COURTVISION_API_LOAD_MODEL_ON_STARTUP", raising=False)
    monkeypatch.delenv("COURTVISION_API_MODEL_ALIAS", raising=False)

    settings = load_api_settings()

    assert settings == ApiSettings(load_model_on_startup=False, model_alias="Candidate")


def test_load_api_settings_enables_startup_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COURTVISION_API_LOAD_MODEL_ON_STARTUP", "true")

    settings = load_api_settings()

    assert settings.load_model_on_startup is True


def test_load_api_settings_overrides_model_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COURTVISION_API_MODEL_ALIAS", "Champion")

    settings = load_api_settings()

    assert settings.model_alias == "Champion"
