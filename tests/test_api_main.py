"""Tests for CourtVision FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from courtvision.api.main import create_app
from courtvision.api.model_service import ShotModelService
from courtvision.api.schemas import ShotPredictionResponse
from courtvision.models.common import FEATURE_COLUMNS
from courtvision.models.registry import REGISTERED_MODEL_NAME


def _feature_payload(*, shot_value: int = 3) -> dict[str, float]:
    features = {column: 0.0 for column in FEATURE_COLUMNS}
    features["shot_value"] = float(shot_value)
    return features


def _loaded_model_service(*, make_probability: float = 0.40) -> ShotModelService:
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1.0 - make_probability, make_probability]])
    return ShotModelService(model=model, model_name=REGISTERED_MODEL_NAME)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(model_service=_loaded_model_service()))


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_shot_returns_probability_and_expected_value(client: TestClient) -> None:
    response = client.post(
        "/predict/shot",
        json={"features": _feature_payload(shot_value=3), "shot_value": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_make_probability"] == pytest.approx(0.40)
    assert payload["expected_shot_value"] == pytest.approx(1.20)
    assert payload["model_name"] == REGISTERED_MODEL_NAME


def test_predict_shot_uses_injected_service_without_mlflow() -> None:
    fake_service = MagicMock(spec=ShotModelService)
    fake_service.predict_shot.return_value = ShotPredictionResponse(
        predicted_make_probability=0.55,
        expected_shot_value=1.10,
        model_name="fake-model",
    )
    client = TestClient(create_app(model_service=fake_service))

    response = client.post(
        "/predict/shot",
        json={"features": {"shot_distance": 15.0}, "shot_value": 2},
    )

    assert response.status_code == 200
    assert response.json()["model_name"] == "fake-model"
    fake_service.predict_shot.assert_called_once_with({"shot_distance": 15.0}, 2)


def test_predict_shot_returns_503_when_model_not_loaded() -> None:
    client = TestClient(create_app(model_service=ShotModelService()))

    response = client.post(
        "/predict/shot",
        json={"features": _feature_payload(shot_value=2), "shot_value": 2},
    )

    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"].lower()


def test_predict_shot_returns_422_for_missing_features(client: TestClient) -> None:
    response = client.post(
        "/predict/shot",
        json={"features": {"shot_distance": 12.0}, "shot_value": 2},
    )

    assert response.status_code == 422
    assert "Missing feature columns" in response.json()["detail"]


def test_predict_shot_returns_422_for_invalid_shot_value(client: TestClient) -> None:
    response = client.post(
        "/predict/shot",
        json={"features": _feature_payload(shot_value=3), "shot_value": 1},
    )

    assert response.status_code == 422
