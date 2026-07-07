"""Tests for CourtVision FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


def _shot_request(*, shot_value: int = 3) -> dict[str, object]:
    return {"features": _feature_payload(shot_value=shot_value), "shot_value": shot_value}


def _batch_request(*shots: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    return {"shots": list(shots)}


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


def test_predict_shots_returns_one_prediction_per_input_shot(client: TestClient) -> None:
    response = client.post(
        "/predict/shots",
        json=_batch_request(_shot_request(shot_value=2), _shot_request(shot_value=3)),
    )

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["expected_shot_value"] == pytest.approx(0.80)
    assert predictions[1]["expected_shot_value"] == pytest.approx(1.20)


def test_predict_shots_preserves_input_order() -> None:
    model = MagicMock()
    model.predict_proba.side_effect = [
        np.array([[0.70, 0.30]]),
        np.array([[0.50, 0.50]]),
        np.array([[0.10, 0.90]]),
    ]
    service = ShotModelService(model=model, model_name=REGISTERED_MODEL_NAME)
    client = TestClient(create_app(model_service=service))

    response = client.post(
        "/predict/shots",
        json=_batch_request(
            _shot_request(shot_value=2),
            _shot_request(shot_value=3),
            _shot_request(shot_value=3),
        ),
    )

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [item["predicted_make_probability"] for item in predictions] == pytest.approx(
        [0.30, 0.50, 0.90]
    )


def test_predict_shots_returns_503_when_model_not_loaded() -> None:
    client = TestClient(create_app(model_service=ShotModelService()))

    response = client.post(
        "/predict/shots",
        json=_batch_request(_shot_request(shot_value=2)),
    )

    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"].lower()


def test_predict_shots_returns_422_for_missing_features(client: TestClient) -> None:
    response = client.post(
        "/predict/shots",
        json=_batch_request({"features": {"shot_distance": 12.0}, "shot_value": 2}),
    )

    assert response.status_code == 422
    assert "Missing feature columns" in response.json()["detail"]


def test_predict_shots_rejects_empty_shot_list(client: TestClient) -> None:
    response = client.post("/predict/shots", json={"shots": []})

    assert response.status_code == 422


def test_predict_shots_rejects_over_max_batch_size(client: TestClient) -> None:
    shots = [_shot_request(shot_value=2) for _ in range(501)]
    response = client.post("/predict/shots", json={"shots": shots})

    assert response.status_code == 422


@patch.object(ShotModelService, "load_from_mlflow")
def test_create_app_lazy_mode_does_not_load_model(load_from_mlflow: MagicMock) -> None:
    create_app(load_model_on_startup=False)

    load_from_mlflow.assert_not_called()


@patch.object(ShotModelService, "load_from_mlflow")
def test_create_app_startup_mode_loads_model(load_from_mlflow: MagicMock) -> None:
    create_app(load_model_on_startup=True)

    load_from_mlflow.assert_called_once_with(alias="Candidate")


@patch.object(ShotModelService, "load_from_mlflow")
def test_create_app_startup_mode_passes_model_alias(load_from_mlflow: MagicMock) -> None:
    create_app(load_model_on_startup=True, model_alias="Champion")

    load_from_mlflow.assert_called_once_with(alias="Champion")


@patch.object(ShotModelService, "load_from_mlflow", side_effect=RuntimeError("mlflow down"))
def test_create_app_startup_mode_raises_when_loading_fails(load_from_mlflow: MagicMock) -> None:
    with pytest.raises(RuntimeError, match="Failed to load MLflow model alias 'Candidate'"):
        create_app(load_model_on_startup=True)

    load_from_mlflow.assert_called_once_with(alias="Candidate")


@patch.object(ShotModelService, "load_from_mlflow")
def test_create_app_with_injected_service_skips_startup_loading(
    load_from_mlflow: MagicMock,
) -> None:
    fake_service = MagicMock(spec=ShotModelService)

    create_app(model_service=fake_service, load_model_on_startup=True)

    load_from_mlflow.assert_not_called()
