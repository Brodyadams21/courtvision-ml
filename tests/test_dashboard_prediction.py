"""Tests for dashboard prediction helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from courtvision.api.model_service import ShotModelService
from courtvision.api.schemas import ShotPredictionResponse
from courtvision.dashboard.data import PreparedPredictionFeatures
from courtvision.dashboard.prediction import (
    PREDICTION_UNAVAILABLE_MESSAGE,
    BatchPredictionResult,
    DashboardPredictionResult,
    PredictionUnavailable,
    create_model_service,
    predict_prepared_shot,
    predict_prepared_shots,
)
from courtvision.models.registry import REGISTERED_MODEL_NAME


def _prepared_shot(**feature_overrides: float) -> PreparedPredictionFeatures:
    features = {f"feature_{index}": float(index) for index in range(30)}
    features.update(feature_overrides)
    return PreparedPredictionFeatures(features=features, shot_value=3)


def test_predict_prepared_shot_calls_service_predict_shot_with_features_and_shot_value() -> None:
    prepared = PreparedPredictionFeatures(
        features={"shot_distance": 15.0, "loc_x": 1.0},
        shot_value=2,
    )
    service = MagicMock(spec=ShotModelService)
    service.is_loaded = True
    service.predict_shot.return_value = ShotPredictionResponse(
        predicted_make_probability=0.417,
        expected_shot_value=0.834,
        model_name=REGISTERED_MODEL_NAME,
    )

    predict_prepared_shot(prepared, service=service)

    service.predict_shot.assert_called_once_with(prepared.features, prepared.shot_value)


def test_predict_prepared_shot_returns_probability_expected_value_and_model_name() -> None:
    prepared = PreparedPredictionFeatures(features={"shot_distance": 24.0}, shot_value=3)
    service = MagicMock(spec=ShotModelService)
    service.is_loaded = True
    service.predict_shot.return_value = ShotPredictionResponse(
        predicted_make_probability=0.417,
        expected_shot_value=1.251,
        model_name=REGISTERED_MODEL_NAME,
    )

    result = predict_prepared_shot(prepared, service=service)

    assert result.predicted_make_probability == pytest.approx(0.417)
    assert result.expected_shot_value == pytest.approx(1.251)
    assert result.model_name == REGISTERED_MODEL_NAME


def test_create_model_service_raises_prediction_unavailable_when_loading_fails() -> None:
    side_effect = RuntimeError("mlflow down")
    with patch.object(ShotModelService, "load_from_mlflow", side_effect=side_effect):
        with pytest.raises(PredictionUnavailable, match="Prediction model is not available"):
            create_model_service()


def test_predict_prepared_shot_raises_prediction_unavailable_for_unloaded_service() -> None:
    prepared = _prepared_shot()
    service = ShotModelService()

    with pytest.raises(PredictionUnavailable, match=PREDICTION_UNAVAILABLE_MESSAGE):
        predict_prepared_shot(prepared, service=service)


def test_predict_prepared_shot_raises_prediction_unavailable_when_runtime_not_loaded() -> None:
    prepared = _prepared_shot()
    service = MagicMock(spec=ShotModelService)
    service.is_loaded = True
    service.predict_shot.side_effect = RuntimeError(
        "Model is not loaded; call load_from_mlflow() first."
    )

    with pytest.raises(PredictionUnavailable, match=PREDICTION_UNAVAILABLE_MESSAGE):
        predict_prepared_shot(prepared, service=service)


def _mock_prediction_response() -> ShotPredictionResponse:
    return ShotPredictionResponse(
        predicted_make_probability=0.417,
        expected_shot_value=1.251,
        model_name=REGISTERED_MODEL_NAME,
    )


def test_predict_prepared_shots_scores_multiple_prepared_rows() -> None:
    prepared_by_row_id = {
        1: PreparedPredictionFeatures(features={"shot_distance": 10.0}, shot_value=2),
        2: PreparedPredictionFeatures(features={"shot_distance": 20.0}, shot_value=3),
    }
    service = MagicMock(spec=ShotModelService)
    service.is_loaded = True
    service.predict_shot.return_value = _mock_prediction_response()

    result = predict_prepared_shots(prepared_by_row_id, service=service)

    assert isinstance(result, BatchPredictionResult)
    assert set(result.predictions_by_row_id) == {1, 2}
    assert result.errors == []
    for prediction in result.predictions_by_row_id.values():
        assert isinstance(prediction, DashboardPredictionResult)
        assert prediction.predicted_make_probability == pytest.approx(0.417)
        assert prediction.expected_shot_value == pytest.approx(1.251)


def test_predict_prepared_shots_continues_when_one_row_fails() -> None:
    prepared_by_row_id = {
        1: PreparedPredictionFeatures(features={"shot_distance": 10.0}, shot_value=2),
        2: PreparedPredictionFeatures(features={"shot_distance": 20.0}, shot_value=3),
    }
    service = MagicMock(spec=ShotModelService)
    service.is_loaded = True
    service.predict_shot.side_effect = [
        _mock_prediction_response(),
        RuntimeError("Model is not loaded; call load_from_mlflow() first."),
    ]

    result = predict_prepared_shots(prepared_by_row_id, service=service)

    assert set(result.predictions_by_row_id) == {1}
    assert len(result.errors) == 1


def test_predict_prepared_shots_returns_row_level_errors() -> None:
    prepared_by_row_id = {
        7: PreparedPredictionFeatures(features={"shot_distance": 10.0}, shot_value=2),
    }
    service = MagicMock(spec=ShotModelService)
    service.is_loaded = True
    service.predict_shot.side_effect = RuntimeError("boom")

    result = predict_prepared_shots(prepared_by_row_id, service=service)

    assert result.predictions_by_row_id == {}
    assert len(result.errors) == 1
    assert result.errors[0].startswith("Row 7:")


def test_predict_prepared_shots_handles_empty_input() -> None:
    service = MagicMock(spec=ShotModelService)

    result = predict_prepared_shots({}, service=service)

    assert result.predictions_by_row_id == {}
    assert result.errors == []
    service.predict_shot.assert_not_called()
