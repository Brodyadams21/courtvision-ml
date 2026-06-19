"""Tests for CourtVision API request/response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from courtvision.api.schemas import ShotPredictionRequest, ShotPredictionResponse


def test_shot_prediction_request_accepts_valid_payload() -> None:
    request = ShotPredictionRequest(features={"shot_distance": 12.5}, shot_value=2)

    assert request.features == {"shot_distance": 12.5}
    assert request.shot_value == 2


@pytest.mark.parametrize("shot_value", [1, 4, 0])
def test_shot_prediction_request_rejects_invalid_shot_value(shot_value: int) -> None:
    with pytest.raises(ValidationError):
        ShotPredictionRequest(features={"shot_distance": 12.5}, shot_value=shot_value)


def test_shot_prediction_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ShotPredictionRequest(
            features={"shot_distance": 12.5},
            shot_value=3,
            extra_field="nope",
        )


def test_shot_prediction_response_accepts_valid_payload() -> None:
    response = ShotPredictionResponse(
        predicted_make_probability=0.42,
        expected_shot_value=1.26,
        model_name="courtvision-shot-make-model",
    )

    assert response.predicted_make_probability == pytest.approx(0.42)
    assert response.expected_shot_value == pytest.approx(1.26)
    assert response.model_name == "courtvision-shot-make-model"


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_shot_prediction_response_rejects_out_of_range_probability(probability: float) -> None:
    with pytest.raises(ValidationError):
        ShotPredictionResponse(
            predicted_make_probability=probability,
            expected_shot_value=1.0,
            model_name="courtvision-shot-make-model",
        )
