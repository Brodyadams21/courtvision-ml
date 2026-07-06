"""Dashboard prediction helpers (Streamlit-free, testable)."""

from __future__ import annotations

from dataclasses import dataclass

from courtvision.api.model_service import ShotModelService
from courtvision.dashboard.data import PreparedPredictionFeatures

PREDICTION_UNAVAILABLE_MESSAGE = (
    "Prediction model is not available locally yet. Train/register a Candidate model "
    "with MLflow, then reload the dashboard."
)


class PredictionUnavailable(Exception):
    """Raised when the dashboard cannot load or use a prediction model."""


@dataclass(frozen=True)
class DashboardPredictionResult:
    predicted_make_probability: float
    expected_shot_value: float
    model_name: str


@dataclass(frozen=True)
class BatchPredictionResult:
    predictions_by_row_id: dict[int, DashboardPredictionResult]
    errors: list[str]


def create_model_service() -> ShotModelService:
    """Create a loaded ``ShotModelService`` from the MLflow Candidate model."""
    service = ShotModelService()
    try:
        service.load_from_mlflow()
    except Exception as exc:
        raise PredictionUnavailable(PREDICTION_UNAVAILABLE_MESSAGE) from exc

    if not service.is_loaded:
        raise PredictionUnavailable(PREDICTION_UNAVAILABLE_MESSAGE)

    return service


def predict_prepared_shot(
    prepared: PreparedPredictionFeatures,
    *,
    service: ShotModelService,
) -> DashboardPredictionResult:
    """Predict make probability and expected value for a prepared test shot."""
    if not service.is_loaded:
        raise PredictionUnavailable(PREDICTION_UNAVAILABLE_MESSAGE)

    try:
        response = service.predict_shot(prepared.features, prepared.shot_value)
    except RuntimeError as exc:
        if "Model is not loaded" in str(exc):
            raise PredictionUnavailable(PREDICTION_UNAVAILABLE_MESSAGE) from exc
        raise

    return DashboardPredictionResult(
        predicted_make_probability=response.predicted_make_probability,
        expected_shot_value=response.expected_shot_value,
        model_name=response.model_name,
    )


def predict_prepared_shots(
    prepared_by_row_id: dict[int, PreparedPredictionFeatures],
    *,
    service: ShotModelService,
) -> BatchPredictionResult:
    """Predict make probability and expected value for a batch of prepared shots."""
    predictions_by_row_id: dict[int, DashboardPredictionResult] = {}
    errors: list[str] = []

    for row_id, prepared in prepared_by_row_id.items():
        try:
            predictions_by_row_id[row_id] = predict_prepared_shot(prepared, service=service)
        except Exception as exc:
            errors.append(f"Row {row_id}: {exc}")

    return BatchPredictionResult(
        predictions_by_row_id=predictions_by_row_id,
        errors=errors,
    )
