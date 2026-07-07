"""Model loading and shot prediction for the inference API."""

from __future__ import annotations

import pandas as pd

from courtvision.api.schemas import ShotPredictionResponse
from courtvision.evaluation.expected_value import expected_shot_value
from courtvision.models.common import FEATURE_COLUMNS
from courtvision.models.registry import REGISTERED_MODEL_NAME


class ShotModelService:
    """Load a shot-make model and serve single-shot predictions."""

    def __init__(
        self,
        *,
        model: object | None = None,
        model_name: str = REGISTERED_MODEL_NAME,
    ) -> None:
        self._model = model
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_from_mlflow(
        self,
        *,
        registered_model_name: str = REGISTERED_MODEL_NAME,
        alias: str = "Candidate",
    ) -> None:
        """Load the registered candidate model from MLflow."""
        from courtvision.evaluation.predict_shots import load_candidate_model

        self._model = load_candidate_model(
            registered_model_name=registered_model_name,
            alias=alias,
        )
        self._model_name = registered_model_name

    def predict_shot(
        self,
        features: dict[str, float],
        shot_value: int,
    ) -> ShotPredictionResponse:
        """Return make probability and expected value for one shot."""
        if self._model is None:
            raise RuntimeError("Model is not loaded; call load_from_mlflow() first.")

        feature_frame = self._prepare_feature_frame(features, shot_value)
        probabilities = self._model.predict_proba(feature_frame)
        predicted_make_probability = float(probabilities[0, 1])
        ev = expected_shot_value(predicted_make_probability, shot_value)

        return ShotPredictionResponse(
            predicted_make_probability=predicted_make_probability,
            expected_shot_value=ev,
            model_name=self._model_name,
        )

    def predict_shots(
        self,
        shots: list[tuple[dict[str, float], int]],
    ) -> list[ShotPredictionResponse]:
        """Return make probability and expected value for each shot in order."""
        return [
            self.predict_shot(features, shot_value)
            for features, shot_value in shots
        ]

    def _prepare_feature_frame(
        self,
        features: dict[str, float],
        shot_value: int,
    ) -> pd.DataFrame:
        """Validate inputs and return a single-row frame ordered by FEATURE_COLUMNS."""
        row = dict(features)
        row["shot_value"] = float(shot_value)

        missing = [column for column in FEATURE_COLUMNS if column not in row]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")

        extra = [column for column in row if column not in FEATURE_COLUMNS]
        if extra:
            raise ValueError(f"Unexpected feature columns: {extra}")

        return pd.DataFrame([{column: row[column] for column in FEATURE_COLUMNS}])
