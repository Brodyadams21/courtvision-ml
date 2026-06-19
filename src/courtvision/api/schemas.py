"""Request and response schemas for the CourtVision inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ShotPredictionRequest(BaseModel):
    """Input for a single-shot make-probability prediction."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        description="Model feature values keyed by column name.",
    )
    shot_value: Literal[2, 3] = Field(
        description="Point value of the attempt (2 for two-pointer, 3 for three-pointer).",
    )


class ShotPredictionResponse(BaseModel):
    """Model output for a single-shot prediction."""

    model_config = ConfigDict(extra="forbid")

    predicted_make_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated probability the shot is made.",
    )
    expected_shot_value: float = Field(
        ge=0.0,
        description="predicted_make_probability * shot_value.",
    )
    model_name: str = Field(description="Registered model identifier used for inference.")
