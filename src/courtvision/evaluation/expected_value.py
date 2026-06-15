"""Expected shot value formulas for make-probability predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALLOWED_SHOT_VALUES = (2, 3)


def _validate_probability(predicted_make_probability: float) -> None:
    if predicted_make_probability < 0.0 or predicted_make_probability > 1.0:
        raise ValueError(
            "predicted_make_probability must be between 0 and 1 inclusive, "
            f"got {predicted_make_probability}"
        )


def _validate_shot_value(shot_value: int) -> None:
    if shot_value not in ALLOWED_SHOT_VALUES:
        raise ValueError(
            f"shot_value must be one of {ALLOWED_SHOT_VALUES}, got {shot_value}"
        )


def expected_shot_value(predicted_make_probability: float, shot_value: int) -> float:
    """Return predicted_make_probability * shot_value."""
    _validate_probability(predicted_make_probability)
    _validate_shot_value(shot_value)
    return predicted_make_probability * shot_value


def actual_points(shot_made_flag: bool, shot_value: int) -> float:
    """Return shot_made_flag * shot_value."""
    _validate_shot_value(shot_value)
    return float(shot_made_flag) * shot_value


def points_above_expected(actual: float, expected: float) -> float:
    """Return actual_points - expected_shot_value."""
    return actual - expected


@dataclass(frozen=True)
class ShotExpectedValueMetrics:
    expected_shot_value: float
    actual_points: float
    points_above_expected: float


def compute_shot_expected_value_metrics(
    predicted_make_probability: float,
    shot_value: int,
    shot_made_flag: bool,
) -> ShotExpectedValueMetrics:
    """Compute expected value, actual points, and points above expectation for one shot."""
    expected = expected_shot_value(predicted_make_probability, shot_value)
    actual = actual_points(shot_made_flag, shot_value)
    return ShotExpectedValueMetrics(
        expected_shot_value=expected,
        actual_points=actual,
        points_above_expected=points_above_expected(actual, expected),
    )


def _validate_probabilities_array(predicted_make_probability: np.ndarray) -> None:
    if np.any(predicted_make_probability < 0.0) or np.any(predicted_make_probability > 1.0):
        raise ValueError(
            "predicted_make_probability must be between 0 and 1 inclusive for all shots"
        )


def _validate_shot_values_array(shot_value: np.ndarray) -> None:
    invalid_mask = ~np.isin(shot_value, ALLOWED_SHOT_VALUES)
    if np.any(invalid_mask):
        invalid_values = sorted({int(value) for value in shot_value[invalid_mask]})
        raise ValueError(
            f"shot_value must be one of {ALLOWED_SHOT_VALUES}, got {invalid_values}"
        )


def compute_expected_value_arrays(
    predicted_make_probability: np.ndarray,
    shot_value: np.ndarray,
    shot_made_flag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized expected value, actual points, and points above expectation."""
    probabilities = np.asarray(predicted_make_probability, dtype=float)
    shot_values = np.asarray(shot_value, dtype=int)
    made_flags = np.asarray(shot_made_flag, dtype=bool)

    if probabilities.shape != shot_values.shape or probabilities.shape != made_flags.shape:
        raise ValueError("predicted_make_probability, shot_value, and shot_made_flag must align")

    _validate_probabilities_array(probabilities)
    _validate_shot_values_array(shot_values)

    expected = probabilities * shot_values
    actual = made_flags.astype(float) * shot_values
    return expected, actual, points_above_expected(actual, expected)
