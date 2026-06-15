"""Tests for expected shot value formulas."""

from __future__ import annotations

import pytest

from courtvision.evaluation.expected_value import (
    actual_points,
    compute_expected_value_arrays,
    compute_shot_expected_value_metrics,
    expected_shot_value,
    points_above_expected,
)


@pytest.mark.parametrize(
    ("predicted_make_probability", "shot_value", "shot_made_flag", "expected", "actual", "above"),
    [
        (0.40, 3, True, 1.20, 3, 1.80),
        (0.40, 3, False, 1.20, 0, -1.20),
        (0.55, 2, True, 1.10, 2, 0.90),
    ],
)
def test_compute_shot_expected_value_metrics(
    predicted_make_probability: float,
    shot_value: int,
    shot_made_flag: bool,
    expected: float,
    actual: float,
    above: float,
) -> None:
    metrics = compute_shot_expected_value_metrics(
        predicted_make_probability=predicted_make_probability,
        shot_value=shot_value,
        shot_made_flag=shot_made_flag,
    )
    assert metrics.expected_shot_value == pytest.approx(expected)
    assert metrics.actual_points == pytest.approx(actual)
    assert metrics.points_above_expected == pytest.approx(above)


def test_expected_shot_value_formula() -> None:
    assert expected_shot_value(0.40, 3) == pytest.approx(1.20)


def test_actual_points_formula() -> None:
    assert actual_points(True, 3) == pytest.approx(3)
    assert actual_points(False, 3) == pytest.approx(0)


def test_points_above_expected_formula() -> None:
    assert points_above_expected(3, 1.20) == pytest.approx(1.80)
    assert points_above_expected(0, 1.20) == pytest.approx(-1.20)


def test_compute_expected_value_arrays_matches_scalar_examples() -> None:
    import numpy as np

    probabilities = np.array([0.40, 0.40, 0.55], dtype=float)
    shot_values = np.array([3, 3, 2], dtype=int)
    made_flags = np.array([True, False, True], dtype=bool)

    expected, actual, above = compute_expected_value_arrays(
        probabilities,
        shot_values,
        made_flags,
    )

    assert expected.tolist() == pytest.approx([1.20, 1.20, 1.10])
    assert actual.tolist() == pytest.approx([3.0, 0.0, 2.0])
    assert above.tolist() == pytest.approx([1.80, -1.20, 0.90])


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_probability_out_of_range_fails(probability: float) -> None:
    with pytest.raises(ValueError, match="predicted_make_probability"):
        expected_shot_value(probability, 2)

    with pytest.raises(ValueError, match="predicted_make_probability"):
        compute_shot_expected_value_metrics(
            predicted_make_probability=probability,
            shot_value=2,
            shot_made_flag=True,
        )


@pytest.mark.parametrize("shot_value", [0, 1, 4, -2])
def test_invalid_shot_value_fails(shot_value: int) -> None:
    with pytest.raises(ValueError, match="shot_value"):
        expected_shot_value(0.5, shot_value)

    with pytest.raises(ValueError, match="shot_value"):
        actual_points(True, shot_value)

    with pytest.raises(ValueError, match="shot_value"):
        compute_shot_expected_value_metrics(
            predicted_make_probability=0.5,
            shot_value=shot_value,
            shot_made_flag=True,
        )
