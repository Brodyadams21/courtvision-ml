"""Tests for MLP spatial feature encodings."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from courtvision.models.common import FEATURE_COLUMNS, TARGET_COLUMN
from courtvision.models.spatial_features import (
    FEATURE_SET_SPATIAL,
    FEATURE_SET_TABULAR,
    LEFT_CORNER_X,
    LEFT_CORNER_Y,
    SPATIAL_ENCODING_COLUMNS,
    add_spatial_features,
    build_mlp_features,
    mlp_feature_columns,
    split_mlp_features_target,
)


def _shot_row(**overrides: object) -> dict[str, object]:
    row = {col: 0.0 for col in FEATURE_COLUMNS}
    row.update(
        {
            TARGET_COLUMN: True,
            "loc_x": 0,
            "loc_y": 100,
            "shot_distance": 10,
            "shot_angle": 0.0,
            "is_home": True,
            "is_corner_three": False,
            "score_margin_missing": False,
        }
    )
    row.update(overrides)
    return row


def test_mlp_feature_columns_tabular_and_spatial() -> None:
    assert mlp_feature_columns(FEATURE_SET_TABULAR) == list(FEATURE_COLUMNS)
    assert len(mlp_feature_columns(FEATURE_SET_SPATIAL)) == len(FEATURE_COLUMNS) + len(
        SPATIAL_ENCODING_COLUMNS
    )


def test_add_spatial_features_at_rim() -> None:
    frame = pd.DataFrame([_shot_row(loc_x=0, loc_y=0, shot_distance=0, shot_angle=0.0)])
    spatial = add_spatial_features(frame)

    assert spatial.loc[0, "loc_x_scaled"] == 0.0
    assert spatial.loc[0, "loc_y_scaled"] == 0.0
    assert spatial.loc[0, "distance_squared"] == 0.0
    assert spatial.loc[0, "sin_shot_angle"] == pytest.approx(0.0)
    assert spatial.loc[0, "cos_shot_angle"] == pytest.approx(1.0)
    assert spatial.loc[0, "is_center"] == 1.0
    assert spatial.loc[0, "dist_to_rim"] == pytest.approx(0.0)


def test_add_spatial_features_left_side_and_corner_distance() -> None:
    frame = pd.DataFrame(
        [
            _shot_row(
                loc_x=LEFT_CORNER_X,
                loc_y=LEFT_CORNER_Y,
                shot_distance=23,
                shot_angle=math.atan2(LEFT_CORNER_X, LEFT_CORNER_Y),
            )
        ]
    )
    spatial = add_spatial_features(frame)

    assert spatial.loc[0, "is_left_side"] == 1.0
    assert spatial.loc[0, "is_right_side"] == 0.0
    assert spatial.loc[0, "dist_to_left_corner"] == pytest.approx(0.0)
    assert spatial.loc[0, "dist_to_right_corner"] > 0.0


def test_build_mlp_features_spatial_includes_all_columns() -> None:
    frame = pd.DataFrame([_shot_row()])
    features = build_mlp_features(frame, FEATURE_SET_SPATIAL)
    expected_columns = mlp_feature_columns(FEATURE_SET_SPATIAL)
    assert list(features.columns) == expected_columns


def test_split_mlp_features_target_returns_target_series() -> None:
    frame = pd.DataFrame([_shot_row(), _shot_row(**{TARGET_COLUMN: False})])
    x, y = split_mlp_features_target(frame, FEATURE_SET_TABULAR)
    assert len(x.columns) == len(FEATURE_COLUMNS)
    assert list(y) == [True, False]
