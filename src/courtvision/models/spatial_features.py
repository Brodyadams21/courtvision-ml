"""Court-location spatial encodings for PyTorch MLP training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from courtvision.data.build_features import CORNER_THREE_ABS_LOC_X_MIN, CORNER_THREE_LOC_Y_MAX
from courtvision.models.common import FEATURE_COLUMNS, TARGET_COLUMN

# NBA shot-chart coordinates are in tenths of feet; basket is at (0, 0).
LOC_X_SCALE = 300.0
LOC_Y_SCALE = 470.0
CENTER_X_THRESHOLD = 50.0

RIM_X = 0.0
RIM_Y = 0.0
LEFT_CORNER_X = -float(CORNER_THREE_ABS_LOC_X_MIN)
LEFT_CORNER_Y = float(CORNER_THREE_LOC_Y_MAX)
RIGHT_CORNER_X = float(CORNER_THREE_ABS_LOC_X_MIN)
RIGHT_CORNER_Y = float(CORNER_THREE_LOC_Y_MAX)
TOP_KEY_X = 0.0
TOP_KEY_Y = 280.0
LEFT_WING_X = -220.0
LEFT_WING_Y = 280.0
RIGHT_WING_X = 220.0
RIGHT_WING_Y = 280.0
PAINT_CENTER_X = 0.0
PAINT_CENTER_Y = 95.0

SPATIAL_ENCODING_COLUMNS: tuple[str, ...] = (
    "loc_x_scaled",
    "loc_y_scaled",
    "loc_x_squared",
    "loc_y_squared",
    "xy_interaction",
    "distance_squared",
    "sin_shot_angle",
    "cos_shot_angle",
    "is_left_side",
    "is_right_side",
    "is_center",
    "dist_to_rim",
    "dist_to_left_corner",
    "dist_to_right_corner",
    "dist_to_top_key",
    "dist_to_left_wing",
    "dist_to_right_wing",
    "dist_to_paint_center",
)

SPATIAL_SOURCE_COLUMNS: tuple[str, ...] = (
    "loc_x",
    "loc_y",
    "shot_distance",
    "shot_angle",
)

FEATURE_SET_TABULAR = "tabular"
FEATURE_SET_SPATIAL = "spatial"
FEATURE_SET_CHOICES: tuple[str, ...] = (FEATURE_SET_TABULAR, FEATURE_SET_SPATIAL)


def mlp_feature_columns(feature_set: str) -> list[str]:
    """Return model input columns for tabular (v1) or tabular+spatial (v2) MLP."""
    if feature_set == FEATURE_SET_TABULAR:
        return list(FEATURE_COLUMNS)
    if feature_set == FEATURE_SET_SPATIAL:
        return [*FEATURE_COLUMNS, *SPATIAL_ENCODING_COLUMNS]
    raise ValueError(f"Unsupported feature_set: {feature_set!r}. Use {FEATURE_SET_CHOICES}.")


def _euclidean_distance(
    loc_x: pd.Series | np.ndarray,
    loc_y: pd.Series | np.ndarray,
    landmark_x: float,
    landmark_y: float,
) -> np.ndarray:
    dx = np.asarray(loc_x, dtype=np.float64) - landmark_x
    dy = np.asarray(loc_y, dtype=np.float64) - landmark_y
    return np.sqrt(dx * dx + dy * dy)


def add_spatial_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add court-location spatial encodings derived from loc_x, loc_y, distance, and angle."""
    missing = [col for col in SPATIAL_SOURCE_COLUMNS if col not in frame.columns]
    if missing:
        raise KeyError(f"Frame missing spatial source columns: {missing}")

    enriched = frame.copy()
    loc_x = pd.to_numeric(enriched["loc_x"], errors="coerce").astype("float64")
    loc_y = pd.to_numeric(enriched["loc_y"], errors="coerce").astype("float64")
    shot_distance = pd.to_numeric(enriched["shot_distance"], errors="coerce").astype("float64")
    shot_angle = pd.to_numeric(enriched["shot_angle"], errors="coerce").astype("float64")

    enriched["loc_x_scaled"] = loc_x / LOC_X_SCALE
    enriched["loc_y_scaled"] = loc_y / LOC_Y_SCALE
    enriched["loc_x_squared"] = loc_x * loc_x
    enriched["loc_y_squared"] = loc_y * loc_y
    enriched["xy_interaction"] = loc_x * loc_y
    enriched["distance_squared"] = shot_distance * shot_distance
    enriched["sin_shot_angle"] = np.sin(shot_angle)
    enriched["cos_shot_angle"] = np.cos(shot_angle)
    enriched["is_left_side"] = (loc_x < -CENTER_X_THRESHOLD).astype("float64")
    enriched["is_right_side"] = (loc_x > CENTER_X_THRESHOLD).astype("float64")
    enriched["is_center"] = (loc_x.abs() <= CENTER_X_THRESHOLD).astype("float64")
    enriched["dist_to_rim"] = _euclidean_distance(loc_x, loc_y, RIM_X, RIM_Y)
    enriched["dist_to_left_corner"] = _euclidean_distance(
        loc_x,
        loc_y,
        LEFT_CORNER_X,
        LEFT_CORNER_Y,
    )
    enriched["dist_to_right_corner"] = _euclidean_distance(
        loc_x,
        loc_y,
        RIGHT_CORNER_X,
        RIGHT_CORNER_Y,
    )
    enriched["dist_to_top_key"] = _euclidean_distance(loc_x, loc_y, TOP_KEY_X, TOP_KEY_Y)
    enriched["dist_to_left_wing"] = _euclidean_distance(
        loc_x,
        loc_y,
        LEFT_WING_X,
        LEFT_WING_Y,
    )
    enriched["dist_to_right_wing"] = _euclidean_distance(
        loc_x,
        loc_y,
        RIGHT_WING_X,
        RIGHT_WING_Y,
    )
    enriched["dist_to_paint_center"] = _euclidean_distance(
        loc_x,
        loc_y,
        PAINT_CENTER_X,
        PAINT_CENTER_Y,
    )
    return enriched


def build_mlp_features(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Build the MLP input matrix for tabular or tabular+spatial feature sets."""
    columns = mlp_feature_columns(feature_set)
    if feature_set == FEATURE_SET_TABULAR:
        return frame[columns].copy()
    if feature_set == FEATURE_SET_SPATIAL:
        return add_spatial_features(frame)[columns].copy()
    raise ValueError(f"Unsupported feature_set: {feature_set!r}. Use {FEATURE_SET_CHOICES}.")


def split_mlp_features_target(
    frame: pd.DataFrame,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return MLP feature matrix and binary target for the requested feature set."""
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(f"Frame missing required column: {TARGET_COLUMN}")

    x = build_mlp_features(frame, feature_set)
    y = frame[TARGET_COLUMN]
    return x, y
