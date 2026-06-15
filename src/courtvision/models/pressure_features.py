"""Shot-clock pressure proxies and sequence summary features for GRU tabular branch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from courtvision.models.sequence_features import EVENT_FEATURE_COLUMNS, SEQUENCE_LENGTH
from courtvision.models.spatial_features import FEATURE_SET_SPATIAL, mlp_feature_columns

# NBA shot clock length; used to scale possession-age into a 0–1 pressure proxy.
POSSESSION_SHOT_CLOCK_SECONDS = 24.0

SHOT_PRESSURE_FEATURE_COLUMNS: tuple[str, ...] = (
    "shot_seconds_remaining_period",
    "late_clock_proxy",
    "sequence_contains_timeout",
    "sequence_contains_offensive_rebound",
    "sequence_contains_turnover",
)

SEQUENCE_SUMMARY_COUNT_COLUMNS: tuple[str, ...] = (
    "prior_5_score_change_total",
    "prior_5_turnover_count",
    "prior_5_steal_count",
    "prior_5_off_rebound_count",
    "prior_5_def_rebound_count",
    "prior_5_foul_count",
    "prior_5_same_team_event_count",
    "prior_5_opponent_event_count",
)

GRU_TABULAR_EXTRA_COLUMNS: tuple[str, ...] = (
    *SHOT_PRESSURE_FEATURE_COLUMNS,
    *SEQUENCE_SUMMARY_COUNT_COLUMNS,
)

_EVENT_ORDER_INDEX = EVENT_FEATURE_COLUMNS.index("event_order_from_shot")
_EVENT_SECONDS_BEFORE_SHOT_INDEX = EVENT_FEATURE_COLUMNS.index("event_seconds_before_shot")
_EVENT_SAME_POSSESSION_INDEX = EVENT_FEATURE_COLUMNS.index("event_same_possession_as_shot")
_EVENT_SCORE_CHANGE_INDEX = EVENT_FEATURE_COLUMNS.index("event_score_change")
_SAME_TEAM_INDEX = EVENT_FEATURE_COLUMNS.index("same_team_as_shooter")
_OPPONENT_TEAM_INDEX = EVENT_FEATURE_COLUMNS.index("event_team_is_opponent")
_IS_TIMEOUT_INDEX = EVENT_FEATURE_COLUMNS.index("is_timeout")
_IS_OFFENSIVE_REBOUND_INDEX = EVENT_FEATURE_COLUMNS.index("is_offensive_rebound")
_IS_DEFENSIVE_REBOUND_INDEX = EVENT_FEATURE_COLUMNS.index("is_defensive_rebound")
_IS_TURNOVER_INDEX = EVENT_FEATURE_COLUMNS.index("is_turnover")
_IS_STEAL_INDEX = EVENT_FEATURE_COLUMNS.index("is_steal")
_IS_FOUL_INDEX = EVENT_FEATURE_COLUMNS.index("is_foul")


def gru_tabular_feature_columns() -> list[str]:
    """Spatial tabular features plus sequence-derived pressure and summary columns."""
    return [*mlp_feature_columns(FEATURE_SET_SPATIAL), *GRU_TABULAR_EXTRA_COLUMNS]


def shot_seconds_remaining_period_from_frame(frame: pd.DataFrame) -> np.ndarray:
    """Period game clock (seconds) at the shot; mirrors gold ``seconds_remaining_period``."""
    if "seconds_remaining_period" not in frame.columns:
        raise KeyError("Frame missing required column: seconds_remaining_period")
    return pd.to_numeric(
        frame["seconds_remaining_period"], errors="coerce"
    ).to_numpy(dtype=np.float32)


def _non_padded_event_mask(sequences: np.ndarray) -> np.ndarray:
    if sequences.ndim != 3 or sequences.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected sequences (n, {SEQUENCE_LENGTH}, features), got {sequences.shape}"
        )
    if sequences.shape[2] != len(EVENT_FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(EVENT_FEATURE_COLUMNS)} event features, got {sequences.shape[2]}"
        )
    return sequences[:, :, _EVENT_ORDER_INDEX] != 0.0


def late_clock_proxy_from_sequences(sequences: np.ndarray) -> np.ndarray:
    """Possession-age proxy for shot-clock pressure, scaled to [0, 1] over 24 seconds."""
    active = _non_padded_event_mask(sequences)
    seconds_before_shot = sequences[:, :, _EVENT_SECONDS_BEFORE_SHOT_INDEX]
    same_possession = sequences[:, :, _EVENT_SAME_POSSESSION_INDEX] == 1.0

    same_possession_active = active & same_possession
    has_same_possession = same_possession_active.any(axis=1)

    same_possession_seconds = np.where(same_possession_active, seconds_before_shot, -np.inf)
    possession_age_same = same_possession_seconds.max(axis=1)
    fallback_age = np.max(np.where(active, seconds_before_shot, 0.0), axis=1)
    possession_age = np.where(has_same_possession, possession_age_same, fallback_age)
    possession_age = np.maximum(possession_age, 0.0)

    return np.minimum(possession_age / POSSESSION_SHOT_CLOCK_SECONDS, 1.0).astype(np.float32)


def sequence_summary_flags_from_sequences(sequences: np.ndarray) -> dict[str, np.ndarray]:
    """Binary flags for whether the prior-event window contains key possession events."""
    active = _non_padded_event_mask(sequences)

    def _contains(flag_index: int) -> np.ndarray:
        flagged = sequences[:, :, flag_index] * active
        return (flagged.max(axis=1) >= 1.0).astype(np.float32)

    return {
        "sequence_contains_timeout": _contains(_IS_TIMEOUT_INDEX),
        "sequence_contains_offensive_rebound": _contains(_IS_OFFENSIVE_REBOUND_INDEX),
        "sequence_contains_turnover": _contains(_IS_TURNOVER_INDEX),
    }


def sequence_summary_counts_from_sequences(sequences: np.ndarray) -> dict[str, np.ndarray]:
    """Aggregate counts from the prior five play-by-play events."""
    active = _non_padded_event_mask(sequences)

    def _sum_feature(feature_index: int) -> np.ndarray:
        return (sequences[:, :, feature_index] * active).sum(axis=1).astype(np.float32)

    return {
        "prior_5_score_change_total": _sum_feature(_EVENT_SCORE_CHANGE_INDEX),
        "prior_5_turnover_count": _sum_feature(_IS_TURNOVER_INDEX),
        "prior_5_steal_count": _sum_feature(_IS_STEAL_INDEX),
        "prior_5_off_rebound_count": _sum_feature(_IS_OFFENSIVE_REBOUND_INDEX),
        "prior_5_def_rebound_count": _sum_feature(_IS_DEFENSIVE_REBOUND_INDEX),
        "prior_5_foul_count": _sum_feature(_IS_FOUL_INDEX),
        "prior_5_same_team_event_count": _sum_feature(_SAME_TEAM_INDEX),
        "prior_5_opponent_event_count": _sum_feature(_OPPONENT_TEAM_INDEX),
    }


def build_shot_pressure_features(
    frame: pd.DataFrame,
    sequences: np.ndarray,
) -> pd.DataFrame:
    """Build GRU tabular extras aligned row-for-row with ``frame`` and ``sequences``."""
    if len(frame) != len(sequences):
        raise ValueError(
            f"Frame rows ({len(frame)}) must match sequence rows ({len(sequences)})"
        )

    summary_flags = sequence_summary_flags_from_sequences(sequences)
    summary_counts = sequence_summary_counts_from_sequences(sequences)
    return pd.DataFrame(
        {
            "shot_seconds_remaining_period": shot_seconds_remaining_period_from_frame(frame),
            "late_clock_proxy": late_clock_proxy_from_sequences(sequences),
            **summary_flags,
            **summary_counts,
        },
        index=frame.index,
    )


def attach_shot_pressure_features(
    tabular_features: pd.DataFrame,
    shot_frame: pd.DataFrame,
    sequences: np.ndarray,
) -> pd.DataFrame:
    """Append sequence-derived tabular extras to an existing feature matrix."""
    extras = build_shot_pressure_features(shot_frame, sequences)
    tabular = tabular_features.reset_index(drop=True)
    return pd.concat([tabular, extras.reset_index(drop=True)], axis=1)
