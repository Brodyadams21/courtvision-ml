"""Tests for shot-clock pressure and sequence summary features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from courtvision.models.common import FEATURE_COLUMNS
from courtvision.models.pressure_features import (
    GRU_TABULAR_EXTRA_COLUMNS,
    POSSESSION_SHOT_CLOCK_SECONDS,
    SEQUENCE_SUMMARY_COUNT_COLUMNS,
    SHOT_PRESSURE_FEATURE_COLUMNS,
    attach_shot_pressure_features,
    build_shot_pressure_features,
    gru_tabular_feature_columns,
    late_clock_proxy_from_sequences,
    sequence_summary_counts_from_sequences,
    sequence_summary_flags_from_sequences,
    shot_seconds_remaining_period_from_frame,
)
from courtvision.models.sequence_features import (
    EVENT_FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    build_shot_sequences,
)
from courtvision.models.spatial_features import (
    FEATURE_SET_SPATIAL,
    build_mlp_features,
    mlp_feature_columns,
)

_FIVE_EVENT_GAME_CLOCK = [
    "PT12M00.00S",
    "PT11M55.00S",
    "PT11M50.00S",
    "PT11M45.00S",
    "PT11M40.00S",
]


def _pressure_frame(seconds_remaining_period: float = 600.0) -> pd.DataFrame:
    row = {col: 0.0 for col in FEATURE_COLUMNS}
    row["seconds_remaining_period"] = seconds_remaining_period
    return pd.DataFrame([row])


def test_gru_tabular_feature_columns_include_sequence_extras() -> None:
    spatial_columns = mlp_feature_columns(FEATURE_SET_SPATIAL)
    gru_columns = gru_tabular_feature_columns()
    assert gru_columns[: len(spatial_columns)] == spatial_columns
    assert gru_columns[len(spatial_columns) :] == list(GRU_TABULAR_EXTRA_COLUMNS)
    assert list(SHOT_PRESSURE_FEATURE_COLUMNS) + list(SEQUENCE_SUMMARY_COUNT_COLUMNS) == list(
        GRU_TABULAR_EXTRA_COLUMNS
    )


def test_shot_seconds_remaining_period_from_frame() -> None:
    frame = _pressure_frame(573.5)
    values = shot_seconds_remaining_period_from_frame(frame)
    assert values[0] == pytest.approx(573.5)


def test_late_clock_proxy_uses_same_possession_age() -> None:
    sequences = np.zeros((1, SEQUENCE_LENGTH, len(EVENT_FEATURE_COLUMNS)), dtype=np.float32)
    order_idx = EVENT_FEATURE_COLUMNS.index("event_order_from_shot")
    before_idx = EVENT_FEATURE_COLUMNS.index("event_seconds_before_shot")
    same_poss_idx = EVENT_FEATURE_COLUMNS.index("event_same_possession_as_shot")

    sequences[0, -2, order_idx] = -2.0
    sequences[0, -2, before_idx] = 8.0
    sequences[0, -2, same_poss_idx] = 1.0
    sequences[0, -1, order_idx] = -1.0
    sequences[0, -1, before_idx] = 20.0
    sequences[0, -1, same_poss_idx] = 1.0

    proxy = late_clock_proxy_from_sequences(sequences)
    assert proxy[0] == pytest.approx(20.0 / POSSESSION_SHOT_CLOCK_SECONDS)


def test_sequence_summary_flags_detect_prior_events() -> None:
    shots = pd.DataFrame(
        {
            "shot_id": [1],
            "game_id": ["0022400001"],
            "game_event_id": [6],
            "team_id": [1],
            "period": [1],
            "minutes_remaining": [11],
            "seconds_remaining": [0],
            "home_team_id": [1],
            "away_team_id": [2],
        }
    )
    play_by_play = pd.DataFrame(
        {
            "game_id": ["0022400001"] * 5,
            "action_number": [1, 2, 3, 4, 5],
            "period": [1, 1, 1, 1, 1],
            "game_clock": _FIVE_EVENT_GAME_CLOCK,
            "team_id": [1, 1, 2, 1, 1],
            "is_field_goal": [False, False, False, False, True],
            "shot_result": [None, None, None, None, "Made"],
            "score_home": [0, 0, 0, 0, 2],
            "score_away": [0, 0, 0, 0, 0],
            "action_type": ["Timeout", "Rebound", "Turnover", "Rebound", "Made Shot"],
            "sub_type": ["", "Offensive", "", "Defensive", "Layup"],
        }
    )

    result = build_shot_sequences(shots, play_by_play)
    flags = sequence_summary_flags_from_sequences(result.sequences)

    assert flags["sequence_contains_timeout"][0] == 1.0
    assert flags["sequence_contains_offensive_rebound"][0] == 1.0
    assert flags["sequence_contains_turnover"][0] == 1.0


def test_sequence_summary_counts_aggregate_prior_events() -> None:
    shots = pd.DataFrame(
        {
            "shot_id": [1],
            "game_id": ["0022400001"],
            "game_event_id": [6],
            "team_id": [1],
            "period": [1],
            "minutes_remaining": [11],
            "seconds_remaining": [0],
            "home_team_id": [1],
            "away_team_id": [2],
        }
    )
    play_by_play = pd.DataFrame(
        {
            "game_id": ["0022400001"] * 5,
            "action_number": [1, 2, 3, 4, 5],
            "period": [1, 1, 1, 1, 1],
            "game_clock": _FIVE_EVENT_GAME_CLOCK,
            "team_id": [1, 1, 2, 1, 1],
            "is_field_goal": [False, False, False, False, True],
            "shot_result": [None, None, None, None, "Made"],
            "score_home": [0, 0, 0, 0, 2],
            "score_away": [0, 0, 0, 0, 0],
            "action_type": ["Timeout", "Rebound", "Turnover", "Rebound", "Made Shot"],
            "sub_type": ["", "Offensive", "", "Defensive", "Layup"],
        }
    )

    result = build_shot_sequences(shots, play_by_play)
    counts = sequence_summary_counts_from_sequences(result.sequences)

    assert counts["prior_5_score_change_total"][0] == pytest.approx(2.0)
    assert counts["prior_5_turnover_count"][0] == pytest.approx(1.0)
    assert counts["prior_5_steal_count"][0] == pytest.approx(0.0)
    assert counts["prior_5_off_rebound_count"][0] == pytest.approx(1.0)
    assert counts["prior_5_def_rebound_count"][0] == pytest.approx(1.0)
    assert counts["prior_5_foul_count"][0] == pytest.approx(0.0)
    assert counts["prior_5_same_team_event_count"][0] == pytest.approx(4.0)
    assert counts["prior_5_opponent_event_count"][0] == pytest.approx(1.0)


def test_build_shot_pressure_features_aligns_with_frame() -> None:
    frame = _pressure_frame(590.0)
    sequences = np.zeros((1, SEQUENCE_LENGTH, len(EVENT_FEATURE_COLUMNS)), dtype=np.float32)
    pressure = build_shot_pressure_features(frame, sequences)

    assert list(pressure.columns) == list(GRU_TABULAR_EXTRA_COLUMNS)
    assert pressure.loc[0, "shot_seconds_remaining_period"] == pytest.approx(590.0)
    assert pressure.loc[0, "late_clock_proxy"] == pytest.approx(0.0)
    assert pressure.loc[0, "sequence_contains_timeout"] == pytest.approx(0.0)
    assert pressure.loc[0, "prior_5_turnover_count"] == pytest.approx(0.0)


def test_attach_shot_pressure_features_appends_columns() -> None:
    frame = _pressure_frame()
    tabular = build_mlp_features(frame, FEATURE_SET_SPATIAL)
    sequences = np.zeros((1, SEQUENCE_LENGTH, len(EVENT_FEATURE_COLUMNS)), dtype=np.float32)

    enriched = attach_shot_pressure_features(tabular, frame, sequences)

    assert len(enriched.columns) == len(tabular.columns) + len(GRU_TABULAR_EXTRA_COLUMNS)
    assert list(enriched.columns[-len(GRU_TABULAR_EXTRA_COLUMNS) :]) == list(
        GRU_TABULAR_EXTRA_COLUMNS
    )
