"""Tests for pre-shot score_margin attachment (no label leakage)."""

from __future__ import annotations

import pandas as pd

from courtvision.data.build_features import (
    CORE_SHOT_FEATURE_COLUMNS,
    SCORE_MARGIN_COLUMN,
    SCORE_MARGIN_MISSING_COLUMN,
    attach_score_margin,
)


def _minimal_shot_features(**overrides) -> pd.DataFrame:
    row = {
        "shot_id": 1,
        "game_id": "0022400001",
        "game_event_id": 10,
        "player_id": 100,
        "team_id": 1,
        "opponent_team_id": 2,
        "season": "2024-25",
        "game_date": pd.Timestamp("2024-10-01"),
        "shot_made_flag": True,
        "shot_value": 3,
        "shot_distance": 24,
        "loc_x": 100,
        "loc_y": 200,
        "abs_loc_x": 100,
        "shot_angle": 0.5,
        "is_corner_three": False,
        "is_home": True,
        "period": 1,
        "seconds_remaining_period": 600.0,
        "seconds_remaining_game": 600.0,
        "shot_zone_basic": "Above the Break 3",
        "shot_zone_area": "Center(C)",
        "shot_zone_range": "24+ ft.",
        "player_recent_fg_pct_5": 0.45,
        "player_recent_fg3_pct_5": 0.35,
        "player_recent_fga_5": 12.0,
        "player_recent_fg3a_5": 4.0,
        "player_recent_minutes_5": 30.0,
        "player_recent_points_5": 15.0,
        "team_recent_off_eff_proxy_5": 1.05,
        "team_recent_pace_proxy_5": 98.0,
        "team_recent_fg_pct_5": 0.47,
        "team_recent_three_point_rate_5": 0.38,
        "team_recent_fga_5": 88.0,
        "team_recent_points_5": 110.0,
        "team_recent_turnovers_5": 12.0,
        "opp_recent_points_allowed_5": 108.0,
        "opp_recent_fg_pct_allowed_5": 0.46,
        "opp_recent_three_point_rate_allowed_5": 0.36,
        "opp_recent_pace_proxy_5": 97.0,
        "opp_recent_fga_allowed_5": 86.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])[list(CORE_SHOT_FEATURE_COLUMNS)]


def test_attach_score_margin_uses_prior_event_not_shot_outcome() -> None:
    """Made and missed shots at the same event get the same pre-shot margin."""
    features = pd.concat(
        [
            _minimal_shot_features(shot_id=1, shot_made_flag=True, shot_value=3),
            _minimal_shot_features(shot_id=2, shot_made_flag=False, shot_value=3),
        ],
        ignore_index=True,
    )
    pbp_scores = pd.DataFrame(
        {
            "game_id": ["0022400001", "0022400001"],
            "action_number": [9, 10],
            "score_home": [50, 53],
            "score_away": [48, 48],
        }
    )

    result = attach_score_margin(features, pbp_scores)

    assert result.loc[0, SCORE_MARGIN_COLUMN] == 2
    assert result.loc[1, SCORE_MARGIN_COLUMN] == 2
    assert not result.loc[0, SCORE_MARGIN_MISSING_COLUMN]
    assert not result.loc[1, SCORE_MARGIN_MISSING_COLUMN]


def test_attach_score_margin_away_team_perspective() -> None:
    features = _minimal_shot_features(is_home=False, game_event_id=20)
    pbp_scores = pd.DataFrame(
        {
            "game_id": ["0022400001", "0022400001"],
            "action_number": [19, 20],
            "score_home": [60, 60],
            "score_away": [55, 58],
        }
    )

    result = attach_score_margin(features, pbp_scores)

    assert result.loc[0, SCORE_MARGIN_COLUMN] == -5
    assert not result.loc[0, SCORE_MARGIN_MISSING_COLUMN]


def test_attach_score_margin_unmatched_shot_is_null_with_missing_flag() -> None:
    features = _minimal_shot_features(game_event_id=999)
    pbp_scores = pd.DataFrame(
        {
            "game_id": ["0022400001"],
            "action_number": [10],
            "score_home": [50],
            "score_away": [48],
        }
    )

    result = attach_score_margin(features, pbp_scores)

    assert pd.isna(result.loc[0, SCORE_MARGIN_COLUMN])
    assert result.loc[0, SCORE_MARGIN_MISSING_COLUMN]


def test_attach_score_margin_first_event_uses_zero_prior() -> None:
    features = _minimal_shot_features(game_event_id=1)
    pbp_scores = pd.DataFrame(
        {
            "game_id": ["0022400001"],
            "action_number": [1],
            "score_home": [0],
            "score_away": [0],
        }
    )

    result = attach_score_margin(features, pbp_scores)

    assert result.loc[0, SCORE_MARGIN_COLUMN] == 0
    assert not result.loc[0, SCORE_MARGIN_MISSING_COLUMN]
