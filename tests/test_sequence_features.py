"""Tests for play-by-play sequence feature construction."""

from __future__ import annotations

import pandas as pd
import pytest

from courtvision.models.sequence_features import (
    EVENT_FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    build_shot_sequences,
    classify_event_type_flags,
    padded_event_rate,
    parse_game_clock_seconds,
    prepare_play_by_play_events,
)


def test_parse_game_clock_seconds() -> None:
    assert parse_game_clock_seconds("PT11M43.00S") == pytest.approx(11 * 60 + 43.0)
    assert parse_game_clock_seconds("PT05.20S") == pytest.approx(5.2)


def test_classify_event_type_flags_made_shot() -> None:
    flags = classify_event_type_flags(
        "Made Shot",
        "Jump Shot",
        is_field_goal=True,
        shot_result="Made",
    )
    assert flags["is_made_shot_event"] == 1.0
    assert flags["is_unknown_event"] == 0.0


def test_build_shot_sequences_uses_only_prior_events() -> None:
    shots = pd.DataFrame(
        {
            "shot_id": [1],
            "game_id": ["0022400001"],
            "game_event_id": [10],
            "team_id": [1],
            "period": [1],
            "minutes_remaining": [10],
            "seconds_remaining": [0],
            "home_team_id": [1],
            "away_team_id": [2],
        }
    )
    play_by_play = pd.DataFrame(
        {
            "game_id": ["0022400001"] * 4,
            "action_number": [7, 8, 9, 10],
            "period": [1, 1, 1, 1],
            "game_clock": ["PT10M00.00S", "PT09M50.00S", "PT09M40.00S", "PT09M30.00S"],
            "team_id": [1, 2, 1, 1],
            "is_field_goal": [False, False, True, True],
            "shot_result": [None, None, "Missed", "Made"],
            "score_home": [0, 0, 0, 2],
            "score_away": [0, 0, 0, 0],
            "action_type": ["Foul", "Rebound", "Missed Shot", "Made Shot"],
            "sub_type": ["", "", "Jump Shot", "Layup"],
        }
    )

    result = build_shot_sequences(shots, play_by_play)

    assert result.sequences.shape == (1, SEQUENCE_LENGTH, len(EVENT_FEATURE_COLUMNS))
    assert result.leakage_check_passed
    assert result.max_prior_action_numbers[0] == 9
    assert result.sequences[0, -1, EVENT_FEATURE_COLUMNS.index("event_order_from_shot")] == -1.0
    assert result.sequences[0, -1, EVENT_FEATURE_COLUMNS.index("is_missed_shot_event")] == 1.0
    # Shot event at action 10 must not appear in the sequence.
    assert result.sequences[0, -1, EVENT_FEATURE_COLUMNS.index("is_made_shot_event")] == 0.0


def test_build_shot_sequences_pads_when_fewer_than_five_prior_events() -> None:
    shots = pd.DataFrame(
        {
            "shot_id": [1],
            "game_id": ["0022400001"],
            "game_event_id": [3],
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
            "game_id": ["0022400001", "0022400001"],
            "action_number": [1, 2],
            "period": [1, 1],
            "game_clock": ["PT11M30.00S", "PT11M20.00S"],
            "team_id": [1, 2],
            "is_field_goal": [False, False],
            "shot_result": [None, None],
            "score_home": [0, 0],
            "score_away": [0, 0],
            "action_type": ["Jump Ball", "Turnover"],
            "sub_type": ["", ""],
        }
    )

    result = build_shot_sequences(shots, play_by_play)

    assert result.sequences[0, 0, 0] == 0.0
    assert result.sequences[0, -1, 0] == -1.0
    assert padded_event_rate(result) == pytest.approx(3 / 5)


def test_prepare_play_by_play_events_adds_prior_scores() -> None:
    play_by_play = pd.DataFrame(
        {
            "game_id": ["0022400001", "0022400001"],
            "action_number": [1, 2],
            "period": [1, 1],
            "game_clock": ["PT12M00.00S", "PT11M50.00S"],
            "team_id": [1, 2],
            "is_field_goal": [True, False],
            "shot_result": ["Made", None],
            "score_home": [2, 2],
            "score_away": [0, 0],
            "action_type": ["Made Shot", "Rebound"],
            "sub_type": ["", ""],
        }
    )

    events = prepare_play_by_play_events(play_by_play)

    assert events.loc[0, "prior_score_home"] == 0
    assert events.loc[1, "prior_score_home"] == 2
