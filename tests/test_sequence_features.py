"""Tests for play-by-play sequence feature construction."""

from __future__ import annotations

import pandas as pd
import pytest

from courtvision.models.sequence_features import (
    EVENT_FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    build_shot_sequences,
    classify_event_type_flags,
    event_likely_possession_change,
    event_same_possession_as_shot,
    event_score_change,
    event_seconds_before_shot,
    padded_event_rate,
    parse_game_clock_seconds,
    prepare_play_by_play_events,
)


def test_event_feature_count_is_twenty_nine() -> None:
    assert len(EVENT_FEATURE_COLUMNS) == 29


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


def test_classify_event_type_flags_v2_context() -> None:
    offensive = classify_event_type_flags("Rebound", "Offensive")
    assert offensive["is_rebound"] == 1.0
    assert offensive["is_offensive_rebound"] == 1.0
    assert offensive["is_defensive_rebound"] == 0.0

    defensive = classify_event_type_flags("Rebound", "Defensive")
    assert defensive["is_defensive_rebound"] == 1.0
    assert defensive["is_offensive_rebound"] == 0.0

    steal = classify_event_type_flags("Steal", "")
    assert steal["is_steal"] == 1.0
    assert steal["is_turnover"] == 0.0

    block = classify_event_type_flags("Block", "Jump Shot")
    assert block["is_block"] == 1.0


def test_event_score_change_uses_total_score_delta() -> None:
    assert event_score_change(0, 0, 2, 0) == pytest.approx(2.0)
    assert event_score_change(2, 0, 2, 0) == pytest.approx(0.0)
    assert event_score_change(0, 0, 3, 0) == pytest.approx(3.0)


def test_event_likely_possession_change_flags() -> None:
    events = prepare_play_by_play_events(
        pd.DataFrame(
            {
                "game_id": ["0022400001"] * 4,
                "action_number": [1, 2, 3, 4],
                "period": [1, 1, 1, 1],
                "game_clock": ["PT12M00.00S"] * 4,
                "team_id": [1, 1, 2, 2],
                "is_field_goal": [True, False, False, False],
                "shot_result": ["Made", None, None, None],
                "score_home": [2, 2, 2, 2],
                "score_away": [0, 0, 0, 0],
                "action_type": ["Made Shot", "Missed Shot", "Rebound", "Steal"],
                "sub_type": ["", "Jump Shot", "Defensive", ""],
            }
        )
    )
    assert event_likely_possession_change(events.iloc[0]) == 1.0
    assert event_likely_possession_change(events.iloc[1]) == 0.0
    assert event_likely_possession_change(events.iloc[2]) == 1.0
    assert event_likely_possession_change(events.iloc[3]) == 1.0


def test_event_same_possession_as_shot_breaks_on_later_change() -> None:
    possession_change = [False, True, False]
    assert (
        event_same_possession_as_shot(
            event_index=0,
            num_prior=3,
            shooter_team_id=1,
            event_team_id=1,
            possession_change_by_index=possession_change,
        )
        == 0.0
    )
    assert (
        event_same_possession_as_shot(
            event_index=2,
            num_prior=3,
            shooter_team_id=1,
            event_team_id=1,
            possession_change_by_index=possession_change,
        )
        == 1.0
    )


def test_event_seconds_before_shot_uses_game_clock_delta() -> None:
    assert event_seconds_before_shot(2740.0, 2760.0) == pytest.approx(20.0)
    assert event_seconds_before_shot(2760.0, 2760.0) == pytest.approx(0.0)
    assert event_seconds_before_shot(2770.0, 2760.0) == pytest.approx(0.0)


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

    seq = result.sequences[0]
    feat = EVENT_FEATURE_COLUMNS.index

    assert result.sequences.shape == (1, SEQUENCE_LENGTH, len(EVENT_FEATURE_COLUMNS))
    assert result.leakage_check_passed
    assert result.max_prior_action_numbers[0] == 9
    assert seq[-1, feat("event_order_from_shot")] == -1.0
    assert seq[-1, feat("is_missed_shot_event")] == 1.0
    # Shot event at action 10 must not appear in the sequence.
    assert seq[-1, feat("is_made_shot_event")] == 0.0
    assert seq[-2, feat("is_defensive_rebound")] == 1.0
    assert seq[-2, feat("event_team_is_opponent")] == 1.0
    assert seq[-2, feat("event_score_change")] == 0.0
    assert seq[-1, feat("event_same_possession_as_shot")] == 1.0
    assert seq[-2, feat("event_likely_possession_change")] == 1.0
    assert seq[-1, feat("event_seconds_before_shot")] == pytest.approx(20.0)
    assert seq[-2, feat("event_seconds_before_shot")] == pytest.approx(10.0)


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
