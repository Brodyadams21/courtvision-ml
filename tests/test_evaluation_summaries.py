"""Tests for evaluation summary tables."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from courtvision.evaluation.summaries import (
    MIN_PLAYER_ATTEMPTS_FOR_INSIGHTS,
    MIN_ZONE_ATTEMPTS_FOR_INSIGHTS,
    PLAYER_SUMMARY_COLUMNS,
    PLAYER_TREND_COLUMNS,
    TEAM_SUMMARY_COLUMNS,
    ZONE_SUMMARY_COLUMNS,
    filter_players_for_insights,
    filter_teams_for_insights,
    filter_zones_for_insights,
    load_shot_predictions,
    run_summaries,
    save_summary_csv,
    summarize_player_trends,
    summarize_players,
    summarize_teams,
    summarize_zones,
)


def _shot_predictions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "shot_id": 1,
                "game_id": 100,
                "player_id": 10,
                "team_id": 100,
                "shot_value": 3,
                "shot_made_flag": True,
                "predicted_make_probability": 0.40,
                "expected_shot_value": 1.20,
                "actual_points": 3.0,
                "points_above_expected": 1.80,
                "shot_zone_basic": "Right Corner 3",
                "shot_zone_area": "Right Side(R)",
                "shot_zone_range": "24+ ft.",
                "shot_distance": 23.0,
                "game_date": "2025-01-15",
            },
            {
                "shot_id": 2,
                "game_id": 100,
                "player_id": 10,
                "team_id": 100,
                "shot_value": 2,
                "shot_made_flag": False,
                "predicted_make_probability": 0.50,
                "expected_shot_value": 1.00,
                "actual_points": 0.0,
                "points_above_expected": -1.00,
                "shot_zone_basic": "Mid-Range",
                "shot_zone_area": "Center(C)",
                "shot_zone_range": "16-24 ft.",
                "shot_distance": 18.0,
                "game_date": "2025-02-10",
            },
            {
                "shot_id": 3,
                "game_id": 200,
                "player_id": 20,
                "team_id": 200,
                "shot_value": 3,
                "shot_made_flag": True,
                "predicted_make_probability": 0.60,
                "expected_shot_value": 1.80,
                "actual_points": 3.0,
                "points_above_expected": 1.20,
                "shot_zone_basic": "Right Corner 3",
                "shot_zone_area": "Right Side(R)",
                "shot_zone_range": "24+ ft.",
                "shot_distance": 24.0,
                "game_date": "2025-02-20",
            },
        ]
    )


def test_summarize_players_aggregates_metrics() -> None:
    summary = summarize_players(_shot_predictions_frame())
    player_10 = summary.loc[summary["player_id"] == 10].iloc[0]

    assert list(summary.columns) == list(PLAYER_SUMMARY_COLUMNS)
    assert player_10["attempts"] == 2
    assert player_10["made_shots"] == 1
    assert player_10["actual_fg_pct"] == pytest.approx(0.5)
    assert player_10["expected_fg_pct"] == pytest.approx(0.45)
    assert player_10["actual_points"] == pytest.approx(3.0)
    assert player_10["expected_points"] == pytest.approx(2.2)
    assert player_10["points_above_expected"] == pytest.approx(0.8)
    assert player_10["points_above_expected_per_100_shots"] == pytest.approx(40.0)
    assert player_10["avg_expected_shot_value"] == pytest.approx(1.1)
    assert player_10["avg_shot_value"] == pytest.approx(2.5)


def test_summarize_teams_aggregates_metrics() -> None:
    summary = summarize_teams(_shot_predictions_frame())
    team_100 = summary.loc[summary["team_id"] == 100].iloc[0]

    assert list(summary.columns) == list(TEAM_SUMMARY_COLUMNS)
    assert team_100["attempts"] == 2
    assert team_100["made_shots"] == 1
    assert team_100["actual_fg_pct"] == pytest.approx(0.5)
    assert team_100["expected_fg_pct"] == pytest.approx(0.45)
    assert team_100["points_above_expected_per_100_shots"] == pytest.approx(40.0)


def test_summarize_zones_aggregates_by_zone_combination() -> None:
    summary = summarize_zones(_shot_predictions_frame())
    corner = summary.loc[summary["shot_zone_basic"] == "Right Corner 3"].iloc[0]

    assert list(summary.columns) == list(ZONE_SUMMARY_COLUMNS)
    assert len(summary) == 2
    assert corner["attempts"] == 2
    assert corner["made_shots"] == 2
    assert corner["actual_fg_pct"] == pytest.approx(1.0)
    assert corner["expected_fg_pct"] == pytest.approx(0.5)
    assert corner["points_above_expected_per_100_shots"] == pytest.approx(150.0)


def test_summarize_player_trends_groups_by_player_and_month() -> None:
    summary = summarize_player_trends(_shot_predictions_frame())

    assert list(summary.columns) == list(PLAYER_TREND_COLUMNS)
    assert len(summary) == 3

    jan_player_10 = summary.loc[
        (summary["player_id"] == 10) & (summary["game_month"] == "2025-01")
    ].iloc[0]
    assert jan_player_10["attempts"] == 1
    assert jan_player_10["made_shots"] == 1
    assert jan_player_10["actual_fg_pct"] == pytest.approx(1.0)
    assert jan_player_10["points_above_expected_per_100_shots"] == pytest.approx(180.0)

    feb_player_10 = summary.loc[
        (summary["player_id"] == 10) & (summary["game_month"] == "2025-02")
    ].iloc[0]
    assert feb_player_10["attempts"] == 1
    assert feb_player_10["made_shots"] == 0
    assert feb_player_10["points_above_expected_per_100_shots"] == pytest.approx(-100.0)


def test_load_shot_predictions_reads_csv(tmp_path: Path) -> None:
    frame = _shot_predictions_frame()
    path = tmp_path / "shot_predictions_2024-25.csv"
    frame.to_csv(path, index=False)

    loaded = load_shot_predictions(path)

    assert len(loaded) == 3
    assert loaded["shot_made_flag"].dtype == bool


def test_save_summary_csv_writes_expected_columns(tmp_path: Path) -> None:
    summary = summarize_players(_shot_predictions_frame())
    output_path = tmp_path / "player_evaluation_2024-25.csv"

    save_summary_csv(summary, output_path)

    loaded = pd.read_csv(output_path)
    assert list(loaded.columns) == list(PLAYER_SUMMARY_COLUMNS)


def test_filter_players_for_insights_drops_low_attempt_players() -> None:
    summary = pd.DataFrame(
        [
            {"player_id": 1, "attempts": 50, "points_above_expected_per_100_shots": 10.0},
            {"player_id": 2, "attempts": 150, "points_above_expected_per_100_shots": 5.0},
        ]
    )

    filtered = filter_players_for_insights(summary, min_attempts=MIN_PLAYER_ATTEMPTS_FOR_INSIGHTS)

    assert list(filtered["player_id"]) == [2]


def test_filter_zones_for_insights_drops_tiny_sample_zones() -> None:
    summary = pd.DataFrame(
        [
            {
                "shot_zone_basic": "Above the Break 3",
                "shot_zone_area": "Back Court(BC)",
                "shot_zone_range": "Back Court Shot",
                "attempts": 6,
                "points_above_expected_per_100_shots": 33.97,
            },
            {
                "shot_zone_basic": "Right Corner 3",
                "shot_zone_area": "Right Side(R)",
                "shot_zone_range": "24+ ft.",
                "attempts": 2500,
                "points_above_expected_per_100_shots": 2.0,
            },
        ]
    )

    filtered = filter_zones_for_insights(summary, min_attempts=MIN_ZONE_ATTEMPTS_FOR_INSIGHTS)

    assert len(filtered) == 1
    assert filtered.iloc[0]["shot_zone_basic"] == "Right Corner 3"


def test_filter_teams_for_insights_keeps_all_teams() -> None:
    summary = pd.DataFrame(
        [
            {"team_id": 1, "attempts": 10},
            {"team_id": 2, "attempts": 1500},
        ]
    )

    filtered = filter_teams_for_insights(summary)

    assert len(filtered) == 2


@patch("courtvision.evaluation.summaries.load_shot_predictions")
def test_run_summaries_prints_checkpoint_lines(
    load_shot_predictions: object,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    load_shot_predictions.return_value = _shot_predictions_frame()

    run_summaries(
        "2024-25",
        tables_dir=tmp_path,
        predictions_path=tmp_path / "shot_predictions_2024-25.csv",
    )

    captured = capsys.readouterr().out
    assert "Loaded shot predictions: 3" in captured
    assert captured.endswith("player_trends_2024-25.csv\n")
    assert (tmp_path / "player_evaluation_2024-25.csv").is_file()
    assert (tmp_path / "team_evaluation_2024-25.csv").is_file()
    assert (tmp_path / "zone_evaluation_2024-25.csv").is_file()
    assert (tmp_path / "player_trends_2024-25.csv").is_file()
