"""Tests for basketball insights report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from courtvision.evaluation.write_insights import (
    SummaryTables,
    build_insights_report,
    finding_top_players_points_above_expected_per_100,
    finding_top_players_total_points_above_expected,
    finding_zone_efficiency_extremes,
    load_summary_tables,
    run_write_insights,
)


def _summary_tables() -> SummaryTables:
    players = pd.DataFrame(
        [
            {
                "player_id": 1,
                "attempts": 150,
                "made_shots": 70,
                "actual_fg_pct": 0.4667,
                "expected_fg_pct": 0.40,
                "actual_points": 180.0,
                "expected_points": 150.0,
                "points_above_expected": 30.0,
                "points_above_expected_per_100_shots": 20.0,
                "avg_expected_shot_value": 1.0,
                "avg_shot_value": 2.5,
            },
            {
                "player_id": 2,
                "attempts": 120,
                "made_shots": 55,
                "actual_fg_pct": 0.4583,
                "expected_fg_pct": 0.42,
                "actual_points": 140.0,
                "expected_points": 130.0,
                "points_above_expected": 10.0,
                "points_above_expected_per_100_shots": 35.0,
                "avg_expected_shot_value": 1.1,
                "avg_shot_value": 2.4,
            },
            {
                "player_id": 3,
                "attempts": 20,
                "made_shots": 10,
                "actual_fg_pct": 0.5,
                "expected_fg_pct": 0.45,
                "actual_points": 30.0,
                "expected_points": 20.0,
                "points_above_expected": 50.0,
                "points_above_expected_per_100_shots": 250.0,
                "avg_expected_shot_value": 1.0,
                "avg_shot_value": 3.0,
            },
        ]
    )
    teams = pd.DataFrame(
        [
            {
                "team_id": 100,
                "attempts": 1000,
                "made_shots": 450,
                "actual_fg_pct": 0.45,
                "expected_fg_pct": 0.43,
                "actual_points": 1100.0,
                "expected_points": 1050.0,
                "points_above_expected": 50.0,
                "points_above_expected_per_100_shots": 5.0,
                "avg_expected_shot_value": 1.05,
                "avg_shot_value": 2.4,
            },
            {
                "team_id": 200,
                "attempts": 900,
                "made_shots": 420,
                "actual_fg_pct": 0.4667,
                "expected_fg_pct": 0.44,
                "actual_points": 1000.0,
                "expected_points": 980.0,
                "points_above_expected": 20.0,
                "points_above_expected_per_100_shots": 2.22,
                "avg_expected_shot_value": 1.20,
                "avg_shot_value": 2.5,
            },
        ]
    )
    zones = pd.DataFrame(
        [
            {
                "shot_zone_basic": "Corner",
                "shot_zone_area": "Right",
                "shot_zone_range": "24+ ft.",
                "attempts": 200,
                "made_shots": 90,
                "actual_fg_pct": 0.45,
                "expected_fg_pct": 0.40,
                "actual_points": 270.0,
                "expected_points": 240.0,
                "points_above_expected": 30.0,
                "points_above_expected_per_100_shots": 15.0,
                "avg_expected_shot_value": 1.2,
                "avg_shot_value": 3.0,
            },
            {
                "shot_zone_basic": "Mid-Range",
                "shot_zone_area": "Center",
                "shot_zone_range": "16-24 ft.",
                "attempts": 150,
                "made_shots": 60,
                "actual_fg_pct": 0.40,
                "expected_fg_pct": 0.42,
                "actual_points": 120.0,
                "expected_points": 130.0,
                "points_above_expected": -10.0,
                "points_above_expected_per_100_shots": -6.67,
                "avg_expected_shot_value": 0.9,
                "avg_shot_value": 2.0,
            },
            {
                "shot_zone_basic": "Back Court",
                "shot_zone_area": "Back Court",
                "shot_zone_range": "Back Court Shot",
                "attempts": 6,
                "made_shots": 1,
                "actual_fg_pct": 0.1667,
                "expected_fg_pct": 0.05,
                "actual_points": 3.0,
                "expected_points": 1.0,
                "points_above_expected": 2.0,
                "points_above_expected_per_100_shots": 33.33,
                "avg_expected_shot_value": 0.16,
                "avg_shot_value": 3.0,
            },
        ]
    )
    trends = pd.DataFrame(
        [
            {
                "player_id": 1,
                "game_month": "2025-03",
                "attempts": 50,
                "made_shots": 20,
                "actual_fg_pct": 0.40,
                "expected_fg_pct": 0.38,
                "actual_points": 50.0,
                "expected_points": 45.0,
                "points_above_expected": 5.0,
                "points_above_expected_per_100_shots": 10.0,
                "avg_expected_shot_value": 0.9,
                "avg_shot_value": 2.4,
            },
            {
                "player_id": 1,
                "game_month": "2025-04",
                "attempts": 55,
                "made_shots": 30,
                "actual_fg_pct": 0.5455,
                "expected_fg_pct": 0.40,
                "actual_points": 75.0,
                "expected_points": 50.0,
                "points_above_expected": 25.0,
                "points_above_expected_per_100_shots": 45.45,
                "avg_expected_shot_value": 0.91,
                "avg_shot_value": 2.5,
            },
        ]
    )
    return SummaryTables(players=players, teams=teams, zones=zones, trends=trends)


def test_player_insight_filters_exclude_low_attempt_players() -> None:
    finding = finding_top_players_total_points_above_expected(
        _summary_tables().players,
        player_names={1: "Player One", 2: "Player Two", 3: "Low Sample"},
    )

    assert "Player One" in finding.body
    assert "Low Sample" not in finding.body


def test_rate_insight_uses_filtered_players() -> None:
    finding = finding_top_players_points_above_expected_per_100(
        _summary_tables().players,
        player_names={1: "Player One", 2: "Player Two"},
    )

    assert "Player Two" in finding.body
    assert "+35.00 per 100 shots" in finding.body


def test_zone_insight_excludes_tiny_sample_zones() -> None:
    finding = finding_zone_efficiency_extremes(_summary_tables().zones)

    assert "Back Court" not in finding.body
    assert "Corner" in finding.body
    assert "Mid-Range" in finding.body


def test_build_insights_report_includes_required_sections() -> None:
    report = build_insights_report(
        _summary_tables(),
        season="2024-25",
        player_names={1: "Player One", 2: "Player Two"},
        team_abbreviations={100: "AAA", 200: "BBB"},
        report_date=__import__("datetime").date(2026, 6, 15),
        evaluation_shots=290,
    )

    assert "# Basketball Insights" in report
    assert "Top players by total points above expected" in report
    assert "Top players by points above expected per 100 shots" in report
    assert "Teams with the best points above expected" in report
    assert "Teams with the best average expected shot value" in report
    assert "Best and worst zones by points above expected" in report
    assert "Player monthly trend" in report
    assert "## Limitations" in report


def test_load_summary_tables_reads_csvs(tmp_path: Path) -> None:
    tables = _summary_tables()
    tables.players.to_csv(tmp_path / "player_evaluation_2024-25.csv", index=False)
    tables.teams.to_csv(tmp_path / "team_evaluation_2024-25.csv", index=False)
    tables.zones.to_csv(tmp_path / "zone_evaluation_2024-25.csv", index=False)
    tables.trends.to_csv(tmp_path / "player_trends_2024-25.csv", index=False)

    loaded = load_summary_tables("2024-25", tables_dir=tmp_path)

    assert len(loaded.players) == 3
    assert len(loaded.zones) == 3


def test_run_write_insights_writes_markdown(tmp_path: Path) -> None:
    tables = _summary_tables()
    tables.players.to_csv(tmp_path / "player_evaluation_2024-25.csv", index=False)
    tables.teams.to_csv(tmp_path / "team_evaluation_2024-25.csv", index=False)
    tables.zones.to_csv(tmp_path / "zone_evaluation_2024-25.csv", index=False)
    tables.trends.to_csv(tmp_path / "player_trends_2024-25.csv", index=False)

    output_path = tmp_path / "basketball_insights.md"
    run_write_insights("2024-25", tables_dir=tmp_path, output_path=output_path, data_dir=tmp_path)

    assert output_path.is_file()
    content = output_path.read_text(encoding="utf-8")
    assert "Player `1`" in content
