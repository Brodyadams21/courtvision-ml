"""Aggregate shot-level predictions into player and team evaluation summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from courtvision.data.collect import DEFAULT_SEASON
from courtvision.data.load_data import PROJECT_ROOT
from courtvision.evaluation.predict_shots import (
    SHOT_PREDICTION_CORE_COLUMNS,
    SHOT_PREDICTION_OPTIONAL_METADATA_COLUMNS,
    shot_predictions_output_path,
)
from courtvision.models.common import DEFAULT_TABLES_DIR

SUMMARY_METRIC_COLUMNS: tuple[str, ...] = (
    "attempts",
    "made_shots",
    "actual_fg_pct",
    "expected_fg_pct",
    "actual_points",
    "expected_points",
    "points_above_expected",
    "points_above_expected_per_100_shots",
    "avg_expected_shot_value",
    "avg_shot_value",
)

PLAYER_SUMMARY_COLUMNS: tuple[str, ...] = ("player_id", *SUMMARY_METRIC_COLUMNS)
TEAM_SUMMARY_COLUMNS: tuple[str, ...] = ("team_id", *SUMMARY_METRIC_COLUMNS)
ZONE_GROUP_COLUMNS: tuple[str, ...] = (
    "shot_zone_basic",
    "shot_zone_area",
    "shot_zone_range",
)
ZONE_SUMMARY_COLUMNS: tuple[str, ...] = (*ZONE_GROUP_COLUMNS, *SUMMARY_METRIC_COLUMNS)
PLAYER_TREND_GROUP_COLUMNS: tuple[str, ...] = ("player_id", "game_month")
PLAYER_TREND_COLUMNS: tuple[str, ...] = (*PLAYER_TREND_GROUP_COLUMNS, *SUMMARY_METRIC_COLUMNS)

# Minimum shot attempts before surfacing a group in basketball insights.
# Full summary CSVs keep every row; apply these filters when writing claims.
MIN_PLAYER_ATTEMPTS_FOR_INSIGHTS = 100
MIN_PLAYER_ATTEMPTS_STRICT_FOR_INSIGHTS = 200
MIN_ZONE_ATTEMPTS_FOR_INSIGHTS = 100


def player_evaluation_output_path(
    season: str,
    *,
    tables_dir: Path | None = None,
) -> Path:
    """Default CSV path for player evaluation summary output."""
    directory = tables_dir or DEFAULT_TABLES_DIR
    return directory / f"player_evaluation_{season}.csv"


def team_evaluation_output_path(
    season: str,
    *,
    tables_dir: Path | None = None,
) -> Path:
    """Default CSV path for team evaluation summary output."""
    directory = tables_dir or DEFAULT_TABLES_DIR
    return directory / f"team_evaluation_{season}.csv"


def zone_evaluation_output_path(
    season: str,
    *,
    tables_dir: Path | None = None,
) -> Path:
    """Default CSV path for zone evaluation summary output."""
    directory = tables_dir or DEFAULT_TABLES_DIR
    return directory / f"zone_evaluation_{season}.csv"


def player_trends_output_path(
    season: str,
    *,
    tables_dir: Path | None = None,
) -> Path:
    """Default CSV path for monthly player trend summary output."""
    directory = tables_dir or DEFAULT_TABLES_DIR
    return directory / f"player_trends_{season}.csv"


def load_shot_predictions(path: Path) -> pd.DataFrame:
    """Load shot-level prediction CSV produced by ``predict_shots``."""
    if not path.is_file():
        raise FileNotFoundError(f"Shot predictions CSV not found: {path}")

    predictions = pd.read_csv(path)
    missing_columns = [
        column for column in SHOT_PREDICTION_CORE_COLUMNS if column not in predictions.columns
    ]
    if missing_columns:
        raise KeyError(f"Shot predictions missing required columns: {missing_columns}")

    optional_columns = [
        column
        for column in SHOT_PREDICTION_OPTIONAL_METADATA_COLUMNS
        if column in predictions.columns
    ]
    predictions = predictions[list(SHOT_PREDICTION_CORE_COLUMNS) + optional_columns].copy()
    predictions["shot_made_flag"] = predictions["shot_made_flag"].astype(bool)
    if "game_date" in predictions.columns:
        predictions["game_date"] = pd.to_datetime(predictions["game_date"]).dt.date
    return predictions


def _with_game_month(predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach a YYYY-MM ``game_month`` column from ``game_date``."""
    frame = predictions.copy()
    game_dates = pd.to_datetime(frame["game_date"])
    frame["game_month"] = game_dates.dt.to_period("M").astype(str)
    return frame


def _summarize_group(
    predictions: pd.DataFrame,
    group_columns: str | tuple[str, ...],
) -> pd.DataFrame:
    """Aggregate shot predictions to one row per group with shared summary metrics."""
    if isinstance(group_columns, str):
        group_columns = (group_columns,)

    grouped = (
        predictions.groupby(list(group_columns), as_index=False)
        .agg(
            attempts=("shot_id", "count"),
            made_shots=("shot_made_flag", "sum"),
            actual_points=("actual_points", "sum"),
            expected_points=("expected_shot_value", "sum"),
            points_above_expected=("points_above_expected", "sum"),
            expected_fg_pct=("predicted_make_probability", "mean"),
            avg_expected_shot_value=("expected_shot_value", "mean"),
            avg_shot_value=("shot_value", "mean"),
        )
        .sort_values(list(group_columns))
        .reset_index(drop=True)
    )

    grouped["made_shots"] = grouped["made_shots"].astype(int)
    grouped["actual_fg_pct"] = grouped["made_shots"] / grouped["attempts"]
    grouped["points_above_expected_per_100_shots"] = (
        grouped["points_above_expected"] / grouped["attempts"] * 100.0
    )
    return grouped


def summarize_players(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build one evaluation row per player."""
    summary = _summarize_group(predictions, "player_id")
    return summary[list(PLAYER_SUMMARY_COLUMNS)]


def summarize_teams(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build one evaluation row per team."""
    summary = _summarize_group(predictions, "team_id")
    return summary[list(TEAM_SUMMARY_COLUMNS)]


def summarize_zones(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build one evaluation row per shot zone combination."""
    missing_columns = [column for column in ZONE_GROUP_COLUMNS if column not in predictions.columns]
    if missing_columns:
        raise KeyError(f"Shot predictions missing zone columns: {missing_columns}")

    summary = _summarize_group(predictions, ZONE_GROUP_COLUMNS)
    return summary[list(ZONE_SUMMARY_COLUMNS)]


def summarize_player_trends(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build one evaluation row per player and calendar month."""
    if "game_date" not in predictions.columns:
        raise KeyError("Shot predictions missing game_date column")

    summary = _summarize_group(_with_game_month(predictions), PLAYER_TREND_GROUP_COLUMNS)
    return summary[list(PLAYER_TREND_COLUMNS)]


def save_summary_csv(summary: pd.DataFrame, output_path: Path) -> Path:
    """Write a summary table to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return output_path


def filter_summary_for_insights(
    summary: pd.DataFrame,
    *,
    min_attempts: int,
) -> pd.DataFrame:
    """Drop rows below an attempt threshold so insights avoid tiny-sample noise."""
    if "attempts" not in summary.columns:
        raise KeyError("Summary missing attempts column")
    return summary.loc[summary["attempts"] >= min_attempts].reset_index(drop=True)


def filter_players_for_insights(
    player_summary: pd.DataFrame,
    *,
    min_attempts: int = MIN_PLAYER_ATTEMPTS_FOR_INSIGHTS,
) -> pd.DataFrame:
    """Keep players with enough evaluation shots for reliable insight claims."""
    return filter_summary_for_insights(player_summary, min_attempts=min_attempts)


def filter_zones_for_insights(
    zone_summary: pd.DataFrame,
    *,
    min_attempts: int = MIN_ZONE_ATTEMPTS_FOR_INSIGHTS,
) -> pd.DataFrame:
    """Keep zones with enough evaluation shots for reliable insight claims."""
    return filter_summary_for_insights(zone_summary, min_attempts=min_attempts)


def filter_teams_for_insights(team_summary: pd.DataFrame) -> pd.DataFrame:
    """Return all team rows; team summaries are not filtered by attempt count."""
    return team_summary.copy()


def _print_saved_path(path: Path) -> None:
    try:
        saved_path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        saved_path = path
    print(f"Saved {saved_path}")


def run_summaries(
    season: str = DEFAULT_SEASON,
    *,
    tables_dir: Path | None = None,
    predictions_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load shot predictions and write player, team, zone, and trend evaluation CSVs."""
    predictions_file = predictions_path or shot_predictions_output_path(
        season, tables_dir=tables_dir
    )
    predictions = load_shot_predictions(predictions_file)
    print(f"Loaded shot predictions: {len(predictions):,}")

    player_summary = summarize_players(predictions)
    team_summary = summarize_teams(predictions)
    zone_summary = summarize_zones(predictions)
    player_trends = summarize_player_trends(predictions)

    player_path = player_evaluation_output_path(season, tables_dir=tables_dir)
    team_path = team_evaluation_output_path(season, tables_dir=tables_dir)
    zone_path = zone_evaluation_output_path(season, tables_dir=tables_dir)
    trends_path = player_trends_output_path(season, tables_dir=tables_dir)

    save_summary_csv(player_summary, player_path)
    save_summary_csv(team_summary, team_path)
    save_summary_csv(zone_summary, zone_path)
    save_summary_csv(player_trends, trends_path)

    _print_saved_path(player_path)
    _print_saved_path(team_path)
    _print_saved_path(zone_path)
    _print_saved_path(trends_path)
    return player_summary, team_summary, zone_summary, player_trends


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate shot predictions into player, team, zone, and trend summaries.",
    )
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season label (e.g. 2024-25)")
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help=f"Directory for summary CSV output (default: {DEFAULT_TABLES_DIR})",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
        help="Input shot predictions CSV (default: reports/tables/shot_predictions_<season>.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_summaries(
        args.season,
        tables_dir=args.tables_dir,
        predictions_path=args.predictions_path,
    )


if __name__ == "__main__":
    main()
