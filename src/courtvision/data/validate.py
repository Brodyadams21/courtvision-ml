"""Phase 3–4 data validation for CourtVision ML tables.

Validation layers:
  Phase 3 (before PostgreSQL load of cleaned tables):
    1. Pandera schemas (schemas.py) — column types, nulls, ranges, allowed values
    2. Custom checks (this module) — season thresholds, duplicates, FK sanity, warnings

  Phase 4 (gold shot features, before gold load):
    1. Pandera gold schema — required columns and shot_value domain
    2. Custom checks — shot_id uniqueness and required field null checks

Critical failures stop the pipeline. Warnings are logged for expected messiness
(especially play-by-play) but do not block inserts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd
from pandera.errors import SchemaErrors

from courtvision.data.collect import DEFAULT_SEASON
from courtvision.data.schemas import GOLD_TABLE_SCHEMAS, TABLE_SCHEMAS

logger = logging.getLogger(__name__)

ValidationLevel = Literal["critical", "warning"]

# Thresholds for a completed NBA regular season (extend per season as needed).
_SEASON_THRESHOLDS: dict[str, dict[str, dict[str, int]]] = {
    "2024-25": {
        "teams": {"exact": 30},
        "games": {"exact": 1230},
        "team_game_logs": {"exact": 2460},
        "shots": {"min": 200_000},
        "player_game_logs": {"min": 20_000},
        "play_by_play": {"min": 500_000},
        "players": {"min": 400},
    },
}

_SEASON_DATE_RANGES: dict[str, tuple[date, date]] = {
    "2024-25": (date(2024, 10, 1), date(2025, 6, 30)),
}


class ValidationFailedError(Exception):
    """Raised when one or more critical validation checks fail."""


@dataclass
class ValidationIssue:
    level: ValidationLevel
    table: str
    check: str
    message: str
    count: int | None = None


@dataclass
class ValidationReport:
    season: str
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def critical_issues(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "critical"]

    @property
    def warning_issues(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def passed(self) -> bool:
        return not self.critical_issues

    def add(
        self,
        level: ValidationLevel,
        table: str,
        check: str,
        message: str,
        *,
        count: int | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(level=level, table=table, check=check, message=message, count=count)
        )

    def log_issues(self) -> None:
        for issue in self.critical_issues:
            logger.error("[%s] %s.%s: %s", issue.level, issue.table, issue.check, issue.message)
        for issue in self.warning_issues:
            logger.warning("[%s] %s.%s: %s", issue.level, issue.table, issue.check, issue.message)

    def raise_if_failed(self) -> None:
        if self.passed:
            logger.info("Validation passed for season %s", self.season)
            return
        summary = "; ".join(
            f"{issue.table}.{issue.check}" for issue in self.critical_issues[:5]
        )
        extra = len(self.critical_issues) - 5
        if extra > 0:
            summary = f"{summary} (+{extra} more)"
        raise ValidationFailedError(
            f"Validation failed for season {self.season}: {len(self.critical_issues)} "
            f"critical issue(s). First checks: {summary}"
        )


def _null_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return len(df)
    series = df[column]
    if series.dtype == object:
        return int(series.isna().sum() + (series.astype(str).str.strip() == "").sum())
    return int(series.isna().sum())


def _duplicate_row_count(df: pd.DataFrame, key_columns: list[str]) -> int:
    if df.empty:
        return 0
    grouped = df.groupby(key_columns, dropna=False).size()
    return int((grouped > 1).sum())


def _duplicate_extra_rows(df: pd.DataFrame, key_columns: list[str]) -> int:
    if df.empty:
        return 0
    grouped = df.groupby(key_columns, dropna=False).size()
    return int((grouped - 1).clip(lower=0).sum())


def _check_required_columns(
    report: ValidationReport,
    table: str,
    df: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    for column in columns:
        missing = _null_count(df, column)
        if missing > 0:
            report.add(
                "critical",
                table,
                f"missing_{column}",
                f"{missing} row(s) missing required column {column}",
                count=missing,
            )


def _check_no_duplicate_keys(
    report: ValidationReport,
    table: str,
    df: pd.DataFrame,
    key_columns: tuple[str, ...],
) -> None:
    duplicate_groups = _duplicate_row_count(df, list(key_columns))
    if duplicate_groups > 0:
        extra_rows = _duplicate_extra_rows(df, list(key_columns))
        report.add(
            "critical",
            table,
            "duplicate_natural_key",
            f"{duplicate_groups} duplicate key group(s) on {key_columns} "
            f"({extra_rows} extra row(s))",
            count=extra_rows,
        )


def _check_row_count_thresholds(
    report: ValidationReport,
    table: str,
    df: pd.DataFrame,
    *,
    season: str,
) -> None:
    thresholds = _SEASON_THRESHOLDS.get(season, {}).get(table)
    if thresholds is None:
        return

    row_count = len(df)
    if "exact" in thresholds and row_count != thresholds["exact"]:
        report.add(
            "critical",
            table,
            "row_count_exact",
            f"expected exactly {thresholds['exact']} rows, found {row_count}",
            count=row_count,
        )
    if "min" in thresholds and row_count < thresholds["min"]:
        report.add(
            "critical",
            table,
            "row_count_min",
            f"expected at least {thresholds['min']} rows, found {row_count}",
            count=row_count,
        )


def _check_date_range(
    report: ValidationReport,
    table: str,
    df: pd.DataFrame,
    *,
    season: str,
    date_column: str = "game_date",
) -> None:
    if date_column not in df.columns or df.empty:
        return

    dates = pd.to_datetime(df[date_column], errors="coerce")
    if dates.isna().any():
        report.add(
            "critical",
            table,
            f"missing_{date_column}",
            f"{int(dates.isna().sum())} row(s) with null {date_column}",
            count=int(dates.isna().sum()),
        )

    valid_dates = dates.dropna()
    if valid_dates.empty:
        return

    bounds = _SEASON_DATE_RANGES.get(season)
    if bounds is None:
        return

    min_allowed, max_allowed = bounds
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    if min_date < min_allowed or max_date > max_allowed:
        report.add(
            "critical",
            table,
            "date_range",
            f"{date_column} range {min_date} to {max_date} outside allowed "
            f"{min_allowed} to {max_allowed} for season {season}",
        )


def _check_foreign_key(
    report: ValidationReport,
    table: str,
    df: pd.DataFrame,
    *,
    column: str,
    reference: pd.DataFrame,
    reference_column: str,
    check_name: str,
) -> None:
    if column not in df.columns or reference_column not in reference.columns:
        return

    left = df[[column]].dropna()
    if left.empty:
        return

    valid_keys = set(reference[reference_column].dropna().unique())
    orphans = left[~left[column].isin(valid_keys)]
    orphan_count = len(orphans)
    if orphan_count > 0:
        report.add(
            "critical",
            table,
            check_name,
            f"{orphan_count} row(s) with {column} not found in "
            f"{reference_column} reference set",
            count=orphan_count,
        )


def validate_teams(teams: pd.DataFrame, *, season: str, report: ValidationReport) -> None:
    table = "teams"
    _check_required_columns(report, table, teams, ("team_id", "abbreviation", "full_name"))
    _check_no_duplicate_keys(report, table, teams, ("team_id",))
    _check_row_count_thresholds(report, table, teams, season=season)

    if "abbreviation" in teams.columns:
        bad_abbr = teams["abbreviation"].astype(str).str.len().gt(3).sum()
        if bad_abbr > 0:
            report.add(
                "critical",
                table,
                "abbreviation_length",
                f"{bad_abbr} row(s) with abbreviation longer than 3 characters",
                count=int(bad_abbr),
            )


def validate_players(
    players: pd.DataFrame,
    *,
    season: str,
    report: ValidationReport,
) -> None:
    table = "players"
    _check_required_columns(report, table, players, ("player_id", "full_name"))
    _check_no_duplicate_keys(report, table, players, ("player_id",))
    _check_row_count_thresholds(report, table, players, season=season)

    missing_first = _null_count(players, "first_name")
    if missing_first > 0:
        report.add(
            "warning",
            table,
            "missing_first_name",
            f"{missing_first} row(s) missing first_name (parsed from full_name)",
            count=missing_first,
        )


def validate_games(
    games: pd.DataFrame,
    *,
    season: str,
    teams: pd.DataFrame,
    report: ValidationReport,
) -> None:
    table = "games"
    _check_no_duplicate_keys(report, table, games, ("game_id",))
    _check_row_count_thresholds(report, table, games, season=season)
    _check_date_range(report, table, games, season=season)

    if {"home_team_id", "away_team_id"}.issubset(games.columns):
        same_team = int((games["home_team_id"] == games["away_team_id"]).sum())
        if same_team > 0:
            report.add(
                "critical",
                table,
                "home_equals_away",
                f"{same_team} game(s) where home_team_id equals away_team_id",
                count=same_team,
            )

    _check_foreign_key(
        report,
        table,
        games,
        column="home_team_id",
        reference=teams,
        reference_column="team_id",
        check_name="orphan_home_team_id",
    )
    _check_foreign_key(
        report,
        table,
        games,
        column="away_team_id",
        reference=teams,
        reference_column="team_id",
        check_name="orphan_away_team_id",
    )


def validate_shots(
    shots: pd.DataFrame,
    *,
    season: str,
    games: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    report: ValidationReport,
) -> None:
    table = "shots"
    _check_no_duplicate_keys(report, table, shots, ("game_id", "game_event_id", "player_id"))
    _check_row_count_thresholds(report, table, shots, season=season)
    _check_date_range(report, table, shots, season=season)

    _check_foreign_key(
        report,
        table,
        shots,
        column="game_id",
        reference=games,
        reference_column="game_id",
        check_name="orphan_game_id",
    )
    _check_foreign_key(
        report,
        table,
        shots,
        column="player_id",
        reference=players,
        reference_column="player_id",
        check_name="orphan_player_id",
    )
    _check_foreign_key(
        report,
        table,
        shots,
        column="team_id",
        reference=teams,
        reference_column="team_id",
        check_name="orphan_team_id",
    )

    for optional_column in ("event_type", "action_type", "shot_zone_basic"):
        missing = _null_count(shots, optional_column)
        if missing > 0:
            report.add(
                "warning",
                table,
                f"missing_{optional_column}",
                f"{missing} row(s) missing optional {optional_column}",
                count=missing,
            )


def validate_player_game_logs(
    player_game_logs: pd.DataFrame,
    *,
    season: str,
    games: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    report: ValidationReport,
) -> None:
    table = "player_game_logs"
    _check_no_duplicate_keys(report, table, player_game_logs, ("game_id", "player_id"))
    _check_row_count_thresholds(report, table, player_game_logs, season=season)
    _check_date_range(report, table, player_game_logs, season=season)

    _check_foreign_key(
        report,
        table,
        player_game_logs,
        column="game_id",
        reference=games,
        reference_column="game_id",
        check_name="orphan_game_id",
    )
    _check_foreign_key(
        report,
        table,
        player_game_logs,
        column="player_id",
        reference=players,
        reference_column="player_id",
        check_name="orphan_player_id",
    )
    _check_foreign_key(
        report,
        table,
        player_game_logs,
        column="team_id",
        reference=teams,
        reference_column="team_id",
        check_name="orphan_team_id",
    )


def validate_team_game_logs(
    team_game_logs: pd.DataFrame,
    *,
    season: str,
    games: pd.DataFrame,
    teams: pd.DataFrame,
    report: ValidationReport,
) -> None:
    table = "team_game_logs"
    _check_no_duplicate_keys(report, table, team_game_logs, ("game_id", "team_id"))
    _check_row_count_thresholds(report, table, team_game_logs, season=season)
    _check_date_range(report, table, team_game_logs, season=season)

    _check_foreign_key(
        report,
        table,
        team_game_logs,
        column="game_id",
        reference=games,
        reference_column="game_id",
        check_name="orphan_game_id",
    )
    _check_foreign_key(
        report,
        table,
        team_game_logs,
        column="team_id",
        reference=teams,
        reference_column="team_id",
        check_name="orphan_team_id",
    )


def validate_play_by_play(
    play_by_play: pd.DataFrame,
    *,
    season: str,
    games: pd.DataFrame,
    report: ValidationReport,
) -> None:
    table = "play_by_play"
    _check_no_duplicate_keys(report, table, play_by_play, ("game_id", "action_number"))
    _check_row_count_thresholds(report, table, play_by_play, season=season)

    _check_foreign_key(
        report,
        table,
        play_by_play,
        column="game_id",
        reference=games,
        reference_column="game_id",
        check_name="orphan_game_id",
    )

    missing_team = _null_count(play_by_play, "team_id")
    if missing_team > 0:
        report.add(
            "warning",
            table,
            "missing_team_id",
            f"{missing_team} row(s) missing team_id (expected for some events)",
            count=missing_team,
        )

    if "person_id" in play_by_play.columns and "player_name" in play_by_play.columns:
        has_person = play_by_play["person_id"].notna()
        missing_name = has_person & play_by_play["player_name"].isna()
        missing_name_count = int(missing_name.sum())
        if missing_name_count > 0:
            report.add(
                "warning",
                table,
                "missing_player_name",
                f"{missing_name_count} row(s) with person_id but no player_name",
                count=missing_name_count,
            )

    missing_description = _null_count(play_by_play, "description")
    if missing_description > 0:
        report.add(
            "warning",
            table,
            "missing_description",
            f"{missing_description} row(s) missing description",
            count=missing_description,
        )

    for optional_column in ("action_type", "sub_type", "shot_result"):
        missing = _null_count(play_by_play, optional_column)
        if missing > 0:
            report.add(
                "warning",
                table,
                f"missing_{optional_column}",
                f"{missing} row(s) missing optional {optional_column}",
                count=missing,
            )


def _record_pandera_schema_errors(
    report: ValidationReport,
    table: str,
    exc: SchemaErrors,
) -> None:
    failure_cases = exc.failure_cases
    if failure_cases is not None and not failure_cases.empty:
        for (column, check), count in (
            failure_cases.groupby(["column", "check"], dropna=False).size().items()
        ):
            column_label = "dataframe" if pd.isna(column) else str(column)
            check_label = "schema" if pd.isna(check) else str(check)
            report.add(
                "critical",
                table,
                f"pandera_{check_label}",
                f"Pandera check {check_label!r} failed on {column_label!r} "
                f"({int(count)} failure case(s))",
                count=int(count),
            )
        return

    report.add(
        "critical",
        table,
        "pandera_schema",
        str(exc).splitlines()[0][:500],
    )


def validate_pandera_schemas(
    datasets: dict[str, pd.DataFrame],
    report: ValidationReport,
) -> None:
    """Layer 1: validate cleaned tables against Pandera column schemas."""
    for table_name, schema in TABLE_SCHEMAS.items():
        dataframe = datasets[table_name]
        if dataframe.empty:
            report.add(
                "critical",
                table_name,
                "pandera_empty",
                "DataFrame is empty; schema validation cannot run",
            )
            continue

        try:
            schema.validate(dataframe, lazy=True)
        except SchemaErrors as exc:
            _record_pandera_schema_errors(report, table_name, exc)


def validate_all_cleaned_datasets(
    datasets: dict[str, pd.DataFrame],
    *,
    season: str = DEFAULT_SEASON,
) -> ValidationReport:
    """Run Pandera schemas, then custom basketball validators."""
    report = ValidationReport(season=season)

    validate_pandera_schemas(datasets, report)

    teams = datasets["teams"]
    players = datasets["players"]
    games = datasets["games"]
    shots = datasets["shots"]
    player_game_logs = datasets["player_game_logs"]
    team_game_logs = datasets["team_game_logs"]
    play_by_play = datasets["play_by_play"]

    validate_teams(teams, season=season, report=report)
    validate_players(players, season=season, report=report)
    validate_games(games, season=season, teams=teams, report=report)
    validate_shots(
        shots,
        season=season,
        games=games,
        players=players,
        teams=teams,
        report=report,
    )
    validate_player_game_logs(
        player_game_logs,
        season=season,
        games=games,
        players=players,
        teams=teams,
        report=report,
    )
    validate_team_game_logs(
        team_game_logs,
        season=season,
        games=games,
        teams=teams,
        report=report,
    )
    validate_play_by_play(
        play_by_play,
        season=season,
        games=games,
        report=report,
    )

    return report


GOLD_SHOT_FEATURES_TABLE = "gold_shot_features"
GOLD_SHOT_FEATURES_UNIQUE_KEY: tuple[str, ...] = ("shot_id",)
GOLD_SHOT_FEATURES_REQUIRED_COLUMNS: tuple[str, ...] = (
    "shot_id",
    "shot_made_flag",
    "game_date",
    "team_id",
    "player_id",
    "opponent_team_id",
    "is_home",
)
_VALID_SHOT_VALUE = frozenset({2, 3})


def _check_values_in_set(
    report: ValidationReport,
    table: str,
    df: pd.DataFrame,
    column: str,
    allowed: frozenset[int] | frozenset[str],
) -> None:
    if df.empty or column not in df.columns:
        return

    invalid_count = int((~df[column].isin(allowed) & df[column].notna()).sum())
    if invalid_count > 0:
        report.add(
            "critical",
            table,
            f"invalid_{column}",
            f"{invalid_count} row(s) with {column} not in {sorted(allowed)}",
            count=invalid_count,
        )


def validate_gold_pandera_schemas(
    datasets: dict[str, pd.DataFrame],
    report: ValidationReport,
) -> None:
    """Layer 1: validate gold tables against Pandera schemas."""
    for table_name, schema in GOLD_TABLE_SCHEMAS.items():
        dataframe = datasets[table_name]
        if dataframe.empty:
            report.add(
                "critical",
                table_name,
                "pandera_empty",
                "DataFrame is empty; schema validation cannot run",
            )
            continue

        try:
            schema.validate(dataframe, lazy=True)
        except SchemaErrors as exc:
            _record_pandera_schema_errors(report, table_name, exc)


def validate_gold_shot_features(
    gold: pd.DataFrame,
    *,
    season: str,
    report: ValidationReport,
) -> None:
    """Phase 4 custom checks for model-ready gold shot features."""
    table = GOLD_SHOT_FEATURES_TABLE

    if gold.empty:
        report.add(
            "critical",
            table,
            "empty",
            "Gold shot features DataFrame is empty",
        )
        return

    _check_required_columns(report, table, gold, GOLD_SHOT_FEATURES_REQUIRED_COLUMNS)
    _check_no_duplicate_keys(report, table, gold, GOLD_SHOT_FEATURES_UNIQUE_KEY)
    _check_values_in_set(report, table, gold, "shot_value", _VALID_SHOT_VALUE)

    thresholds = _SEASON_THRESHOLDS.get(season, {}).get("shots")
    if thresholds is not None:
        row_count = len(gold)
        if "min" in thresholds and row_count < thresholds["min"]:
            report.add(
                "critical",
                table,
                "row_count_min",
                f"expected at least {thresholds['min']} rows, found {row_count}",
                count=row_count,
            )


def validate_all_gold_shot_features(
    gold: pd.DataFrame,
    *,
    season: str = DEFAULT_SEASON,
    feature_set_version: str = "base_v1",
) -> ValidationReport:
    """Run Phase 4 Pandera schema and custom validators on gold shot features."""
    report = ValidationReport(season=season)

    validation_frame = gold.copy()
    if "feature_set_version" not in validation_frame.columns:
        validation_frame["feature_set_version"] = feature_set_version

    validate_gold_pandera_schemas({GOLD_SHOT_FEATURES_TABLE: validation_frame}, report)
    validate_gold_shot_features(gold, season=season, report=report)
    return report
