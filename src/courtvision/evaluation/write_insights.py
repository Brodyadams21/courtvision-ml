"""Generate basketball-facing insights from evaluation summary tables."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from nba_api.stats.static import teams as nba_teams

from courtvision.data.collect import DEFAULT_DATA_DIR, DEFAULT_SEASON, shot_chart_output_paths
from courtvision.data.load_data import PROJECT_ROOT
from courtvision.evaluation.summaries import (
    MIN_PLAYER_ATTEMPTS_FOR_INSIGHTS,
    MIN_ZONE_ATTEMPTS_FOR_INSIGHTS,
    filter_players_for_insights,
    filter_teams_for_insights,
    filter_zones_for_insights,
    player_evaluation_output_path,
    player_trends_output_path,
    team_evaluation_output_path,
    zone_evaluation_output_path,
)
from courtvision.models.registry import CANDIDATE_ALIAS, REGISTERED_MODEL_NAME

DEFAULT_INSIGHTS_PATH = PROJECT_ROOT / "reports" / "basketball_insights.md"
TOP_N_DEFAULT = 5
MIN_PLAYER_ATTEMPTS_PER_MONTH_FOR_TREND = 40


@dataclass(frozen=True)
class SummaryTables:
    players: pd.DataFrame
    teams: pd.DataFrame
    zones: pd.DataFrame
    trends: pd.DataFrame


@dataclass(frozen=True)
class InsightFinding:
    title: str
    body: str


def basketball_insights_output_path(
    season: str = DEFAULT_SEASON,
    *,
    reports_dir: Path | None = None,
) -> Path:
    """Default markdown path for basketball insights."""
    directory = reports_dir or (PROJECT_ROOT / "reports")
    return directory / "basketball_insights.md"


def load_player_evaluation(path: Path) -> pd.DataFrame:
    """Load the player evaluation summary CSV."""
    return pd.read_csv(path)


def load_team_evaluation(path: Path) -> pd.DataFrame:
    """Load the team evaluation summary CSV."""
    return pd.read_csv(path)


def load_zone_evaluation(path: Path) -> pd.DataFrame:
    """Load the zone evaluation summary CSV."""
    return pd.read_csv(path)


def load_player_trends(path: Path) -> pd.DataFrame:
    """Load the monthly player trend summary CSV."""
    return pd.read_csv(path)


def load_summary_tables(
    season: str = DEFAULT_SEASON,
    *,
    tables_dir: Path | None = None,
) -> SummaryTables:
    """Load all evaluation summary tables for a season."""
    return SummaryTables(
        players=load_player_evaluation(
            player_evaluation_output_path(season, tables_dir=tables_dir)
        ),
        teams=load_team_evaluation(team_evaluation_output_path(season, tables_dir=tables_dir)),
        zones=load_zone_evaluation(zone_evaluation_output_path(season, tables_dir=tables_dir)),
        trends=load_player_trends(player_trends_output_path(season, tables_dir=tables_dir)),
    )


def load_player_name_map(
    season: str = DEFAULT_SEASON,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[int, str]:
    """Map ``player_id`` to display name from raw shot-chart parquet when available."""
    path = shot_chart_output_paths(data_dir, season)["parquet"]
    if not path.is_file():
        return {}

    names = pd.read_parquet(path, columns=["PLAYER_ID", "PLAYER_NAME"]).drop_duplicates()
    names = names.rename(columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "player_name"})
    return {
        int(player_id): str(player_name)
        for player_id, player_name in zip(names["player_id"], names["player_name"], strict=True)
    }


def load_team_abbreviation_map() -> dict[int, str]:
    """Map NBA ``team_id`` to tricode abbreviation."""
    return {int(team["id"]): str(team["abbreviation"]) for team in nba_teams.get_teams()}


def format_player(player_id: int, player_names: dict[int, str]) -> str:
    """Return a readable player label."""
    name = player_names.get(int(player_id))
    if name:
        return f"{name} (`{int(player_id)}`)"
    return f"Player `{int(player_id)}`"


def format_team(team_id: int, team_abbreviations: dict[int, str]) -> str:
    """Return a readable team label."""
    abbreviation = team_abbreviations.get(int(team_id))
    if abbreviation:
        return f"{abbreviation} (`{int(team_id)}`)"
    return f"Team `{int(team_id)}`"


def format_zone(row: pd.Series) -> str:
    """Return a readable shot-zone label."""
    return (
        f"{row['shot_zone_basic']} / {row['shot_zone_area']} / {row['shot_zone_range']}"
    )


def _format_rate(value: float) -> str:
    return f"{value:+.2f}"


def _format_points(value: float) -> str:
    return f"{value:+.1f}"


def finding_top_players_total_points_above_expected(
    players: pd.DataFrame,
    *,
    player_names: dict[int, str],
    min_attempts: int = MIN_PLAYER_ATTEMPTS_FOR_INSIGHTS,
    top_n: int = TOP_N_DEFAULT,
) -> InsightFinding:
    """Top players by total points above expected on the evaluation split."""
    filtered = filter_players_for_insights(players, min_attempts=min_attempts)
    leaders = filtered.nlargest(top_n, "points_above_expected")

    lines = [
        (
            f"- **{format_player(int(row.player_id), player_names)}** — "
            f"{int(row.attempts):,} attempts, "
            f"{_format_points(row.points_above_expected)} total points above expected "
            f"({_format_rate(row.points_above_expected_per_100_shots)} per 100 shots)."
        )
        for row in leaders.itertuples(index=False)
    ]
    body = (
        f"Among players with at least {min_attempts} evaluation shots, these players added "
        "the most total scoring value relative to model expectation:\n\n"
        + "\n".join(lines)
    )
    return InsightFinding(
        title="Top players by total points above expected",
        body=body,
    )


def finding_top_players_points_above_expected_per_100(
    players: pd.DataFrame,
    *,
    player_names: dict[int, str],
    min_attempts: int = MIN_PLAYER_ATTEMPTS_FOR_INSIGHTS,
    top_n: int = TOP_N_DEFAULT,
) -> InsightFinding:
    """Top players by points above expected per 100 shots with an attempt filter."""
    filtered = filter_players_for_insights(players, min_attempts=min_attempts)
    leaders = filtered.nlargest(top_n, "points_above_expected_per_100_shots")

    lines = [
        (
            f"- **{format_player(int(row.player_id), player_names)}** — "
            f"{_format_rate(row.points_above_expected_per_100_shots)} per 100 shots "
            f"on {int(row.attempts):,} attempts "
            f"({_format_points(row.points_above_expected)} total)."
        )
        for row in leaders.itertuples(index=False)
    ]
    body = (
        f"Efficiency leaders on the evaluation split (minimum {min_attempts} attempts):\n\n"
        + "\n".join(lines)
    )
    return InsightFinding(
        title="Top players by points above expected per 100 shots",
        body=body,
    )


def finding_top_teams_points_above_expected(
    teams: pd.DataFrame,
    *,
    team_abbreviations: dict[int, str],
    top_n: int = TOP_N_DEFAULT,
) -> InsightFinding:
    """Teams with the most total points above expected."""
    leaders = filter_teams_for_insights(teams).nlargest(top_n, "points_above_expected")

    lines = [
        (
            f"- **{format_team(int(row.team_id), team_abbreviations)}** — "
            f"{_format_points(row.points_above_expected)} total "
            f"({_format_rate(row.points_above_expected_per_100_shots)} per 100 shots) "
            f"across {int(row.attempts):,} team shots."
        )
        for row in leaders.itertuples(index=False)
    ]
    intro = "Team shot-making value relative to expectation on the held-out test games:\n\n"
    body = intro + "\n".join(lines)
    return InsightFinding(
        title="Teams with the best points above expected",
        body=body,
    )


def finding_top_teams_average_expected_shot_value(
    teams: pd.DataFrame,
    *,
    team_abbreviations: dict[int, str],
    top_n: int = TOP_N_DEFAULT,
) -> InsightFinding:
    """Teams whose shot diet produced the highest average expected shot value."""
    leaders = filter_teams_for_insights(teams).nlargest(top_n, "avg_expected_shot_value")

    lines = [
        (
            f"- **{format_team(int(row.team_id), team_abbreviations)}** — "
            f"average expected shot value {row.avg_expected_shot_value:.3f} "
            f"on {int(row.attempts):,} shots "
            f"({_format_points(row.points_above_expected)} total above expected)."
        )
        for row in leaders.itertuples(index=False)
    ]
    body = (
        "Teams whose shot mix generated the highest average expected points per attempt "
        "from the model:\n\n" + "\n".join(lines)
    )
    return InsightFinding(
        title="Teams with the best average expected shot value",
        body=body,
    )


def finding_zone_efficiency_extremes(
    zones: pd.DataFrame,
    *,
    min_attempts: int = MIN_ZONE_ATTEMPTS_FOR_INSIGHTS,
    top_n: int = 3,
) -> InsightFinding:
    """Best and worst zones by points above expected per 100 shots."""
    filtered = filter_zones_for_insights(zones, min_attempts=min_attempts)
    best = filtered.nlargest(top_n, "points_above_expected_per_100_shots")
    worst = filtered.nsmallest(top_n, "points_above_expected_per_100_shots")

    best_lines = [
        (
            f"- **{format_zone(row)}** — "
            f"{_format_rate(row.points_above_expected_per_100_shots)} per 100 shots "
            f"({int(row.attempts):,} attempts)."
        )
        for _, row in best.iterrows()
    ]
    worst_lines = [
        (
            f"- **{format_zone(row)}** — "
            f"{_format_rate(row.points_above_expected_per_100_shots)} per 100 shots "
            f"({int(row.attempts):,} attempts)."
        )
        for _, row in worst.iterrows()
    ]
    body = (
        f"Zone combinations with at least {min_attempts} evaluation shots:\n\n"
        "**Best zones**\n\n"
        + "\n".join(best_lines)
        + "\n\n**Worst zones**\n\n"
        + "\n".join(worst_lines)
    )
    return InsightFinding(
        title="Best and worst zones by points above expected",
        body=body,
    )


def finding_player_monthly_trend(
    trends: pd.DataFrame,
    *,
    player_names: dict[int, str],
    min_attempts_per_month: int = MIN_PLAYER_ATTEMPTS_PER_MONTH_FOR_TREND,
) -> InsightFinding:
    """Highlight the largest month-over-month swing in per-100-shot value."""
    eligible = trends.loc[trends["attempts"] >= min_attempts_per_month].copy()
    eligible = eligible.sort_values(["player_id", "game_month"])

    deltas: list[dict[str, object]] = []
    for player_id, group in eligible.groupby("player_id", sort=False):
        ordered = group.reset_index(drop=True)
        for index in range(1, len(ordered)):
            previous = ordered.iloc[index - 1]
            current = ordered.iloc[index]
            deltas.append(
                {
                    "player_id": int(player_id),
                    "from_month": str(previous["game_month"]),
                    "to_month": str(current["game_month"]),
                    "delta_per_100": float(
                        current["points_above_expected_per_100_shots"]
                        - previous["points_above_expected_per_100_shots"]
                    ),
                    "from_attempts": int(previous["attempts"]),
                    "to_attempts": int(current["attempts"]),
                    "to_total_per_100": float(current["points_above_expected_per_100_shots"]),
                }
            )

    if not deltas:
        body = (
            "No consecutive player-month pairs met the minimum attempt threshold "
            f"({min_attempts_per_month} shots per month)."
        )
        return InsightFinding(title="Player scoring trend", body=body)

    frame = pd.DataFrame(deltas)
    best = frame.loc[frame["delta_per_100"].idxmax()]
    worst = frame.loc[frame["delta_per_100"].idxmin()]

    best_player = format_player(int(best["player_id"]), player_names)
    worst_player = format_player(int(worst["player_id"]), player_names)
    body = (
        f"Monthly trend scan using at least {min_attempts_per_month} shots in each month:\n\n"
        f"- **Largest improvement:** {best_player} moved from "
        f"{best['from_month']} to {best['to_month']} at "
        f"{_format_rate(float(best['delta_per_100']))} per 100 shots "
        f"({int(best['from_attempts'])} → {int(best['to_attempts'])} attempts; "
        f"ending month rate {_format_rate(float(best['to_total_per_100']))} per 100).\n"
        f"- **Largest decline:** {worst_player} moved from "
        f"{worst['from_month']} to {worst['to_month']} at "
        f"{_format_rate(float(worst['delta_per_100']))} per 100 shots "
        f"({int(worst['from_attempts'])} → {int(worst['to_attempts'])} attempts)."
    )
    return InsightFinding(title="Player monthly trend", body=body)


def limitations_section(
    *,
    season: str,
    evaluation_shots: int,
    min_player_attempts: int = MIN_PLAYER_ATTEMPTS_FOR_INSIGHTS,
    min_zone_attempts: int = MIN_ZONE_ATTEMPTS_FOR_INSIGHTS,
) -> str:
    """Return the report limitations section."""
    model_ref = f"`{REGISTERED_MODEL_NAME}` (`{CANDIDATE_ALIAS}`)"
    return (
        "## Limitations\n\n"
        "- **Held-out test split only:** Findings come from the latest ~20% of "
        f"{season} games by date, not the full season.\n"
        "- **Model-driven expectation:** Expected shot value uses raw LightGBM make "
        f"probabilities from {model_ref} without a separate calibration pass.\n"
        "- **Sample-size filters:** Player insights require at least "
        f"{min_player_attempts} evaluation shots; zone insights require at least "
        f"{min_zone_attempts}. Tiny zones (for example backcourt heaves) are excluded "
        "from zone rankings.\n"
        "- **Feature scope:** The registered LightGBM Candidate uses shot geometry, "
        "game context, and rolling player/team form — not matchup-specific scouting, "
        "player tracking, lineup, or defender-distance features.\n"
        "- **Evaluation volume:** This report is based on "
        f"{evaluation_shots:,} evaluation shots for season `{season}`.\n"
    )


def build_insights_report(
    tables: SummaryTables,
    *,
    season: str = DEFAULT_SEASON,
    player_names: dict[int, str] | None = None,
    team_abbreviations: dict[int, str] | None = None,
    report_date: date | None = None,
    evaluation_shots: int | None = None,
) -> str:
    """Build the basketball insights markdown report from summary tables."""
    names = player_names or {}
    abbreviations = team_abbreviations or load_team_abbreviation_map()
    generated_on = report_date or datetime.now(UTC).date()
    if evaluation_shots is not None:
        shot_count = evaluation_shots
    else:
        shot_count = int(tables.players["attempts"].sum())

    findings = [
        finding_top_players_total_points_above_expected(tables.players, player_names=names),
        finding_top_players_points_above_expected_per_100(tables.players, player_names=names),
        finding_top_teams_points_above_expected(
            tables.teams, team_abbreviations=abbreviations
        ),
        finding_top_teams_average_expected_shot_value(
            tables.teams, team_abbreviations=abbreviations
        ),
        finding_zone_efficiency_extremes(tables.zones),
        finding_player_monthly_trend(tables.trends, player_names=names),
    ]

    finding_blocks = "\n\n".join(
        f"### {index}. {finding.title}\n\n{finding.body}"
        for index, finding in enumerate(findings, start=1)
    )

    return f"""# Basketball Insights — Expected Shot Value ({season})

**Project:** CourtVision ML  
**Phase:** 8 — Expected Shot Value and Player Evaluation  
**Season:** {season}  
**Model:** `{REGISTERED_MODEL_NAME}` (`{CANDIDATE_ALIAS}`)  
**Report date:** {generated_on.isoformat()}  
**Evaluation shots:** {shot_count:,}

---

## Overview

This report translates shot-level expected value results into basketball-facing takeaways from the
held-out evaluation period. Positive **points above expected** means a player, team, or zone scored
more than the model anticipated given shot difficulty and context.

---

## Findings

{finding_blocks}

---

{limitations_section(season=season, evaluation_shots=shot_count)}
"""


def write_insights_report(
    report: str,
    output_path: Path,
) -> Path:
    """Write the insights markdown report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def run_write_insights(
    season: str = DEFAULT_SEASON,
    *,
    tables_dir: Path | None = None,
    output_path: Path | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Path:
    """Load summary tables and write the basketball insights markdown report."""
    tables = load_summary_tables(season, tables_dir=tables_dir)
    evaluation_shots = int(tables.players["attempts"].sum())
    print(f"Loaded summary tables for {season} ({evaluation_shots:,} player-summary attempts)")

    report = build_insights_report(
        tables,
        season=season,
        player_names=load_player_name_map(season, data_dir=data_dir),
        evaluation_shots=evaluation_shots,
    )
    destination = output_path or basketball_insights_output_path(season)
    write_insights_report(report, destination)

    try:
        saved_path = destination.relative_to(PROJECT_ROOT)
    except ValueError:
        saved_path = destination
    print(f"Saved {saved_path}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write basketball insights markdown from evaluation summary tables.",
    )
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season label (e.g. 2024-25)")
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="Directory containing evaluation summary CSV files",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output markdown path (default: reports/basketball_insights.md)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_write_insights(
        args.season,
        tables_dir=args.tables_dir,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
