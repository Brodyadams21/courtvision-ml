"""Download NBA raw data and persist files with collection metadata."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from nba_api.stats.endpoints import playergamelogs, shotchartdetail, teamgamelogs
from nba_api.stats.static import teams

logger = logging.getLogger(__name__)

DEFAULT_SEASON = "2024-25"
DEFAULT_DATA_DIR = Path("data")
METADATA_SUBDIR = Path("metadata")
METADATA_FILENAME = "data_collection_metadata.json"
REQUEST_DELAY_SECONDS = 0.75
MAX_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 120

SHOT_CHART_DATASET = "shot_chart"
PLAYER_GAME_LOGS_DATASET = "player_game_logs"
TEAM_GAME_LOGS_DATASET = "team_game_logs"

SHOT_CHART_SOURCE = "https://stats.nba.com/stats/shotchartdetail"
PLAYER_GAME_LOGS_SOURCE = "https://stats.nba.com/stats/playergamelogs"
TEAM_GAME_LOGS_SOURCE = "https://stats.nba.com/stats/teamgamelogs"

SHOTS_SUBDIR = Path("raw") / "shots"
PLAYER_GAME_LOGS_SUBDIR = Path("raw") / "player_game_logs"
TEAM_GAME_LOGS_SUBDIR = Path("raw") / "team_game_logs"
ALL_DATASETS = {SHOT_CHART_DATASET, PLAYER_GAME_LOGS_DATASET, TEAM_GAME_LOGS_DATASET}


def shot_chart_output_stem(season: str) -> str:
    return f"{season}_shot_chart_raw"


def player_game_logs_output_stem(season: str) -> str:
    return f"{season}_player_game_logs_raw"


def team_game_logs_output_stem(season: str) -> str:
    return f"{season}_team_game_logs_raw"


def dataset_output_paths(data_dir: Path, season: str, subdir: Path, stem: str) -> dict[str, Path]:
    output_dir = data_dir / subdir
    return {
        "csv": output_dir / f"{stem}.csv",
        "parquet": output_dir / f"{stem}.parquet",
    }


def shot_chart_output_paths(data_dir: Path, season: str) -> dict[str, Path]:
    return dataset_output_paths(
        data_dir,
        season,
        SHOTS_SUBDIR,
        shot_chart_output_stem(season),
    )


def player_game_logs_output_paths(data_dir: Path, season: str) -> dict[str, Path]:
    return dataset_output_paths(
        data_dir,
        season,
        PLAYER_GAME_LOGS_SUBDIR,
        player_game_logs_output_stem(season),
    )


def team_game_logs_output_paths(data_dir: Path, season: str) -> dict[str, Path]:
    return dataset_output_paths(
        data_dir,
        season,
        TEAM_GAME_LOGS_SUBDIR,
        team_game_logs_output_stem(season),
    )


def metadata_path(data_dir: Path) -> Path:
    return data_dir / METADATA_SUBDIR / METADATA_FILENAME


def infer_dataset(entry: dict[str, Any]) -> str:
    dataset = entry.get("dataset")
    if isinstance(dataset, str):
        return dataset

    source = entry.get("source", "")
    if "shotchartdetail" in source:
        return SHOT_CHART_DATASET
    if "playergamelogs" in source:
        return PLAYER_GAME_LOGS_DATASET
    if "teamgamelogs" in source:
        return TEAM_GAME_LOGS_DATASET
    return "unknown"


def metadata_entry_key(entry: dict[str, Any]) -> tuple[str, str]:
    season = entry.get("season")
    if not isinstance(season, str):
        raise ValueError(f"Metadata entry missing season: {entry}")
    return infer_dataset(entry), season


def fetch_with_retries(
    fetch: Callable[[], pd.DataFrame],
    *,
    label: str,
    max_retries: int = MAX_RETRIES,
) -> pd.DataFrame:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return fetch()
        except Exception as exc:
            last_error = exc
            wait_seconds = REQUEST_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "%s request failed (attempt %s/%s): %s",
                label,
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(wait_seconds)

    assert last_error is not None
    raise RuntimeError(f"Failed to fetch {label}") from last_error


def fetch_team_shot_chart(
    team_id: int,
    season: str,
    *,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Fetch regular-season shot chart rows for one team."""

    def fetch() -> pd.DataFrame:
        response = shotchartdetail.ShotChartDetail(
            team_id=team_id,
            player_id=0,
            season_nullable=season,
            season_type_all_star="Regular Season",
            context_measure_simple="FGA",
            timeout=timeout,
        )
        frames = response.get_data_frames()
        if not frames:
            return pd.DataFrame()
        return frames[0]

    return fetch_with_retries(fetch, label=f"shot chart for team {team_id}")


def fetch_season_player_game_logs(
    season: str,
    *,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Fetch regular-season player game logs for the full league."""

    def fetch() -> pd.DataFrame:
        response = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable="Regular Season",
            timeout=timeout,
        )
        frames = response.get_data_frames()
        if not frames:
            return pd.DataFrame()
        return frames[0]

    return fetch_with_retries(fetch, label=f"player game logs for {season}")


def fetch_season_team_game_logs(
    season: str,
    *,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Fetch regular-season team game logs for the full league."""

    def fetch() -> pd.DataFrame:
        response = teamgamelogs.TeamGameLogs(
            season_nullable=season,
            season_type_nullable="Regular Season",
            timeout=timeout,
        )
        frames = response.get_data_frames()
        if not frames:
            return pd.DataFrame()
        return frames[0]

    return fetch_with_retries(fetch, label=f"team game logs for {season}")


def collect_season_shot_charts(season: str = DEFAULT_SEASON) -> pd.DataFrame:
    """Download shot chart detail for every NBA team in a season."""
    nba_teams = teams.get_teams()
    frames: list[pd.DataFrame] = []

    logger.info("Collecting %s regular-season shot charts for %s teams", season, len(nba_teams))

    for index, team in enumerate(nba_teams, start=1):
        team_id = int(team["id"])
        team_name = team["full_name"]
        logger.info("Fetching team %s/%s: %s (%s)", index, len(nba_teams), team_name, team_id)

        team_frame = fetch_team_shot_chart(team_id, season)
        if not team_frame.empty:
            frames.append(team_frame)
            logger.info("Retrieved %s rows for %s", len(team_frame), team_name)
        else:
            logger.info("No rows returned for %s", team_name)

        if index < len(nba_teams):
            time.sleep(REQUEST_DELAY_SECONDS)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    dedupe_keys = [col for col in ("GAME_ID", "GAME_EVENT_ID", "PLAYER_ID") if col in combined.columns]
    if dedupe_keys:
        combined = combined.drop_duplicates(subset=dedupe_keys, keep="first")

    logger.info("Collected %s total shot rows for season %s", len(combined), season)
    return combined


def collect_season_player_game_logs(season: str = DEFAULT_SEASON) -> pd.DataFrame:
    """Download regular-season player game logs for the full league."""
    logger.info("Collecting %s regular-season player game logs", season)
    game_logs = fetch_season_player_game_logs(season)

    if game_logs.empty:
        logger.info("No player game log rows returned for season %s", season)
        return game_logs

    dedupe_keys = [col for col in ("GAME_ID", "PLAYER_ID") if col in game_logs.columns]
    if dedupe_keys:
        game_logs = game_logs.drop_duplicates(subset=dedupe_keys, keep="first")

    logger.info("Collected %s player game log rows for season %s", len(game_logs), season)
    return game_logs


def collect_season_team_game_logs(season: str = DEFAULT_SEASON) -> pd.DataFrame:
    """Download regular-season team game logs for the full league."""
    logger.info("Collecting %s regular-season team game logs", season)
    game_logs = fetch_season_team_game_logs(season)

    if game_logs.empty:
        logger.info("No team game log rows returned for season %s", season)
        return game_logs

    dedupe_keys = [col for col in ("GAME_ID", "TEAM_ID") if col in game_logs.columns]
    if dedupe_keys:
        game_logs = game_logs.drop_duplicates(subset=dedupe_keys, keep="first")

    logger.info("Collected %s team game log rows for season %s", len(game_logs), season)
    return game_logs


def save_raw_outputs(df: pd.DataFrame, output_paths: dict[str, Path]) -> dict[str, Path]:
    """Write a dataframe to CSV and Parquet."""
    output_paths["csv"].parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_paths["csv"], index=False)
    df.to_parquet(output_paths["parquet"], index=False)

    logger.info("Saved CSV to %s", output_paths["csv"])
    logger.info("Saved Parquet to %s", output_paths["parquet"])
    return output_paths


def load_metadata_entries(data_dir: Path) -> list[dict[str, Any]]:
    path = metadata_path(data_dir)
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as metadata_file:
        payload = json.load(metadata_file)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        return payload["entries"]

    raise ValueError(f"Unexpected metadata format in {path}")


def write_metadata_entry(
    data_dir: Path,
    *,
    dataset: str,
    source: str,
    season: str,
    row_count: int,
    output_paths: dict[str, Path],
    downloaded_at: datetime | None = None,
) -> dict[str, Any]:
    """Append or replace a metadata record for the given dataset and season."""
    downloaded_at = downloaded_at or datetime.now(UTC)
    metadata_file = metadata_path(data_dir)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "dataset": dataset,
        "source": source,
        "season": season,
        "downloaded_at": downloaded_at.isoformat(),
        "row_count": row_count,
        "output_paths": {
            "csv": output_paths["csv"].as_posix(),
            "parquet": output_paths["parquet"].as_posix(),
        },
    }

    entries = load_metadata_entries(data_dir)
    entries = [
        existing
        for existing in entries
        if metadata_entry_key(existing) != (dataset, season)
    ]
    entries.append(entry)

    with metadata_file.open("w", encoding="utf-8") as metadata_handle:
        json.dump(entries, metadata_handle, indent=2)
        metadata_handle.write("\n")

    logger.info("Updated metadata at %s", metadata_file)
    return entry


def collect_and_save_shot_charts(
    season: str,
    data_dir: Path,
) -> dict[str, Any]:
    shot_data = collect_season_shot_charts(season)
    output_paths = save_raw_outputs(shot_data, shot_chart_output_paths(data_dir, season))
    return write_metadata_entry(
        data_dir,
        dataset=SHOT_CHART_DATASET,
        source=SHOT_CHART_SOURCE,
        season=season,
        row_count=len(shot_data),
        output_paths=output_paths,
    )


def collect_and_save_player_game_logs(
    season: str,
    data_dir: Path,
) -> dict[str, Any]:
    game_log_data = collect_season_player_game_logs(season)
    output_paths = save_raw_outputs(
        game_log_data,
        player_game_logs_output_paths(data_dir, season),
    )
    return write_metadata_entry(
        data_dir,
        dataset=PLAYER_GAME_LOGS_DATASET,
        source=PLAYER_GAME_LOGS_SOURCE,
        season=season,
        row_count=len(game_log_data),
        output_paths=output_paths,
    )


def collect_and_save_team_game_logs(
    season: str,
    data_dir: Path,
) -> dict[str, Any]:
    game_log_data = collect_season_team_game_logs(season)
    output_paths = save_raw_outputs(
        game_log_data,
        team_game_logs_output_paths(data_dir, season),
    )
    return write_metadata_entry(
        data_dir,
        dataset=TEAM_GAME_LOGS_DATASET,
        source=TEAM_GAME_LOGS_SOURCE,
        season=season,
        row_count=len(game_log_data),
        output_paths=output_paths,
    )


def collect_and_save(
    season: str = DEFAULT_SEASON,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    datasets: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Download selected raw datasets, save files, and record metadata."""
    resolved_data_dir = Path(data_dir)
    selected = datasets or ALL_DATASETS
    entries: list[dict[str, Any]] = []

    if SHOT_CHART_DATASET in selected:
        entries.append(collect_and_save_shot_charts(season, resolved_data_dir))
    if PLAYER_GAME_LOGS_DATASET in selected:
        entries.append(collect_and_save_player_game_logs(season, resolved_data_dir))
    if TEAM_GAME_LOGS_DATASET in selected:
        entries.append(collect_and_save_team_game_logs(season, resolved_data_dir))

    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NBA raw data for a season.")
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        help=f"NBA season string (default: {DEFAULT_SEASON})",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Project data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *sorted(ALL_DATASETS)],
        default="all",
        help="Which dataset to collect (default: all)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    if args.dataset == "all":
        datasets = ALL_DATASETS
    else:
        datasets = {args.dataset}

    entries = collect_and_save(season=args.season, data_dir=args.data_dir, datasets=datasets)
    for entry in entries:
        logger.info(
            "Finished %s collection for %s (%s rows)",
            entry["dataset"],
            entry["season"],
            entry["row_count"],
        )


if __name__ == "__main__":
    main()
