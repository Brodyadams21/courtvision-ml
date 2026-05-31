"""Download NBA shot chart data and persist raw files with collection metadata."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams

logger = logging.getLogger(__name__)

DEFAULT_SEASON = "2024-25"
DEFAULT_DATA_DIR = Path("data")
SHOTS_SUBDIR = Path("raw") / "shots"
METADATA_SUBDIR = Path("metadata")
METADATA_FILENAME = "data_collection_metadata.json"
SOURCE = "https://stats.nba.com/stats/shotchartdetail"
REQUEST_DELAY_SECONDS = 0.75
MAX_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 120


def season_output_stem(season: str) -> str:
    return f"{season}_shot_chart_raw"


def shots_output_paths(data_dir: Path, season: str) -> dict[str, Path]:
    stem = season_output_stem(season)
    shots_dir = data_dir / SHOTS_SUBDIR
    return {
        "csv": shots_dir / f"{stem}.csv",
        "parquet": shots_dir / f"{stem}.parquet",
    }


def metadata_path(data_dir: Path) -> Path:
    return data_dir / METADATA_SUBDIR / METADATA_FILENAME


def fetch_team_shot_chart(
    team_id: int,
    season: str,
    *,
    max_retries: int = MAX_RETRIES,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Fetch regular-season shot chart rows for one team."""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
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
        except Exception as exc:
            last_error = exc
            wait_seconds = REQUEST_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Shot chart request failed for team %s (attempt %s/%s): %s",
                team_id,
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(wait_seconds)

    assert last_error is not None
    raise RuntimeError(f"Failed to fetch shot chart for team {team_id}") from last_error


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


def save_raw_outputs(df: pd.DataFrame, data_dir: Path, season: str) -> dict[str, Path]:
    """Write combined shot data to CSV and Parquet under data/raw/shots/."""
    paths = shots_output_paths(data_dir, season)
    paths["csv"].parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(paths["csv"], index=False)
    df.to_parquet(paths["parquet"], index=False)

    logger.info("Saved CSV to %s", paths["csv"])
    logger.info("Saved Parquet to %s", paths["parquet"])
    return paths


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
    season: str,
    row_count: int,
    output_paths: dict[str, Path],
    downloaded_at: datetime | None = None,
) -> dict[str, Any]:
    """Append or replace a metadata record for the given season."""
    downloaded_at = downloaded_at or datetime.now(UTC)
    metadata_file = metadata_path(data_dir)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "source": SOURCE,
        "season": season,
        "downloaded_at": downloaded_at.isoformat(),
        "row_count": row_count,
        "output_paths": {
            "csv": output_paths["csv"].as_posix(),
            "parquet": output_paths["parquet"].as_posix(),
        },
    }

    entries = load_metadata_entries(data_dir)
    entries = [existing for existing in entries if existing.get("season") != season]
    entries.append(entry)

    with metadata_file.open("w", encoding="utf-8") as metadata_handle:
        json.dump(entries, metadata_handle, indent=2)
        metadata_handle.write("\n")

    logger.info("Updated metadata at %s", metadata_file)
    return entry


def collect_and_save(
    season: str = DEFAULT_SEASON,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    """Download season shot charts, save raw files, and record metadata."""
    resolved_data_dir = Path(data_dir)
    shot_data = collect_season_shot_charts(season)
    output_paths = save_raw_outputs(shot_data, resolved_data_dir, season)
    metadata_entry = write_metadata_entry(
        resolved_data_dir,
        season=season,
        row_count=len(shot_data),
        output_paths=output_paths,
    )
    return metadata_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NBA shot chart data for a season.")
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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    entry = collect_and_save(season=args.season, data_dir=args.data_dir)
    logger.info(
        "Finished collection for %s (%s rows)",
        entry["season"],
        entry["row_count"],
    )


if __name__ == "__main__":
    main()
