"""Load raw NBA parquet/CSV files into PostgreSQL (Phase 2).

Development reload flow (Option B):
  1. Apply sql/schema.sql once when setting up or changing the schema.
  2. Run this module; it truncates loaded tables, then inserts cleaned data.

Tables must be inserted in LOAD_ORDER so foreign keys succeed.
Column renames live in RAW_*_COLUMN_MAP; coerce types before insert.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from dotenv import load_dotenv
from nba_api.stats.static import teams as nba_teams
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

from courtvision.data.collect import (
    DEFAULT_DATA_DIR,
    DEFAULT_SEASON,
    normalize_game_id,
    player_game_logs_output_paths,
    play_by_play_output_paths,
    shot_chart_output_paths,
    team_game_logs_output_paths,
)

# --- Raw Parquet paths (season-based; mirrors collect.py output layout) --------
# Example for season="2024-25":
#   data/raw/shots/2024-25_shot_chart_raw.parquet
#   data/raw/player_game_logs/2024-25_player_game_logs_raw.parquet
#   data/raw/team_game_logs/2024-25_team_game_logs_raw.parquet
#   data/raw/play_by_play/2024-25_play_by_play_raw.parquet

RAW_SHOTS_PARQUET = "shots"
RAW_PLAYER_GAME_LOGS_PARQUET = "player_game_logs"
RAW_TEAM_GAME_LOGS_PARQUET = "team_game_logs"
RAW_PLAY_BY_PLAY_PARQUET = "play_by_play"

RAW_PARQUET_DATASETS: tuple[str, ...] = (
    RAW_SHOTS_PARQUET,
    RAW_PLAYER_GAME_LOGS_PARQUET,
    RAW_TEAM_GAME_LOGS_PARQUET,
    RAW_PLAY_BY_PLAY_PARQUET,
)


def raw_shot_chart_parquet_path(
    season: str = DEFAULT_SEASON,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Path:
    return shot_chart_output_paths(data_dir, season)["parquet"]


def raw_player_game_logs_parquet_path(
    season: str = DEFAULT_SEASON,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Path:
    return player_game_logs_output_paths(data_dir, season)["parquet"]


def raw_team_game_logs_parquet_path(
    season: str = DEFAULT_SEASON,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Path:
    return team_game_logs_output_paths(data_dir, season)["parquet"]


def raw_play_by_play_parquet_path(
    season: str = DEFAULT_SEASON,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Path:
    return play_by_play_output_paths(data_dir, season)["parquet"]


def season_raw_parquet_paths(
    season: str = DEFAULT_SEASON,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, Path]:
    """Return all raw Parquet paths for a season keyed by dataset name."""
    return {
        RAW_SHOTS_PARQUET: raw_shot_chart_parquet_path(season, data_dir=data_dir),
        RAW_PLAYER_GAME_LOGS_PARQUET: raw_player_game_logs_parquet_path(
            season, data_dir=data_dir
        ),
        RAW_TEAM_GAME_LOGS_PARQUET: raw_team_game_logs_parquet_path(season, data_dir=data_dir),
        RAW_PLAY_BY_PLAY_PARQUET: raw_play_by_play_parquet_path(season, data_dir=data_dir),
    }


class RawDatasets(TypedDict):
    shots_raw: pd.DataFrame
    player_logs_raw: pd.DataFrame
    team_logs_raw: pd.DataFrame
    play_by_play_raw: pd.DataFrame


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw Parquet not found: {path}. "
            "Run collect for this season first (python -m courtvision.data.collect --season ...)."
        )
    return pd.read_parquet(path)


def read_raw_datasets(
    season: str = DEFAULT_SEASON,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> RawDatasets:
    """Load all four raw Parquet files for a season into pandas DataFrames."""
    paths = season_raw_parquet_paths(season, data_dir=data_dir)
    return RawDatasets(
        shots_raw=_read_parquet(paths[RAW_SHOTS_PARQUET]),
        player_logs_raw=_read_parquet(paths[RAW_PLAYER_GAME_LOGS_PARQUET]),
        team_logs_raw=_read_parquet(paths[RAW_TEAM_GAME_LOGS_PARQUET]),
        play_by_play_raw=_read_parquet(paths[RAW_PLAY_BY_PLAY_PARQUET]),
    )


# --- Identity table builders --------------------------------------------------

NBA_STATIC_TEAM_COLUMN_MAP: dict[str, str] = {
    "id": "team_id",
    "abbreviation": "abbreviation",
    "full_name": "full_name",
    "nickname": "nickname",
    "city": "city",
    "state": "state",
    "year_founded": "year_founded",
}

TEAMS_TABLE_COLUMNS: tuple[str, ...] = tuple(NBA_STATIC_TEAM_COLUMN_MAP.values())


def build_teams_dataframe() -> pd.DataFrame:
    """Build the teams table from nba_api.stats.static.teams.get_teams()."""
    raw = pd.DataFrame(nba_teams.get_teams())
    teams_df = raw.rename(columns=NBA_STATIC_TEAM_COLUMN_MAP)[list(TEAMS_TABLE_COLUMNS)]
    teams_df["team_id"] = teams_df["team_id"].astype(int)
    teams_df["year_founded"] = pd.to_numeric(
        teams_df["year_founded"], errors="coerce"
    ).astype("Int64")
    return teams_df.sort_values("team_id", ignore_index=True)


PLAYERS_TABLE_COLUMNS: tuple[str, ...] = (
    "player_id",
    "full_name",
    "first_name",
    "last_name",
    "is_active",
)


def build_players_dataframe(
    shots_raw: pd.DataFrame,
    player_logs_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per player from shot chart and player game log raw data."""
    combined = pd.concat(
        [
            _player_id_name_frame(shots_raw),
            _player_id_name_frame(player_logs_raw),
        ],
        ignore_index=True,
    )
    if combined.empty:
        return pd.DataFrame(columns=list(PLAYERS_TABLE_COLUMNS))

    combined["full_name"] = combined["full_name"].astype(str).str.strip()
    combined = combined[combined["full_name"].ne("") & combined["full_name"].ne("nan")]
    combined["_name_len"] = combined["full_name"].str.len()
    combined = combined.sort_values(["player_id", "_name_len"], ascending=[True, False])
    combined = combined.drop_duplicates("player_id", keep="first").drop(columns="_name_len")

    first_last = combined["full_name"].map(_split_full_name)
    combined["first_name"] = [parts[0] for parts in first_last]
    combined["last_name"] = [parts[1] for parts in first_last]
    combined["is_active"] = True

    players_df = combined[list(PLAYERS_TABLE_COLUMNS)]
    return players_df.sort_values("player_id", ignore_index=True)


def _player_id_name_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Extract PLAYER_ID and PLAYER_NAME from a raw NBA Stats API dataframe."""
    if "PLAYER_ID" not in df.columns or "PLAYER_NAME" not in df.columns:
        return pd.DataFrame(columns=["player_id", "full_name"])

    frame = df[["PLAYER_ID", "PLAYER_NAME"]].rename(
        columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "full_name"}
    )
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    frame = frame.dropna(subset=["player_id", "full_name"])
    frame["player_id"] = frame["player_id"].astype(int)
    return frame.loc[frame["player_id"] > 0]


def _split_full_name(full_name: str) -> tuple[str | None, str | None]:
    parts = full_name.split()
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], " ".join(parts[1:])


GAMES_TABLE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "season_type",
    "game_date",
    "home_team_id",
    "away_team_id",
)

_TEAM_GAME_LOG_GAME_COLUMNS: tuple[str, ...] = (
    "GAME_ID",
    "SEASON_YEAR",
    "GAME_DATE",
    "MATCHUP",
)


def team_abbreviation_to_id_map(teams_df: pd.DataFrame) -> dict[str, int]:
    """Map team tricode -> team_id from the teams dimension table."""
    return teams_df.set_index("abbreviation")["team_id"].astype(int).to_dict()


def build_games_dataframe(
    team_logs_raw: pd.DataFrame,
    teams_df: pd.DataFrame,
    *,
    season_type: str = "Regular Season",
    shots_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one row per game from team game logs using MATCHUP home/away rules."""
    missing = [col for col in _TEAM_GAME_LOG_GAME_COLUMNS if col not in team_logs_raw.columns]
    if missing:
        raise ValueError(f"team_logs_raw missing required columns: {missing}")

    abbreviation_map = team_abbreviation_to_id_map(teams_df)
    logs = team_logs_raw[list(_TEAM_GAME_LOG_GAME_COLUMNS)].copy()
    logs["team_abbreviation"] = _team_abbreviation_series(team_logs_raw, teams_df)
    logs["game_id"] = logs["GAME_ID"].map(normalize_game_id)
    logs["game_date"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce").dt.date
    logs = logs.dropna(subset=["game_id", "game_date", "MATCHUP", "team_abbreviation"])

    game_rows: list[dict[str, Any]] = []
    for game_id, group in logs.groupby("game_id", sort=False):
        home_team_id, away_team_id = _resolve_game_home_away(
            group, abbreviation_map, shots_raw=shots_raw
        )
        first = group.iloc[0]
        game_rows.append(
            {
                "game_id": game_id,
                "season": str(first["SEASON_YEAR"]),
                "season_type": season_type,
                "game_date": first["game_date"],
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
            }
        )

    games = pd.DataFrame(game_rows, columns=list(GAMES_TABLE_COLUMNS))
    return games.sort_values("game_date", ignore_index=True)


def _team_abbreviation_series(team_logs_raw: pd.DataFrame, teams_df: pd.DataFrame) -> pd.Series:
    if "TEAM_ABBREVIATION" in team_logs_raw.columns:
        return team_logs_raw["TEAM_ABBREVIATION"].astype(str).str.upper()
    if "TEAM_ID" not in team_logs_raw.columns:
        raise ValueError("team_logs_raw needs TEAM_ABBREVIATION or TEAM_ID")

    id_to_abbreviation = teams_df.set_index("team_id")["abbreviation"]
    return team_logs_raw["TEAM_ID"].map(id_to_abbreviation).astype(str).str.upper()


def _resolve_game_home_away(
    game_rows: pd.DataFrame,
    abbreviation_to_team_id: dict[str, int],
    *,
    shots_raw: pd.DataFrame | None,
) -> tuple[int, int]:
    """Resolve home/away for one game from its team log rows."""
    parsed = [
        home_and_away_team_ids(
            str(row["MATCHUP"]),
            str(row["team_abbreviation"]),
            abbreviation_to_team_id,
        )
        for _, row in game_rows.iterrows()
    ]
    if len(parsed) == 1 or all(result == parsed[0] for result in parsed[1:]):
        return parsed[0]

    home_rows = game_rows[
        game_rows["MATCHUP"].str.contains(r"\bvs\.?\b", case=False, regex=True)
    ]
    if len(home_rows) == 1:
        row = home_rows.iloc[0]
        return home_and_away_team_ids(
            str(row["MATCHUP"]),
            str(row["team_abbreviation"]),
            abbreviation_to_team_id,
        )

    if shots_raw is not None:
        game_id = str(game_rows["game_id"].iloc[0])
        return _home_away_from_shot_chart(game_id, shots_raw, abbreviation_to_team_id)

    game_id = game_rows["game_id"].iloc[0]
    raise ValueError(
        f"Could not resolve home/away for game_id {game_id}. "
        "MATCHUP rows disagree and no shots_raw HTM/VTM fallback was provided."
    )


def _home_away_from_shot_chart(
    game_id: str,
    shots_raw: pd.DataFrame,
    abbreviation_to_team_id: dict[str, int],
) -> tuple[int, int]:
    """Use shot chart HTM (home) and VTM (away) when team logs are ambiguous."""
    if "HTM" not in shots_raw.columns or "VTM" not in shots_raw.columns:
        raise ValueError("shots_raw missing HTM/VTM columns for home/away fallback")

    game_shots = shots_raw[shots_raw["GAME_ID"].map(normalize_game_id) == game_id]
    if game_shots.empty:
        raise ValueError(f"No shot chart rows found for game_id {game_id}")

    home_tri = str(game_shots["HTM"].iloc[0]).upper()
    away_tri = str(game_shots["VTM"].iloc[0]).upper()
    return abbreviation_to_team_id[home_tri], abbreviation_to_team_id[away_tri]


# Foreign-key-safe insert order. ML tables are omitted until modeling phases.
LOAD_ORDER: tuple[str, ...] = (
    "teams",
    "players",
    "games",
    "shots",
    "player_game_logs",
    "team_game_logs",
    "play_by_play",
)

# --- Raw NBA Stats API column name -> SQL column name -----------------------

RAW_SHOT_COLUMN_MAP: dict[str, str] = {
    "GAME_ID": "game_id",
    "GAME_EVENT_ID": "game_event_id",
    "PLAYER_ID": "player_id",
    "TEAM_ID": "team_id",
    "PERIOD": "period",
    "MINUTES_REMAINING": "minutes_remaining",
    "SECONDS_REMAINING": "seconds_remaining",
    "EVENT_TYPE": "event_type",
    "ACTION_TYPE": "action_type",
    "SHOT_TYPE": "shot_type",
    "SHOT_ZONE_BASIC": "shot_zone_basic",
    "SHOT_ZONE_AREA": "shot_zone_area",
    "SHOT_ZONE_RANGE": "shot_zone_range",
    "SHOT_DISTANCE": "shot_distance",
    "LOC_X": "loc_x",
    "LOC_Y": "loc_y",
    "SHOT_ATTEMPTED_FLAG": "shot_attempted_flag",
    "SHOT_MADE_FLAG": "shot_made_flag",
    "GAME_DATE": "game_date",
    "HTM": "home_team_abbreviation",
    "VTM": "away_team_abbreviation",
}

RAW_PLAYER_GAME_LOG_COLUMN_MAP: dict[str, str] = {
    "SEASON_YEAR": "season_year",
    "GAME_ID": "game_id",
    "PLAYER_ID": "player_id",
    "TEAM_ID": "team_id",
    "GAME_DATE": "game_date",
    "MATCHUP": "matchup",
    "WL": "win_loss",
    "MIN": "minutes",
    "FGM": "field_goals_made",
    "FGA": "field_goals_attempted",
    "FG_PCT": "field_goal_pct",
    "FG3M": "three_pointers_made",
    "FG3A": "three_pointers_attempted",
    "FG3_PCT": "three_point_pct",
    "FTM": "free_throws_made",
    "FTA": "free_throws_attempted",
    "FT_PCT": "free_throw_pct",
    "OREB": "offensive_rebounds",
    "DREB": "defensive_rebounds",
    "REB": "rebounds",
    "AST": "assists",
    "TOV": "turnovers",
    "STL": "steals",
    "BLK": "blocks",
    "BLKA": "blocked_att",
    "PF": "personal_fouls",
    "PFD": "fouls_drawn",
    "PTS": "points",
    "PLUS_MINUS": "plus_minus",
}

RAW_TEAM_GAME_LOG_COLUMN_MAP: dict[str, str] = {
    "SEASON_YEAR": "season_year",
    "GAME_ID": "game_id",
    "TEAM_ID": "team_id",
    "GAME_DATE": "game_date",
    "MATCHUP": "matchup",
    "WL": "win_loss",
    "MIN": "minutes",
    "FGM": "field_goals_made",
    "FGA": "field_goals_attempted",
    "FG_PCT": "field_goal_pct",
    "FG3M": "three_pointers_made",
    "FG3A": "three_pointers_attempted",
    "FG3_PCT": "three_point_pct",
    "FTM": "free_throws_made",
    "FTA": "free_throws_attempted",
    "FT_PCT": "free_throw_pct",
    "OREB": "offensive_rebounds",
    "DREB": "defensive_rebounds",
    "REB": "rebounds",
    "AST": "assists",
    "TOV": "turnovers",
    "STL": "steals",
    "BLK": "blocks",
    "BLKA": "blocked_att",
    "PF": "personal_fouls",
    "PFD": "fouls_drawn",
    "PTS": "points",
    "PLUS_MINUS": "plus_minus",
}

RAW_PLAY_BY_PLAY_COLUMN_MAP: dict[str, str] = {
    "gameId": "game_id",
    "actionNumber": "action_number",
    "actionId": "action_id",
    "clock": "game_clock",
    "period": "period",
    "teamId": "team_id",
    "teamTricode": "team_tricode",
    "personId": "person_id",
    "playerName": "player_name",
    "xLegacy": "x_legacy",
    "yLegacy": "y_legacy",
    "shotDistance": "shot_distance",
    "shotResult": "shot_result",
    "isFieldGoal": "is_field_goal",
    "scoreHome": "score_home",
    "scoreAway": "score_away",
    "pointsTotal": "points_total",
    "location": "court_location",
    "description": "description",
    "actionType": "action_type",
    "subType": "sub_type",
    "videoAvailable": "video_available",
}

MATCHUP_PATTERN = re.compile(
    r"^(?P<team>[A-Z]{2,3})\s+(?P<sep>vs\.?|@)\s+(?P<opponent>[A-Z]{2,3})$",
    re.IGNORECASE,
)


def rename_raw_columns(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """Keep only mapped raw columns and rename them to SQL names."""
    present = {raw: sql for raw, sql in column_map.items() if raw in df.columns}
    return df[list(present)].rename(columns=present)


def parse_matchup_team_ids(
    matchup: str,
    team_abbreviation: str,
    abbreviation_to_team_id: dict[str, int],
) -> tuple[int, int]:
    """Return (row_team_id, opponent_team_id) for one team_game_logs row."""
    match = MATCHUP_PATTERN.match(matchup.strip())
    if not match:
        raise ValueError(f"Unrecognized MATCHUP format: {matchup!r}")

    row_tri = match.group("team").upper()
    opponent_tri = match.group("opponent").upper()

    if row_tri != team_abbreviation.upper():
        raise ValueError(
            f"MATCHUP team {row_tri} does not match row TEAM_ABBREVIATION {team_abbreviation}"
        )

    return (
        abbreviation_to_team_id[row_tri],
        abbreviation_to_team_id[opponent_tri],
    )


def home_and_away_team_ids(
    matchup: str,
    team_abbreviation: str,
    abbreviation_to_team_id: dict[str, int],
) -> tuple[int, int]:
    """Return (home_team_id, away_team_id) for a team_game_logs row.

    MATCHUP is relative to the row team (first tricode in the string):
    - "LAL vs. BOS" with row LAL -> LAL home, BOS away
    - "LAL @ BOS" with row LAL -> BOS home, LAL away
    """
    match = MATCHUP_PATTERN.match(matchup.strip())
    if not match:
        raise ValueError(f"Unrecognized MATCHUP format: {matchup!r}")

    row_team_id, opponent_team_id = parse_matchup_team_ids(
        matchup, team_abbreviation, abbreviation_to_team_id
    )
    sep = match.group("sep").lower()
    if sep.startswith("vs"):
        return row_team_id, opponent_team_id
    return opponent_team_id, row_team_id


def normalize_person_id(value: Any) -> int | None:
    """Map PBP personId for insert; 0 and missing mean no player."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    person_id = int(value)
    return person_id if person_id > 0 else None


def coerce_int(series: pd.Series) -> pd.Series:
    """Nullable integer series for SMALLINT columns."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def coerce_bool_from_flag(series: pd.Series) -> pd.Series:
    """NBA 0/1 flags -> boolean."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(lambda v: bool(v) if pd.notna(v) else pd.NA).astype("boolean")


def coerce_minutes(series: pd.Series) -> pd.Series:
    """MIN may be numeric or 'MM:SS' depending on endpoint/version."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.5:
        return numeric.round(1)
    parts = series.astype(str).str.split(":", n=1, expand=True)
    minutes = pd.to_numeric(parts[0], errors="coerce")
    seconds = pd.to_numeric(parts[1], errors="coerce")
    return (minutes + seconds / 60).round(1)


def coerce_game_date(series: pd.Series) -> pd.Series:
    """GAME_DATE may be YYYY-MM-DD, YYYYMMDD, or parsed strings."""
    if series.dtype == object and series.str.fullmatch(r"\d{8}", na=False).any():
        return pd.to_datetime(series, format="%Y%m%d", errors="coerce").dt.date
    return pd.to_datetime(series, errors="coerce").dt.date


SHOTS_TABLE_COLUMNS: tuple[str, ...] = tuple(RAW_SHOT_COLUMN_MAP.values())
SHOTS_NATURAL_KEY_COLUMNS: tuple[str, ...] = ("game_id", "game_event_id", "player_id")

_SHOTS_REQUIRED_NOT_NULL_COLUMNS: tuple[str, ...] = (
    *SHOTS_NATURAL_KEY_COLUMNS,
    "team_id",
    "period",
    "minutes_remaining",
    "seconds_remaining",
    "shot_made_flag",
    "game_date",
)

_SHOTS_INTEGER_COLUMNS: tuple[str, ...] = (
    "game_event_id",
    "player_id",
    "team_id",
    "period",
    "minutes_remaining",
    "seconds_remaining",
    "shot_distance",
    "loc_x",
    "loc_y",
)


def build_shots_dataframe(shots_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean shot chart rows for the shots table (shot quality model source)."""
    shots = rename_raw_columns(shots_raw, RAW_SHOT_COLUMN_MAP)

    shots["game_id"] = shots["game_id"].map(normalize_game_id)
    shots["game_date"] = coerce_game_date(shots["game_date"])
    shots["shot_attempted_flag"] = coerce_bool_from_flag(shots["shot_attempted_flag"])
    shots["shot_made_flag"] = coerce_bool_from_flag(shots["shot_made_flag"])

    for column in _SHOTS_INTEGER_COLUMNS:
        if column in shots.columns:
            shots[column] = coerce_int(shots[column])

    if "shot_attempted_flag" in shots.columns:
        shots["shot_attempted_flag"] = shots["shot_attempted_flag"].fillna(True)

    shots = shots.dropna(subset=list(_SHOTS_REQUIRED_NOT_NULL_COLUMNS))
    shots = shots[shots["player_id"] > 0]

    for column in _SHOTS_INTEGER_COLUMNS:
        if column in shots.columns:
            shots[column] = shots[column].astype(int)

    shots = shots.drop_duplicates(subset=list(SHOTS_NATURAL_KEY_COLUMNS), keep="first")
    return shots[list(SHOTS_TABLE_COLUMNS)].reset_index(drop=True)


PLAYER_GAME_LOGS_TABLE_COLUMNS: tuple[str, ...] = tuple(RAW_PLAYER_GAME_LOG_COLUMN_MAP.values())
PLAYER_GAME_LOGS_NATURAL_KEY_COLUMNS: tuple[str, ...] = ("game_id", "player_id")

TEAM_GAME_LOGS_TABLE_COLUMNS: tuple[str, ...] = tuple(RAW_TEAM_GAME_LOG_COLUMN_MAP.values())
TEAM_GAME_LOGS_NATURAL_KEY_COLUMNS: tuple[str, ...] = ("game_id", "team_id")

_GAME_LOG_BOX_SCORE_INTEGER_COLUMNS: tuple[str, ...] = (
    "field_goals_made",
    "field_goals_attempted",
    "three_pointers_made",
    "three_pointers_attempted",
    "free_throws_made",
    "free_throws_attempted",
    "offensive_rebounds",
    "defensive_rebounds",
    "rebounds",
    "assists",
    "turnovers",
    "steals",
    "blocks",
    "blocked_att",
    "personal_fouls",
    "fouls_drawn",
    "points",
    "plus_minus",
)

_GAME_LOG_PCT_COLUMNS: tuple[str, ...] = (
    "field_goal_pct",
    "three_point_pct",
    "free_throw_pct",
)


def build_player_game_logs_dataframe(player_logs_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean player game logs for rolling player context features."""
    return _clean_game_logs_dataframe(
        player_logs_raw,
        RAW_PLAYER_GAME_LOG_COLUMN_MAP,
        table_columns=PLAYER_GAME_LOGS_TABLE_COLUMNS,
        natural_key=PLAYER_GAME_LOGS_NATURAL_KEY_COLUMNS,
        require_player_id=True,
    )


def build_team_game_logs_dataframe(team_logs_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean team game logs for rolling team context features."""
    return _clean_game_logs_dataframe(
        team_logs_raw,
        RAW_TEAM_GAME_LOG_COLUMN_MAP,
        table_columns=TEAM_GAME_LOGS_TABLE_COLUMNS,
        natural_key=TEAM_GAME_LOGS_NATURAL_KEY_COLUMNS,
        require_player_id=False,
    )


def _clean_game_logs_dataframe(
    logs_raw: pd.DataFrame,
    column_map: dict[str, str],
    *,
    table_columns: tuple[str, ...],
    natural_key: tuple[str, ...],
    require_player_id: bool,
) -> pd.DataFrame:
    logs = rename_raw_columns(logs_raw, column_map)

    logs["game_id"] = logs["game_id"].map(normalize_game_id)
    logs["game_date"] = coerce_game_date(logs["game_date"])
    logs["season_year"] = logs["season_year"].astype(str)
    logs["matchup"] = logs["matchup"].astype(str).str.strip()
    logs["win_loss"] = logs["win_loss"].astype(str).str.strip().str.upper()
    logs["minutes"] = coerce_minutes(logs["minutes"])

    for column in ("player_id", "team_id"):
        if column in logs.columns:
            logs[column] = coerce_int(logs[column])

    for column in _GAME_LOG_BOX_SCORE_INTEGER_COLUMNS:
        if column in logs.columns:
            logs[column] = coerce_int(logs[column])

    for column in _GAME_LOG_PCT_COLUMNS:
        if column in logs.columns:
            logs[column] = pd.to_numeric(logs[column], errors="coerce").round(3)

    required_columns = [
        "season_year",
        "game_id",
        "game_date",
        "matchup",
        "win_loss",
        *natural_key,
    ]
    if "team_id" in logs.columns and "team_id" not in natural_key:
        required_columns.append("team_id")

    logs = logs.dropna(subset=required_columns)
    logs = logs[logs["win_loss"].isin(["W", "L"])]

    if require_player_id:
        logs = logs[logs["player_id"] > 0]
    if "team_id" in logs.columns:
        logs = logs[logs["team_id"] > 0]

    for column in ("player_id", "team_id"):
        if column in logs.columns:
            logs[column] = logs[column].astype(int)

    logs = logs.drop_duplicates(subset=list(natural_key), keep="first")
    return logs[list(table_columns)].reset_index(drop=True)


PLAY_BY_PLAY_TABLE_COLUMNS: tuple[str, ...] = tuple(RAW_PLAY_BY_PLAY_COLUMN_MAP.values())
PLAY_BY_PLAY_NATURAL_KEY_COLUMNS: tuple[str, ...] = ("game_id", "action_number")

_PLAY_BY_PLAY_REQUIRED_COLUMNS: tuple[str, ...] = (
    *PLAY_BY_PLAY_NATURAL_KEY_COLUMNS,
    "period",
    "game_clock",
)

_PLAY_BY_PLAY_INTEGER_COLUMNS: tuple[str, ...] = (
    "action_number",
    "period",
    "team_id",
    "x_legacy",
    "y_legacy",
    "shot_distance",
    "score_home",
    "score_away",
    "points_total",
)

_PLAY_BY_PLAY_BOOLEAN_COLUMNS: tuple[str, ...] = (
    "is_field_goal",
    "video_available",
)


def build_play_by_play_dataframe(play_by_play_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean play-by-play rows for sequence and game-context features."""
    pbp = rename_raw_columns(play_by_play_raw, RAW_PLAY_BY_PLAY_COLUMN_MAP)

    pbp["game_id"] = pbp["game_id"].map(normalize_game_id)
    pbp["game_clock"] = pbp["game_clock"].astype(str).str.strip()
    pbp["person_id"] = pd.array(
        pbp["person_id"].map(normalize_person_id),
        dtype="Int64",
    )

    if "action_id" in pbp.columns:
        pbp["action_id"] = pd.to_numeric(pbp["action_id"], errors="coerce").astype("Int64")

    for column in _PLAY_BY_PLAY_INTEGER_COLUMNS:
        if column in pbp.columns:
            pbp[column] = coerce_int(pbp[column])

    if "team_id" in pbp.columns:
        pbp["team_id"] = pbp["team_id"].where(pbp["team_id"] > 0)

    for column in _PLAY_BY_PLAY_BOOLEAN_COLUMNS:
        if column in pbp.columns:
            pbp[column] = coerce_bool_from_flag(pbp[column])

    if "player_name" in pbp.columns:
        pbp["player_name"] = pbp["player_name"].astype(str).str.strip()
        pbp.loc[pbp["player_name"].isin(["", "nan", "None"]), "player_name"] = pd.NA

    pbp = pbp.dropna(subset=list(_PLAY_BY_PLAY_REQUIRED_COLUMNS))

    for column in ("action_number", "period"):
        pbp[column] = pbp[column].astype(int)

    pbp = pbp.drop_duplicates(subset=list(PLAY_BY_PLAY_NATURAL_KEY_COLUMNS), keep="first")
    return pbp[list(PLAY_BY_PLAY_TABLE_COLUMNS)].reset_index(drop=True)


class CleanedDatasets(TypedDict):
    teams: pd.DataFrame
    players: pd.DataFrame
    games: pd.DataFrame
    shots: pd.DataFrame
    player_game_logs: pd.DataFrame
    team_game_logs: pd.DataFrame
    play_by_play: pd.DataFrame


# Clears basketball fact/dimension data before each load. ML placeholder tables are omitted.
_CLEAR_LOADED_TABLES_SQL = """
TRUNCATE TABLE
    play_by_play,
    player_game_logs,
    team_game_logs,
    shots,
    games,
    players,
    teams
RESTART IDENTITY CASCADE
"""


def load_environment(*, env_path: Path | None = None) -> None:
    """Load environment variables from the project .env file."""
    dotenv_path = env_path or DEFAULT_ENV_PATH
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
    else:
        load_dotenv()


def get_database_url(*, env_path: Path | None = None) -> str:
    """Read DATABASE_URL from the environment (after loading .env)."""
    load_environment(env_path=env_path)
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL is not set. Copy .env.example to .env and configure PostgreSQL, "
            "or export DATABASE_URL in your shell."
        )
    return database_url


def create_database_engine(
    *,
    database_url: str | None = None,
    env_path: Path | None = None,
) -> Engine:
    """Create a SQLAlchemy engine from DATABASE_URL."""
    url = database_url or get_database_url(env_path=env_path)
    return create_engine(url)


def build_cleaned_datasets(
    season: str = DEFAULT_SEASON,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> CleanedDatasets:
    """Read raw files and return cleaned DataFrames keyed by table name."""
    raw = read_raw_datasets(season, data_dir=data_dir)
    teams_df = build_teams_dataframe()
    players_df = build_players_dataframe(raw["shots_raw"], raw["player_logs_raw"])
    games_df = build_games_dataframe(
        raw["team_logs_raw"],
        teams_df,
        shots_raw=raw["shots_raw"],
    )
    return CleanedDatasets(
        teams=teams_df,
        players=players_df,
        games=games_df,
        shots=build_shots_dataframe(raw["shots_raw"]),
        player_game_logs=build_player_game_logs_dataframe(raw["player_logs_raw"]),
        team_game_logs=build_team_game_logs_dataframe(raw["team_logs_raw"]),
        play_by_play=build_play_by_play_dataframe(raw["play_by_play_raw"]),
    )


def clear_loaded_tables(engine: Engine) -> None:
    """Delete existing basketball rows before a reload (TRUNCATE ... RESTART IDENTITY)."""
    logger.info("Clearing loaded tables before insert")
    with engine.begin() as connection:
        connection.execute(text(_CLEAR_LOADED_TABLES_SQL))


def truncate_loaded_tables(engine: Engine) -> None:
    """Backward-compatible alias for :func:`clear_loaded_tables`."""
    clear_loaded_tables(engine)


def insert_dataframe(
    engine: Engine,
    table_name: str,
    dataframe: pd.DataFrame,
    *,
    chunksize: int = 5_000,
) -> None:
    """Append a cleaned DataFrame to a PostgreSQL table."""
    if dataframe.empty:
        logger.warning("Skipping empty insert for %s", table_name)
        return

    logger.info("Inserting %s rows into %s", len(dataframe), table_name)
    dataframe.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
        chunksize=chunksize,
        method="multi",
    )


def load_cleaned_datasets_to_postgres(
    datasets: CleanedDatasets | dict[str, pd.DataFrame],
    engine: Engine,
    *,
    chunksize: int = 5_000,
) -> None:
    """Insert cleaned tables in foreign-key-safe order."""
    for table_name in LOAD_ORDER:
        insert_dataframe(
            engine,
            table_name,
            datasets[table_name],
            chunksize=_chunksize_for_table(table_name, default=chunksize),
        )


def load_season_to_postgres(
    season: str = DEFAULT_SEASON,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    database_url: str | None = None,
    env_path: Path | None = None,
    clear_before_load: bool = True,
    chunksize: int = 5_000,
) -> CleanedDatasets:
    """Load .env, connect to PostgreSQL, and insert a season's cleaned data.

    By default, clears loaded tables first so you do not need to rerun schema.sql
    on every reload during development.
    """
    from courtvision.data.validate import validate_all_cleaned_datasets

    engine = create_database_engine(database_url=database_url, env_path=env_path)
    datasets = build_cleaned_datasets(season, data_dir=data_dir)

    validation_report = validate_all_cleaned_datasets(datasets, season=season)
    validation_report.log_issues()
    validation_report.raise_if_failed()

    if clear_before_load:
        clear_loaded_tables(engine)

    load_cleaned_datasets_to_postgres(datasets, engine, chunksize=chunksize)
    return datasets


def _chunksize_for_table(table_name: str, *, default: int) -> int:
    if table_name == "play_by_play":
        return max(default, 10_000)
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load cleaned NBA data into PostgreSQL.")
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season label (e.g. 2024-25)")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Project data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_PATH),
        help=f"Path to .env file (default: {DEFAULT_ENV_PATH})",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows without clearing tables first (default: clear then load)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=5_000,
        help="Rows per insert batch (default: 5000)",
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

    datasets = load_season_to_postgres(
        season=args.season,
        data_dir=Path(args.data_dir),
        env_path=Path(args.env_file),
        clear_before_load=not args.append,
        chunksize=args.chunksize,
    )
    for table_name in LOAD_ORDER:
        logger.info("Loaded %s rows into %s", len(datasets[table_name]), table_name)


if __name__ == "__main__":
    main()
