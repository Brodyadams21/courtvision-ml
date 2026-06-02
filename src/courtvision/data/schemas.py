"""Pandera DataFrame schemas for cleaned CourtVision ML tables (Layer 1 validation).

Column types, nullability, allowed values, and numeric ranges live here.
Season thresholds, duplicate keys, FK sanity, and play-by-play warnings stay in validate.py.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

VALID_SHOT_TYPES: frozenset[str] = frozenset({"2PT Field Goal", "3PT Field Goal"})

LOC_X_BOUNDS = (-300, 300)
LOC_Y_BOUNDS = (-100, 900)
SHOT_DISTANCE_BOUNDS = (0, 100)

_WIN_LOSS_VALUES = ("W", "L")


def _game_date_is_valid(series: pd.Series) -> bool:
    parsed = pd.to_datetime(series, errors="coerce")
    return bool(parsed.notna().all())


shots_schema = DataFrameSchema(
    {
        "game_id": Column(str, nullable=False),
        "game_event_id": Column(int, nullable=False),
        "player_id": Column(int, nullable=False),
        "team_id": Column(int, nullable=False),
        "shot_made_flag": Column(bool, nullable=False),
        "shot_type": Column(str, Check.isin(sorted(VALID_SHOT_TYPES)), nullable=False),
        "loc_x": Column(int, Check.in_range(*LOC_X_BOUNDS), nullable=True),
        "loc_y": Column(int, Check.in_range(*LOC_Y_BOUNDS), nullable=True),
        "shot_distance": Column(int, Check.in_range(*SHOT_DISTANCE_BOUNDS), nullable=True),
        "game_date": Column(object, Check(_game_date_is_valid), nullable=False),
    },
    coerce=True,
    strict=False,
    name="shots",
)

games_schema = DataFrameSchema(
    {
        "game_id": Column(str, nullable=False),
        "season": Column(str, nullable=False),
        "season_type": Column(str, nullable=False),
        "game_date": Column(object, Check(_game_date_is_valid), nullable=False),
        "home_team_id": Column(int, nullable=False),
        "away_team_id": Column(int, nullable=False),
    },
    coerce=True,
    strict=False,
    name="games",
)

player_game_logs_schema = DataFrameSchema(
    {
        "season_year": Column(str, nullable=False),
        "game_id": Column(str, nullable=False),
        "player_id": Column(int, nullable=False),
        "team_id": Column(int, nullable=False),
        "game_date": Column(object, Check(_game_date_is_valid), nullable=False),
        "matchup": Column(str, nullable=False),
        "win_loss": Column(str, Check.isin(_WIN_LOSS_VALUES), nullable=False),
    },
    coerce=True,
    strict=False,
    name="player_game_logs",
)

team_game_logs_schema = DataFrameSchema(
    {
        "season_year": Column(str, nullable=False),
        "game_id": Column(str, nullable=False),
        "team_id": Column(int, nullable=False),
        "game_date": Column(object, Check(_game_date_is_valid), nullable=False),
        "matchup": Column(str, nullable=False),
        "win_loss": Column(str, Check.isin(_WIN_LOSS_VALUES), nullable=False),
    },
    coerce=True,
    strict=False,
    name="team_game_logs",
)

play_by_play_schema = DataFrameSchema(
    {
        "game_id": Column(str, nullable=False),
        "action_number": Column(int, nullable=False),
        "period": Column(int, nullable=False),
        "game_clock": Column(
            str,
            Check.str_length(min_value=1),
            nullable=False,
        ),
        "team_id": Column("Int64", nullable=True),
        "person_id": Column("Int64", nullable=True),
        "player_name": Column(str, nullable=True),
        "description": Column(str, nullable=True),
    },
    coerce=True,
    strict=False,
    name="play_by_play",
)

TABLE_SCHEMAS: dict[str, DataFrameSchema] = {
    "shots": shots_schema,
    "games": games_schema,
    "player_game_logs": player_game_logs_schema,
    "team_game_logs": team_game_logs_schema,
    "play_by_play": play_by_play_schema,
}
