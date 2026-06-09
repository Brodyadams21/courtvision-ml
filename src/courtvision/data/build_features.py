"""Build model-ready shot features from PostgreSQL (Phase 4).

Base shot features join ``shots``, ``games``, and ``teams``. Player, team, and
opponent rolling features follow. ``score_margin`` is optional and non-blocking:
it joins ``play_by_play`` on prior-event scores (never the shot outcome) and stays
null when PBP alignment is missing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from courtvision.data.collect import DEFAULT_SEASON
from courtvision.data.load_data import (
    DEFAULT_ENV_PATH,
    PROJECT_ROOT,
    create_database_engine,
    insert_dataframe,
)
from courtvision.data.validate import validate_all_gold_shot_features

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_SET_VERSION = "base_v1"
GOLD_SHOT_FEATURES_TABLE = "gold_shot_features"
SCORE_MARGIN_COLUMN = "score_margin"
SCORE_MARGIN_MISSING_COLUMN = "score_margin_missing"
PLAYER_ROLLING_WINDOW = 5
TEAM_ROLLING_WINDOW = 5
ROLLING_WINDOW = PLAYER_ROLLING_WINDOW
FREE_THROW_REBOUND_FACTOR = 0.44
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_PROCESSED_FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"

PLAYER_ROLLING_FEATURE_COLUMNS: tuple[str, ...] = (
    "player_recent_fg_pct_5",
    "player_recent_fg3_pct_5",
    "player_recent_fga_5",
    "player_recent_fg3a_5",
    "player_recent_minutes_5",
    "player_recent_points_5",
)

TEAM_ROLLING_FEATURE_COLUMNS: tuple[str, ...] = (
    "team_recent_off_eff_proxy_5",
    "team_recent_pace_proxy_5",
    "team_recent_fg_pct_5",
    "team_recent_three_point_rate_5",
    "team_recent_fga_5",
    "team_recent_points_5",
    "team_recent_turnovers_5",
)

OPPONENT_ROLLING_FEATURE_COLUMNS: tuple[str, ...] = (
    "opp_recent_points_allowed_5",
    "opp_recent_fg_pct_allowed_5",
    "opp_recent_three_point_rate_allowed_5",
    "opp_recent_pace_proxy_5",
    "opp_recent_fga_allowed_5",
)

GOLD_SHOT_FEATURES_INSERT_COLUMNS: tuple[str, ...] = (
    "shot_id",
    "game_id",
    "game_event_id",
    "player_id",
    "team_id",
    "opponent_team_id",
    "season",
    "game_date",
    "feature_set_version",
    "shot_made_flag",
    "shot_value",
    "shot_distance",
    "loc_x",
    "loc_y",
    "abs_loc_x",
    "shot_angle",
    "is_corner_three",
    "is_home",
    "shot_zone_basic",
    "shot_zone_area",
    "shot_zone_range",
    "period",
    "seconds_remaining_period",
    "seconds_remaining_game",
    *PLAYER_ROLLING_FEATURE_COLUMNS,
    *TEAM_ROLLING_FEATURE_COLUMNS,
    *OPPONENT_ROLLING_FEATURE_COLUMNS,
    SCORE_MARGIN_COLUMN,
    SCORE_MARGIN_MISSING_COLUMN,
)

CORE_SHOT_FEATURE_COLUMNS: tuple[str, ...] = (
    "shot_id",
    "game_id",
    "game_event_id",
    "player_id",
    "team_id",
    "opponent_team_id",
    "season",
    "game_date",
    "shot_made_flag",
    "shot_value",
    "shot_distance",
    "loc_x",
    "loc_y",
    "abs_loc_x",
    "shot_angle",
    "is_corner_three",
    "is_home",
    "period",
    "seconds_remaining_period",
    "seconds_remaining_game",
    "shot_zone_basic",
    "shot_zone_area",
    "shot_zone_range",
    *PLAYER_ROLLING_FEATURE_COLUMNS,
    *TEAM_ROLLING_FEATURE_COLUMNS,
    *OPPONENT_ROLLING_FEATURE_COLUMNS,
)

REGULATION_PERIOD_SECONDS = 12 * 60
NUM_REGULATION_PERIODS = 4

CORNER_ZONE_BASIC_VALUES: frozenset[str] = frozenset({"Left Corner 3", "Right Corner 3"})
CORNER_THREE_ABS_LOC_X_MIN = 220
CORNER_THREE_LOC_Y_MAX = 92.5

BASE_SHOT_FEATURE_COLUMNS: tuple[str, ...] = (
    "shot_id",
    "game_id",
    "game_event_id",
    "player_id",
    "team_id",
    "opponent_team_id",
    "season",
    "game_date",
    "shot_made_flag",
    "shot_value",
    "shot_distance",
    "loc_x",
    "loc_y",
    "abs_loc_x",
    "shot_angle",
    "is_corner_three",
    "is_home",
    "period",
    "seconds_remaining_period",
    "seconds_remaining_game",
    "shot_zone_basic",
    "shot_zone_area",
    "shot_zone_range",
)

SHOT_FEATURE_COLUMNS: tuple[str, ...] = CORE_SHOT_FEATURE_COLUMNS + (
    SCORE_MARGIN_COLUMN,
    SCORE_MARGIN_MISSING_COLUMN,
)

_SHOT_SOURCES_SQL = """
SELECT
    s.shot_id,
    s.game_id,
    s.game_event_id,
    s.player_id,
    s.team_id,
    g.home_team_id,
    g.away_team_id,
    g.season,
    g.game_date,
    s.shot_made_flag,
    s.shot_type,
    s.shot_distance,
    s.loc_x,
    s.loc_y,
    s.period,
    s.minutes_remaining,
    s.seconds_remaining,
    s.shot_zone_basic,
    s.shot_zone_area,
    s.shot_zone_range
FROM shots AS s
INNER JOIN games AS g ON g.game_id = s.game_id
INNER JOIN teams AS t ON t.team_id = s.team_id
"""

_PLAYER_GAME_LOGS_SQL = """
SELECT
    pgl.game_id,
    pgl.player_id,
    pgl.game_date,
    pgl.field_goals_made,
    pgl.field_goals_attempted,
    pgl.three_pointers_made,
    pgl.three_pointers_attempted,
    pgl.minutes,
    pgl.points
FROM player_game_logs AS pgl
INNER JOIN games AS g ON g.game_id = pgl.game_id
"""

_TEAM_GAME_LOGS_SQL = """
SELECT
    tgl.game_id,
    tgl.team_id,
    tgl.game_date,
    tgl.field_goals_made,
    tgl.field_goals_attempted,
    tgl.three_pointers_attempted,
    tgl.free_throws_attempted,
    tgl.offensive_rebounds,
    tgl.turnovers,
    tgl.points
FROM team_game_logs AS tgl
INNER JOIN games AS g ON g.game_id = tgl.game_id
"""

_PLAY_BY_PLAY_SCORES_SQL = """
SELECT
    pbp.game_id,
    pbp.action_number,
    pbp.score_home,
    pbp.score_away
FROM play_by_play AS pbp
INNER JOIN games AS g ON g.game_id = pbp.game_id
"""


def query_shot_sources(
    engine: Engine,
    *,
    season: str | None = None,
) -> pd.DataFrame:
    """Load one row per shot from ``shots``, ``games``, and ``teams``."""
    sql = _SHOT_SOURCES_SQL
    params: dict[str, str] = {}
    if season is not None:
        sql += "\nWHERE g.season = :season"
        params["season"] = season

    with engine.connect() as connection:
        frame = pd.read_sql(text(sql), connection, params=params or None)

    if frame.empty:
        logger.warning("Shot source query returned no rows (season=%s)", season)
    else:
        logger.info("Loaded %s shot rows from PostgreSQL (season=%s)", len(frame), season)
    return frame


def query_play_by_play_scores(
    engine: Engine,
    *,
    season: str | None = None,
) -> pd.DataFrame:
    """Load PBP score snapshots keyed by ``game_id`` + ``action_number``."""
    sql = _PLAY_BY_PLAY_SCORES_SQL
    params: dict[str, str] = {}
    if season is not None:
        sql += "\n  AND g.season = :season"
        params["season"] = season

    with engine.connect() as connection:
        frame = pd.read_sql(text(sql), connection, params=params or None)

    logger.info("Loaded %s play-by-play rows with score snapshots", len(frame))
    return frame


def _add_null_score_margin(features: pd.DataFrame) -> pd.DataFrame:
    """Return ``features`` with null ``score_margin`` and missingness flag set."""
    if features.empty:
        return pd.DataFrame(columns=list(SHOT_FEATURE_COLUMNS))

    output = features.copy()
    output[SCORE_MARGIN_COLUMN] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output[SCORE_MARGIN_MISSING_COLUMN] = pd.Series(True, index=output.index, dtype=bool)
    return output[list(SHOT_FEATURE_COLUMNS)]


def _prepare_prior_pbp_scores(pbp_scores: pd.DataFrame) -> pd.DataFrame:
    """Build pre-shot score snapshots from the prior PBP event in each game."""
    pbp = pbp_scores.sort_values(["game_id", "action_number"]).copy()
    for column in ("score_home", "score_away"):
        pbp[column] = pbp.groupby("game_id")[column].ffill()
    pbp["prior_score_home"] = pbp.groupby("game_id")["score_home"].shift(1)
    pbp["prior_score_away"] = pbp.groupby("game_id")["score_away"].shift(1)
    pbp["prior_score_home"] = pbp["prior_score_home"].fillna(0)
    pbp["prior_score_away"] = pbp["prior_score_away"].fillna(0)
    return pbp.rename(columns={"action_number": "game_event_id"})[
        ["game_id", "game_event_id", "prior_score_home", "prior_score_away"]
    ]


def attach_score_margin(
    features: pd.DataFrame,
    pbp_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Join prior PBP scores and compute pre-shot margin from the shooting team's perspective.

    Matches shots on ``game_id`` + ``game_event_id`` = ``action_number``. Uses the score
    from the previous PBP event in the same game — never ``shot_made_flag`` or
    ``shot_value``. Unmatched shots keep ``score_margin`` null.
    """
    if features.empty:
        return _add_null_score_margin(features)

    prior_scores = _prepare_prior_pbp_scores(pbp_scores)
    merged = features.merge(
        prior_scores,
        on=["game_id", "game_event_id"],
        how="left",
        validate="m:1",
    )

    matched = merged["prior_score_home"].notna() & merged["prior_score_away"].notna()
    is_home = merged["is_home"].astype(bool)
    prior_home = merged["prior_score_home"].astype(float)
    prior_away = merged["prior_score_away"].astype(float)
    margin = np.where(is_home, prior_home - prior_away, prior_away - prior_home)

    merged[SCORE_MARGIN_COLUMN] = pd.array(
        np.where(matched, margin, np.nan),
        dtype=pd.Int64Dtype(),
    )
    merged[SCORE_MARGIN_MISSING_COLUMN] = merged[SCORE_MARGIN_COLUMN].isna().astype(bool)

    matched_count = int(matched.sum())
    total = len(features)
    logger.info(
        "Score margin attached for %s/%s shots (%.1f%%); unmatched remain null",
        matched_count,
        total,
        100.0 * matched_count / total if total else 0.0,
    )
    return merged.drop(columns=["prior_score_home", "prior_score_away"])[list(SHOT_FEATURE_COLUMNS)]


def attach_score_margin_from_db(
    features: pd.DataFrame,
    engine: Engine,
    *,
    season: str | None = None,
) -> pd.DataFrame:
    """Best-effort score margin attachment; failures leave the column null."""
    try:
        scores = query_play_by_play_scores(engine, season=season)
        return attach_score_margin(features, scores)
    except Exception as exc:
        logger.warning(
            "Score margin attachment failed (non-blocking): %s",
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return _add_null_score_margin(features)


def query_player_game_logs(engine: Engine) -> pd.DataFrame:
    """Load player game logs for rolling feature computation (all seasons in DB)."""
    with engine.connect() as connection:
        frame = pd.read_sql(text(_PLAYER_GAME_LOGS_SQL), connection)

    if frame.empty:
        logger.warning("Player game logs query returned no rows")
    else:
        logger.info("Loaded %s player game log rows for rolling features", len(frame))
    return frame


def query_team_game_logs(engine: Engine) -> pd.DataFrame:
    """Load team game logs for rolling feature computation (all seasons in DB)."""
    with engine.connect() as connection:
        frame = pd.read_sql(text(_TEAM_GAME_LOGS_SQL), connection)

    if frame.empty:
        logger.warning("Team game logs query returned no rows")
    else:
        logger.info("Loaded %s team game log rows for rolling features", len(frame))
    return frame


def _team_possessions_proxy(frame: pd.DataFrame) -> pd.Series:
    """Dean Oliver possession estimate from box score counting stats."""
    return (
        frame["field_goals_attempted"].astype(float)
        + FREE_THROW_REBOUND_FACTOR * frame["free_throws_attempted"].astype(float)
        + frame["turnovers"].astype(float)
        - frame["offensive_rebounds"].astype(float)
    )


def _shifted_rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """Shift by one row (exclude current game), then sum over ``window`` prior games."""
    return series.astype(float).shift(1).rolling(window, min_periods=1).sum()


def _shifted_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Shift by one row (exclude current game), then mean over ``window`` prior games."""
    return series.astype(float).shift(1).rolling(window, min_periods=1).mean()


def build_player_rolling_features(logs: pd.DataFrame) -> pd.DataFrame:
    """Build previous-5-game rolling player stats keyed by ``game_id`` + ``player_id``.

    Stats use only games before the current row (shift-then-roll within each player).
    """
    columns = ["game_id", "player_id", *PLAYER_ROLLING_FEATURE_COLUMNS]
    if logs.empty:
        return pd.DataFrame(columns=columns)

    window = ROLLING_WINDOW
    frame = logs.sort_values(["player_id", "game_date", "game_id"]).copy()
    grouped = frame.groupby("player_id", sort=False)

    fgm_sum = grouped["field_goals_made"].transform(
        lambda series: _shifted_rolling_sum(series, window)
    )
    fga_sum = grouped["field_goals_attempted"].transform(
        lambda series: _shifted_rolling_sum(series, window)
    )
    fg3m_sum = grouped["three_pointers_made"].transform(
        lambda series: _shifted_rolling_sum(series, window)
    )
    fg3a_sum = grouped["three_pointers_attempted"].transform(
        lambda series: _shifted_rolling_sum(series, window)
    )

    frame["player_recent_fg_pct_5"] = np.where(fga_sum > 0, fgm_sum / fga_sum, np.nan)
    frame["player_recent_fg3_pct_5"] = np.where(fg3a_sum > 0, fg3m_sum / fg3a_sum, np.nan)
    frame["player_recent_fga_5"] = grouped["field_goals_attempted"].transform(
        lambda series: _shifted_rolling_mean(series, window)
    )
    frame["player_recent_fg3a_5"] = grouped["three_pointers_attempted"].transform(
        lambda series: _shifted_rolling_mean(series, window)
    )
    frame["player_recent_minutes_5"] = grouped["minutes"].transform(
        lambda series: _shifted_rolling_mean(series, window)
    )
    frame["player_recent_points_5"] = grouped["points"].transform(
        lambda series: _shifted_rolling_mean(series, window)
    )

    rolling = frame[columns].drop_duplicates(subset=["game_id", "player_id"], keep="first")
    logger.info("Built player rolling features for %s game-player rows", len(rolling))
    return rolling.reset_index(drop=True)


def join_player_rolling_features(
    shot_features: pd.DataFrame,
    player_rolling: pd.DataFrame,
) -> pd.DataFrame:
    """Attach player rolling features to shots on ``game_id`` + ``player_id``."""
    if shot_features.empty:
        return shot_features

    join_cols = ["game_id", "player_id"]
    rolling_cols = list(PLAYER_ROLLING_FEATURE_COLUMNS)
    merged = shot_features.merge(
        player_rolling[join_cols + rolling_cols],
        on=join_cols,
        how="left",
        validate="m:1",
    )
    return merged


def build_team_rolling_features(logs: pd.DataFrame) -> pd.DataFrame:
    """Build previous-5-game rolling team stats keyed by ``game_id`` + ``team_id``.

    Stats use only games before the current row (shift-then-roll within each team).
    """
    columns = ["game_id", "team_id", *TEAM_ROLLING_FEATURE_COLUMNS]
    if logs.empty:
        return pd.DataFrame(columns=columns)

    window = ROLLING_WINDOW
    frame = logs.sort_values(["team_id", "game_date", "game_id"]).copy()
    frame["possessions"] = _team_possessions_proxy(frame)
    grouped = frame.groupby("team_id", sort=False)

    fgm_sum = grouped["field_goals_made"].transform(
        lambda series: _shifted_rolling_sum(series, window)
    )
    fga_sum = grouped["field_goals_attempted"].transform(
        lambda series: _shifted_rolling_sum(series, window)
    )
    fg3a_sum = grouped["three_pointers_attempted"].transform(
        lambda series: _shifted_rolling_sum(series, window)
    )
    points_sum = grouped["points"].transform(lambda series: _shifted_rolling_sum(series, window))
    poss_sum = grouped["possessions"].transform(lambda series: _shifted_rolling_sum(series, window))

    frame["team_recent_off_eff_proxy_5"] = np.where(poss_sum > 0, points_sum / poss_sum, np.nan)
    frame["team_recent_pace_proxy_5"] = grouped["possessions"].transform(
        lambda series: _shifted_rolling_mean(series, window)
    )
    frame["team_recent_fg_pct_5"] = np.where(fga_sum > 0, fgm_sum / fga_sum, np.nan)
    frame["team_recent_three_point_rate_5"] = np.where(fga_sum > 0, fg3a_sum / fga_sum, np.nan)
    frame["team_recent_fga_5"] = grouped["field_goals_attempted"].transform(
        lambda series: _shifted_rolling_mean(series, window)
    )
    frame["team_recent_points_5"] = grouped["points"].transform(
        lambda series: _shifted_rolling_mean(series, window)
    )
    frame["team_recent_turnovers_5"] = grouped["turnovers"].transform(
        lambda series: _shifted_rolling_mean(series, window)
    )

    rolling = frame[columns].drop_duplicates(subset=["game_id", "team_id"], keep="first")
    logger.info("Built team rolling features for %s game-team rows", len(rolling))
    return rolling.reset_index(drop=True)


def join_team_rolling_features(
    shot_features: pd.DataFrame,
    team_rolling: pd.DataFrame,
) -> pd.DataFrame:
    """Attach team rolling features to shots on ``game_id`` + ``team_id``."""
    if shot_features.empty:
        return shot_features

    join_cols = ["game_id", "team_id"]
    rolling_cols = list(TEAM_ROLLING_FEATURE_COLUMNS)
    return shot_features.merge(
        team_rolling[join_cols + rolling_cols],
        on=join_cols,
        how="left",
        validate="m:1",
    )


def _build_team_allowed_game_logs(logs: pd.DataFrame) -> pd.DataFrame:
    """Flip ``team_game_logs`` to a defensive view: opponent box-score stats allowed."""
    offense = logs.rename(
        columns={
            "team_id": "offense_team_id",
            "field_goals_made": "allowed_fgm",
            "field_goals_attempted": "allowed_fga",
            "three_pointers_attempted": "allowed_fg3a",
            "free_throws_attempted": "allowed_fta",
            "offensive_rebounds": "allowed_oreb",
            "turnovers": "allowed_turnovers",
            "points": "allowed_points",
        }
    )
    allowed = logs[["game_id", "team_id", "game_date"]].merge(
        offense[
            [
                "game_id",
                "offense_team_id",
                "allowed_fgm",
                "allowed_fga",
                "allowed_fg3a",
                "allowed_fta",
                "allowed_oreb",
                "allowed_turnovers",
                "allowed_points",
            ]
        ],
        on="game_id",
        how="inner",
    )
    allowed = allowed[allowed["team_id"] != allowed["offense_team_id"]].drop(
        columns=["offense_team_id"]
    )
    possessions_input = allowed.rename(
        columns={
            "allowed_fga": "field_goals_attempted",
            "allowed_fta": "free_throws_attempted",
            "allowed_oreb": "offensive_rebounds",
            "allowed_turnovers": "turnovers",
        }
    )
    allowed["allowed_possessions"] = _team_possessions_proxy(possessions_input)
    return allowed.reset_index(drop=True)


def build_opponent_rolling_features(logs: pd.DataFrame) -> pd.DataFrame:
    """Build previous-5-game rolling allowed stats keyed by ``game_id`` + ``team_id``.

    Each row reflects what that team (defense) allowed opponents to produce in prior games
    (shift-then-roll within each team).
    """
    columns = ["game_id", "team_id", *OPPONENT_ROLLING_FEATURE_COLUMNS]
    if logs.empty:
        return pd.DataFrame(columns=columns)

    window = ROLLING_WINDOW
    frame = _build_team_allowed_game_logs(logs)
    frame = frame.sort_values(["team_id", "game_date", "game_id"]).copy()
    grouped = frame.groupby("team_id", sort=False)

    fgm_sum = grouped["allowed_fgm"].transform(lambda series: _shifted_rolling_sum(series, window))
    fga_sum = grouped["allowed_fga"].transform(lambda series: _shifted_rolling_sum(series, window))
    fg3a_sum = grouped["allowed_fg3a"].transform(
        lambda series: _shifted_rolling_sum(series, window)
    )

    frame["opp_recent_points_allowed_5"] = grouped["allowed_points"].transform(
        lambda series: _shifted_rolling_mean(series, window)
    )
    frame["opp_recent_fg_pct_allowed_5"] = np.where(fga_sum > 0, fgm_sum / fga_sum, np.nan)
    frame["opp_recent_three_point_rate_allowed_5"] = np.where(
        fga_sum > 0,
        fg3a_sum / fga_sum,
        np.nan,
    )
    frame["opp_recent_pace_proxy_5"] = grouped["allowed_possessions"].transform(
        lambda series: _shifted_rolling_mean(series, window)
    )
    frame["opp_recent_fga_allowed_5"] = grouped["allowed_fga"].transform(
        lambda series: _shifted_rolling_mean(series, window)
    )

    rolling = frame[columns].drop_duplicates(subset=["game_id", "team_id"], keep="first")
    logger.info("Built opponent rolling features for %s game-team rows", len(rolling))
    return rolling.reset_index(drop=True)


def join_opponent_rolling_features(
    shot_features: pd.DataFrame,
    opponent_rolling: pd.DataFrame,
) -> pd.DataFrame:
    """Attach opponent rolling features on ``game_id`` + ``opponent_team_id``."""
    if shot_features.empty:
        return shot_features

    rolling_cols = list(OPPONENT_ROLLING_FEATURE_COLUMNS)
    opponent_view = opponent_rolling.rename(columns={"team_id": "opponent_team_id"})
    return shot_features.merge(
        opponent_view[["game_id", "opponent_team_id", *rolling_cols]],
        on=["game_id", "opponent_team_id"],
        how="left",
        validate="m:1",
    )


def _compute_opponent_team_id(frame: pd.DataFrame) -> pd.Series:
    is_home = frame["team_id"].eq(frame["home_team_id"])
    return np.where(is_home, frame["away_team_id"], frame["home_team_id"])


def _compute_shot_value(frame: pd.DataFrame) -> pd.Series:
    """Points value of the attempt: 3 for 3PT, 2 otherwise (independent of make/miss)."""
    is_three = frame["shot_type"].eq("3PT Field Goal")
    return pd.Series(np.where(is_three, 3, 2), index=frame.index, dtype="int64")


def _compute_seconds_remaining_period(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["minutes_remaining"].astype(float) * 60
        + frame["seconds_remaining"].astype(float)
    )


def _compute_seconds_remaining_game(frame: pd.DataFrame) -> pd.Series:
    period_seconds = _compute_seconds_remaining_period(frame)
    period = frame["period"].astype(int)
    regulation_remaining = (NUM_REGULATION_PERIODS - period) * REGULATION_PERIOD_SECONDS
    in_regulation = period <= NUM_REGULATION_PERIODS
    return pd.Series(
        np.where(in_regulation, regulation_remaining + period_seconds, period_seconds),
        index=frame.index,
        dtype="float64",
    )


def _compute_is_corner_three(frame: pd.DataFrame) -> pd.Series:
    zone_label = frame["shot_zone_basic"].isin(CORNER_ZONE_BASIC_VALUES)
    geometric = (
        frame["shot_type"].eq("3PT Field Goal")
        & frame["loc_x"].notna()
        & frame["loc_y"].notna()
        & frame["loc_x"].abs().ge(CORNER_THREE_ABS_LOC_X_MIN)
        & frame["loc_y"].le(CORNER_THREE_LOC_Y_MAX)
    )
    has_zone = frame["shot_zone_basic"].notna()
    return pd.Series(np.where(has_zone, zone_label, geometric), index=frame.index)


def build_base_shot_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive base shot features from a :func:`query_shot_sources` frame."""
    if frame.empty:
        return pd.DataFrame(columns=list(BASE_SHOT_FEATURE_COLUMNS))

    features = pd.DataFrame(index=frame.index)
    features["shot_id"] = frame["shot_id"].astype("int64")
    features["game_id"] = frame["game_id"].astype(str)
    features["game_event_id"] = frame["game_event_id"].astype("int64")
    features["player_id"] = frame["player_id"].astype("int64")
    features["team_id"] = frame["team_id"].astype("int64")
    features["opponent_team_id"] = _compute_opponent_team_id(frame).astype("int64")
    features["season"] = frame["season"].astype(str)
    features["game_date"] = pd.to_datetime(frame["game_date"]).dt.date
    features["shot_made_flag"] = frame["shot_made_flag"].astype(bool)
    features["shot_value"] = _compute_shot_value(frame)
    features["shot_distance"] = pd.to_numeric(frame["shot_distance"], errors="coerce").astype(
        "Int64"
    )
    features["loc_x"] = pd.to_numeric(frame["loc_x"], errors="coerce").astype("Int64")
    features["loc_y"] = pd.to_numeric(frame["loc_y"], errors="coerce").astype("Int64")
    features["abs_loc_x"] = features["loc_x"].abs()
    features["shot_angle"] = np.arctan2(
        features["loc_x"].astype("float64"),
        features["loc_y"].astype("float64"),
    )
    features["is_corner_three"] = _compute_is_corner_three(frame)
    features["is_home"] = frame["team_id"].eq(frame["home_team_id"])
    features["period"] = frame["period"].astype("int64")
    features["seconds_remaining_period"] = _compute_seconds_remaining_period(frame)
    features["seconds_remaining_game"] = _compute_seconds_remaining_game(frame)
    features["shot_zone_basic"] = frame["shot_zone_basic"]
    features["shot_zone_area"] = frame["shot_zone_area"]
    features["shot_zone_range"] = frame["shot_zone_range"]

    return features[list(BASE_SHOT_FEATURE_COLUMNS)].reset_index(drop=True)


def build_base_shot_features_from_db(
    engine: Engine,
    *,
    season: str | None = None,
) -> pd.DataFrame:
    """Query shot sources and return base shot features."""
    sources = query_shot_sources(engine, season=season)
    return build_base_shot_features(sources)


def build_shot_features_from_db(
    engine: Engine,
    *,
    season: str | None = None,
    include_score_margin: bool = True,
) -> pd.DataFrame:
    """Build core shot features, rolling joins, then optional score margin."""
    base = build_base_shot_features_from_db(engine, season=season)
    if base.empty:
        return pd.DataFrame(columns=list(SHOT_FEATURE_COLUMNS))

    player_logs = query_player_game_logs(engine)
    player_rolling = build_player_rolling_features(player_logs)
    merged = join_player_rolling_features(base, player_rolling)

    team_logs = query_team_game_logs(engine)
    team_rolling = build_team_rolling_features(team_logs)
    merged = join_team_rolling_features(merged, team_rolling)

    opponent_rolling = build_opponent_rolling_features(team_logs)
    merged = join_opponent_rolling_features(merged, opponent_rolling)

    if include_score_margin:
        return attach_score_margin_from_db(merged, engine, season=season)
    return _add_null_score_margin(merged)


def prepare_gold_shot_features_frame(
    base_features: pd.DataFrame,
    *,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
) -> pd.DataFrame:
    """Map base features to ``gold_shot_features`` insert columns."""
    if base_features.empty:
        return pd.DataFrame(columns=list(GOLD_SHOT_FEATURES_INSERT_COLUMNS))

    gold = base_features.copy()
    gold["feature_set_version"] = feature_set_version
    return gold[list(GOLD_SHOT_FEATURES_INSERT_COLUMNS)].reset_index(drop=True)


def clear_gold_shot_features(
    engine: Engine,
    *,
    season: str,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
) -> None:
    """Remove existing gold rows for a season and feature set before reload."""
    sql = text(
        """
        DELETE FROM gold_shot_features
        WHERE season = :season
          AND feature_set_version = :feature_set_version
        """
    )
    with engine.begin() as connection:
        result = connection.execute(
            sql,
            {"season": season, "feature_set_version": feature_set_version},
        )
    logger.info(
        "Cleared %s rows from %s (season=%s, version=%s)",
        result.rowcount,
        GOLD_SHOT_FEATURES_TABLE,
        season,
        feature_set_version,
    )


def load_base_shot_features_to_gold(
    engine: Engine,
    base_features: pd.DataFrame,
    *,
    season: str,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
    replace: bool = True,
    chunksize: int = 5_000,
) -> pd.DataFrame:
    """Insert base shot features into ``gold_shot_features``."""
    gold = prepare_gold_shot_features_frame(
        base_features,
        feature_set_version=feature_set_version,
    )
    if gold.empty:
        logger.warning("No base features to load into %s", GOLD_SHOT_FEATURES_TABLE)
        return gold

    if replace:
        clear_gold_shot_features(
            engine,
            season=season,
            feature_set_version=feature_set_version,
        )

    insert_dataframe(engine, GOLD_SHOT_FEATURES_TABLE, gold, chunksize=chunksize)
    logger.info(
        "Loaded %s rows into %s (season=%s, version=%s)",
        len(gold),
        GOLD_SHOT_FEATURES_TABLE,
        season,
        feature_set_version,
    )
    return gold


class GoldShotFeatureInspection:
    """Quality-check results for loaded gold shot features."""

    def __init__(self, *, season: str, feature_set_version: str) -> None:
        self.season = season
        self.feature_set_version = feature_set_version
        self.shots_row_count = 0
        self.gold_row_count = 0
        self.duplicate_shot_ids = 0
        self.invalid_shot_value_rows = 0
        self.missing_shot_made_flag = 0
        self.missing_opponent_team_id = 0
        self.is_home_mismatches = 0

    @property
    def row_count_delta(self) -> int:
        return self.gold_row_count - self.shots_row_count

    @property
    def passed(self) -> bool:
        return (
            self.gold_row_count > 0
            and self.row_count_delta == 0
            and self.duplicate_shot_ids == 0
            and self.invalid_shot_value_rows == 0
            and self.missing_shot_made_flag == 0
            and self.missing_opponent_team_id == 0
            and self.is_home_mismatches == 0
        )

    def log_summary(self) -> None:
        logger.info(
            "Gold inspection season=%s version=%s: shots=%s gold=%s delta=%s "
            "dup_shot_id=%s bad_shot_value=%s missing_made=%s missing_opp=%s "
            "is_home_mismatch=%s passed=%s",
            self.season,
            self.feature_set_version,
            self.shots_row_count,
            self.gold_row_count,
            self.row_count_delta,
            self.duplicate_shot_ids,
            self.invalid_shot_value_rows,
            self.missing_shot_made_flag,
            self.missing_opponent_team_id,
            self.is_home_mismatches,
            self.passed,
        )


def inspect_gold_shot_features(
    engine: Engine,
    *,
    season: str,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
) -> GoldShotFeatureInspection:
    """Run post-load checks on ``gold_shot_features``."""
    inspection = GoldShotFeatureInspection(
        season=season,
        feature_set_version=feature_set_version,
    )
    params = {"season": season, "feature_set_version": feature_set_version}

    with engine.connect() as connection:
        inspection.shots_row_count = int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM shots AS s
                    INNER JOIN games AS g ON g.game_id = s.game_id
                    WHERE g.season = :season
                    """
                ),
                params,
            ).scalar_one()
        )
        inspection.gold_row_count = int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM gold_shot_features
                    WHERE season = :season
                      AND feature_set_version = :feature_set_version
                    """
                ),
                params,
            ).scalar_one()
        )
        inspection.duplicate_shot_ids = int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*) - COUNT(DISTINCT shot_id)
                    FROM gold_shot_features
                    WHERE season = :season
                      AND feature_set_version = :feature_set_version
                    """
                ),
                params,
            ).scalar_one()
        )
        inspection.invalid_shot_value_rows = int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM gold_shot_features
                    WHERE season = :season
                      AND feature_set_version = :feature_set_version
                      AND shot_value NOT IN (2, 3)
                    """
                ),
                params,
            ).scalar_one()
        )
        inspection.missing_shot_made_flag = int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM gold_shot_features
                    WHERE season = :season
                      AND feature_set_version = :feature_set_version
                      AND shot_made_flag IS NULL
                    """
                ),
                params,
            ).scalar_one()
        )
        inspection.missing_opponent_team_id = int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM gold_shot_features
                    WHERE season = :season
                      AND feature_set_version = :feature_set_version
                      AND opponent_team_id IS NULL
                    """
                ),
                params,
            ).scalar_one()
        )
        inspection.is_home_mismatches = int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM gold_shot_features AS g
                    INNER JOIN games AS gl ON gl.game_id = g.game_id
                    WHERE g.season = :season
                      AND g.feature_set_version = :feature_set_version
                      AND (
                          (g.is_home IS TRUE AND g.team_id <> gl.home_team_id)
                          OR (g.is_home IS FALSE AND g.team_id <> gl.away_team_id)
                      )
                    """
                ),
                params,
            ).scalar_one()
        )

    inspection.log_summary()
    return inspection


def processed_train_test_paths(
    season: str,
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Default train/test Parquet paths for a season."""
    directory = output_dir or DEFAULT_PROCESSED_FEATURES_DIR
    return {
        "train": directory / f"train_shot_features_{season}.parquet",
        "test": directory / f"test_shot_features_{season}.parquet",
    }


def query_gold_shot_features(
    engine: Engine,
    *,
    season: str,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
) -> pd.DataFrame:
    """Load model-ready rows from ``gold_shot_features``."""
    columns_sql = ", ".join(SHOT_FEATURE_COLUMNS)
    sql = text(
        f"""
        SELECT {columns_sql}
        FROM gold_shot_features
        WHERE season = :season
          AND feature_set_version = :feature_set_version
        """
    )
    with engine.connect() as connection:
        frame = pd.read_sql(
            sql,
            connection,
            params={"season": season, "feature_set_version": feature_set_version},
        )

    if frame.empty:
        logger.warning(
            "No gold shot features found (season=%s, version=%s)",
            season,
            feature_set_version,
        )
    else:
        logger.info(
            "Loaded %s gold shot feature rows (season=%s, version=%s)",
            len(frame),
            season,
            feature_set_version,
        )
    return frame


def time_based_train_test_split(
    features: pd.DataFrame,
    *,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split shots by game date: earliest ``train_fraction`` of games -> train."""
    if features.empty:
        return (
            features.copy(),
            features.copy(),
            {"train_games": 0, "test_games": 0, "train_rows": 0, "test_rows": 0},
        )

    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1 (got {train_fraction})")

    game_dates = (
        features[["game_id", "game_date"]]
        .drop_duplicates(subset=["game_id"])
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )
    n_games = len(game_dates)
    if n_games == 0:
        raise ValueError("No games available for time-based split")

    if n_games == 1:
        n_train_games = 1
    else:
        n_train_games = max(1, min(n_games - 1, int(n_games * train_fraction)))

    train_game_ids = set(game_dates.iloc[:n_train_games]["game_id"])
    test_game_ids = set(game_dates.iloc[n_train_games:]["game_id"])
    if train_game_ids & test_game_ids:
        raise RuntimeError("Train and test game sets overlap")

    train = features[features["game_id"].isin(train_game_ids)].copy()
    test = features[features["game_id"].isin(test_game_ids)].copy()
    sort_cols = ["game_date", "game_id", "shot_id"]
    train = train.sort_values(sort_cols).reset_index(drop=True)
    test = test.sort_values(sort_cols).reset_index(drop=True)

    train_dates = game_dates.iloc[:n_train_games]["game_date"]
    test_dates = game_dates.iloc[n_train_games:]["game_date"]
    metadata: dict[str, Any] = {
        "train_games": len(train_game_ids),
        "test_games": len(test_game_ids),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_fraction": train_fraction,
        "train_min_game_date": train_dates.min() if not train_dates.empty else None,
        "train_max_game_date": train_dates.max() if not train_dates.empty else None,
        "test_min_game_date": test_dates.min() if not test_dates.empty else None,
        "test_max_game_date": test_dates.max() if not test_dates.empty else None,
    }
    logger.info(
        "Time-based split: train %s games (%s rows, %s to %s) | test %s games (%s rows, %s to %s)",
        metadata["train_games"],
        metadata["train_rows"],
        metadata["train_min_game_date"],
        metadata["train_max_game_date"],
        metadata["test_games"],
        metadata["test_rows"],
        metadata["test_min_game_date"],
        metadata["test_max_game_date"],
    )
    return train, test, metadata


def export_train_test_shot_features(
    features: pd.DataFrame,
    *,
    season: str,
    output_dir: Path | None = None,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> dict[str, Any]:
    """Export time-based train/test Parquet files for modeling."""
    paths = processed_train_test_paths(season, output_dir=output_dir)
    train, test, split_meta = time_based_train_test_split(
        features,
        train_fraction=train_fraction,
    )

    paths["train"].parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(paths["train"], index=False)
    test.to_parquet(paths["test"], index=False)
    logger.info("Wrote train features to %s", paths["train"])
    logger.info("Wrote test features to %s", paths["test"])

    return {
        **split_meta,
        "train_path": paths["train"],
        "test_path": paths["test"],
    }


def export_train_test_from_gold(
    engine: Engine,
    *,
    season: str,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
    output_dir: Path | None = None,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> dict[str, Any]:
    """Load ``gold_shot_features`` and export train/test Parquet files."""
    features = query_gold_shot_features(
        engine,
        season=season,
        feature_set_version=feature_set_version,
    )
    if features.empty:
        raise ValueError(
            f"No gold shot features to export for season={season!r}, "
            f"version={feature_set_version!r}"
        )
    return export_train_test_shot_features(
        features,
        season=season,
        output_dir=output_dir,
        train_fraction=train_fraction,
    )


def build_load_and_inspect_base_shot_features(
    engine: Engine,
    *,
    season: str,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
    replace: bool = True,
    chunksize: int = 5_000,
    include_score_margin: bool = True,
) -> tuple[pd.DataFrame, GoldShotFeatureInspection]:
    """Build shot features, validate, load into gold, and inspect (score margin optional)."""
    features = build_shot_features_from_db(
        engine,
        season=season,
        include_score_margin=include_score_margin,
    )
    validation_report = validate_all_gold_shot_features(
        features,
        season=season,
        feature_set_version=feature_set_version,
    )
    validation_report.log_issues()
    validation_report.raise_if_failed()

    load_base_shot_features_to_gold(
        engine,
        features,
        season=season,
        feature_set_version=feature_set_version,
        replace=replace,
        chunksize=chunksize,
    )
    inspection = inspect_gold_shot_features(
        engine,
        season=season,
        feature_set_version=feature_set_version,
    )
    if not inspection.passed:
        raise RuntimeError(
            f"Gold shot feature inspection failed for season={season!r}, "
            f"version={feature_set_version!r}. See log for details."
        )
    return features, inspection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build base shot features from PostgreSQL.")
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season label (e.g. 2024-25)")
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_PATH),
        help=f"Path to .env file (default: {DEFAULT_ENV_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional Parquet path for the feature frame",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Insert base features into gold_shot_features",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Run post-load inspection checks (implies --load unless data already loaded)",
    )
    parser.add_argument(
        "--feature-set-version",
        default=DEFAULT_FEATURE_SET_VERSION,
        help=f"Feature set version label (default: {DEFAULT_FEATURE_SET_VERSION})",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append gold rows without clearing the season/version first",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=5_000,
        help=(
            "Target rows per insert batch when loading gold (default: 5000). "
            "Automatically reduced if needed for PostgreSQL bind limits."
        ),
    )
    parser.add_argument(
        "--skip-score-margin",
        action="store_true",
        help="Skip play-by-play score margin join (column stays null)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export time-based train/test Parquet files after gold is ready",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export train/test Parquet from existing gold_shot_features (no rebuild)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_FEATURES_DIR,
        help=f"Output directory for train/test Parquet (default: {DEFAULT_PROCESSED_FEATURES_DIR})",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help="Fraction of earliest games by date for train split (default: 0.8)",
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

    engine = create_database_engine(env_path=Path(args.env_file))
    do_load = args.load or args.inspect
    features: pd.DataFrame | None = None

    if args.export_only:
        export_train_test_from_gold(
            engine,
            season=args.season,
            feature_set_version=args.feature_set_version,
            output_dir=args.processed_dir,
            train_fraction=args.train_fraction,
        )
        return

    if do_load:
        features, inspection = build_load_and_inspect_base_shot_features(
            engine,
            season=args.season,
            feature_set_version=args.feature_set_version,
            replace=not args.append,
            chunksize=args.chunksize,
            include_score_margin=not args.skip_score_margin,
        )
        logger.info("Gold inspection passed for %s rows", len(features))
    else:
        features = build_shot_features_from_db(
            engine,
            season=args.season,
            include_score_margin=not args.skip_score_margin,
        )
        logger.info("Built %s shot feature rows", len(features))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(args.output, index=False)
        logger.info("Wrote %s", args.output)

    if args.export:
        export_source = features
        if do_load:
            export_source = query_gold_shot_features(
                engine,
                season=args.season,
                feature_set_version=args.feature_set_version,
            )
        export_train_test_shot_features(
            export_source,
            season=args.season,
            output_dir=args.processed_dir,
            train_fraction=args.train_fraction,
        )


if __name__ == "__main__":
    main()
