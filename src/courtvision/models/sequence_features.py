"""Play-by-play sequence features for GRU shot-make models."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from courtvision.data.build_features import NUM_REGULATION_PERIODS, REGULATION_PERIOD_SECONDS
from courtvision.data.collect import DEFAULT_SEASON
from courtvision.data.load_data import DEFAULT_ENV_PATH, create_database_engine

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 5

EVENT_FEATURE_COLUMNS: tuple[str, ...] = (
    "event_order_from_shot",
    "period",
    "seconds_remaining_period",
    "seconds_remaining_game",
    "seconds_since_next_event_or_shot",
    "same_team_as_shooter",
    "event_team_is_home",
    "score_margin_before_event",
    "is_field_goal",
    "is_made_shot_event",
    "is_missed_shot_event",
    "is_rebound",
    "is_turnover",
    "is_foul",
    "is_free_throw",
    "is_timeout",
    "is_substitution",
    "is_violation",
    "is_jump_ball",
    "is_unknown_event",
)

EVENT_TYPE_FLAG_COLUMNS: tuple[str, ...] = (
    "is_field_goal",
    "is_made_shot_event",
    "is_missed_shot_event",
    "is_rebound",
    "is_turnover",
    "is_foul",
    "is_free_throw",
    "is_timeout",
    "is_substitution",
    "is_violation",
    "is_jump_ball",
    "is_unknown_event",
)

_GAME_CLOCK_PATTERN = re.compile(
    r"^PT(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?$",
    re.IGNORECASE,
)

_SHOTS_FOR_SEQUENCES_SQL = """
SELECT
    s.shot_id,
    s.game_id,
    s.game_event_id,
    s.team_id,
    s.period,
    s.minutes_remaining,
    s.seconds_remaining,
    g.home_team_id,
    g.away_team_id
FROM shots AS s
INNER JOIN games AS g ON g.game_id = s.game_id
WHERE g.season = :season
ORDER BY s.game_id, s.game_event_id, s.shot_id
"""

_PLAY_BY_PLAY_FOR_SEQUENCES_SQL = """
SELECT
    pbp.game_id,
    pbp.action_number,
    pbp.period,
    pbp.game_clock,
    pbp.team_id,
    pbp.is_field_goal,
    pbp.shot_result,
    pbp.score_home,
    pbp.score_away,
    pbp.action_type,
    pbp.sub_type
FROM play_by_play AS pbp
INNER JOIN games AS g ON g.game_id = pbp.game_id
WHERE g.season = :season
ORDER BY pbp.game_id, pbp.action_number
"""


@dataclass(frozen=True)
class SequenceBuildResult:
    """Shot-aligned play-by-play sequences for GRU input."""

    sequences: np.ndarray
    shot_ids: np.ndarray
    game_ids: np.ndarray
    game_event_ids: np.ndarray
    sequence_length: int
    event_feature_count: int
    padded_slot_count: int
    total_slots: int
    max_prior_action_numbers: np.ndarray
    leakage_check_passed: bool


def parse_game_clock_seconds(game_clock: str) -> float:
    """Parse NBA PBP clock strings such as ``PT11M43.00S`` into period seconds."""
    match = _GAME_CLOCK_PATTERN.match(str(game_clock).strip())
    if match is None:
        return float("nan")
    minutes = int(match.group("minutes") or 0)
    seconds = float(match.group("seconds") or 0.0)
    return minutes * 60.0 + seconds


def seconds_remaining_game(period: int | float, seconds_remaining_period: float) -> float:
    """Regulation-aware seconds left in the game."""
    period_value = int(period)
    regulation_remaining = (NUM_REGULATION_PERIODS - period_value) * REGULATION_PERIOD_SECONDS
    if period_value <= NUM_REGULATION_PERIODS:
        return regulation_remaining + seconds_remaining_period
    return seconds_remaining_period


def shot_seconds_remaining_period(minutes_remaining: float, seconds_remaining: float) -> float:
    return float(minutes_remaining) * 60.0 + float(seconds_remaining)


def score_margin_from_shooter_perspective(
    score_home: float,
    score_away: float,
    *,
    shooter_team_id: int,
    home_team_id: int,
) -> float:
    """Pre-event margin from the shooting team's perspective."""
    if int(shooter_team_id) == int(home_team_id):
        return float(score_home) - float(score_away)
    return float(score_away) - float(score_home)


def _event_label(action_type: object, sub_type: object) -> str:
    return f"{action_type or ''} {sub_type or ''}".strip().upper()


def classify_event_type_flags(
    action_type: object,
    sub_type: object,
    *,
    is_field_goal: object = None,
    shot_result: object = None,
) -> dict[str, float]:
    """Map one play-by-play row to numeric/boolean event-type indicators."""
    label = _event_label(action_type, sub_type)
    shot_result_value = str(shot_result or "").strip().lower()

    is_made_shot = float(
        "MADE SHOT" in label
        or shot_result_value == "made"
        or (bool(is_field_goal) and shot_result_value == "made")
    )
    is_missed_shot = float(
        "MISSED SHOT" in label
        or shot_result_value == "missed"
        or (bool(is_field_goal) and shot_result_value == "missed")
    )
    is_rebound = float("REBOUND" in label)
    is_turnover = float("TURNOVER" in label)
    is_foul = float("FOUL" in label and "FREE THROW" not in label)
    is_free_throw = float("FREE THROW" in label)
    is_timeout = float("TIMEOUT" in label)
    is_substitution = float("SUBSTITUTION" in label or "SUB:" in label)
    is_violation = float("VIOLATION" in label)
    is_jump_ball = float("JUMP BALL" in label)

    is_fg = float(bool(is_field_goal) or is_made_shot or is_missed_shot)
    known = (
        is_fg
        or is_made_shot
        or is_missed_shot
        or is_rebound
        or is_turnover
        or is_foul
        or is_free_throw
        or is_timeout
        or is_substitution
        or is_violation
        or is_jump_ball
    )
    is_unknown = float(not known)

    return {
        "is_field_goal": is_fg,
        "is_made_shot_event": is_made_shot,
        "is_missed_shot_event": is_missed_shot,
        "is_rebound": is_rebound,
        "is_turnover": is_turnover,
        "is_foul": is_foul,
        "is_free_throw": is_free_throw,
        "is_timeout": is_timeout,
        "is_substitution": is_substitution,
        "is_violation": is_violation,
        "is_jump_ball": is_jump_ball,
        "is_unknown_event": is_unknown,
    }


def prepare_play_by_play_events(play_by_play: pd.DataFrame) -> pd.DataFrame:
    """Derive timing and pre-event score snapshots for play-by-play rows."""
    if play_by_play.empty:
        return play_by_play.copy()

    events = play_by_play.sort_values(["game_id", "action_number"]).copy()
    events["seconds_remaining_period"] = events["game_clock"].map(parse_game_clock_seconds)
    events["seconds_remaining_game"] = [
        seconds_remaining_game(period, period_seconds)
        for period, period_seconds in zip(
            events["period"],
            events["seconds_remaining_period"],
            strict=True,
        )
    ]

    for column in ("score_home", "score_away"):
        events[column] = events.groupby("game_id")[column].ffill()
    events["prior_score_home"] = events.groupby("game_id")["score_home"].shift(1).fillna(0)
    events["prior_score_away"] = events.groupby("game_id")["score_away"].shift(1).fillna(0)

    type_flags = events.apply(
        lambda row: classify_event_type_flags(
            row.get("action_type"),
            row.get("sub_type"),
            is_field_goal=row.get("is_field_goal"),
            shot_result=row.get("shot_result"),
        ),
        axis=1,
        result_type="expand",
    )
    # Drop raw columns that share names with encoded flags (e.g. ``is_field_goal``).
    events = events.drop(
        columns=[column for column in EVENT_TYPE_FLAG_COLUMNS if column in events.columns]
    )
    return pd.concat([events.reset_index(drop=True), type_flags.reset_index(drop=True)], axis=1)


def load_shots_for_sequences(
    engine: Engine,
    *,
    season: str = DEFAULT_SEASON,
) -> pd.DataFrame:
    """Load shot rows needed to align prior play-by-play events."""
    with engine.connect() as connection:
        frame = pd.read_sql(text(_SHOTS_FOR_SEQUENCES_SQL), connection, params={"season": season})
    return frame


def load_play_by_play_for_sequences(
    engine: Engine,
    *,
    season: str = DEFAULT_SEASON,
) -> pd.DataFrame:
    """Load play-by-play rows for sequence construction."""
    with engine.connect() as connection:
        frame = pd.read_sql(
            text(_PLAY_BY_PLAY_FOR_SEQUENCES_SQL),
            connection,
            params={"season": season},
        )
    return frame


def _series_scalar(event: pd.Series, column: str) -> object:
    """Return a single scalar from a row, even if duplicate column names exist."""
    value = event[column]
    if isinstance(value, pd.Series):
        return value.iloc[0]
    return value


def _encode_event_vector(
    event: pd.Series,
    *,
    event_order_from_shot: float,
    shooter_team_id: int,
    home_team_id: int,
    seconds_since_next_event_or_shot: float,
) -> np.ndarray:
    event_team_id = _series_scalar(event, "team_id")
    same_team = (
        float(int(event_team_id) == int(shooter_team_id))
        if pd.notna(event_team_id)
        else 0.0
    )
    event_team_is_home = (
        float(int(event_team_id) == int(home_team_id)) if pd.notna(event_team_id) else 0.0
    )
    score_margin = score_margin_from_shooter_perspective(
        _series_scalar(event, "prior_score_home"),
        _series_scalar(event, "prior_score_away"),
        shooter_team_id=shooter_team_id,
        home_team_id=home_team_id,
    )

    values: dict[str, float] = {
        "event_order_from_shot": event_order_from_shot,
        "period": float(_series_scalar(event, "period")),
        "seconds_remaining_period": float(_series_scalar(event, "seconds_remaining_period")),
        "seconds_remaining_game": float(_series_scalar(event, "seconds_remaining_game")),
        "seconds_since_next_event_or_shot": float(seconds_since_next_event_or_shot),
        "same_team_as_shooter": same_team,
        "event_team_is_home": event_team_is_home,
        "score_margin_before_event": score_margin,
    }
    for column in EVENT_TYPE_FLAG_COLUMNS:
        values[column] = float(_series_scalar(event, column))

    return np.asarray([values[column] for column in EVENT_FEATURE_COLUMNS], dtype=np.float32)


def _build_sequences_for_game(
    game_shots: pd.DataFrame,
    game_events: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sequence tensors for all shots in one game."""
    if game_shots.empty:
        return (
            np.empty((0, SEQUENCE_LENGTH, len(EVENT_FEATURE_COLUMNS)), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    action_numbers = game_events["action_number"].to_numpy(dtype=np.int64)
    shot_seconds_remaining_game = [
        seconds_remaining_game(
            row["period"],
            shot_seconds_remaining_period(row["minutes_remaining"], row["seconds_remaining"]),
        )
        for _, row in game_shots.iterrows()
    ]

    sequences: list[np.ndarray] = []
    max_prior_actions: list[int] = []
    feature_count = len(EVENT_FEATURE_COLUMNS)

    for shot_index, shot in enumerate(game_shots.itertuples(index=False)):
        prior_end = int(np.searchsorted(action_numbers, shot.game_event_id, side="left"))
        prior_events = game_events.iloc[max(0, prior_end - SEQUENCE_LENGTH) : prior_end]
        num_prior = len(prior_events)
        pad_count = SEQUENCE_LENGTH - num_prior

        sequence = np.zeros((SEQUENCE_LENGTH, feature_count), dtype=np.float32)
        shot_srg = float(shot_seconds_remaining_game[shot_index])

        for slot_index in range(SEQUENCE_LENGTH):
            if slot_index < pad_count:
                continue

            event_index = slot_index - pad_count
            event = prior_events.iloc[event_index]
            order = float(-(num_prior - event_index))
            if event_index < num_prior - 1:
                next_srg = float(prior_events.iloc[event_index + 1]["seconds_remaining_game"])
            else:
                next_srg = shot_srg
            seconds_since_next = float(event["seconds_remaining_game"]) - next_srg
            if seconds_since_next < 0.0:
                seconds_since_next = 0.0

            sequence[slot_index] = _encode_event_vector(
                event,
                event_order_from_shot=order,
                shooter_team_id=int(shot.team_id),
                home_team_id=int(shot.home_team_id),
                seconds_since_next_event_or_shot=seconds_since_next,
            )

        sequences.append(sequence)
        if num_prior == 0:
            max_prior_actions.append(-1)
        else:
            max_prior_actions.append(int(prior_events["action_number"].max()))

    return (
        np.stack(sequences, axis=0),
        game_shots["shot_id"].to_numpy(dtype=np.int64),
        np.asarray(max_prior_actions, dtype=np.int64),
    )


def build_shot_sequences(
    shots: pd.DataFrame,
    play_by_play: pd.DataFrame,
) -> SequenceBuildResult:
    """Build padded prior-event sequences for every shot.

    For a shot at ``game_event_id = N``, only play-by-play rows with
    ``action_number < N`` are eligible. The shot event itself is never included.
    """
    if shots.empty:
        return SequenceBuildResult(
            sequences=np.empty((0, SEQUENCE_LENGTH, len(EVENT_FEATURE_COLUMNS)), dtype=np.float32),
            shot_ids=np.empty(0, dtype=np.int64),
            game_ids=np.empty(0, dtype=object),
            game_event_ids=np.empty(0, dtype=np.int64),
            sequence_length=SEQUENCE_LENGTH,
            event_feature_count=len(EVENT_FEATURE_COLUMNS),
            padded_slot_count=0,
            total_slots=0,
            max_prior_action_numbers=np.empty(0, dtype=np.int64),
            leakage_check_passed=True,
        )

    events = prepare_play_by_play_events(play_by_play)
    events_by_game = {
        game_id: group.reset_index(drop=True)
        for game_id, group in events.groupby("game_id", sort=False)
    }

    sequence_chunks: list[np.ndarray] = []
    shot_id_chunks: list[np.ndarray] = []
    game_id_chunks: list[np.ndarray] = []
    game_event_id_chunks: list[np.ndarray] = []
    max_prior_chunks: list[np.ndarray] = []
    padded_slot_count = 0

    for game_id, game_shots in shots.groupby("game_id", sort=False):
        game_events = events_by_game.get(game_id)
        if game_events is None:
            game_events = pd.DataFrame(columns=events.columns)

        game_sequences, game_shot_ids, game_max_prior = _build_sequences_for_game(
            game_shots.reset_index(drop=True),
            game_events,
        )
        if len(game_sequences) == 0:
            continue

        num_prior_by_shot = np.sum(game_sequences[:, :, 0] != 0.0, axis=1)
        padded_slot_count += int(SEQUENCE_LENGTH * len(game_sequences) - num_prior_by_shot.sum())

        sequence_chunks.append(game_sequences)
        shot_id_chunks.append(game_shot_ids)
        game_id_chunks.append(np.full(len(game_shot_ids), game_id, dtype=object))
        game_event_id_chunks.append(game_shots["game_event_id"].to_numpy(dtype=np.int64))
        max_prior_chunks.append(game_max_prior)

    sequences = np.concatenate(sequence_chunks, axis=0)
    shot_ids = np.concatenate(shot_id_chunks, axis=0)
    game_ids = np.concatenate(game_id_chunks, axis=0)
    game_event_ids = np.concatenate(game_event_id_chunks, axis=0)
    max_prior_action_numbers = np.concatenate(max_prior_chunks, axis=0)

    has_prior = max_prior_action_numbers >= 0
    leakage_check_passed = bool(
        np.all(max_prior_action_numbers[has_prior] < game_event_ids[has_prior])
    )

    total_slots = SEQUENCE_LENGTH * len(sequences)
    return SequenceBuildResult(
        sequences=sequences,
        shot_ids=shot_ids,
        game_ids=game_ids,
        game_event_ids=game_event_ids,
        sequence_length=SEQUENCE_LENGTH,
        event_feature_count=len(EVENT_FEATURE_COLUMNS),
        padded_slot_count=padded_slot_count,
        total_slots=total_slots,
        max_prior_action_numbers=max_prior_action_numbers,
        leakage_check_passed=leakage_check_passed,
    )


def padded_event_rate(result: SequenceBuildResult) -> float:
    if result.total_slots == 0:
        return 0.0
    return result.padded_slot_count / result.total_slots


def print_sequence_build_checkpoint(
    *,
    loaded_shots: int,
    result: SequenceBuildResult,
) -> None:
    """Print the first GRU sequence-build checkpoint summary."""
    print(f"Loaded shots: {loaded_shots:,}")
    print(f"Built sequences: {len(result.sequences):,}")
    print(f"Sequence length: {result.sequence_length}")
    print(f"Event feature count: {result.event_feature_count}")
    print(f"Missing/padded event rate: {padded_event_rate(result):.4f}")
    if result.leakage_check_passed:
        print("Confirmed max previous action_number < shot game_event_id")
    else:
        print("Leakage check FAILED: prior action_number >= shot game_event_id detected")


def run_sequence_build_checkpoint(
    *,
    season: str = DEFAULT_SEASON,
    engine: Engine | None = None,
) -> SequenceBuildResult:
    """Load data, build sequences, and print the GRU v0 checkpoint."""
    db_engine = engine or create_database_engine(env_path=DEFAULT_ENV_PATH)
    shots = load_shots_for_sequences(db_engine, season=season)
    play_by_play = load_play_by_play_for_sequences(db_engine, season=season)
    result = build_shot_sequences(shots, play_by_play)
    print_sequence_build_checkpoint(loaded_shots=len(shots), result=result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prior play-by-play sequences for GRU shot-make models.",
    )
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season label (e.g. 2024-25)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    run_sequence_build_checkpoint(season=args.season)


if __name__ == "__main__":
    main()
