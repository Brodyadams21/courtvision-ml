"""Tests for LightGBM training helpers."""

from __future__ import annotations

import pandas as pd

from courtvision.models.common import FEATURE_COLUMNS, TARGET_COLUMN
from courtvision.models.train_lgbm import (
    INNER_TRAIN_FRACTION,
    SEARCH_CONFIGS,
    build_lgbm_classifier,
    split_inner_train_validation,
)


def _synthetic_train_frame(n_games: int = 10) -> pd.DataFrame:
    rows = []
    for game_idx in range(n_games):
        game_id = f"002240{game_idx:04d}"
        game_date = pd.Timestamp("2024-10-01") + pd.Timedelta(days=game_idx)
        for shot_id in range(2):
            row = {col: 0.0 for col in FEATURE_COLUMNS}
            row.update(
                {
                    "shot_id": game_idx * 2 + shot_id,
                    "game_id": game_id,
                    "game_date": game_date.date(),
                    TARGET_COLUMN: True,
                    "is_home": True,
                    "is_corner_three": False,
                    "score_margin_missing": False,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_search_config_count_is_in_requested_range() -> None:
    assert 8 <= len(SEARCH_CONFIGS) <= 12


def test_build_lgbm_classifier_accepts_overrides() -> None:
    model = build_lgbm_classifier(num_leaves=15, n_estimators=100)
    assert model.get_params()["num_leaves"] == 15
    assert model.get_params()["n_estimators"] == 100


def test_inner_train_validation_split_is_time_based_and_non_overlapping() -> None:
    train = _synthetic_train_frame(n_games=10)
    inner_train, validation, meta = split_inner_train_validation(
        train,
        inner_train_fraction=INNER_TRAIN_FRACTION,
    )

    inner_game_ids = set(inner_train["game_id"])
    validation_game_ids = set(validation["game_id"])
    assert not inner_game_ids & validation_game_ids
    assert meta["train_games"] + meta["test_games"] == 10
    assert len(inner_train) + len(validation) == len(train)
