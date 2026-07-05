"""Tests for dashboard data helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from courtvision.dashboard.data import compute_overview_stats
from courtvision.models.common import FEATURE_COLUMNS, TARGET_COLUMN


def _shot_frame(
    *,
    n_rows: int,
    make_flags: list[bool] | None = None,
    include_target: bool = True,
) -> pd.DataFrame:
    rows = []
    for idx in range(n_rows):
        row = {col: 0.0 for col in FEATURE_COLUMNS}
        if include_target:
            if make_flags is not None:
                row[TARGET_COLUMN] = make_flags[idx]
            else:
                row[TARGET_COLUMN] = idx % 2 == 0
        rows.append(row)
    return pd.DataFrame(rows)


def test_overview_stats_counts_train_and_test_rows() -> None:
    train = _shot_frame(n_rows=100)
    test = _shot_frame(n_rows=25)

    stats = compute_overview_stats(train, test)

    assert stats.train_shots == 100
    assert stats.test_shots == 25
    assert stats.total_shots == 125


def test_overview_stats_make_rates_use_shot_made_flag() -> None:
    train = _shot_frame(n_rows=4, make_flags=[True, True, False, False])
    test = _shot_frame(n_rows=2, make_flags=[True, False])

    stats = compute_overview_stats(train, test)

    assert stats.train_make_rate == pytest.approx(0.5)
    assert stats.test_make_rate == pytest.approx(0.5)
    assert stats.overall_make_rate == pytest.approx(0.5)


def test_overview_stats_feature_count_matches_feature_columns() -> None:
    train = _shot_frame(n_rows=1)
    test = _shot_frame(n_rows=1)

    stats = compute_overview_stats(train, test)

    assert stats.feature_count == len(FEATURE_COLUMNS)
    assert stats.target_column == TARGET_COLUMN


def test_overview_stats_missing_target_raises_key_error() -> None:
    train = _shot_frame(n_rows=2, include_target=False)
    test = _shot_frame(n_rows=1)

    with pytest.raises(KeyError, match=TARGET_COLUMN):
        compute_overview_stats(train, test)
