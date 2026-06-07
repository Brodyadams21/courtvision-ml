"""Tests for PostgreSQL-safe insert batch sizing."""

from __future__ import annotations

import pytest

from courtvision.data.build_features import GOLD_SHOT_FEATURES_INSERT_COLUMNS
from courtvision.data.load_data import POSTGRES_MAX_BIND_PARAMS, effective_insert_chunksize


def test_effective_insert_chunksize_unchanged_when_under_limit() -> None:
    assert effective_insert_chunksize(500, num_columns=10) == 500


def test_effective_insert_chunksize_caps_at_bind_limit() -> None:
    num_columns = 44
    expected = POSTGRES_MAX_BIND_PARAMS // num_columns
    assert effective_insert_chunksize(5_000, num_columns=num_columns) == expected


def test_gold_shot_features_default_batch_is_safe() -> None:
    num_columns = len(GOLD_SHOT_FEATURES_INSERT_COLUMNS)
    effective = effective_insert_chunksize(5_000, num_columns=num_columns)
    assert effective * num_columns <= POSTGRES_MAX_BIND_PARAMS


def test_effective_insert_chunksize_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="chunksize"):
        effective_insert_chunksize(0, num_columns=10)
    with pytest.raises(ValueError, match="num_columns"):
        effective_insert_chunksize(100, num_columns=0)


def test_effective_insert_chunksize_rejects_impossibly_wide_rows() -> None:
    with pytest.raises(ValueError, match="bind limit"):
        effective_insert_chunksize(100, num_columns=POSTGRES_MAX_BIND_PARAMS + 1)
