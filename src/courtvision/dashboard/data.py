"""Dashboard data helpers (Streamlit-free, testable)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from courtvision.data.build_features import DEFAULT_PROCESSED_FEATURES_DIR
from courtvision.data.collect import DEFAULT_SEASON
from courtvision.models.common import FEATURE_COLUMNS, TARGET_COLUMN, load_train_test_parquet


@dataclass(frozen=True)
class OverviewStats:
    train_shots: int
    test_shots: int
    total_shots: int
    train_make_rate: float
    test_make_rate: float
    overall_make_rate: float
    feature_count: int
    target_column: str


def load_dashboard_splits(
    season: str = DEFAULT_SEASON,
    *,
    processed_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test shot-feature Parquet splits for the dashboard."""
    directory = processed_dir or DEFAULT_PROCESSED_FEATURES_DIR
    return load_train_test_parquet(season, processed_dir=directory)


def _make_rate(frame: pd.DataFrame) -> float:
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(f"Frame missing required column: {TARGET_COLUMN}")
    return float(frame[TARGET_COLUMN].mean())


def compute_overview_stats(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> OverviewStats:
    """Compute dataset overview metrics for the dashboard."""
    train_shots = len(train)
    test_shots = len(test)
    total_shots = train_shots + test_shots

    train_make_rate = _make_rate(train)
    test_make_rate = _make_rate(test)

    if total_shots == 0:
        overall_make_rate = 0.0
    else:
        combined = pd.concat([train[TARGET_COLUMN], test[TARGET_COLUMN]], ignore_index=True)
        overall_make_rate = float(combined.mean())

    return OverviewStats(
        train_shots=train_shots,
        test_shots=test_shots,
        total_shots=total_shots,
        train_make_rate=train_make_rate,
        test_make_rate=test_make_rate,
        overall_make_rate=overall_make_rate,
        feature_count=len(FEATURE_COLUMNS),
        target_column=TARGET_COLUMN,
    )
