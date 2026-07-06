"""Dashboard data helpers (Streamlit-free, testable)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from courtvision.data.build_features import DEFAULT_PROCESSED_FEATURES_DIR
from courtvision.data.collect import DEFAULT_SEASON
from courtvision.models.common import FEATURE_COLUMNS, TARGET_COLUMN, load_train_test_parquet

DISTANCE_BUCKET_LABELS: tuple[str, ...] = (
    "0-5 ft",
    "5-10 ft",
    "10-15 ft",
    "15-20 ft",
    "20-25 ft",
    "25-30 ft",
    "30+ ft",
)
DISTANCE_BUCKET_BINS: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, float("inf"))
DISTANCE_BUCKET_COLUMN = "distance_bucket"

SHOT_VALUE_COLUMN = "shot_value"
PERIOD_COLUMN = "period"
SHOT_DISTANCE_COLUMN = "shot_distance"


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


@dataclass(frozen=True)
class ShotQualitySummary:
    shot_count: int
    make_rate: float
    avg_shot_value: float
    avg_shot_distance: float
    avg_expected_points_baseline: float


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
    if frame.empty:
        return 0.0
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


def filter_shots(
    frame: pd.DataFrame,
    *,
    shot_value: int | None = None,
    periods: list[int] | None = None,
    min_distance: float | None = None,
    max_distance: float | None = None,
) -> pd.DataFrame:
    """Filter shots by shot type, period, and distance range."""
    filtered = frame.copy()

    if shot_value is not None:
        filtered = filtered.loc[filtered[SHOT_VALUE_COLUMN] == shot_value]

    if periods is not None:
        filtered = filtered.loc[filtered[PERIOD_COLUMN].isin(periods)]

    if min_distance is not None:
        filtered = filtered.loc[filtered[SHOT_DISTANCE_COLUMN] >= min_distance]

    if max_distance is not None:
        filtered = filtered.loc[filtered[SHOT_DISTANCE_COLUMN] <= max_distance]

    return filtered.reset_index(drop=True)


def compute_shot_quality_summary(frame: pd.DataFrame) -> ShotQualitySummary:
    """Compute shot-quality metrics for a filtered shot dataframe."""
    shot_count = len(frame)
    if shot_count == 0:
        return ShotQualitySummary(
            shot_count=0,
            make_rate=0.0,
            avg_shot_value=0.0,
            avg_shot_distance=0.0,
            avg_expected_points_baseline=0.0,
        )

    make_rate = _make_rate(frame)
    avg_shot_value = float(frame[SHOT_VALUE_COLUMN].mean())
    avg_shot_distance = float(frame[SHOT_DISTANCE_COLUMN].mean())
    avg_expected_points_baseline = make_rate * avg_shot_value

    return ShotQualitySummary(
        shot_count=shot_count,
        make_rate=make_rate,
        avg_shot_value=avg_shot_value,
        avg_shot_distance=avg_shot_distance,
        avg_expected_points_baseline=avg_expected_points_baseline,
    )


def add_distance_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    """Add an ordered distance-bucket column for charting and grouping."""
    output = frame.copy()
    if output.empty:
        output[DISTANCE_BUCKET_COLUMN] = pd.Categorical(
            [],
            categories=list(DISTANCE_BUCKET_LABELS),
            ordered=True,
        )
        return output

    output[DISTANCE_BUCKET_COLUMN] = pd.cut(
        output[SHOT_DISTANCE_COLUMN],
        bins=list(DISTANCE_BUCKET_BINS),
        labels=list(DISTANCE_BUCKET_LABELS),
        right=False,
        include_lowest=True,
    )
    return output


def summarize_by_distance_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    """Return shot-quality metrics grouped by distance bucket."""
    columns = [
        DISTANCE_BUCKET_COLUMN,
        "shot_count",
        "make_rate",
        "avg_shot_value",
        "avg_expected_points_baseline",
    ]
    bucketed = add_distance_bucket(frame)
    if bucketed.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for bucket in DISTANCE_BUCKET_LABELS:
        bucket_frame = bucketed.loc[bucketed[DISTANCE_BUCKET_COLUMN] == bucket]
        summary = compute_shot_quality_summary(bucket_frame)
        rows.append(
            {
                DISTANCE_BUCKET_COLUMN: bucket,
                "shot_count": summary.shot_count,
                "make_rate": summary.make_rate,
                "avg_shot_value": summary.avg_shot_value,
                "avg_expected_points_baseline": summary.avg_expected_points_baseline,
            }
        )

    return pd.DataFrame(rows, columns=columns)
