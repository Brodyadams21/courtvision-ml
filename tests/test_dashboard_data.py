"""Tests for dashboard data helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from courtvision.dashboard.data import (
    DISTANCE_BUCKET_LABELS,
    add_distance_bucket,
    compute_overview_stats,
    compute_shot_quality_summary,
    filter_shots,
    summarize_by_distance_bucket,
)
from courtvision.models.common import FEATURE_COLUMNS, TARGET_COLUMN


def _shot_frame(
    *,
    rows: list[dict[str, object]] | None = None,
    n_rows: int | None = None,
    make_flags: list[bool] | None = None,
    include_target: bool = True,
) -> pd.DataFrame:
    if rows is not None:
        return pd.DataFrame(rows)

    built_rows = []
    for idx in range(n_rows or 0):
        row: dict[str, object] = {col: 0.0 for col in FEATURE_COLUMNS}
        if include_target:
            if make_flags is not None:
                row[TARGET_COLUMN] = make_flags[idx]
            else:
                row[TARGET_COLUMN] = idx % 2 == 0
        built_rows.append(row)
    return pd.DataFrame(built_rows)


def _shot_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {col: 0.0 for col in FEATURE_COLUMNS}
    row.update(
        {
            TARGET_COLUMN: True,
            "shot_value": 2.0,
            "shot_distance": 4.0,
            "period": 1.0,
        }
    )
    row.update(overrides)
    return row


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


def test_filter_shots_filters_by_shot_value() -> None:
    frame = _shot_frame(
        rows=[
            _shot_row(shot_value=2.0),
            _shot_row(shot_value=3.0),
            _shot_row(shot_value=2.0),
        ]
    )

    filtered = filter_shots(frame, shot_value=2)

    assert len(filtered) == 2
    assert filtered["shot_value"].eq(2).all()


def test_filter_shots_filters_by_period() -> None:
    frame = _shot_frame(
        rows=[
            _shot_row(period=1.0),
            _shot_row(period=2.0),
            _shot_row(period=3.0),
        ]
    )

    filtered = filter_shots(frame, periods=[1, 3])

    assert len(filtered) == 2
    assert set(filtered["period"].astype(int)) == {1, 3}


def test_filter_shots_filters_by_distance_range() -> None:
    frame = _shot_frame(
        rows=[
            _shot_row(shot_distance=3.0),
            _shot_row(shot_distance=12.0),
            _shot_row(shot_distance=28.0),
        ]
    )

    filtered = filter_shots(frame, min_distance=5.0, max_distance=20.0)

    assert len(filtered) == 1
    assert filtered.iloc[0]["shot_distance"] == pytest.approx(12.0)


def test_compute_shot_quality_summary_calculates_make_rate_correctly() -> None:
    frame = _shot_frame(
        rows=[
            _shot_row(shot_made_flag=True, shot_value=2.0, shot_distance=4.0),
            _shot_row(shot_made_flag=False, shot_value=3.0, shot_distance=24.0),
        ]
    )

    summary = compute_shot_quality_summary(frame)

    assert summary.shot_count == 2
    assert summary.make_rate == pytest.approx(0.5)
    assert summary.avg_shot_value == pytest.approx(2.5)
    assert summary.avg_shot_distance == pytest.approx(14.0)
    assert summary.avg_expected_points_baseline == pytest.approx(0.5 * 2.5)


def test_add_distance_bucket_assigns_expected_buckets() -> None:
    frame = _shot_frame(
        rows=[
            _shot_row(shot_distance=2.0),
            _shot_row(shot_distance=7.0),
            _shot_row(shot_distance=31.0),
        ]
    )

    bucketed = add_distance_bucket(frame)

    assert bucketed["distance_bucket"].tolist() == [
        "0-5 ft",
        "5-10 ft",
        "30+ ft",
    ]


def test_summarize_by_distance_bucket_returns_grouped_rows() -> None:
    frame = _shot_frame(
        rows=[
            _shot_row(shot_made_flag=True, shot_value=2.0, shot_distance=2.0),
            _shot_row(shot_made_flag=False, shot_value=2.0, shot_distance=4.0),
            _shot_row(shot_made_flag=True, shot_value=3.0, shot_distance=27.0),
        ]
    )

    summary = summarize_by_distance_bucket(frame)

    assert list(summary.columns) == [
        "distance_bucket",
        "shot_count",
        "make_rate",
        "avg_shot_value",
        "avg_expected_points_baseline",
    ]
    assert list(summary["distance_bucket"]) == list(DISTANCE_BUCKET_LABELS)

    zero_to_five = summary.loc[summary["distance_bucket"] == "0-5 ft"].iloc[0]
    assert zero_to_five["shot_count"] == 2
    assert zero_to_five["make_rate"] == pytest.approx(0.5)
    assert zero_to_five["avg_shot_value"] == pytest.approx(2.0)
    assert zero_to_five["avg_expected_points_baseline"] == pytest.approx(1.0)

    twenty_five_to_thirty = summary.loc[summary["distance_bucket"] == "25-30 ft"].iloc[0]
    assert twenty_five_to_thirty["shot_count"] == 1
    assert twenty_five_to_thirty["make_rate"] == pytest.approx(1.0)
    assert twenty_five_to_thirty["avg_shot_value"] == pytest.approx(3.0)


def test_empty_filtered_data_does_not_crash() -> None:
    frame = _shot_frame(rows=[_shot_row(shot_value=2.0, shot_distance=4.0)])

    filtered = filter_shots(frame, shot_value=3)
    summary = compute_shot_quality_summary(filtered)
    bucketed = add_distance_bucket(filtered)
    bucket_summary = summarize_by_distance_bucket(filtered)

    assert filtered.empty
    assert summary.shot_count == 0
    assert summary.make_rate == 0.0
    assert bucketed.empty
    assert bucket_summary["shot_count"].eq(0).all()
