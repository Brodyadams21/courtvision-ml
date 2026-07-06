"""Tests for dashboard data helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from courtvision.dashboard.data import (
    DISTANCE_BUCKET_LABELS,
    EDGE_BUCKET_LABELS,
    PREDICTION_DISPLAY_COLUMNS,
    PREDICTION_FEATURE_COLUMNS,
    PREDICTION_ROW_ID_COLUMN,
    BaselineShotProfile,
    actual_points_for_row,
    add_distance_bucket,
    add_edge_bucket,
    assign_edge_bucket,
    baseline_for_similar_shots,
    build_shot_edge_row,
    build_shot_edge_table,
    compare_prediction_to_baseline,
    compute_overview_stats,
    compute_shot_quality_summary,
    existing_model_artifact_paths,
    filter_shots,
    get_prediction_row,
    load_feature_importance,
    load_training_summary,
    prepare_prediction_features,
    sample_prediction_rows,
    summarize_by_distance_bucket,
    summarize_edge_backtest,
    top_feature_importance,
)
from courtvision.dashboard.prediction import DashboardPredictionResult
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


def _write_training_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_load_training_summary_returns_none_when_file_is_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "model_artifacts" / "training_summary.json"

    assert load_training_summary(missing_path) is None


def test_load_training_summary_reads_default_mode_metrics_correctly(tmp_path: Path) -> None:
    summary_path = tmp_path / "training_summary.json"
    _write_training_summary(
        summary_path,
        {
            "environment": "local",
            "model": "lightgbm",
            "mode": "default",
            "season": "2024-25",
            "metrics": {
                "auc": 0.6479,
                "log_loss": 0.6495,
                "brier_score": 0.2292,
                "accuracy": 0.6213,
            },
        },
    )

    summary = load_training_summary(summary_path)

    assert summary is not None
    assert summary.environment == "local"
    assert summary.model == "lightgbm"
    assert summary.mode == "default"
    assert summary.season == "2024-25"
    assert summary.auc == pytest.approx(0.6479)
    assert summary.log_loss == pytest.approx(0.6495)
    assert summary.brier_score == pytest.approx(0.2292)
    assert summary.accuracy == pytest.approx(0.6213)
    assert summary.validation_auc is None
    assert summary.validation_log_loss is None
    assert summary.best_config_index is None
    assert summary.summary_path == summary_path.resolve()


def test_load_training_summary_reads_search_mode_summary_correctly(tmp_path: Path) -> None:
    summary_path = tmp_path / "training_summary.json"
    _write_training_summary(
        summary_path,
        {
            "environment": "local",
            "model": "lightgbm",
            "mode": "search",
            "season": "2024-25",
            "best_config_index": 3,
            "validation_log_loss": 0.648,
            "validation_auc": 0.651,
            "test_log_loss": 0.646,
            "test_auc": 0.652,
        },
    )

    summary = load_training_summary(summary_path)

    assert summary is not None
    assert summary.mode == "search"
    assert summary.auc == pytest.approx(0.652)
    assert summary.log_loss == pytest.approx(0.646)
    assert summary.validation_auc == pytest.approx(0.651)
    assert summary.validation_log_loss == pytest.approx(0.648)
    assert summary.best_config_index == 3


def test_load_training_summary_handles_optional_missing_brier_and_accuracy_for_search_mode(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "training_summary.json"
    _write_training_summary(
        summary_path,
        {
            "environment": "local",
            "model": "lightgbm",
            "mode": "search",
            "season": "2024-25",
            "test_log_loss": 0.646,
            "test_auc": 0.652,
        },
    )

    summary = load_training_summary(summary_path)

    assert summary is not None
    assert summary.brier_score is None
    assert summary.accuracy is None
    assert summary.validation_auc is None
    assert summary.validation_log_loss is None
    assert summary.best_config_index is None


def test_load_training_summary_raises_value_error_for_malformed_json(tmp_path: Path) -> None:
    summary_path = tmp_path / "training_summary.json"
    summary_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed training summary JSON"):
        load_training_summary(summary_path)


def test_load_training_summary_raises_value_error_when_top_level_is_not_object(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "training_summary.json"
    summary_path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected training summary object"):
        load_training_summary(summary_path)


def test_load_training_summary_raises_value_error_when_metadata_fields_are_missing(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "training_summary.json"
    _write_training_summary(
        summary_path,
        {
            "model": "lightgbm",
            "mode": "default",
            "season": "2024-25",
            "metrics": {
                "auc": 0.6479,
                "log_loss": 0.6495,
                "brier_score": 0.2292,
                "accuracy": 0.6213,
            },
        },
    )

    with pytest.raises(ValueError, match="missing required field: environment"):
        load_training_summary(summary_path)


def test_load_training_summary_raises_value_error_when_metrics_are_missing(tmp_path: Path) -> None:
    summary_path = tmp_path / "training_summary.json"
    _write_training_summary(
        summary_path,
        {
            "environment": "local",
            "model": "lightgbm",
            "mode": "default",
            "season": "2024-25",
            "metrics": {
                "auc": 0.6479,
                "log_loss": 0.6495,
            },
        },
    )

    with pytest.raises(ValueError, match="metrics missing required field"):
        load_training_summary(summary_path)


def test_load_feature_importance_returns_none_when_file_is_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "lightgbm_feature_importance_gain.csv"

    assert load_feature_importance(missing_path) is None


def test_load_feature_importance_reads_valid_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "importance.csv"
    csv_path.write_text(
        "feature,importance\nshot_distance,10.0\nloc_x,5.0\n",
        encoding="utf-8",
    )

    importance = load_feature_importance(csv_path)

    assert importance is not None
    assert list(importance.columns) == ["feature", "importance"]
    assert len(importance) == 2


def test_load_feature_importance_sorts_by_importance_descending(tmp_path: Path) -> None:
    csv_path = tmp_path / "importance.csv"
    csv_path.write_text(
        "feature,importance\nlow,1.0\nhigh,3.0\nmid,2.0\n",
        encoding="utf-8",
    )

    importance = load_feature_importance(csv_path)

    assert importance is not None
    assert importance["feature"].tolist() == ["high", "mid", "low"]


def test_load_feature_importance_raises_value_error_when_required_columns_are_missing(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "importance.csv"
    csv_path.write_text("feature,gain\nshot_distance,10.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required column"):
        load_feature_importance(csv_path)


def test_top_feature_importance_returns_requested_number_of_rows() -> None:
    importance = pd.DataFrame(
        {
            "feature": ["a", "b", "c", "d", "e"],
            "importance": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )

    top = top_feature_importance(importance, n=3)

    assert len(top) == 3
    assert top["feature"].tolist() == ["a", "b", "c"]


def test_top_feature_importance_handles_n_larger_than_row_count() -> None:
    importance = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "importance": [2.0, 1.0],
        }
    )

    top = top_feature_importance(importance, n=10)

    assert len(top) == 2
    assert top["feature"].tolist() == ["a", "b"]


def test_existing_model_artifact_paths_returns_empty_dict_when_files_are_missing(
    tmp_path: Path,
) -> None:
    artifacts = existing_model_artifact_paths(
        calibration_curve_path=tmp_path / "missing_calibration.png",
        probability_distribution_path=tmp_path / "missing_distribution.png",
        feature_importance_gain_path=tmp_path / "missing_importance.png",
    )

    assert artifacts == {}


def test_existing_model_artifact_paths_returns_only_files_that_exist(tmp_path: Path) -> None:
    importance_path = tmp_path / "lightgbm_feature_importance_gain.png"
    importance_path.write_bytes(b"png")

    artifacts = existing_model_artifact_paths(
        calibration_curve_path=tmp_path / "missing_calibration.png",
        probability_distribution_path=tmp_path / "missing_distribution.png",
        feature_importance_gain_path=importance_path,
    )

    assert set(artifacts) == {"feature_importance_gain"}
    assert artifacts["feature_importance_gain"] == importance_path.resolve()


def test_existing_model_artifact_paths_includes_calibration_and_distribution_when_present(
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "lightgbm_calibration_curve.png"
    distribution_path = tmp_path / "lightgbm_probability_distribution.png"
    calibration_path.write_bytes(b"calibration")
    distribution_path.write_bytes(b"distribution")

    artifacts = existing_model_artifact_paths(
        calibration_curve_path=calibration_path,
        probability_distribution_path=distribution_path,
        feature_importance_gain_path=tmp_path / "missing_importance.png",
    )

    assert set(artifacts) == {"calibration_curve", "probability_distribution"}
    assert artifacts["calibration_curve"] == calibration_path.resolve()
    assert artifacts["probability_distribution"] == distribution_path.resolve()


def test_sample_prediction_rows_returns_expected_display_columns() -> None:
    test = _shot_frame(n_rows=5)

    samples = sample_prediction_rows(test, n=3, random_state=0)

    assert list(samples.columns) == list(PREDICTION_DISPLAY_COLUMNS)
    assert len(samples) == 3


def test_sample_prediction_rows_includes_row_id() -> None:
    test = _shot_frame(n_rows=4)
    test.index = [100, 101, 102, 103]

    samples = sample_prediction_rows(test, n=2, random_state=0)

    assert PREDICTION_ROW_ID_COLUMN in samples.columns
    assert samples[PREDICTION_ROW_ID_COLUMN].isin(test.index).all()


def test_get_prediction_row_returns_correct_row() -> None:
    test = _shot_frame(
        rows=[
            _shot_row(shot_value=2.0, shot_distance=4.0),
            _shot_row(shot_value=3.0, shot_distance=24.0),
        ]
    )
    test.index = [10, 20]

    row = get_prediction_row(test, 20)

    assert row["shot_value"] == pytest.approx(3.0)
    assert row["shot_distance"] == pytest.approx(24.0)


def test_get_prediction_row_raises_key_error_for_invalid_row_id() -> None:
    test = _shot_frame(n_rows=2)
    test.index = [1, 2]

    with pytest.raises(KeyError, match="Prediction row_id not found in test set: 99"):
        get_prediction_row(test, 99)


def test_prepare_prediction_features_excludes_shot_value_from_features() -> None:
    row = get_prediction_row(
        _shot_frame(rows=[_shot_row(shot_value=3.0, shot_distance=24.0)]),
        0,
    )

    prepared = prepare_prediction_features(row)

    assert "shot_value" not in prepared.features


def test_prepare_prediction_features_returns_shot_value_as_int() -> None:
    row = get_prediction_row(
        _shot_frame(rows=[_shot_row(shot_value=3.0, shot_distance=24.0)]),
        0,
    )

    prepared = prepare_prediction_features(row)

    assert prepared.shot_value == 3
    assert isinstance(prepared.shot_value, int)


def test_prepare_prediction_features_includes_all_remaining_expected_feature_columns() -> None:
    row = get_prediction_row(_shot_frame(n_rows=1), 0)

    prepared = prepare_prediction_features(row)

    assert set(prepared.features) == set(PREDICTION_FEATURE_COLUMNS)
    assert len(prepared.features) == len(PREDICTION_FEATURE_COLUMNS)


def test_actual_points_for_row_returns_shot_value_for_made_shot() -> None:
    row = _shot_row(shot_made_flag=True, shot_value=3.0)

    assert actual_points_for_row(row) == pytest.approx(3.0)


def test_actual_points_for_row_returns_zero_for_missed_shot() -> None:
    row = _shot_row(shot_made_flag=False, shot_value=2.0)

    assert actual_points_for_row(row) == pytest.approx(0.0)


def test_baseline_for_similar_shots_uses_same_shot_value_and_distance_bucket() -> None:
    test = _shot_frame(
        rows=[
            _shot_row(shot_made_flag=True, shot_value=2.0, shot_distance=4.0),
            _shot_row(shot_made_flag=False, shot_value=2.0, shot_distance=4.5),
            _shot_row(shot_made_flag=True, shot_value=3.0, shot_distance=4.0),
            _shot_row(shot_made_flag=True, shot_value=2.0, shot_distance=20.0),
        ]
    )
    selected = _shot_row(shot_made_flag=True, shot_value=2.0, shot_distance=3.5)

    baseline = baseline_for_similar_shots(test, selected)

    assert baseline.shot_count == 2
    assert baseline.make_rate == pytest.approx(0.5)


def test_baseline_for_similar_shots_returns_make_rate_and_ev() -> None:
    test = _shot_frame(
        rows=[
            _shot_row(shot_made_flag=True, shot_value=3.0, shot_distance=26.0),
            _shot_row(shot_made_flag=False, shot_value=3.0, shot_distance=27.0),
        ]
    )
    selected = _shot_row(shot_made_flag=True, shot_value=3.0, shot_distance=26.5)

    baseline = baseline_for_similar_shots(test, selected)

    assert baseline.make_rate == pytest.approx(0.5)
    assert baseline.expected_value == pytest.approx(1.5)


def test_compare_prediction_to_baseline_computes_probability_edge() -> None:
    row = _shot_row(shot_made_flag=True, shot_value=2.0, shot_distance=4.0)
    result = DashboardPredictionResult(
        predicted_make_probability=0.432,
        expected_shot_value=0.86,
        model_name="test-model",
    )
    baseline = baseline_for_similar_shots(
        _shot_frame(rows=[_shot_row(shot_made_flag=False, shot_value=2.0, shot_distance=4.0)]),
        row,
    )

    comparison = compare_prediction_to_baseline(result, row, baseline)

    assert comparison.probability_edge_vs_baseline == pytest.approx(0.432)


def test_compare_prediction_to_baseline_computes_ev_edge() -> None:
    row = _shot_row(shot_made_flag=False, shot_value=3.0, shot_distance=26.0)
    result = DashboardPredictionResult(
        predicted_make_probability=0.40,
        expected_shot_value=1.20,
        model_name="test-model",
    )
    baseline = BaselineShotProfile(shot_count=10, make_rate=0.35, expected_value=1.05)

    comparison = compare_prediction_to_baseline(result, row, baseline)

    assert comparison.ev_edge_vs_baseline == pytest.approx(0.15)
    assert comparison.actual_points == pytest.approx(0.0)
    assert comparison.actual_made is False


def test_build_shot_edge_row_returns_expected_fields() -> None:
    test = _shot_frame(
        rows=[
            _shot_row(shot_made_flag=True, shot_value=3.0, shot_distance=26.0, period=2.0),
            _shot_row(shot_made_flag=False, shot_value=3.0, shot_distance=27.0, period=2.0),
        ]
    )
    test.index = [10, 20]
    result = DashboardPredictionResult(
        predicted_make_probability=0.42,
        expected_shot_value=1.26,
        model_name="test-model",
    )

    edge_row = build_shot_edge_row(test, 10, result)

    assert edge_row.row_id == 10
    assert edge_row.shot_value == 3
    assert edge_row.shot_distance == pytest.approx(26.0)
    assert edge_row.period == 2
    assert edge_row.actual_made is True
    assert edge_row.predicted_make_probability == pytest.approx(0.42)
    assert edge_row.baseline_make_rate == pytest.approx(0.5)
    assert edge_row.similar_shot_count == 2


def test_build_shot_edge_row_computes_ev_edge() -> None:
    test = _shot_frame(rows=[_shot_row(shot_made_flag=True, shot_value=2.0, shot_distance=4.0)])
    test.index = [5]
    result = DashboardPredictionResult(
        predicted_make_probability=0.60,
        expected_shot_value=1.20,
        model_name="test-model",
    )

    edge_row = build_shot_edge_row(test, 5, result)

    assert edge_row.ev_edge_vs_baseline == pytest.approx(
        edge_row.expected_shot_value - edge_row.baseline_expected_value
    )


def test_build_shot_edge_table_returns_one_row_per_prediction() -> None:
    test = _shot_frame(
        rows=[
            _shot_row(shot_made_flag=True, shot_value=2.0, shot_distance=4.0),
            _shot_row(shot_made_flag=False, shot_value=3.0, shot_distance=26.0),
        ]
    )
    test.index = [1, 2]
    predictions = {
        1: DashboardPredictionResult(0.55, 1.10, "test-model"),
        2: DashboardPredictionResult(0.35, 1.05, "test-model"),
    }

    table = build_shot_edge_table(test, predictions)

    assert len(table) == 2
    assert set(table["row_id"]) == {1, 2}
    assert "actual_points" in table.columns
    assert table.loc[table["row_id"] == 1, "actual_points"].iloc[0] == pytest.approx(2.0)
    assert table.loc[table["row_id"] == 2, "actual_points"].iloc[0] == pytest.approx(0.0)


def test_build_shot_edge_table_can_be_sorted_by_ev_edge() -> None:
    test = _shot_frame(
        rows=[
            _shot_row(shot_made_flag=True, shot_value=2.0, shot_distance=4.0),
            _shot_row(shot_made_flag=False, shot_value=3.0, shot_distance=26.0),
        ]
    )
    test.index = [1, 2]
    predictions = {
        1: DashboardPredictionResult(0.40, 0.80, "test-model"),
        2: DashboardPredictionResult(0.50, 1.50, "test-model"),
    }

    table = build_shot_edge_table(test, predictions)
    sorted_table = table.sort_values("ev_edge_vs_baseline", ascending=False)

    assert sorted_table.iloc[0]["row_id"] == 2


def _edge_table_row(
    *,
    row_id: int,
    ev_edge: float,
    actual_made: bool,
    actual_points: float,
    predicted_make_probability: float = 0.5,
    expected_shot_value: float = 1.0,
    baseline_expected_value: float = 0.9,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "shot_value": 2,
        "shot_distance": 10.0,
        "period": 1,
        "actual_made": actual_made,
        "actual_points": actual_points,
        "predicted_make_probability": predicted_make_probability,
        "expected_shot_value": expected_shot_value,
        "baseline_make_rate": 0.45,
        "baseline_expected_value": baseline_expected_value,
        "probability_edge_vs_baseline": predicted_make_probability - 0.45,
        "ev_edge_vs_baseline": ev_edge,
        "similar_shot_count": 10,
    }


def test_add_edge_bucket_assigns_expected_buckets() -> None:
    edge_table = pd.DataFrame(
        [
            _edge_table_row(row_id=1, ev_edge=-0.20, actual_made=False, actual_points=0.0),
            _edge_table_row(row_id=2, ev_edge=-0.05, actual_made=False, actual_points=0.0),
            _edge_table_row(row_id=3, ev_edge=0.05, actual_made=True, actual_points=2.0),
            _edge_table_row(row_id=4, ev_edge=0.15, actual_made=True, actual_points=2.0),
        ]
    )

    bucketed = add_edge_bucket(edge_table)

    assert bucketed["edge_bucket"].tolist() == list(EDGE_BUCKET_LABELS)
    assert assign_edge_bucket(-0.10) == "Strong negative edge"
    assert assign_edge_bucket(0.0) == "Slight negative edge"
    assert assign_edge_bucket(0.10) == "Strong positive edge"


def test_summarize_edge_backtest_returns_one_row_per_bucket() -> None:
    edge_table = pd.DataFrame(
        [
            _edge_table_row(row_id=1, ev_edge=-0.20, actual_made=False, actual_points=0.0),
            _edge_table_row(row_id=2, ev_edge=0.15, actual_made=True, actual_points=2.0),
        ]
    )

    summary = summarize_edge_backtest(edge_table)

    assert len(summary) == len(EDGE_BUCKET_LABELS)
    assert list(summary["bucket"]) == list(EDGE_BUCKET_LABELS)


def test_summarize_edge_backtest_computes_actual_make_rate() -> None:
    edge_table = pd.DataFrame(
        [
            _edge_table_row(row_id=1, ev_edge=0.15, actual_made=True, actual_points=2.0),
            _edge_table_row(row_id=2, ev_edge=0.12, actual_made=False, actual_points=0.0),
        ]
    )

    summary = summarize_edge_backtest(edge_table)
    strong_positive = summary.loc[summary["bucket"] == "Strong positive edge"].iloc[0]

    assert strong_positive["shot_count"] == 2
    assert strong_positive["actual_make_rate"] == pytest.approx(0.5)


def test_summarize_edge_backtest_computes_average_model_ev() -> None:
    edge_table = pd.DataFrame(
        [
            _edge_table_row(
                row_id=1,
                ev_edge=-0.20,
                actual_made=False,
                actual_points=0.0,
                expected_shot_value=0.80,
            ),
            _edge_table_row(
                row_id=2,
                ev_edge=-0.15,
                actual_made=True,
                actual_points=2.0,
                expected_shot_value=1.00,
            ),
        ]
    )

    summary = summarize_edge_backtest(edge_table)
    strong_negative = summary.loc[summary["bucket"] == "Strong negative edge"].iloc[0]

    assert strong_negative["avg_model_ev"] == pytest.approx(0.90)


def test_summarize_edge_backtest_computes_average_baseline_ev() -> None:
    edge_table = pd.DataFrame(
        [
            _edge_table_row(
                row_id=1,
                ev_edge=0.05,
                actual_made=True,
                actual_points=2.0,
                baseline_expected_value=0.80,
            ),
            _edge_table_row(
                row_id=2,
                ev_edge=0.04,
                actual_made=False,
                actual_points=0.0,
                baseline_expected_value=1.00,
            ),
        ]
    )

    summary = summarize_edge_backtest(edge_table)
    slight_positive = summary.loc[summary["bucket"] == "Slight positive edge"].iloc[0]

    assert slight_positive["avg_baseline_ev"] == pytest.approx(0.90)


def test_summarize_edge_backtest_handles_empty_edge_table() -> None:
    summary = summarize_edge_backtest(pd.DataFrame())

    assert summary.empty
    assert list(summary.columns) == [
        "bucket",
        "shot_count",
        "avg_predicted_make_probability",
        "avg_model_ev",
        "avg_baseline_ev",
        "avg_ev_edge",
        "actual_make_rate",
        "avg_actual_points",
        "model_ev_minus_actual_points",
        "baseline_ev_minus_actual_points",
    ]


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
