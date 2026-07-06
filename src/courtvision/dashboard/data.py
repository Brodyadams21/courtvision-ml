"""Dashboard data helpers (Streamlit-free, testable)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from courtvision.data.build_features import DEFAULT_PROCESSED_FEATURES_DIR
from courtvision.data.collect import DEFAULT_SEASON
from courtvision.models.common import FEATURE_COLUMNS, TARGET_COLUMN, load_train_test_parquet
from courtvision.utils.config import PROJECT_ROOT

DEFAULT_TRAINING_SUMMARY_PATH = PROJECT_ROOT / "model_artifacts" / "training_summary.json"
DEFAULT_FEATURE_IMPORTANCE_GAIN_CSV = (
    PROJECT_ROOT / "reports" / "tables" / "lightgbm_feature_importance_gain.csv"
)
DEFAULT_FEATURE_IMPORTANCE_GAIN_PNG = (
    PROJECT_ROOT / "reports" / "figures" / "lightgbm_feature_importance_gain.png"
)
DEFAULT_LIGHTGBM_CALIBRATION_CURVE_PNG = (
    PROJECT_ROOT / "reports" / "figures" / "lightgbm_calibration_curve.png"
)
DEFAULT_LIGHTGBM_PROBABILITY_DISTRIBUTION_PNG = (
    PROJECT_ROOT / "reports" / "figures" / "lightgbm_probability_distribution.png"
)
REQUIRED_SUMMARY_FIELDS: tuple[str, ...] = ("environment", "model", "mode", "season")
REQUIRED_DEFAULT_METRIC_FIELDS: tuple[str, ...] = ("auc", "log_loss", "brier_score", "accuracy")
REQUIRED_SEARCH_TEST_FIELDS: tuple[str, ...] = ("test_auc", "test_log_loss")
FEATURE_IMPORTANCE_FEATURE_COLUMN = "feature"
FEATURE_IMPORTANCE_VALUE_COLUMN = "importance"
REQUIRED_FEATURE_IMPORTANCE_COLUMNS: tuple[str, ...] = (
    FEATURE_IMPORTANCE_FEATURE_COLUMN,
    FEATURE_IMPORTANCE_VALUE_COLUMN,
)

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
SCORE_MARGIN_COLUMN = "score_margin"
LOC_X_COLUMN = "loc_x"
LOC_Y_COLUMN = "loc_y"
PREDICTION_ROW_ID_COLUMN = "row_id"
PREDICTION_DISPLAY_COLUMNS: tuple[str, ...] = (
    PREDICTION_ROW_ID_COLUMN,
    TARGET_COLUMN,
    SHOT_VALUE_COLUMN,
    SHOT_DISTANCE_COLUMN,
    PERIOD_COLUMN,
    SCORE_MARGIN_COLUMN,
    LOC_X_COLUMN,
    LOC_Y_COLUMN,
)
PREDICTION_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column for column in FEATURE_COLUMNS if column != SHOT_VALUE_COLUMN
)


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


@dataclass(frozen=True)
class PreparedPredictionFeatures:
    features: dict[str, float]
    shot_value: int


@dataclass(frozen=True)
class BaselineShotProfile:
    shot_count: int
    make_rate: float
    expected_value: float


@dataclass(frozen=True)
class PredictionComparison:
    predicted_make_probability: float
    expected_shot_value: float
    actual_made: bool | None
    actual_points: float | None
    baseline_make_rate: float
    baseline_expected_value: float
    probability_edge_vs_baseline: float
    ev_edge_vs_baseline: float
    similar_shot_count: int


@dataclass(frozen=True)
class ShotEdgeRow:
    row_id: int
    shot_value: int
    shot_distance: float
    period: int
    actual_made: bool | None
    actual_points: float | None
    predicted_make_probability: float
    expected_shot_value: float
    baseline_make_rate: float
    baseline_expected_value: float
    probability_edge_vs_baseline: float
    ev_edge_vs_baseline: float
    similar_shot_count: int


SHOT_EDGE_TABLE_COLUMNS: tuple[str, ...] = (
    "row_id",
    "shot_value",
    "shot_distance",
    "period",
    "actual_made",
    "actual_points",
    "predicted_make_probability",
    "expected_shot_value",
    "baseline_make_rate",
    "baseline_expected_value",
    "probability_edge_vs_baseline",
    "ev_edge_vs_baseline",
    "similar_shot_count",
)

EDGE_BUCKET_COLUMN = "edge_bucket"
STRONG_NEGATIVE_EV_EDGE_THRESHOLD = -0.10
STRONG_POSITIVE_EV_EDGE_THRESHOLD = 0.10
EDGE_BUCKET_LABELS: tuple[str, ...] = (
    "Strong negative edge",
    "Slight negative edge",
    "Slight positive edge",
    "Strong positive edge",
)
EDGE_BACKTEST_COLUMNS: tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class EdgeBacktestSummary:
    bucket: str
    shot_count: int
    avg_predicted_make_probability: float
    avg_model_ev: float
    avg_baseline_ev: float
    avg_ev_edge: float
    actual_make_rate: float
    avg_actual_points: float
    model_ev_minus_actual_points: float
    baseline_ev_minus_actual_points: float


class PredictionResultLike(Protocol):
    predicted_make_probability: float
    expected_shot_value: float


@dataclass(frozen=True)
class ModelPerformanceSummary:
    environment: str
    model: str
    mode: str
    season: str
    auc: float
    log_loss: float
    brier_score: float | None
    accuracy: float | None
    validation_auc: float | None
    validation_log_loss: float | None
    best_config_index: int | None
    summary_path: Path


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


def load_training_summary(
    path: Path | None = None,
) -> ModelPerformanceSummary | None:
    """Load LightGBM training metrics from ``training_summary.json``."""
    summary_path = (path or DEFAULT_TRAINING_SUMMARY_PATH).resolve()
    if not summary_path.is_file():
        return None

    payload = _read_training_summary_payload(summary_path)
    return _parse_training_summary(payload, summary_path)


def _read_training_summary_payload(summary_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed training summary JSON at {summary_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected training summary object at {summary_path}, got {type(payload).__name__}"
        )

    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in payload:
            raise ValueError(f"Training summary missing required field: {field}")

    return payload


def _parse_training_summary(
    payload: dict[str, object],
    summary_path: Path,
) -> ModelPerformanceSummary:
    mode = str(payload["mode"])
    base_kwargs = {
        "environment": str(payload["environment"]),
        "model": str(payload["model"]),
        "mode": mode,
        "season": str(payload["season"]),
        "summary_path": summary_path,
    }

    if mode == "default":
        return _parse_default_training_summary(payload, **base_kwargs)
    if mode == "search":
        return _parse_search_training_summary(payload, **base_kwargs)

    raise ValueError(f"Unsupported training summary mode: {mode}")


def _parse_default_training_summary(
    payload: dict[str, object],
    **base_kwargs: object,
) -> ModelPerformanceSummary:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Training summary missing required field: metrics")

    missing_metrics = [
        field for field in REQUIRED_DEFAULT_METRIC_FIELDS if field not in metrics
    ]
    if missing_metrics:
        missing = ", ".join(missing_metrics)
        raise ValueError(f"Training summary metrics missing required field(s): {missing}")

    return ModelPerformanceSummary(
        auc=float(metrics["auc"]),
        log_loss=float(metrics["log_loss"]),
        brier_score=float(metrics["brier_score"]),
        accuracy=float(metrics["accuracy"]),
        validation_auc=None,
        validation_log_loss=None,
        best_config_index=None,
        **base_kwargs,
    )


def _parse_search_training_summary(
    payload: dict[str, object],
    **base_kwargs: object,
) -> ModelPerformanceSummary:
    missing_fields = [field for field in REQUIRED_SEARCH_TEST_FIELDS if field not in payload]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Training summary missing required search-mode field(s): {missing}")

    metrics = payload.get("metrics")
    brier_score: float | None = None
    accuracy: float | None = None
    if isinstance(metrics, dict):
        if "brier_score" in metrics:
            brier_score = float(metrics["brier_score"])
        if "accuracy" in metrics:
            accuracy = float(metrics["accuracy"])

    validation_auc = (
        float(payload["validation_auc"]) if "validation_auc" in payload else None
    )
    validation_log_loss = (
        float(payload["validation_log_loss"]) if "validation_log_loss" in payload else None
    )
    best_config_index = (
        int(payload["best_config_index"]) if "best_config_index" in payload else None
    )

    return ModelPerformanceSummary(
        auc=float(payload["test_auc"]),
        log_loss=float(payload["test_log_loss"]),
        brier_score=brier_score,
        accuracy=accuracy,
        validation_auc=validation_auc,
        validation_log_loss=validation_log_loss,
        best_config_index=best_config_index,
        **base_kwargs,
    )


def load_feature_importance(path: Path | None = None) -> pd.DataFrame | None:
    """Load LightGBM gain feature importance from the reports CSV."""
    csv_path = (path or DEFAULT_FEATURE_IMPORTANCE_GAIN_CSV).resolve()
    if not csv_path.is_file():
        return None

    try:
        frame = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Malformed feature importance CSV at {csv_path}: {exc}") from exc

    missing_columns = [
        column
        for column in REQUIRED_FEATURE_IMPORTANCE_COLUMNS
        if column not in frame.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Feature importance CSV missing required column(s): {missing}")

    if frame.empty:
        raise ValueError(f"Feature importance CSV at {csv_path} is empty")

    return (
        frame[list(REQUIRED_FEATURE_IMPORTANCE_COLUMNS)]
        .sort_values(FEATURE_IMPORTANCE_VALUE_COLUMN, ascending=False)
        .reset_index(drop=True)
    )


def top_feature_importance(frame: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Return the top ``n`` features by importance."""
    if n <= 0:
        return frame.iloc[0:0].copy()
    return frame.head(min(n, len(frame))).copy()


def existing_model_artifact_paths(
    *,
    calibration_curve_path: Path | None = None,
    probability_distribution_path: Path | None = None,
    feature_importance_gain_path: Path | None = None,
) -> dict[str, Path]:
    """Return local model artifact paths that exist on disk."""
    candidates = {
        "calibration_curve": calibration_curve_path or DEFAULT_LIGHTGBM_CALIBRATION_CURVE_PNG,
        "probability_distribution": (
            probability_distribution_path or DEFAULT_LIGHTGBM_PROBABILITY_DISTRIBUTION_PNG
        ),
        "feature_importance_gain": (
            feature_importance_gain_path or DEFAULT_FEATURE_IMPORTANCE_GAIN_PNG
        ),
    }
    return {
        key: path.resolve()
        for key, path in candidates.items()
        if path.is_file()
    }


def sample_prediction_rows(
    test: pd.DataFrame,
    n: int = 100,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a lightweight sample of test shots for the prediction playground UI."""
    if test.empty:
        return pd.DataFrame(columns=list(PREDICTION_DISPLAY_COLUMNS))

    sample_size = min(n, len(test))
    sampled = test.sample(n=sample_size, random_state=random_state).copy()
    sampled[PREDICTION_ROW_ID_COLUMN] = sampled.index
    return sampled[list(PREDICTION_DISPLAY_COLUMNS)].reset_index(drop=True)


def get_prediction_row(test: pd.DataFrame, row_id: int) -> pd.Series:
    """Return the full test-set row for a selected prediction playground row."""
    if row_id not in test.index:
        raise KeyError(f"Prediction row_id not found in test set: {row_id}")
    return test.loc[row_id]


def prepare_prediction_features(row: pd.Series) -> PreparedPredictionFeatures:
    """Build API-style model inputs from a full test-set row."""
    missing_columns = [column for column in FEATURE_COLUMNS if column not in row.index]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Row missing required feature columns: {missing}")

    shot_value = int(row[SHOT_VALUE_COLUMN])
    features = {column: float(row[column]) for column in PREDICTION_FEATURE_COLUMNS}
    return PreparedPredictionFeatures(features=features, shot_value=shot_value)


def _series_row(row: pd.Series | dict[str, object]) -> pd.Series:
    if isinstance(row, pd.Series):
        return row
    return pd.Series(row)


def actual_points_for_row(row: pd.Series | dict[str, object]) -> float | None:
    """Return points scored on the attempt, or None when the outcome is unknown."""
    series = _series_row(row)
    if TARGET_COLUMN not in series.index or pd.isna(series[TARGET_COLUMN]):
        return None
    if bool(series[TARGET_COLUMN]):
        return float(series[SHOT_VALUE_COLUMN])
    return 0.0


def baseline_for_similar_shots(
    test: pd.DataFrame,
    row: pd.Series | dict[str, object],
) -> BaselineShotProfile:
    """Compute historical make rate and EV for same shot value and distance bucket."""
    series = _series_row(row)
    shot_value = int(series[SHOT_VALUE_COLUMN])
    row_bucket = add_distance_bucket(pd.DataFrame([series])).iloc[0][DISTANCE_BUCKET_COLUMN]
    bucketed_test = add_distance_bucket(test)
    similar = bucketed_test.loc[
        (bucketed_test[SHOT_VALUE_COLUMN] == shot_value)
        & (bucketed_test[DISTANCE_BUCKET_COLUMN] == row_bucket)
    ]

    shot_count = len(similar)
    if shot_count == 0:
        return BaselineShotProfile(shot_count=0, make_rate=0.0, expected_value=0.0)

    make_rate = float(similar[TARGET_COLUMN].mean())
    return BaselineShotProfile(
        shot_count=shot_count,
        make_rate=make_rate,
        expected_value=make_rate * shot_value,
    )


def compare_prediction_to_baseline(
    result: PredictionResultLike,
    row: pd.Series | dict[str, object],
    baseline: BaselineShotProfile,
) -> PredictionComparison:
    """Compare model prediction against a similar-shot historical baseline."""
    series = _series_row(row)
    actual_made: bool | None
    if TARGET_COLUMN not in series.index or pd.isna(series[TARGET_COLUMN]):
        actual_made = None
    else:
        actual_made = bool(series[TARGET_COLUMN])

    return PredictionComparison(
        predicted_make_probability=result.predicted_make_probability,
        expected_shot_value=result.expected_shot_value,
        actual_made=actual_made,
        actual_points=actual_points_for_row(series),
        baseline_make_rate=baseline.make_rate,
        baseline_expected_value=baseline.expected_value,
        probability_edge_vs_baseline=result.predicted_make_probability - baseline.make_rate,
        ev_edge_vs_baseline=result.expected_shot_value - baseline.expected_value,
        similar_shot_count=baseline.shot_count,
    )


def build_shot_edge_row(
    test: pd.DataFrame,
    row_id: int,
    result: PredictionResultLike,
) -> ShotEdgeRow:
    """Build a single shot-edge record from a prediction and held-out test row."""
    row = get_prediction_row(test, row_id)
    series = _series_row(row)
    baseline = baseline_for_similar_shots(test, series)
    comparison = compare_prediction_to_baseline(result, series, baseline)

    return ShotEdgeRow(
        row_id=row_id,
        shot_value=int(series[SHOT_VALUE_COLUMN]),
        shot_distance=float(series[SHOT_DISTANCE_COLUMN]),
        period=int(series[PERIOD_COLUMN]),
        actual_made=comparison.actual_made,
        actual_points=comparison.actual_points,
        predicted_make_probability=comparison.predicted_make_probability,
        expected_shot_value=comparison.expected_shot_value,
        baseline_make_rate=comparison.baseline_make_rate,
        baseline_expected_value=comparison.baseline_expected_value,
        probability_edge_vs_baseline=comparison.probability_edge_vs_baseline,
        ev_edge_vs_baseline=comparison.ev_edge_vs_baseline,
        similar_shot_count=comparison.similar_shot_count,
    )


def build_shot_edge_table(
    test: pd.DataFrame,
    predictions_by_row_id: dict[int, PredictionResultLike],
) -> pd.DataFrame:
    """Build a shot-edge table with one row per prediction."""
    if not predictions_by_row_id:
        return pd.DataFrame(columns=list(SHOT_EDGE_TABLE_COLUMNS))

    rows = [
        build_shot_edge_row(test, row_id, result)
        for row_id, result in predictions_by_row_id.items()
    ]
    return pd.DataFrame(rows, columns=list(SHOT_EDGE_TABLE_COLUMNS))


def assign_edge_bucket(ev_edge: float) -> str:
    """Map an EV edge value to a backtest bucket label."""
    if ev_edge <= STRONG_NEGATIVE_EV_EDGE_THRESHOLD:
        return EDGE_BUCKET_LABELS[0]
    if ev_edge <= 0:
        return EDGE_BUCKET_LABELS[1]
    if ev_edge < STRONG_POSITIVE_EV_EDGE_THRESHOLD:
        return EDGE_BUCKET_LABELS[2]
    return EDGE_BUCKET_LABELS[3]


def add_edge_bucket(edge_table: pd.DataFrame) -> pd.DataFrame:
    """Add an ordered edge-bucket column for backtest grouping."""
    output = edge_table.copy()
    if output.empty:
        output[EDGE_BUCKET_COLUMN] = pd.Categorical(
            [],
            categories=list(EDGE_BUCKET_LABELS),
            ordered=True,
        )
        return output

    output[EDGE_BUCKET_COLUMN] = output["ev_edge_vs_baseline"].map(assign_edge_bucket)
    output[EDGE_BUCKET_COLUMN] = pd.Categorical(
        output[EDGE_BUCKET_COLUMN],
        categories=list(EDGE_BUCKET_LABELS),
        ordered=True,
    )
    return output


def _empty_edge_backtest_summary(bucket: str) -> EdgeBacktestSummary:
    return EdgeBacktestSummary(
        bucket=bucket,
        shot_count=0,
        avg_predicted_make_probability=0.0,
        avg_model_ev=0.0,
        avg_baseline_ev=0.0,
        avg_ev_edge=0.0,
        actual_make_rate=0.0,
        avg_actual_points=0.0,
        model_ev_minus_actual_points=0.0,
        baseline_ev_minus_actual_points=0.0,
    )


def _mean_actual_make_rate(frame: pd.DataFrame) -> float:
    known_outcomes = frame["actual_made"].dropna()
    if known_outcomes.empty:
        return 0.0
    return float(known_outcomes.astype(bool).mean())


def _mean_actual_points(frame: pd.DataFrame) -> float:
    known_points = frame["actual_points"].dropna()
    if known_points.empty:
        return 0.0
    return float(known_points.mean())


def summarize_edge_backtest(edge_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize scored shots by model EV edge bucket."""
    if edge_table.empty:
        return pd.DataFrame(columns=list(EDGE_BACKTEST_COLUMNS))

    bucketed = add_edge_bucket(edge_table)
    summaries: list[EdgeBacktestSummary] = []
    for bucket in EDGE_BUCKET_LABELS:
        subset = bucketed.loc[bucketed[EDGE_BUCKET_COLUMN] == bucket]
        if subset.empty:
            summaries.append(_empty_edge_backtest_summary(bucket))
            continue

        avg_model_ev = float(subset["expected_shot_value"].mean())
        avg_baseline_ev = float(subset["baseline_expected_value"].mean())
        avg_actual_points = _mean_actual_points(subset)

        summaries.append(
            EdgeBacktestSummary(
                bucket=bucket,
                shot_count=len(subset),
                avg_predicted_make_probability=float(
                    subset["predicted_make_probability"].mean()
                ),
                avg_model_ev=avg_model_ev,
                avg_baseline_ev=avg_baseline_ev,
                avg_ev_edge=float(subset["ev_edge_vs_baseline"].mean()),
                actual_make_rate=_mean_actual_make_rate(subset),
                avg_actual_points=avg_actual_points,
                model_ev_minus_actual_points=avg_model_ev - avg_actual_points,
                baseline_ev_minus_actual_points=avg_baseline_ev - avg_actual_points,
            )
        )

    return pd.DataFrame(summaries, columns=list(EDGE_BACKTEST_COLUMNS))
