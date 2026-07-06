"""Dashboard data helpers (Streamlit-free, testable)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
