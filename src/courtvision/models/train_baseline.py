"""Baseline logistic regression: load Parquet, build X/y, train sklearn pipeline."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from courtvision.data.build_features import (
    DEFAULT_PROCESSED_FEATURES_DIR,
    processed_train_test_paths,
)
from courtvision.data.collect import DEFAULT_SEASON
from courtvision.models.common import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    evaluate_classification_metrics,
    load_train_test_parquet,
    print_missing_by_feature,
    save_baseline_figures,
    split_features_target,
)

DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "courtvision-baseline"

MODEL_TYPE = "logistic_regression"
IMPUTER_STRATEGY = "median"
SCALER_NAME = "StandardScaler"
CLASSIFIER_NAME = "LogisticRegression"
MAX_ITER = 1000


def build_baseline_pipeline() -> Pipeline:
    """Median impute → scale → logistic regression baseline."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy=IMPUTER_STRATEGY)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=MAX_ITER)),
        ]
    )


def train_and_predict_proba(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> tuple[Pipeline, np.ndarray]:
    """Fit pipeline on train and return test make probabilities."""
    pipeline = build_baseline_pipeline()
    pipeline.fit(x_train, y_train)
    test_proba = pipeline.predict_proba(x_test)[:, 1]
    return pipeline, test_proba


def print_baseline_metrics(metrics: dict[str, float]) -> None:
    print("\nBaseline logistic regression metrics")
    print(f"  auc: {metrics['auc']:.4f}")
    print(f"  log_loss: {metrics['log_loss']:.4f}")
    print(f"  brier_score: {metrics['brier_score']:.4f}")
    print(f"  accuracy: {metrics['accuracy']:.4f}")


def to_mlflow_tracking_uri(tracking_uri: Path | str) -> str:
    """Return an MLflow tracking URI (HTTP passes through; local paths become file URIs)."""
    if isinstance(tracking_uri, Path):
        return tracking_uri.resolve().as_uri()
    uri = str(tracking_uri)
    if "://" not in uri:
        return Path(uri).resolve().as_uri()
    return uri


def write_feature_list_artifact(output_path: Path) -> Path:
    """Write model feature names for MLflow artifact logging."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def log_baseline_mlflow_run(
    *,
    season: str,
    pipeline: Pipeline,
    metrics: dict[str, float],
    train_rows: int,
    test_rows: int,
    calibration_path: Path,
    distribution_path: Path,
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT_NAME,
) -> None:
    """Log baseline parameters, metrics, model, plots, and feature list to MLflow."""
    mlflow.set_tracking_uri(to_mlflow_tracking_uri(tracking_uri))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"baseline-logistic-{season}"):
        mlflow.log_params(
            {
                "model_type": MODEL_TYPE,
                "season": season,
                "feature_count": len(FEATURE_COLUMNS),
                "train_rows": train_rows,
                "test_rows": test_rows,
                "imputer_strategy": IMPUTER_STRATEGY,
                "scaler": SCALER_NAME,
                "classifier": CLASSIFIER_NAME,
                "max_iter": MAX_ITER,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(calibration_path))
        mlflow.log_artifact(str(distribution_path))
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        with tempfile.TemporaryDirectory() as temp_dir:
            feature_list_path = write_feature_list_artifact(
                Path(temp_dir) / "feature_columns.json",
            )
            mlflow.log_artifact(str(feature_list_path))


def main(
    season: str = DEFAULT_SEASON,
    *,
    processed_dir: Path | None = None,
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    mlflow_experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT_NAME,
) -> None:
    paths = processed_train_test_paths(season, output_dir=processed_dir)
    print(f"Season: {season}")
    print(f"Train path: {paths['train']}")
    print(f"Test path:  {paths['test']}")

    train, test = load_train_test_parquet(season, processed_dir=processed_dir)
    x_train, y_train = split_features_target(train)
    x_test, y_test = split_features_target(test)

    print(f"\nX_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"target mean in train: {y_train.mean():.4f}")
    print(f"target mean in test: {y_test.mean():.4f}")

    print_missing_by_feature(x_train, x_test)

    pipeline, test_proba = train_and_predict_proba(x_train, y_train, x_test)

    print("\nmodel trained successfully")
    print(f"test predicted probability min/max: {test_proba.min():.4f} / {test_proba.max():.4f}")
    predicted_probability_mean = float(test_proba.mean())
    print(f"test predicted probability mean: {predicted_probability_mean:.4f}")

    metrics = evaluate_classification_metrics(y_test, test_proba)
    print_baseline_metrics(metrics)

    calibration_path, distribution_path = save_baseline_figures(y_test, test_proba)
    print(f"\nSaved calibration curve to {calibration_path}")
    print(f"Saved probability distribution to {distribution_path}")

    mlflow_metrics = {
        **metrics,
        "train_target_mean": float(y_train.mean()),
        "test_target_mean": float(y_test.mean()),
        "predicted_probability_mean": predicted_probability_mean,
    }
    log_baseline_mlflow_run(
        season=season,
        pipeline=pipeline,
        metrics=mlflow_metrics,
        train_rows=len(x_train),
        test_rows=len(x_test),
        calibration_path=calibration_path,
        distribution_path=distribution_path,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment_name,
    )
    print("\nMLflow run logged successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline logistic regression on shot make features.",
    )
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season label (e.g. 2024-25)")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_FEATURES_DIR,
        help=f"Directory with train/test Parquet (default: {DEFAULT_PROCESSED_FEATURES_DIR})",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=DEFAULT_MLFLOW_TRACKING_URI,
        help=(
            "MLflow tracking server URI "
            f"(default: {DEFAULT_MLFLOW_TRACKING_URI}; server stores to PostgreSQL backend)"
        ),
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=DEFAULT_MLFLOW_EXPERIMENT_NAME,
        help=f"MLflow experiment name (default: {DEFAULT_MLFLOW_EXPERIMENT_NAME})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.season,
        processed_dir=args.processed_dir,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment,
    )
