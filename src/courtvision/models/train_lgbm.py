"""LightGBM shot-make model: default training or small hyperparameter search with MLflow."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier

from courtvision.data.build_features import (
    DEFAULT_PROCESSED_FEATURES_DIR,
    processed_train_test_paths,
    time_based_train_test_split,
)
from courtvision.data.collect import DEFAULT_SEASON
from courtvision.data.load_data import DEFAULT_ENV_PATH
from courtvision.models.common import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_TABLES_DIR,
    FEATURE_COLUMNS,
    evaluate_classification_metrics,
    load_train_test_parquet,
    save_calibration_curve,
    save_probability_distribution,
    split_features_target,
)
from courtvision.models.registry import (
    CANDIDATE_ALIAS,
    REGISTERED_MODEL_NAME,
    promote_model_version_to_candidate,
)

MODEL_TYPE = "lightgbm"
MLFLOW_EXPERIMENT = "courtvision-lightgbm"
INNER_TRAIN_FRACTION = 0.8
RANDOM_STATE = 42

DEFAULT_LGBM_PARAMS: dict[str, float | int] = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 0.0,
}

# Hand-picked grid (10 configs) — not full Cartesian product.
SEARCH_CONFIGS: tuple[dict[str, float | int], ...] = (
    {
        "num_leaves": 15,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.0,
    },
    {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.0,
    },
    {
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.0,
    },
    {
        "num_leaves": 31,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.0,
    },
    {
        "num_leaves": 31,
        "learning_rate": 0.03,
        "n_estimators": 600,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.0,
    },
    {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 600,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.0,
    },
    {
        "num_leaves": 63,
        "learning_rate": 0.03,
        "n_estimators": 600,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.0,
    },
    {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.0,
    },
    {
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 600,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
)


class SearchTrialResult(TypedDict):
    config_index: int
    params: dict[str, float | int]
    validation_log_loss: float
    validation_auc: float
    test_log_loss: float
    test_auc: float


def configure_mlflow(tracking_uri: str | None = None) -> None:
    """Load ``.env`` and set MLflow tracking URI / experiment."""
    load_dotenv(DEFAULT_ENV_PATH)
    resolved_tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if resolved_tracking_uri:
        mlflow.set_tracking_uri(resolved_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def build_lgbm_classifier(**overrides: float | int) -> LGBMClassifier:
    """Binary LightGBM classifier; raw features only (no imputation or scaling)."""
    params = {**DEFAULT_LGBM_PARAMS, **overrides}
    return LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        verbose=-1,
        **params,
    )


def train_and_predict_proba(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
    **lgbm_params: float | int,
) -> tuple[LGBMClassifier, np.ndarray]:
    """Fit LightGBM and return predicted make probabilities on ``x_eval``."""
    model = build_lgbm_classifier(**lgbm_params)
    model.fit(x_train, y_train)
    eval_proba = model.predict_proba(x_eval)[:, 1]
    return model, eval_proba


def split_inner_train_validation(
    train: pd.DataFrame,
    *,
    inner_train_fraction: float = INNER_TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Time-based inner split of exported train: earlier games -> inner train."""
    return time_based_train_test_split(train, train_fraction=inner_train_fraction)


def _row_counts(
    *,
    train_rows: int,
    validation_rows: int = 0,
    test_rows: int,
) -> dict[str, int]:
    return {
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
    }


def _hyperparam_log_payload(params: dict[str, float | int]) -> dict[str, float | int]:
    keys = (
        "num_leaves",
        "learning_rate",
        "n_estimators",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    )
    return {key: params[key] for key in keys}


def _log_common_params(
    *,
    season: str,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    params: dict[str, float | int],
) -> None:
    mlflow.log_param("model_type", MODEL_TYPE)
    mlflow.log_param("season", season)
    mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
    for key, value in _row_counts(
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
    ).items():
        mlflow.log_param(key, value)
    for key, value in _hyperparam_log_payload(params).items():
        mlflow.log_param(key, value)


def _log_validation_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metric("validation_auc", metrics["auc"])
    mlflow.log_metric("validation_log_loss", metrics["log_loss"])
    mlflow.log_metric("validation_brier_score", metrics["brier_score"])


def _log_test_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metric("test_auc", metrics["auc"])
    mlflow.log_metric("test_log_loss", metrics["log_loss"])
    mlflow.log_metric("test_brier_score", metrics["brier_score"])
    mlflow.log_metric("test_accuracy", metrics["accuracy"])


LGBM_GAIN_IMPORTANCE = "gain"
LGBM_SPLIT_IMPORTANCE = "split"


def lgbm_feature_importance(
    model: LGBMClassifier,
    importance_type: str,
) -> np.ndarray:
    """Return per-feature importance from the fitted LightGBM booster."""
    return np.asarray(
        model.booster_.feature_importance(importance_type=importance_type),
        dtype=float,
    )


def _importance_axis_label(importance_type: str) -> str:
    if importance_type == LGBM_GAIN_IMPORTANCE:
        return "Importance (gain)"
    return "Importance (split count)"


def _importance_plot_title(importance_type: str) -> str:
    if importance_type == LGBM_GAIN_IMPORTANCE:
        return "LightGBM feature importance (gain)"
    return "LightGBM feature importance (split count)"


def save_feature_importance_plot(
    model: LGBMClassifier,
    feature_names: list[str],
    output_path: Path,
    *,
    importance_type: str = LGBM_GAIN_IMPORTANCE,
) -> Path:
    """Save horizontal bar chart of LightGBM feature importances."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = lgbm_feature_importance(model, importance_type)
    importance = pd.Series(values, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(8, max(6, len(feature_names) * 0.25)))
    importance.plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel(_importance_axis_label(importance_type))
    ax.set_title(_importance_plot_title(importance_type))
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_feature_importance_csv(
    model: LGBMClassifier,
    feature_names: list[str],
    output_path: Path,
    *,
    importance_type: str = LGBM_GAIN_IMPORTANCE,
) -> Path:
    """Write feature importances to CSV sorted descending."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = lgbm_feature_importance(model, importance_type)
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
        }
    ).sort_values("importance", ascending=False)
    frame.to_csv(output_path, index=False)
    return output_path


def log_best_model_artifacts(
    model: LGBMClassifier,
    *,
    y_test: pd.Series,
    test_proba: np.ndarray,
    season: str,
    register_candidate: bool = False,
) -> int | None:
    """Log calibration, probability distribution, importances, model, and feature list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        calibration_path = save_calibration_curve(
            y_test,
            test_proba,
            artifact_dir / "lightgbm_calibration_curve.png",
            model_label="LightGBM",
            title=f"LightGBM calibration curve (test, {season})",
        )
        distribution_path = save_probability_distribution(
            test_proba,
            artifact_dir / "lightgbm_probability_distribution.png",
            title=f"LightGBM predicted probability distribution (test, {season})",
        )
        gain_plot_path = save_feature_importance_plot(
            model,
            FEATURE_COLUMNS,
            DEFAULT_FIGURES_DIR / "lightgbm_feature_importance_gain.png",
            importance_type=LGBM_GAIN_IMPORTANCE,
        )
        gain_csv_report_path = save_feature_importance_csv(
            model,
            FEATURE_COLUMNS,
            DEFAULT_TABLES_DIR / "lightgbm_feature_importance_gain.csv",
            importance_type=LGBM_GAIN_IMPORTANCE,
        )
        gain_csv_path = save_feature_importance_csv(
            model,
            FEATURE_COLUMNS,
            artifact_dir / "lightgbm_feature_importance_gain.csv",
            importance_type=LGBM_GAIN_IMPORTANCE,
        )
        split_csv_path = save_feature_importance_csv(
            model,
            FEATURE_COLUMNS,
            artifact_dir / "lightgbm_feature_importance_split.csv",
            importance_type=LGBM_SPLIT_IMPORTANCE,
        )
        feature_list_path = artifact_dir / "feature_columns.json"
        feature_list_path.write_text(
            json.dumps(
                {"target": "shot_made_flag", "features": FEATURE_COLUMNS},
                indent=2,
            ),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(calibration_path))
        mlflow.log_artifact(str(distribution_path))
        mlflow.log_artifact(str(gain_plot_path))
        mlflow.log_artifact(str(gain_csv_report_path))
        mlflow.log_artifact(str(gain_csv_path))
        mlflow.log_artifact(str(split_csv_path))
        mlflow.log_artifact(str(feature_list_path))

    model_info = mlflow.sklearn.log_model(
        model,
        name="model",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        registered_model_name=REGISTERED_MODEL_NAME if register_candidate else None,
        tags={"candidate": "true"} if register_candidate else None,
    )
    if not register_candidate:
        return None

    version = int(model_info.registered_model_version)
    promote_model_version_to_candidate(version)
    return version


def print_metrics_block(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    print(f"  auc: {metrics['auc']:.4f}")
    print(f"  log_loss: {metrics['log_loss']:.4f}")
    print(f"  brier_score: {metrics['brier_score']:.4f}")
    print(f"  accuracy: {metrics['accuracy']:.4f}")


def run_default(
    season: str,
    *,
    processed_dir: Path | None = None,
    log_mlflow: bool = True,
    mlflow_tracking_uri: str | None = None,
) -> dict[str, float]:
    """Train default LightGBM on full train parquet; evaluate on held-out test."""
    paths = processed_train_test_paths(season, output_dir=processed_dir)
    print(f"Season: {season}")
    print("Mode: default")
    print(f"Model: {MODEL_TYPE}")
    print(f"Train path: {paths['train']}")
    print(f"Test path:  {paths['test']}")

    train, test = load_train_test_parquet(season, processed_dir=processed_dir)
    x_train, y_train = split_features_target(train)
    x_test, y_test = split_features_target(test)

    print(f"\nX_train shape: {x_train.shape}")
    print(f"X_test shape: {x_test.shape}")
    print(f"target mean in train: {y_train.mean():.4f}")
    print(f"target mean in test: {y_test.mean():.4f}")

    model, test_proba = train_and_predict_proba(x_train, y_train, x_test)
    test_metrics = evaluate_classification_metrics(y_test, test_proba)
    print_metrics_block("Test metrics", test_metrics)

    if log_mlflow:
        configure_mlflow(tracking_uri=mlflow_tracking_uri)
        run_name = f"lightgbm-default-{season}"
        with mlflow.start_run(run_name=run_name):
            _log_common_params(
                season=season,
                train_rows=len(x_train),
                validation_rows=0,
                test_rows=len(x_test),
                params=DEFAULT_LGBM_PARAMS,
            )
            _log_test_metrics(test_metrics)
            log_best_model_artifacts(
                model,
                y_test=y_test,
                test_proba=test_proba,
                season=season,
            )

    return test_metrics


def _evaluate_config(
    params: dict[str, float | int],
    *,
    x_inner: pd.DataFrame,
    y_inner: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[LGBMClassifier, dict[str, float], dict[str, float], np.ndarray]:
    model, val_proba = train_and_predict_proba(x_inner, y_inner, x_val, **params)
    val_metrics = evaluate_classification_metrics(y_val, val_proba)
    test_proba = model.predict_proba(x_test)[:, 1]
    test_metrics = evaluate_classification_metrics(y_test, test_proba)
    return model, val_metrics, test_metrics, test_proba


def run_search(
    season: str,
    *,
    processed_dir: Path | None = None,
    inner_train_fraction: float = INNER_TRAIN_FRACTION,
    log_mlflow: bool = True,
    register_candidate: bool = False,
    mlflow_tracking_uri: str | None = None,
) -> SearchTrialResult:
    """Small hyperparameter search with inner validation; retrain best on full train."""
    paths = processed_train_test_paths(season, output_dir=processed_dir)
    print(f"Season: {season}")
    print(f"Mode: search ({len(SEARCH_CONFIGS)} configs)")
    print(f"Model: {MODEL_TYPE}")
    print(f"Train path: {paths['train']}")
    print(f"Test path:  {paths['test']}")
    print(f"Inner train fraction (by game date): {inner_train_fraction:.2f}")

    train, test = load_train_test_parquet(season, processed_dir=processed_dir)
    inner_train, validation, split_meta = split_inner_train_validation(
        train,
        inner_train_fraction=inner_train_fraction,
    )
    print(
        f"\nInner split: train {split_meta['train_games']} games "
        f"({split_meta['train_rows']:,} rows) | "
        f"validation {split_meta['test_games']} games ({split_meta['test_rows']:,} rows)"
    )

    x_inner, y_inner = split_features_target(inner_train)
    x_val, y_val = split_features_target(validation)
    x_train_full, y_train_full = split_features_target(train)
    x_test, y_test = split_features_target(test)

    if log_mlflow:
        configure_mlflow(tracking_uri=mlflow_tracking_uri)

    trial_results: list[SearchTrialResult] = []

    for index, params in enumerate(SEARCH_CONFIGS):
        print(
            f"\n--- Config {index + 1}/{len(SEARCH_CONFIGS)}: "
            f"leaves={params['num_leaves']}, lr={params['learning_rate']}, "
            f"trees={params['n_estimators']}, min_child={params['min_child_samples']}, "
            f"lambda={params['reg_lambda']} ---"
        )

        _, val_metrics, test_metrics, _ = _evaluate_config(
            params,
            x_inner=x_inner,
            y_inner=y_inner,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
        )
        print_metrics_block("Validation metrics", val_metrics)
        print_metrics_block("Test metrics (not used for selection)", test_metrics)

        trial: SearchTrialResult = {
            "config_index": index,
            "params": params,
            "validation_log_loss": val_metrics["log_loss"],
            "validation_auc": val_metrics["auc"],
            "test_log_loss": test_metrics["log_loss"],
            "test_auc": test_metrics["auc"],
        }
        trial_results.append(trial)

        if log_mlflow:
            with mlflow.start_run(run_name=f"lightgbm-search-{season}-{index:02d}"):
                _log_common_params(
                    season=season,
                    train_rows=len(x_inner),
                    validation_rows=len(x_val),
                    test_rows=len(x_test),
                    params=params,
                )
                _log_validation_metrics(val_metrics)
                _log_test_metrics(test_metrics)

    best = min(trial_results, key=lambda row: row["validation_log_loss"])
    print("\n=== Search summary (sorted by validation log loss) ===")
    header = (
        f"{'rank':<5} {'idx':<4} {'val_log_loss':>14} {'val_auc':>10} "
        f"{'test_log_loss':>14} {'leaves':>7} {'lr':>6}"
    )
    print(header)
    print("-" * 72)
    for rank, row in enumerate(
        sorted(trial_results, key=lambda item: item["validation_log_loss"]),
        start=1,
    ):
        params = row["params"]
        print(
            f"{rank:<5} {row['config_index']:<4} {row['validation_log_loss']:14.4f} "
            f"{row['validation_auc']:10.4f} {row['test_log_loss']:14.4f} "
            f"{params['num_leaves']:7d} {params['learning_rate']:6.2f}"
        )

    print(
        f"\nBest config (lowest validation log loss): index={best['config_index']} "
        f"val_log_loss={best['validation_log_loss']:.4f}"
    )

    best_params = best["params"]
    final_model, test_proba = train_and_predict_proba(
        x_train_full,
        y_train_full,
        x_test,
        **best_params,
    )
    final_test_metrics = evaluate_classification_metrics(y_test, test_proba)
    print_metrics_block("Final test metrics (retrained on full train)", final_test_metrics)

    if log_mlflow:
        with mlflow.start_run(run_name=f"lightgbm-best-{season}") as run:
            mlflow.set_tag("best_model", "true")
            mlflow.set_tag("selected_by", "validation_log_loss")
            if register_candidate:
                mlflow.set_tag("candidate", "true")
            mlflow.log_param("best_config_index", best["config_index"])
            _log_common_params(
                season=season,
                train_rows=len(x_train_full),
                validation_rows=len(x_val),
                test_rows=len(x_test),
                params=best_params,
            )
            _log_test_metrics(final_test_metrics)
            registered_version = log_best_model_artifacts(
                final_model,
                y_test=y_test,
                test_proba=test_proba,
                season=season,
                register_candidate=register_candidate,
            )
            if register_candidate and registered_version is not None:
                print(
                    f"\nRegistered {REGISTERED_MODEL_NAME} "
                    f"version {registered_version} with alias '{CANDIDATE_ALIAS}' "
                    f"(run_id={run.info.run_id})"
                )

    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train LightGBM on shot make features (default or small hyperparameter search)."
        ),
    )
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season label (e.g. 2024-25)")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_FEATURES_DIR,
        help=f"Directory with train/test Parquet (default: {DEFAULT_PROCESSED_FEATURES_DIR})",
    )
    parser.add_argument(
        "--mode",
        choices=("default", "search"),
        default="default",
        help="default: train once on full train split; search: tune on inner validation",
    )
    parser.add_argument(
        "--inner-train-fraction",
        type=float,
        default=INNER_TRAIN_FRACTION,
        help="Fraction of earliest train games for inner train (search mode only, default: 0.8)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )
    parser.add_argument(
        "--register-candidate",
        action="store_true",
        help=(
            "Search mode only: register the final best model as "
            f"{REGISTERED_MODEL_NAME} and set the {CANDIDATE_ALIAS} alias"
        ),
    )
    return parser.parse_args()


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.register_candidate and args.mode != "search":
        raise SystemExit("--register-candidate requires --mode search")
    if args.register_candidate and args.no_mlflow:
        raise SystemExit("--register-candidate cannot be used with --no-mlflow")


def main() -> None:
    args = parse_args()
    _validate_cli_args(args)
    log_mlflow = not args.no_mlflow

    if args.mode == "search":
        run_search(
            args.season,
            processed_dir=args.processed_dir,
            inner_train_fraction=args.inner_train_fraction,
            log_mlflow=log_mlflow,
            register_candidate=args.register_candidate,
        )
        return

    run_default(
        args.season,
        processed_dir=args.processed_dir,
        log_mlflow=log_mlflow,
    )


if __name__ == "__main__":
    main()
