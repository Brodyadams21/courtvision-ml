"""Build shot-level predictions and expected value for the evaluation period."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from courtvision.data.build_features import (
    DEFAULT_PROCESSED_FEATURES_DIR,
    processed_train_test_paths,
)
from courtvision.data.collect import DEFAULT_SEASON
from courtvision.data.load_data import DEFAULT_ENV_PATH, PROJECT_ROOT
from courtvision.evaluation.expected_value import compute_expected_value_arrays
from courtvision.models.common import DEFAULT_TABLES_DIR, FEATURE_COLUMNS
from courtvision.models.registry import CANDIDATE_ALIAS, REGISTERED_MODEL_NAME

MODEL_TYPE_CANDIDATE = "candidate"
MODEL_TYPE_GRU = "gru"
SUPPORTED_MODEL_TYPES: tuple[str, ...] = (MODEL_TYPE_CANDIDATE, MODEL_TYPE_GRU)

SHOT_PREDICTION_CORE_COLUMNS: tuple[str, ...] = (
    "shot_id",
    "game_id",
    "player_id",
    "team_id",
    "shot_value",
    "shot_made_flag",
    "predicted_make_probability",
    "expected_shot_value",
    "actual_points",
    "points_above_expected",
)

SHOT_PREDICTION_OPTIONAL_METADATA_COLUMNS: tuple[str, ...] = (
    "game_date",
    "shot_zone_basic",
    "shot_zone_area",
    "shot_zone_range",
    "shot_distance",
)

SHOT_PREDICTION_COLUMNS: tuple[str, ...] = (
    *SHOT_PREDICTION_CORE_COLUMNS,
    *SHOT_PREDICTION_OPTIONAL_METADATA_COLUMNS,
)

EVALUATION_METADATA_COLUMNS: tuple[str, ...] = (
    "shot_id",
    "game_id",
    "player_id",
    "team_id",
    "shot_value",
    "shot_made_flag",
)


def configure_mlflow() -> None:
    """Load ``.env`` and set MLflow tracking URI when configured."""
    load_dotenv(DEFAULT_ENV_PATH)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def evaluation_shots_path(
    season: str,
    *,
    processed_dir: Path | None = None,
) -> Path:
    """Return the held-out test Parquet path for a season."""
    return processed_train_test_paths(season, output_dir=processed_dir)["test"]


def shot_predictions_output_path(
    season: str,
    *,
    model_type: str = MODEL_TYPE_CANDIDATE,
    tables_dir: Path | None = None,
) -> Path:
    """Default CSV path for shot-level prediction output."""
    if model_type not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(SUPPORTED_MODEL_TYPES)
        raise ValueError(f"Unsupported model_type {model_type!r}; expected one of: {supported}")

    directory = tables_dir or DEFAULT_TABLES_DIR
    return directory / f"shot_predictions_{season}_{model_type}.csv"


def load_evaluation_shots(
    season: str = DEFAULT_SEASON,
    *,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Load evaluation-period shots from the season test Parquet export."""
    path = evaluation_shots_path(season, processed_dir=processed_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation shots parquet not found: {path}")
    return pd.read_parquet(path)


def load_candidate_model(
    *,
    registered_model_name: str = REGISTERED_MODEL_NAME,
    alias: str = CANDIDATE_ALIAS,
) -> object:
    """Load the registered Candidate alias from the MLflow model registry."""
    configure_mlflow()
    model_uri = f"models:/{registered_model_name}@{alias}"
    return mlflow.sklearn.load_model(model_uri)


def predict_make_probabilities(model: object, shots: pd.DataFrame) -> np.ndarray:
    """Return predicted make probabilities for evaluation shots."""
    missing_features = [column for column in FEATURE_COLUMNS if column not in shots.columns]
    if missing_features:
        raise KeyError(f"Evaluation shots missing feature columns: {missing_features}")

    features = shots[FEATURE_COLUMNS]
    probabilities = model.predict_proba(features)[:, 1]
    return np.asarray(probabilities, dtype=float)


def build_shot_predictions_table(
    shots: pd.DataFrame,
    predicted_make_probability: np.ndarray,
) -> pd.DataFrame:
    """Combine identifiers, model probabilities, and expected-value metrics."""
    missing_columns = [
        column for column in EVALUATION_METADATA_COLUMNS if column not in shots.columns
    ]
    if missing_columns:
        raise KeyError(f"Evaluation shots missing required columns: {missing_columns}")

    probabilities = np.asarray(predicted_make_probability, dtype=float)
    if len(shots) != len(probabilities):
        raise ValueError(
            "Shot count and prediction count must match: "
            f"{len(shots):,} shots vs {len(probabilities):,} predictions"
        )

    expected_shot_value, actual_points, points_above_expected = compute_expected_value_arrays(
        probabilities,
        shots["shot_value"].to_numpy(),
        shots["shot_made_flag"].to_numpy(dtype=bool),
    )

    result: dict[str, object] = {
        "shot_id": shots["shot_id"].values,
        "game_id": shots["game_id"].values,
        "player_id": shots["player_id"].values,
        "team_id": shots["team_id"].values,
        "shot_value": shots["shot_value"].astype(int).values,
        "shot_made_flag": shots["shot_made_flag"].astype(bool).values,
        "predicted_make_probability": probabilities,
        "expected_shot_value": expected_shot_value,
        "actual_points": actual_points,
        "points_above_expected": points_above_expected,
    }
    output_columns = list(SHOT_PREDICTION_CORE_COLUMNS)
    for column in SHOT_PREDICTION_OPTIONAL_METADATA_COLUMNS:
        if column in shots.columns:
            result[column] = shots[column].values
            output_columns.append(column)

    return pd.DataFrame(result, columns=output_columns)


def save_shot_predictions_csv(
    predictions: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write shot-level predictions to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return output_path


def run_predict_shots(
    season: str = DEFAULT_SEASON,
    *,
    model_type: str = MODEL_TYPE_CANDIDATE,
    gru_run_id: str | None = None,
    processed_dir: Path | None = None,
    output_path: Path | None = None,
    registered_model_name: str = REGISTERED_MODEL_NAME,
    alias: str = CANDIDATE_ALIAS,
    gru_cache_root: Path | None = None,
    gru_force_download: bool = False,
    gru_device: str | None = None,
    gru_batch_size: int | None = None,
) -> pd.DataFrame:
    """Load evaluation shots, score with the selected model, and save shot predictions."""
    if model_type not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(SUPPORTED_MODEL_TYPES)
        raise ValueError(f"Unsupported model_type {model_type!r}; expected one of: {supported}")

    shots = load_evaluation_shots(season, processed_dir=processed_dir)
    print(f"Loaded evaluation shots: {len(shots):,}")

    if model_type == MODEL_TYPE_CANDIDATE:
        model = load_candidate_model(
            registered_model_name=registered_model_name,
            alias=alias,
        )
        probabilities = predict_make_probabilities(model, shots)
    else:
        if not gru_run_id:
            raise ValueError("--gru-run-id is required when --model-type gru")
        from courtvision.evaluation.predict_gru import score_evaluation_shots_with_gru

        print(f"Using GRU model from run {gru_run_id}")
        probabilities = score_evaluation_shots_with_gru(
            shots,
            gru_run_id,
            season=season,
            cache_root=gru_cache_root,
            force_download=gru_force_download,
            device=gru_device,
            batch_size=gru_batch_size,
        )

    print(f"Generated predictions: {len(probabilities):,}")

    predictions = build_shot_predictions_table(shots, probabilities)
    destination = output_path or shot_predictions_output_path(season, model_type=model_type)
    save_shot_predictions_csv(predictions, destination)
    try:
        saved_path = destination.relative_to(PROJECT_ROOT)
    except ValueError:
        saved_path = destination
    print(f"Saved {saved_path}")
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score evaluation-period shots and write shot-level expected value CSV.",
    )
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season label (e.g. 2024-25)")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_FEATURES_DIR,
        help=f"Directory with train/test Parquet (default: {DEFAULT_PROCESSED_FEATURES_DIR})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output CSV path "
            "(default: reports/tables/shot_predictions_<season>_<model_type>.csv)"
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=SUPPORTED_MODEL_TYPES,
        default=MODEL_TYPE_CANDIDATE,
        help="Model backend to score evaluation shots (default: candidate)",
    )
    parser.add_argument(
        "--gru-run-id",
        default=None,
        help="MLflow run id for GRU artifacts (required when --model-type gru)",
    )
    parser.add_argument(
        "--model-name",
        default=REGISTERED_MODEL_NAME,
        help=f"Registered MLflow model name (default: {REGISTERED_MODEL_NAME})",
    )
    parser.add_argument(
        "--model-alias",
        default=CANDIDATE_ALIAS,
        help=f"Registered model alias to load (default: {CANDIDATE_ALIAS})",
    )
    parser.add_argument(
        "--gru-cache-root",
        type=Path,
        default=None,
        help="Local cache directory for GRU artifacts (gru model type only)",
    )
    parser.add_argument(
        "--gru-force-download",
        action="store_true",
        help="Re-download GRU artifacts even if cached locally",
    )
    parser.add_argument(
        "--gru-device",
        default=None,
        help="Torch device for GRU inference (default: cuda when available)",
    )
    parser.add_argument(
        "--gru-batch-size",
        type=int,
        default=None,
        help="Inference batch size for GRU scoring",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_predict_shots(
        args.season,
        model_type=args.model_type,
        gru_run_id=args.gru_run_id,
        processed_dir=args.processed_dir,
        output_path=args.output_path,
        registered_model_name=args.model_name,
        alias=args.model_alias,
        gru_cache_root=args.gru_cache_root,
        gru_force_download=args.gru_force_download,
        gru_device=args.gru_device,
        gru_batch_size=args.gru_batch_size,
    )


if __name__ == "__main__":
    main()
