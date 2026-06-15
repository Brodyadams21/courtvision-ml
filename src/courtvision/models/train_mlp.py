"""Train PyTorch MLP on shot make features; evaluate on held-out test."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader

from courtvision.data.build_features import (
    DEFAULT_PROCESSED_FEATURES_DIR,
    time_based_train_test_split,
)
from courtvision.data.collect import DEFAULT_SEASON
from courtvision.data.load_data import DEFAULT_ENV_PATH
from courtvision.models.common import evaluate_classification_metrics, load_train_test_parquet
from courtvision.models.spatial_features import (
    FEATURE_SET_SPATIAL,
    FEATURE_SET_TABULAR,
    mlp_feature_columns,
    split_mlp_features_target,
)
from courtvision.models.torch_data import DataLoaderBundle, prepare_dataloaders
from courtvision.models.torch_models import (
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_DIMS,
    ShotMakeMLP,
    build_shot_make_mlp,
)

MODEL_TYPE = "pytorch_mlp"
MLFLOW_EXPERIMENT = "courtvision-mlp"
INNER_TRAIN_FRACTION = 0.8
RANDOM_STATE = 42

DEFAULT_BATCH_SIZE = 1024
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5


@dataclass(frozen=True)
class MLPTrainConfig:
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS
    dropout: float = DEFAULT_DROPOUT
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE


def configure_mlflow() -> None:
    """Load ``.env`` and set MLflow tracking URI / experiment."""
    import mlflow

    load_dotenv(DEFAULT_ENV_PATH)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = RANDOM_STATE) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_inner_train_validation(
    train: pd.DataFrame,
    *,
    inner_train_fraction: float = INNER_TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Time-based inner split of exported train: earlier games -> inner train."""
    return time_based_train_test_split(train, train_fraction=inner_train_fraction)


def train_one_epoch(
    model: ShotMakeMLP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = len(targets)
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    return total_loss / sample_count


@torch.no_grad()
def predict_loader(
    model: ShotMakeMLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for features, targets in loader:
        features = features.to(device)
        batch_proba = model.predict_proba(features).cpu().numpy()
        probabilities.append(batch_proba)
        labels.append(targets.numpy())

    return np.concatenate(labels), np.concatenate(probabilities)


@torch.no_grad()
def evaluate_loader_loss(
    model: ShotMakeMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    sample_count = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        loss = criterion(logits, targets)
        batch_size = len(targets)
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    return total_loss / sample_count


@dataclass(frozen=True)
class FitResult:
    model: ShotMakeMLP
    best_validation_log_loss: float


def fit_mlp(
    bundle: DataLoaderBundle,
    config: MLPTrainConfig,
    *,
    device: torch.device,
) -> FitResult:
    """Train MLP with validation log-loss early stopping."""
    model = build_shot_make_mlp(
        bundle.feature_count,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_validation_log_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_one_epoch(model, bundle.train, criterion, optimizer, device)

        if bundle.validation is not None:
            y_val_true, val_proba = predict_loader(model, bundle.validation, device)
            validation_log_loss = evaluate_classification_metrics(y_val_true, val_proba)["log_loss"]
            print(f"Epoch {epoch} | validation log loss: {validation_log_loss:.4f}")

            if validation_log_loss < best_validation_log_loss:
                best_validation_log_loss = validation_log_loss
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    break
        else:
            print(f"Epoch {epoch}")
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return FitResult(
        model=model,
        best_validation_log_loss=best_validation_log_loss,
    )


def print_test_metrics(metrics: dict[str, float]) -> None:
    print(f"Test AUC: {metrics['auc']:.4f}")
    print(f"Test log loss: {metrics['log_loss']:.4f}")
    print(f"Test Brier score: {metrics['brier_score']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")


def log_mlflow_run(
    *,
    season: str,
    feature_set: str,
    feature_count: int,
    config: MLPTrainConfig,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    train_metrics: dict[str, float] | None,
    validation_metrics: dict[str, float] | None,
    test_metrics: dict[str, float],
    model: ShotMakeMLP,
) -> None:
    """Log MLP parameters, metrics, and model artifacts to MLflow.

    Artifact logging (calibration curves, probability distributions, registered
    model) will be expanded in a follow-up change.
    """
    import mlflow

    configure_mlflow()
    run_name = f"mlp-{feature_set}-{season}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("season", season)
        mlflow.log_param("feature_count", feature_count)
        mlflow.log_param("train_rows", train_rows)
        mlflow.log_param("validation_rows", validation_rows)
        mlflow.log_param("test_rows", test_rows)
        mlflow.log_param("hidden_dims", list(config.hidden_dims))
        mlflow.log_param("dropout", config.dropout)
        mlflow.log_param("learning_rate", config.learning_rate)
        mlflow.log_param("weight_decay", config.weight_decay)
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("epochs", config.epochs)
        mlflow.log_param("patience", config.patience)

        if train_metrics is not None:
            for key, value in train_metrics.items():
                mlflow.log_metric(f"train_{key}", value)
        if validation_metrics is not None:
            for key, value in validation_metrics.items():
                mlflow.log_metric(f"validation_{key}", value)
        for key, value in test_metrics.items():
            mlflow.log_metric(f"test_{key}", value)

        # Model and plot artifacts will be added when mlflow.pytorch integration is wired up.


def run_default(
    season: str,
    *,
    feature_set: str = FEATURE_SET_TABULAR,
    processed_dir: Path | None = None,
    config: MLPTrainConfig | None = None,
    inner_train_fraction: float = INNER_TRAIN_FRACTION,
    device: str | None = None,
    log_mlflow: bool = True,
) -> dict[str, float]:
    """Train MLP with inner validation for early stopping; evaluate on held-out test."""
    train_config = config or MLPTrainConfig()
    torch_device = resolve_device(device)
    feature_columns = mlp_feature_columns(feature_set)
    set_seed()

    print(f"Season: {season}")
    print(f"Model: {MODEL_TYPE}")
    print(f"Feature set: {feature_set}")

    train, test = load_train_test_parquet(season, processed_dir=processed_dir)
    inner_train, validation, _split_meta = split_inner_train_validation(
        train,
        inner_train_fraction=inner_train_fraction,
    )

    x_inner, y_inner = split_mlp_features_target(inner_train, feature_set)
    x_val, y_val = split_mlp_features_target(validation, feature_set)
    x_test, y_test = split_mlp_features_target(test, feature_set)

    print(f"Inner train rows: {len(x_inner):,}")
    print(f"Validation rows: {len(x_val):,}")
    print(f"Test rows: {len(x_test):,}")
    print(f"Feature count: {len(feature_columns)}")

    bundle = prepare_dataloaders(
        x_inner,
        y_inner,
        feature_columns=feature_columns,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        batch_size=train_config.batch_size,
    )

    fit_result = fit_mlp(bundle, train_config, device=torch_device)
    model = fit_result.model

    print(f"Best validation log loss: {fit_result.best_validation_log_loss:.4f}")

    y_val_true, val_proba = predict_loader(model, bundle.validation, torch_device)
    validation_metrics = evaluate_classification_metrics(y_val_true, val_proba)

    y_test_true, test_proba = predict_loader(model, bundle.test, torch_device)
    test_metrics = evaluate_classification_metrics(y_test_true, test_proba)
    print_test_metrics(test_metrics)

    if log_mlflow:
        log_mlflow_run(
            season=season,
            feature_set=feature_set,
            feature_count=len(feature_columns),
            config=train_config,
            train_rows=len(x_inner),
            validation_rows=len(x_val),
            test_rows=len(x_test),
            train_metrics=None,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            model=model,
        )

    return test_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PyTorch MLP on shot make features.",
    )
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season label (e.g. 2024-25)")
    parser.add_argument(
        "--mode",
        choices=("default",),
        default="default",
        help="default: train MLP with inner validation and test evaluation",
    )
    parser.add_argument(
        "--feature-set",
        choices=(FEATURE_SET_TABULAR, FEATURE_SET_SPATIAL),
        default=FEATURE_SET_TABULAR,
        help=(
            "tabular: MLP v1 on the 31 LightGBM features; "
            "spatial: MLP v2 with tabular + court-location encodings"
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_FEATURES_DIR,
        help=f"Directory with train/test Parquet (default: {DEFAULT_PROCESSED_FEATURES_DIR})",
    )
    parser.add_argument(
        "--inner-train-fraction",
        type=float,
        default=INNER_TRAIN_FRACTION,
        help="Fraction of earliest train games for inner train (default: 0.8)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode != "default":
        raise SystemExit(f"Unsupported mode: {args.mode}")

    config = MLPTrainConfig(
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )
    run_default(
        args.season,
        feature_set=args.feature_set,
        processed_dir=args.processed_dir,
        config=config,
        inner_train_fraction=args.inner_train_fraction,
        device=args.device,
        log_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
