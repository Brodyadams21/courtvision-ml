"""Train PyTorch GRU on tabular/spatial features plus prior play-by-play sequences."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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
from courtvision.data.load_data import DEFAULT_ENV_PATH, create_database_engine
from courtvision.models.common import (
    DEFAULT_FIGURES_DIR,
    TARGET_COLUMN,
    evaluate_classification_metrics,
    load_train_test_parquet,
    save_calibration_curve,
    save_probability_distribution,
)
from courtvision.models.sequence_features import (
    EVENT_FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    SequenceBuildResult,
    build_shot_sequences,
    load_play_by_play_for_sequences,
    load_shots_for_sequences,
)
from courtvision.models.spatial_features import FEATURE_SET_SPATIAL, mlp_feature_columns, split_mlp_features_target
from courtvision.models.torch_models import (
    DEFAULT_DROPOUT,
    DEFAULT_GRU_HEAD_DIMS,
    DEFAULT_GRU_HIDDEN_SIZE,
    DEFAULT_TABULAR_EMBED_DIM,
    ShotMakeGRU,
    build_shot_make_gru,
)
from courtvision.models.torch_sequence_data import (
    FEATURE_SET_SPATIAL_SEQUENCE,
    SequenceDataLoaderBundle,
    align_sequences_to_shot_ids,
    build_shot_id_sequence_map,
    prepare_sequence_dataloaders,
    verify_split_shot_sequence_coverage,
)

MODEL_TYPE = "pytorch_gru"
MLFLOW_EXPERIMENT = "courtvision-gru"
MODEL_ROLE = "Challenger"
LIGHTGBM_CANDIDATE_TEST_LOG_LOSS = 0.6495
INNER_TRAIN_FRACTION = 0.8
RANDOM_STATE = 42

DEFAULT_BATCH_SIZE = 1024
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5


@dataclass(frozen=True)
class GRUTrainConfig:
    gru_hidden_size: int = DEFAULT_GRU_HIDDEN_SIZE
    tabular_embed_dim: int = DEFAULT_TABULAR_EMBED_DIM
    head_dims: tuple[int, ...] = DEFAULT_GRU_HEAD_DIMS
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


def load_or_build_sequences(
    season: str,
    *,
    engine=None,
) -> SequenceBuildResult:
    """Build prior-event sequences for all shots in a season."""
    db_engine = engine or create_database_engine(env_path=DEFAULT_ENV_PATH)
    shots = load_shots_for_sequences(db_engine, season=season)
    play_by_play = load_play_by_play_for_sequences(db_engine, season=season)
    return build_shot_sequences(shots, play_by_play)


def split_spatial_sequence_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return spatial tabular features, target, and shot_ids for one split."""
    if "shot_id" not in frame.columns:
        raise KeyError("Frame missing required column: shot_id")
    x_tabular, y = split_mlp_features_target(frame, FEATURE_SET_SPATIAL)
    shot_ids = frame["shot_id"].astype(np.int64)
    return x_tabular, y, shot_ids


def train_one_epoch(
    model: ShotMakeGRU,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0

    for tabular_features, sequence_features, targets in loader:
        tabular_features = tabular_features.to(device)
        sequence_features = sequence_features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(tabular_features, sequence_features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = len(targets)
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    return total_loss / sample_count


@torch.no_grad()
def predict_loader(
    model: ShotMakeGRU,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for tabular_features, sequence_features, targets in loader:
        tabular_features = tabular_features.to(device)
        sequence_features = sequence_features.to(device)
        batch_proba = model.predict_proba(tabular_features, sequence_features).cpu().numpy()
        probabilities.append(batch_proba)
        labels.append(targets.numpy())

    return np.concatenate(labels), np.concatenate(probabilities)


@dataclass(frozen=True)
class TrainingHistory:
    epochs: list[int]
    train_loss: list[float]
    validation_log_loss: list[float]


@dataclass(frozen=True)
class FitResult:
    model: ShotMakeGRU
    best_validation_log_loss: float
    history: TrainingHistory


def save_gru_training_curve(history: TrainingHistory, output_path: Path) -> Path:
    """Plot train BCE loss and validation log loss across epochs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(history.epochs, history.train_loss, marker="o", label="Train loss (BCE)")
    ax.plot(
        history.epochs,
        history.validation_log_loss,
        marker="o",
        label="Validation log loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GRU training curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def fit_gru(
    bundle: SequenceDataLoaderBundle,
    config: GRUTrainConfig,
    *,
    device: torch.device,
) -> FitResult:
    """Train GRU with validation log-loss early stopping."""
    model = build_shot_make_gru(
        bundle.tabular_feature_count,
        bundle.event_feature_count,
        gru_hidden_size=config.gru_hidden_size,
        tabular_embed_dim=config.tabular_embed_dim,
        head_dims=config.head_dims,
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
    history_epochs: list[int] = []
    history_train_loss: list[float] = []
    history_validation_log_loss: list[float] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, bundle.train, criterion, optimizer, device)
        history_epochs.append(epoch)
        history_train_loss.append(train_loss)

        if bundle.validation is not None:
            y_val_true, val_proba = predict_loader(model, bundle.validation, device)
            validation_log_loss = evaluate_classification_metrics(y_val_true, val_proba)["log_loss"]
            history_validation_log_loss.append(validation_log_loss)
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
            history_validation_log_loss.append(float("nan"))
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return FitResult(
        model=model,
        best_validation_log_loss=best_validation_log_loss,
        history=TrainingHistory(
            epochs=history_epochs,
            train_loss=history_train_loss,
            validation_log_loss=history_validation_log_loss,
        ),
    )


def print_test_metrics(metrics: dict[str, float]) -> None:
    print(f"Test AUC: {metrics['auc']:.4f}")
    print(f"Test log loss: {metrics['log_loss']:.4f}")
    print(f"Test Brier score: {metrics['brier_score']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")


def _beats_lightgbm(test_metrics: dict[str, float]) -> bool:
    return test_metrics["log_loss"] < LIGHTGBM_CANDIDATE_TEST_LOG_LOSS


def log_gru_mlflow_run(
    *,
    season: str,
    config: GRUTrainConfig,
    model: ShotMakeGRU,
    bundle: SequenceDataLoaderBundle,
    tabular_feature_columns: list[str],
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    history: TrainingHistory,
    y_test: pd.Series | np.ndarray,
    test_proba: np.ndarray,
    validation_metrics: dict[str, float] | None,
    test_metrics: dict[str, float],
) -> None:
    """Log GRU parameters, metrics, preprocessors, and evaluation artifacts to MLflow."""
    import mlflow

    configure_mlflow()
    beats_lightgbm = _beats_lightgbm(test_metrics)
    run_name = f"gru-{FEATURE_SET_SPATIAL_SEQUENCE}-{season}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_role", MODEL_ROLE)
        mlflow.set_tag("beats_lightgbm", str(beats_lightgbm).lower())
        mlflow.set_tag("sequence_length", str(SEQUENCE_LENGTH))

        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_param("feature_set", FEATURE_SET_SPATIAL_SEQUENCE)
        mlflow.log_param("season", season)
        mlflow.log_param("tabular_feature_count", len(tabular_feature_columns))
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("event_feature_count", len(EVENT_FEATURE_COLUMNS))
        mlflow.log_param("train_rows", train_rows)
        mlflow.log_param("validation_rows", validation_rows)
        mlflow.log_param("test_rows", test_rows)
        mlflow.log_param("gru_hidden_size", config.gru_hidden_size)
        mlflow.log_param("tabular_embed_dim", config.tabular_embed_dim)
        mlflow.log_param("head_dims", list(config.head_dims))
        mlflow.log_param("dropout", config.dropout)
        mlflow.log_param("learning_rate", config.learning_rate)
        mlflow.log_param("weight_decay", config.weight_decay)
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("epochs", config.epochs)
        mlflow.log_param("patience", config.patience)
        mlflow.log_param("lightgbm_candidate_test_log_loss", LIGHTGBM_CANDIDATE_TEST_LOG_LOSS)

        if validation_metrics is not None:
            for key, value in validation_metrics.items():
                mlflow.log_metric(f"validation_{key}", value)
        for key, value in test_metrics.items():
            mlflow.log_metric(f"test_{key}", value)
        mlflow.log_metric("beats_lightgbm", float(beats_lightgbm))

        figures_dir = DEFAULT_FIGURES_DIR
        figures_dir.mkdir(parents=True, exist_ok=True)

        save_gru_training_curve(history, figures_dir / "gru_training_curve.png")
        save_calibration_curve(
            y_test,
            test_proba,
            figures_dir / "gru_calibration_curve.png",
            model_label="GRU (tabular + sequence)",
            title=f"GRU calibration curve (test, {season})",
        )
        save_probability_distribution(
            test_proba,
            figures_dir / "gru_probability_distribution.png",
            title=f"GRU predicted probability distribution (test, {season})",
        )

        model_config = {
            "model_type": MODEL_TYPE,
            "feature_set": FEATURE_SET_SPATIAL_SEQUENCE,
            "model_role": MODEL_ROLE,
            "tabular_feature_count": len(tabular_feature_columns),
            "sequence_length": SEQUENCE_LENGTH,
            "event_feature_count": len(EVENT_FEATURE_COLUMNS),
            "training": asdict(config),
            "architecture": {
                "gru_hidden_size": config.gru_hidden_size,
                "tabular_embed_dim": config.tabular_embed_dim,
                "head_dims": list(config.head_dims),
                "dropout": config.dropout,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)

            training_curve_path = save_gru_training_curve(
                history,
                artifact_dir / "gru_training_curve.png",
            )
            calibration_path = save_calibration_curve(
                y_test,
                test_proba,
                artifact_dir / "gru_calibration_curve.png",
                model_label="GRU (tabular + sequence)",
                title=f"GRU calibration curve (test, {season})",
            )
            distribution_path = save_probability_distribution(
                test_proba,
                artifact_dir / "gru_probability_distribution.png",
                title=f"GRU predicted probability distribution (test, {season})",
            )

            feature_columns_path = artifact_dir / "feature_columns.json"
            feature_columns_path.write_text(
                json.dumps(
                    {
                        "target": TARGET_COLUMN,
                        "feature_set": FEATURE_SET_SPATIAL,
                        "features": tabular_feature_columns,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            event_feature_columns_path = artifact_dir / "event_feature_columns.json"
            event_feature_columns_path.write_text(
                json.dumps(
                    {
                        "sequence_length": SEQUENCE_LENGTH,
                        "features": list(EVENT_FEATURE_COLUMNS),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            model_config_path = artifact_dir / "model_config.json"
            model_config_path.write_text(
                json.dumps(model_config, indent=2),
                encoding="utf-8",
            )

            state_dict_path = artifact_dir / "gru_state_dict.pt"
            torch.save(model.state_dict(), state_dict_path)

            tabular_preprocessor_path = artifact_dir / "tabular_preprocessor.joblib"
            joblib.dump(bundle.tabular_preprocessor, tabular_preprocessor_path)

            sequence_preprocessor_path = artifact_dir / "sequence_preprocessor.joblib"
            joblib.dump(bundle.sequence_preprocessor, sequence_preprocessor_path)

            for artifact_path in (
                training_curve_path,
                calibration_path,
                distribution_path,
                feature_columns_path,
                event_feature_columns_path,
                model_config_path,
                state_dict_path,
                tabular_preprocessor_path,
                sequence_preprocessor_path,
            ):
                mlflow.log_artifact(str(artifact_path))


def run_default(
    season: str,
    *,
    processed_dir: Path | None = None,
    sequence_result: SequenceBuildResult | None = None,
    config: GRUTrainConfig | None = None,
    inner_train_fraction: float = INNER_TRAIN_FRACTION,
    device: str | None = None,
    log_mlflow: bool = True,
    engine=None,
) -> dict[str, float]:
    """Train GRU with inner validation for early stopping; evaluate on held-out test."""
    train_config = config or GRUTrainConfig()
    torch_device = resolve_device(device)
    tabular_feature_columns = mlp_feature_columns(FEATURE_SET_SPATIAL)
    set_seed()

    print(f"Season: {season}")
    print(f"Model: {MODEL_TYPE}")
    print(f"Feature set: {FEATURE_SET_SPATIAL_SEQUENCE}")
    print(f"Tabular feature count: {len(tabular_feature_columns)}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Event feature count: {len(EVENT_FEATURE_COLUMNS)}")

    train, test = load_train_test_parquet(season, processed_dir=processed_dir)
    inner_train, validation, _split_meta = split_inner_train_validation(
        train,
        inner_train_fraction=inner_train_fraction,
    )

    x_inner, y_inner, inner_shot_ids = split_spatial_sequence_data(inner_train)
    x_val, y_val, val_shot_ids = split_spatial_sequence_data(validation)
    x_test, y_test, test_shot_ids = split_spatial_sequence_data(test)

    print(f"Inner train rows: {len(x_inner):,}")
    print(f"Validation rows: {len(x_val):,}")
    print(f"Test rows: {len(x_test):,}")

    print("Building/loading play-by-play sequences...")
    sequences = sequence_result or load_or_build_sequences(season, engine=engine)
    if not sequences.leakage_check_passed:
        raise RuntimeError("Sequence leakage check failed: prior action_number >= shot game_event_id")

    print("Building shot_id → sequence map...")
    sequence_map = build_shot_id_sequence_map(sequences.shot_ids, sequences.sequences)

    print("Verifying sequence coverage...")
    verify_split_shot_sequence_coverage(inner_shot_ids, sequence_map, split_name="inner_train")
    verify_split_shot_sequence_coverage(val_shot_ids, sequence_map, split_name="validation")
    verify_split_shot_sequence_coverage(test_shot_ids, sequence_map, split_name="test")

    inner_sequences = align_sequences_to_shot_ids(inner_shot_ids, sequence_map)
    val_sequences = align_sequences_to_shot_ids(val_shot_ids, sequence_map)
    test_sequences = align_sequences_to_shot_ids(test_shot_ids, sequence_map)

    print(f"Aligned inner train sequences: {len(inner_sequences):,}")
    print(f"Aligned validation sequences: {len(val_sequences):,}")
    print(f"Aligned test sequences: {len(test_sequences):,}")
    print(f"Sequence tensor shape: {inner_sequences.shape}")

    print("Preparing GRU dataloaders...")
    bundle = prepare_sequence_dataloaders(
        x_inner,
        y_inner,
        inner_sequences,
        feature_columns=tabular_feature_columns,
        x_val=x_val,
        y_val=y_val,
        val_sequences=val_sequences,
        x_test=x_test,
        y_test=y_test,
        test_sequences=test_sequences,
        batch_size=train_config.batch_size,
    )

    print("Starting GRU training...")
    fit_result = fit_gru(bundle, train_config, device=torch_device)
    model = fit_result.model

    print(f"Best validation log loss: {fit_result.best_validation_log_loss:.4f}")

    y_val_true, val_proba = predict_loader(model, bundle.validation, torch_device)
    validation_metrics = evaluate_classification_metrics(y_val_true, val_proba)

    y_test_true, test_proba = predict_loader(model, bundle.test, torch_device)
    test_metrics = evaluate_classification_metrics(y_test_true, test_proba)
    print_test_metrics(test_metrics)

    if log_mlflow:
        log_gru_mlflow_run(
            season=season,
            config=train_config,
            model=model,
            bundle=bundle,
            tabular_feature_columns=tabular_feature_columns,
            train_rows=len(x_inner),
            validation_rows=len(x_val),
            test_rows=len(x_test),
            history=fit_result.history,
            y_test=y_test_true,
            test_proba=test_proba,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
        )

    return test_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PyTorch GRU on spatial tabular + prior play-by-play sequences.",
    )
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season label (e.g. 2024-25)")
    parser.add_argument(
        "--mode",
        choices=("default",),
        default="default",
        help="default: train GRU with inner validation and test evaluation",
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
    parser.add_argument("--gru-hidden-size", type=int, default=DEFAULT_GRU_HIDDEN_SIZE)
    parser.add_argument("--tabular-embed-dim", type=int, default=DEFAULT_TABULAR_EMBED_DIM)
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

    config = GRUTrainConfig(
        gru_hidden_size=args.gru_hidden_size,
        tabular_embed_dim=args.tabular_embed_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )
    run_default(
        args.season,
        processed_dir=args.processed_dir,
        config=config,
        inner_train_fraction=args.inner_train_fraction,
        device=args.device,
        log_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
