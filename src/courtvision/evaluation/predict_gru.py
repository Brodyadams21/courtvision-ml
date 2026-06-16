"""Load GRU model artifacts from MLflow for evaluation and inference."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from mlflow.artifacts import download_artifacts

from courtvision.data.build_features import DEFAULT_PROCESSED_FEATURES_DIR
from courtvision.data.collect import DEFAULT_SEASON
from courtvision.data.load_data import DEFAULT_ENV_PATH, PROJECT_ROOT, create_database_engine
from courtvision.evaluation.predict_shots import load_evaluation_shots
from courtvision.models.pressure_features import attach_shot_pressure_features
from courtvision.models.sequence_features import SequenceBuildResult
from courtvision.models.spatial_features import FEATURE_SET_SPATIAL, split_mlp_features_target
from courtvision.models.torch_models import ShotMakeGRU, build_shot_make_gru
from courtvision.models.torch_sequence_data import (
    align_sequences_to_shot_ids,
    build_shot_id_sequence_map,
    verify_split_shot_sequence_coverage,
)
from courtvision.models.train_gru import load_or_build_sequences

GRU_STATE_DICT_FILENAME = "gru_state_dict.pt"
TABULAR_PREPROCESSOR_FILENAME = "tabular_preprocessor.joblib"
SEQUENCE_PREPROCESSOR_FILENAME = "sequence_preprocessor.joblib"
FEATURE_COLUMNS_FILENAME = "feature_columns.json"
EVENT_FEATURE_COLUMNS_FILENAME = "event_feature_columns.json"
MODEL_CONFIG_FILENAME = "model_config.json"

REQUIRED_GRU_ARTIFACT_FILENAMES: tuple[str, ...] = (
    GRU_STATE_DICT_FILENAME,
    TABULAR_PREPROCESSOR_FILENAME,
    SEQUENCE_PREPROCESSOR_FILENAME,
    FEATURE_COLUMNS_FILENAME,
    EVENT_FEATURE_COLUMNS_FILENAME,
    MODEL_CONFIG_FILENAME,
)

DEFAULT_GRU_ARTIFACT_CACHE_DIR = PROJECT_ROOT / "model_artifacts" / "gru"
DEFAULT_INFERENCE_BATCH_SIZE = 1024


@dataclass(frozen=True)
class GRUArtifactPaths:
    """Local paths to the trained GRU bundle for one MLflow run."""

    run_id: str
    artifact_root: Path
    state_dict: Path
    tabular_preprocessor: Path
    sequence_preprocessor: Path
    feature_columns: Path
    event_feature_columns: Path
    model_config: Path

    def as_filename_map(self) -> dict[str, Path]:
        """Map artifact filenames to resolved local paths."""
        return {
            GRU_STATE_DICT_FILENAME: self.state_dict,
            TABULAR_PREPROCESSOR_FILENAME: self.tabular_preprocessor,
            SEQUENCE_PREPROCESSOR_FILENAME: self.sequence_preprocessor,
            FEATURE_COLUMNS_FILENAME: self.feature_columns,
            EVENT_FEATURE_COLUMNS_FILENAME: self.event_feature_columns,
            MODEL_CONFIG_FILENAME: self.model_config,
        }


@dataclass(frozen=True)
class LoadedGRUModel:
    """Rehydrated GRU model bundle for inference."""

    model: ShotMakeGRU
    tabular_preprocessor: object
    sequence_preprocessor: object
    feature_columns: list[str]
    event_feature_columns: list[str]
    model_config: dict[str, object]
    device: torch.device


def resolve_device(device: str | None = None) -> torch.device:
    """Return the torch device used for GRU inference."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json_file(path: Path) -> dict[str, object]:
    """Load a JSON artifact file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _load_string_list(payload: dict[str, object], key: str, *, path: Path) -> list[str]:
    values = payload.get(key)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Expected {key} to be a list of strings in {path}")
    return values


def load_gru_model_bundle(
    paths: GRUArtifactPaths,
    *,
    device: str | None = None,
) -> LoadedGRUModel:
    """Load preprocessors, config, and PyTorch weights for a trained GRU run."""
    torch_device = resolve_device(device)

    model_config = load_json_file(paths.model_config)
    feature_payload = load_json_file(paths.feature_columns)
    event_payload = load_json_file(paths.event_feature_columns)
    feature_columns = _load_string_list(feature_payload, "features", path=paths.feature_columns)
    event_feature_columns = _load_string_list(
        event_payload,
        "features",
        path=paths.event_feature_columns,
    )

    tabular_preprocessor = joblib.load(paths.tabular_preprocessor)
    sequence_preprocessor = joblib.load(paths.sequence_preprocessor)

    architecture = model_config.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("model_config.json missing architecture block")

    tabular_feature_count = model_config.get("tabular_feature_count")
    event_feature_count = model_config.get("event_feature_count")
    if not isinstance(tabular_feature_count, int) or not isinstance(event_feature_count, int):
        raise ValueError("model_config.json missing tabular_feature_count/event_feature_count")

    head_dims_raw = architecture.get("head_dims")
    if not isinstance(head_dims_raw, list):
        raise ValueError("model_config architecture.head_dims must be a list of integers")
    if not all(isinstance(dim, int) for dim in head_dims_raw):
        raise ValueError("model_config architecture.head_dims must be a list of integers")

    model = build_shot_make_gru(
        tabular_feature_count,
        event_feature_count,
        gru_hidden_size=int(architecture["gru_hidden_size"]),
        tabular_embed_dim=int(architecture["tabular_embed_dim"]),
        head_dims=tuple(head_dims_raw),
        dropout=float(architecture["dropout"]),
    )
    state_dict = torch.load(paths.state_dict, map_location=torch_device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(torch_device)
    model.eval()

    return LoadedGRUModel(
        model=model,
        tabular_preprocessor=tabular_preprocessor,
        sequence_preprocessor=sequence_preprocessor,
        feature_columns=feature_columns,
        event_feature_columns=event_feature_columns,
        model_config=model_config,
        device=torch_device,
    )


def _inference_batch_size(bundle: LoadedGRUModel, batch_size: int | None = None) -> int:
    if batch_size is not None:
        return batch_size
    training = bundle.model_config.get("training")
    if isinstance(training, dict):
        configured = training.get("batch_size")
        if isinstance(configured, int) and configured > 0:
            return configured
    return DEFAULT_INFERENCE_BATCH_SIZE


@torch.no_grad()
def predict_gru_make_probabilities(
    shots: pd.DataFrame,
    bundle: LoadedGRUModel,
    sequences: np.ndarray,
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    """Return predicted make probabilities for aligned tabular rows and sequences."""
    if len(shots) != len(sequences):
        raise ValueError(
            "Shot count and sequence count must match: "
            f"{len(shots):,} shots vs {len(sequences):,} sequences"
        )

    missing_columns = [
        column for column in bundle.feature_columns if column not in shots.columns
    ]
    if missing_columns:
        raise KeyError(f"Shots missing required tabular feature columns: {missing_columns}")

    tabular_values = bundle.tabular_preprocessor.transform(shots[bundle.feature_columns])
    sequence_values = bundle.sequence_preprocessor.transform(sequences)

    inference_batch_size = _inference_batch_size(bundle, batch_size=batch_size)
    probabilities: list[np.ndarray] = []
    model = bundle.model
    device = bundle.device

    for start in range(0, len(shots), inference_batch_size):
        end = start + inference_batch_size
        tabular_batch = torch.as_tensor(
            tabular_values[start:end],
            dtype=torch.float32,
            device=device,
        )
        sequence_batch = torch.as_tensor(
            sequence_values[start:end],
            dtype=torch.float32,
            device=device,
        )
        batch_proba = model.predict_proba(tabular_batch, sequence_batch).cpu().numpy()
        probabilities.append(batch_proba)

    return np.concatenate(probabilities)


def prepare_gru_tabular_features(
    shots: pd.DataFrame,
    sequences: np.ndarray,
) -> pd.DataFrame:
    """Build GRU tabular features from spatial shot columns and aligned sequences."""
    x_spatial, _target = split_mlp_features_target(shots, FEATURE_SET_SPATIAL)
    return attach_shot_pressure_features(x_spatial, shots, sequences)


def align_evaluation_sequences(
    shots: pd.DataFrame,
    sequence_result: SequenceBuildResult,
) -> np.ndarray:
    """Align season play-by-play sequences to evaluation-shot row order."""
    if "shot_id" not in shots.columns:
        raise KeyError("Evaluation shots missing required column: shot_id")

    sequence_map = build_shot_id_sequence_map(
        sequence_result.shot_ids,
        sequence_result.sequences,
    )
    shot_ids = shots["shot_id"].astype(np.int64)
    verify_split_shot_sequence_coverage(shot_ids, sequence_map, split_name="evaluation")
    return align_sequences_to_shot_ids(shot_ids, sequence_map)


def score_evaluation_shots_with_gru(
    shots: pd.DataFrame,
    run_id: str,
    *,
    season: str = DEFAULT_SEASON,
    cache_root: Path | None = None,
    force_download: bool = False,
    device: str | None = None,
    batch_size: int | None = None,
    engine=None,
) -> np.ndarray:
    """Score pre-loaded evaluation shots with a trained GRU MLflow run."""
    paths = load_gru_artifacts(
        run_id,
        cache_root=cache_root,
        force_download=force_download,
    )
    bundle = load_gru_model_bundle(paths, device=device)
    return _score_loaded_gru_on_shots(
        shots,
        bundle,
        season=season,
        engine=engine,
        batch_size=batch_size,
    )


def _score_loaded_gru_on_shots(
    shots: pd.DataFrame,
    bundle: LoadedGRUModel,
    *,
    season: str,
    engine=None,
    batch_size: int | None = None,
) -> np.ndarray:
    db_engine = engine or create_database_engine(env_path=DEFAULT_ENV_PATH)
    sequence_result = load_or_build_sequences(season, engine=db_engine)
    if not sequence_result.leakage_check_passed:
        raise RuntimeError(
            "Sequence leakage check failed: prior action_number >= shot game_event_id"
        )

    sequences = align_evaluation_sequences(shots, sequence_result)
    tabular_features = prepare_gru_tabular_features(shots, sequences)
    return predict_gru_make_probabilities(
        tabular_features,
        bundle,
        sequences,
        batch_size=batch_size,
    )


def run_predict_gru_evaluation(
    run_id: str,
    *,
    season: str = DEFAULT_SEASON,
    processed_dir: Path | None = None,
    cache_root: Path | None = None,
    force_download: bool = False,
    device: str | None = None,
    batch_size: int | None = None,
    engine=None,
) -> np.ndarray:
    """Load evaluation shots, rebuild sequences, and score with a trained GRU run."""
    shots = load_evaluation_shots(season, processed_dir=processed_dir)
    print(f"Loaded evaluation shots: {len(shots):,}")

    paths = load_gru_artifacts(
        run_id,
        cache_root=cache_root,
        force_download=force_download,
    )
    print(f"Loaded GRU artifacts from run {run_id}")
    bundle = load_gru_model_bundle(paths, device=device)
    print("Loaded GRU model bundle")

    print("Building/loading play-by-play sequences...")
    probabilities = _score_loaded_gru_on_shots(
        shots,
        bundle,
        season=season,
        engine=engine,
        batch_size=batch_size,
    )
    print(f"Aligned evaluation sequences: {len(shots):,}")
    print(f"Generated GRU predictions: {len(probabilities):,}")
    print(
        "Probability range: "
        f"{float(np.min(probabilities)):.3f} to {float(np.max(probabilities)):.3f}"
    )
    return probabilities


def configure_mlflow() -> None:
    """Load ``.env`` and set MLflow tracking URI when configured."""
    load_dotenv(DEFAULT_ENV_PATH)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def gru_artifact_cache_dir(
    run_id: str,
    *,
    cache_root: Path | None = None,
) -> Path:
    """Return the local cache directory for one GRU MLflow run."""
    root = cache_root or DEFAULT_GRU_ARTIFACT_CACHE_DIR
    return root / run_id


def _resolve_downloaded_artifact_path(
    cache_dir: Path,
    filename: str,
    downloaded_path: str,
) -> Path:
    """Normalize MLflow download output to the cached artifact file path."""
    path = Path(downloaded_path)
    if path.is_file() and path.name == filename:
        return path

    direct = cache_dir / filename
    if direct.is_file():
        return direct

    matches = [candidate for candidate in cache_dir.rglob(filename) if candidate.is_file()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"MLflow run artifact not found after download: {filename} (run cache: {cache_dir})"
        )
    raise FileNotFoundError(
        f"Multiple copies of {filename} found under {cache_dir}: "
        f"{[str(match) for match in matches]}"
    )


def download_gru_artifact(
    run_id: str,
    filename: str,
    *,
    cache_dir: Path,
    force: bool = False,
) -> Path:
    """Download one GRU artifact file from MLflow into the run cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / filename
    if cached_path.is_file() and not force:
        return cached_path

    downloaded_path = download_artifacts(
        run_id=run_id,
        artifact_path=filename,
        dst_path=str(cache_dir),
    )
    return _resolve_downloaded_artifact_path(cache_dir, filename, downloaded_path)


def load_gru_artifacts(
    run_id: str,
    *,
    cache_root: Path | None = None,
    force_download: bool = False,
) -> GRUArtifactPaths:
    """Resolve or download all required GRU artifacts for an MLflow run."""
    configure_mlflow()
    cache_dir = gru_artifact_cache_dir(run_id, cache_root=cache_root)

    paths = {
        filename: download_gru_artifact(
            run_id,
            filename,
            cache_dir=cache_dir,
            force=force_download,
        )
        for filename in REQUIRED_GRU_ARTIFACT_FILENAMES
    }

    return GRUArtifactPaths(
        run_id=run_id,
        artifact_root=cache_dir,
        state_dict=paths[GRU_STATE_DICT_FILENAME],
        tabular_preprocessor=paths[TABULAR_PREPROCESSOR_FILENAME],
        sequence_preprocessor=paths[SEQUENCE_PREPROCESSOR_FILENAME],
        feature_columns=paths[FEATURE_COLUMNS_FILENAME],
        event_feature_columns=paths[EVENT_FEATURE_COLUMNS_FILENAME],
        model_config=paths[MODEL_CONFIG_FILENAME],
    )


def print_gru_artifact_checkpoint(paths: GRUArtifactPaths, *, run_id: str | None = None) -> None:
    """Print the first-load checkpoint lines for a GRU artifact bundle."""
    resolved_run_id = run_id or paths.run_id
    print(f"Loaded GRU artifacts from run {resolved_run_id}")
    for filename in REQUIRED_GRU_ARTIFACT_FILENAMES:
        print(f"Found {filename}")


def print_gru_model_checkpoint(paths: GRUArtifactPaths, *, run_id: str | None = None) -> None:
    """Print checkpoint lines after a GRU model bundle is loaded."""
    resolved_run_id = run_id or paths.run_id
    print(f"Loaded GRU artifacts from run {resolved_run_id}")
    print(f"Loaded {MODEL_CONFIG_FILENAME}")
    print(f"Loaded {FEATURE_COLUMNS_FILENAME}")
    print(f"Loaded {EVENT_FEATURE_COLUMNS_FILENAME}")
    print(f"Loaded {TABULAR_PREPROCESSOR_FILENAME}")
    print(f"Loaded {SEQUENCE_PREPROCESSOR_FILENAME}")
    print("Loaded GRU state_dict")
    print("Built ShotMakeGRU model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify GRU MLflow run artifacts.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="MLflow run ID containing GRU training artifacts",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_GRU_ARTIFACT_CACHE_DIR,
        help=(
            "Local cache root for downloaded artifacts "
            f"(default: {DEFAULT_GRU_ARTIFACT_CACHE_DIR})"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        choices=("artifacts", "model", "predict"),
        default="artifacts",
        help="artifacts: verify files; model: load bundle; predict: score evaluation shots",
    )
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        help="Season label for evaluation shots (predict checkpoint only)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_FEATURES_DIR,
        help="Directory with train/test Parquet (predict checkpoint only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size (predict checkpoint only)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for model loading (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download artifacts even if they already exist in the cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_gru_artifacts(
        args.run_id,
        cache_root=args.cache_root,
        force_download=args.force_download,
    )
    if args.checkpoint == "artifacts":
        print_gru_artifact_checkpoint(paths, run_id=args.run_id)
        return

    if args.checkpoint == "model":
        load_gru_model_bundle(paths, device=args.device)
        print_gru_model_checkpoint(paths, run_id=args.run_id)
        return

    run_predict_gru_evaluation(
        args.run_id,
        season=args.season,
        processed_dir=args.processed_dir,
        cache_root=args.cache_root,
        force_download=args.force_download,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
