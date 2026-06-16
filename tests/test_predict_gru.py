"""Tests for GRU artifact loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from courtvision.evaluation.predict_gru import (
    REQUIRED_GRU_ARTIFACT_FILENAMES,
    GRUArtifactPaths,
    download_gru_artifact,
    gru_artifact_cache_dir,
    load_gru_artifacts,
    load_gru_model_bundle,
    predict_gru_make_probabilities,
    print_gru_artifact_checkpoint,
    print_gru_model_checkpoint,
)
from courtvision.models.torch_data import FeaturePreprocessor
from courtvision.models.torch_models import build_shot_make_gru
from courtvision.models.torch_sequence_data import SequencePreprocessor


def _touch_artifacts(cache_dir: Path, run_id: str = "test-run") -> GRUArtifactPaths:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {filename: cache_dir / filename for filename in REQUIRED_GRU_ARTIFACT_FILENAMES}
    for path in paths.values():
        path.write_text("stub", encoding="utf-8")
    return GRUArtifactPaths(
        run_id=run_id,
        artifact_root=cache_dir,
        state_dict=paths["gru_state_dict.pt"],
        tabular_preprocessor=paths["tabular_preprocessor.joblib"],
        sequence_preprocessor=paths["sequence_preprocessor.joblib"],
        feature_columns=paths["feature_columns.json"],
        event_feature_columns=paths["event_feature_columns.json"],
        model_config=paths["model_config.json"],
    )


def test_gru_artifact_cache_dir_uses_run_subdirectory(tmp_path: Path) -> None:
    cache_dir = gru_artifact_cache_dir("abc123", cache_root=tmp_path)

    assert cache_dir == tmp_path / "abc123"


@patch("courtvision.evaluation.predict_gru.download_artifacts")
def test_download_gru_artifact_uses_cache_when_present(
    download_artifacts: object,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "run"
    cached = cache_dir / "gru_state_dict.pt"
    cache_dir.mkdir(parents=True)
    cached.write_text("cached", encoding="utf-8")

    path = download_gru_artifact("run", "gru_state_dict.pt", cache_dir=cache_dir)

    assert path == cached
    download_artifacts.assert_not_called()


@patch("courtvision.evaluation.predict_gru.download_artifacts")
def test_download_gru_artifact_fetches_missing_file(
    download_artifacts: object,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "run"
    cache_dir.mkdir(parents=True)
    target = cache_dir / "model_config.json"

    def _download(**_kwargs: object) -> str:
        target.write_text("{}", encoding="utf-8")
        return str(target)

    download_artifacts.side_effect = _download

    path = download_gru_artifact("run", "model_config.json", cache_dir=cache_dir)

    assert path == target
    download_artifacts.assert_called_once_with(
        run_id="run",
        artifact_path="model_config.json",
        dst_path=str(cache_dir),
    )


@patch("courtvision.evaluation.predict_gru.download_gru_artifact")
@patch("courtvision.evaluation.predict_gru.configure_mlflow")
def test_load_gru_artifacts_returns_all_required_paths(
    configure_mlflow: object,
    download_gru_artifact: object,
    tmp_path: Path,
) -> None:
    run_id = "40fe1b8851f7423f831a77fce30b770d"
    cache_dir = tmp_path / run_id
    expected = _touch_artifacts(cache_dir, run_id=run_id)
    download_gru_artifact.side_effect = lambda _run_id, filename, *, cache_dir, force: (
        cache_dir / filename
    )

    paths = load_gru_artifacts(
        run_id,
        cache_root=tmp_path,
    )

    configure_mlflow.assert_called_once()
    assert paths == expected
    assert download_gru_artifact.call_count == len(REQUIRED_GRU_ARTIFACT_FILENAMES)


def test_print_gru_artifact_checkpoint_lines(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _touch_artifacts(tmp_path / "run")

    print_gru_artifact_checkpoint(paths, run_id="40fe1b8851f7423f831a77fce30b770d")

    captured = capsys.readouterr().out
    assert captured.startswith("Loaded GRU artifacts from run 40fe1b8851f7423f831a77fce30b770d\n")
    assert captured.endswith("Found model_config.json\n")
    for filename in REQUIRED_GRU_ARTIFACT_FILENAMES:
        assert f"Found {filename}" in captured


def _write_model_bundle(cache_dir: Path, run_id: str) -> GRUArtifactPaths:
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = ["f1", "f2", "f3", "f4"]
    event_feature_columns = ["e1", "e2", "e3"]
    tabular_dim = len(feature_columns)
    event_dim = len(event_feature_columns)

    model_config = {
        "model_type": "pytorch_gru",
        "tabular_feature_count": tabular_dim,
        "event_feature_count": event_dim,
        "architecture": {
            "gru_hidden_size": 8,
            "tabular_embed_dim": 8,
            "head_dims": [8, 4],
            "dropout": 0.1,
        },
    }
    (cache_dir / "model_config.json").write_text(json.dumps(model_config), encoding="utf-8")
    (cache_dir / "feature_columns.json").write_text(
        json.dumps({"features": feature_columns}),
        encoding="utf-8",
    )
    (cache_dir / "event_feature_columns.json").write_text(
        json.dumps({"sequence_length": 5, "features": event_feature_columns}),
        encoding="utf-8",
    )

    tabular_preprocessor = FeaturePreprocessor(feature_columns)
    tabular_preprocessor.fit(np.zeros((2, tabular_dim), dtype=np.float64))
    joblib.dump(tabular_preprocessor, cache_dir / "tabular_preprocessor.joblib")

    sequence_preprocessor = SequencePreprocessor(event_dim)
    sequence_preprocessor.fit(np.zeros((2, 5, event_dim), dtype=np.float32))
    joblib.dump(sequence_preprocessor, cache_dir / "sequence_preprocessor.joblib")

    model = build_shot_make_gru(
        tabular_dim,
        event_dim,
        gru_hidden_size=8,
        tabular_embed_dim=8,
        head_dims=(8, 4),
        dropout=0.1,
    )
    torch.save(model.state_dict(), cache_dir / "gru_state_dict.pt")

    return GRUArtifactPaths(
        run_id=run_id,
        artifact_root=cache_dir,
        state_dict=cache_dir / "gru_state_dict.pt",
        tabular_preprocessor=cache_dir / "tabular_preprocessor.joblib",
        sequence_preprocessor=cache_dir / "sequence_preprocessor.joblib",
        feature_columns=cache_dir / "feature_columns.json",
        event_feature_columns=cache_dir / "event_feature_columns.json",
        model_config=cache_dir / "model_config.json",
    )


def test_load_gru_model_bundle_rehydrates_model(tmp_path: Path) -> None:
    paths = _write_model_bundle(tmp_path / "run", "run")

    bundle = load_gru_model_bundle(paths, device="cpu")

    assert bundle.feature_columns == ["f1", "f2", "f3", "f4"]
    assert bundle.event_feature_columns == ["e1", "e2", "e3"]
    assert bundle.model_config["tabular_feature_count"] == 4
    assert bundle.device.type == "cpu"
    assert not bundle.model.training


def test_print_gru_model_checkpoint_lines(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_model_bundle(tmp_path / "run", "40fe1b8851f7423f831a77fce30b770d")

    print_gru_model_checkpoint(paths, run_id="40fe1b8851f7423f831a77fce30b770d")

    captured = capsys.readouterr().out
    assert captured.startswith("Loaded GRU artifacts from run 40fe1b8851f7423f831a77fce30b770d\n")
    assert captured.endswith("Built ShotMakeGRU model\n")
    assert "Loaded model_config.json" in captured
    assert "Loaded GRU state_dict" in captured


def test_predict_gru_make_probabilities_returns_valid_probabilities(
    tmp_path: Path,
) -> None:
    paths = _write_model_bundle(tmp_path / "run", "run")
    bundle = load_gru_model_bundle(paths, device="cpu")
    shots = pd.DataFrame({column: [0.0, 1.0] for column in bundle.feature_columns})
    sequences = np.zeros(
        (2, 5, len(bundle.event_feature_columns)),
        dtype=np.float32,
    )

    probabilities = predict_gru_make_probabilities(shots, bundle, sequences)

    assert len(probabilities) == len(shots)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
