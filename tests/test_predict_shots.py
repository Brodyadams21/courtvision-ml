"""Tests for shot-level prediction builder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from courtvision.evaluation.predict_shots import (
    MODEL_TYPE_GRU,
    SHOT_PREDICTION_CORE_COLUMNS,
    SHOT_PREDICTION_OPTIONAL_METADATA_COLUMNS,
    build_shot_predictions_table,
    predict_make_probabilities,
    run_predict_shots,
    save_shot_predictions_csv,
    shot_predictions_output_path,
)
from courtvision.models.common import FEATURE_COLUMNS


def _evaluation_shots_frame() -> pd.DataFrame:
    row = {
        "shot_id": 1,
        "game_id": 100,
        "player_id": 10,
        "team_id": 20,
        "shot_value": 3,
        "shot_made_flag": True,
    }
    for column in FEATURE_COLUMNS:
        row[column] = 0.0
    row["shot_value"] = 3
    return pd.DataFrame([row])


def _minimal_prediction_columns() -> tuple[str, ...]:
    return (*SHOT_PREDICTION_CORE_COLUMNS, "shot_distance")


def test_predict_make_probabilities_uses_feature_columns() -> None:
    shots = _evaluation_shots_frame()
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.25, 0.75]])

    probabilities = predict_make_probabilities(model, shots)

    pd.testing.assert_frame_equal(model.predict_proba.call_args.args[0], shots[FEATURE_COLUMNS])
    assert probabilities == pytest.approx(np.array([0.75]))


def test_build_shot_predictions_table_shapes_output() -> None:
    shots = _evaluation_shots_frame()
    predictions = build_shot_predictions_table(shots, np.array([0.40]))

    assert list(predictions.columns) == list(_minimal_prediction_columns())
    assert predictions.loc[0, "expected_shot_value"] == pytest.approx(1.20)
    assert predictions.loc[0, "actual_points"] == pytest.approx(3.0)
    assert predictions.loc[0, "points_above_expected"] == pytest.approx(1.80)


def test_save_shot_predictions_csv_writes_expected_columns(tmp_path: Path) -> None:
    predictions = build_shot_predictions_table(_evaluation_shots_frame(), np.array([0.40]))
    output_path = tmp_path / "shot_predictions.csv"

    save_shot_predictions_csv(predictions, output_path)

    loaded = pd.read_csv(output_path)
    assert list(loaded.columns) == list(_minimal_prediction_columns())


def test_build_shot_predictions_table_includes_zone_metadata_when_present() -> None:
    shots = _evaluation_shots_frame()
    shots["shot_zone_basic"] = "Right Corner 3"
    shots["shot_zone_area"] = "Right Side(R)"
    shots["shot_zone_range"] = "24+ ft."
    shots["shot_distance"] = 23.0
    shots["game_date"] = "2025-03-13"

    predictions = build_shot_predictions_table(shots, np.array([0.40]))

    assert list(predictions.columns) == list(SHOT_PREDICTION_CORE_COLUMNS) + list(
        SHOT_PREDICTION_OPTIONAL_METADATA_COLUMNS
    )
    assert predictions.loc[0, "shot_zone_basic"] == "Right Corner 3"
    assert predictions.loc[0, "shot_distance"] == pytest.approx(23.0)


@patch("courtvision.evaluation.predict_shots.load_candidate_model")
@patch("courtvision.evaluation.predict_shots.load_evaluation_shots")
def test_run_predict_shots_prints_checkpoint_lines(
    load_evaluation_shots: MagicMock,
    load_candidate_model: MagicMock,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shots = _evaluation_shots_frame()
    load_evaluation_shots.return_value = shots

    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.60, 0.40]])
    load_candidate_model.return_value = model

    output_path = tmp_path / "shot_predictions_2024-25_candidate.csv"
    run_predict_shots("2024-25", output_path=output_path)

    captured = capsys.readouterr().out
    assert "Loaded evaluation shots: 1" in captured
    assert "Generated predictions: 1" in captured
    assert captured.endswith("shot_predictions_2024-25_candidate.csv\n")
    assert output_path.is_file()


def test_shot_predictions_output_path_includes_model_type(tmp_path: Path) -> None:
    path = shot_predictions_output_path("2024-25", model_type=MODEL_TYPE_GRU, tables_dir=tmp_path)

    assert path == tmp_path / "shot_predictions_2024-25_gru.csv"


@patch("courtvision.evaluation.predict_gru.score_evaluation_shots_with_gru")
@patch("courtvision.evaluation.predict_shots.load_evaluation_shots")
def test_run_predict_shots_gru_path_prints_checkpoint_lines(
    load_evaluation_shots: MagicMock,
    score_evaluation_shots_with_gru: MagicMock,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shots = _evaluation_shots_frame()
    load_evaluation_shots.return_value = shots
    score_evaluation_shots_with_gru.return_value = np.array([0.55])

    run_id = "40fe1b8851f7423f831a77fce30b770d"
    output_path = tmp_path / "shot_predictions_2024-25_gru.csv"
    run_predict_shots(
        "2024-25",
        model_type=MODEL_TYPE_GRU,
        gru_run_id=run_id,
        output_path=output_path,
    )

    captured = capsys.readouterr().out
    assert "Loaded evaluation shots: 1" in captured
    assert f"Using GRU model from run {run_id}" in captured
    assert "Generated predictions: 1" in captured
    assert captured.endswith("shot_predictions_2024-25_gru.csv\n")
    score_evaluation_shots_with_gru.assert_called_once()
    assert output_path.is_file()
