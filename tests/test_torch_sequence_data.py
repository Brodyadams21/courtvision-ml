"""Tests for tabular + sequence GRU data utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from courtvision.models.sequence_features import EVENT_FEATURE_COLUMNS, SEQUENCE_LENGTH
from courtvision.models.spatial_features import FEATURE_SET_SPATIAL, mlp_feature_columns
from courtvision.models.torch_models import ShotMakeGRU
from courtvision.models.torch_sequence_data import (
    ShotMakeSequenceDataset,
    align_sequences_to_shot_ids,
    build_shot_id_sequence_map,
    prepare_sequence_dataloaders,
    verify_split_shot_sequence_coverage,
)


def _synthetic_tabular_frame(shot_ids: list[int]) -> pd.DataFrame:
    rows = []
    for shot_id in shot_ids:
        row = {col: 0.0 for col in mlp_feature_columns(FEATURE_SET_SPATIAL)}
        row.update({"shot_id": shot_id, "shot_made_flag": True})
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_shot_id_sequence_map_rejects_duplicates() -> None:
    shot_ids = np.array([1, 1], dtype=np.int64)
    sequences = np.zeros((2, SEQUENCE_LENGTH, len(EVENT_FEATURE_COLUMNS)), dtype=np.float32)
    with pytest.raises(ValueError, match="duplicate shot_ids"):
        build_shot_id_sequence_map(shot_ids, sequences)


def test_align_sequences_to_shot_ids_preserves_requested_order() -> None:
    sequences = np.arange(435, dtype=np.float32).reshape(
        3, SEQUENCE_LENGTH, len(EVENT_FEATURE_COLUMNS)
    )
    shot_ids = np.array([10, 20, 30], dtype=np.int64)
    sequence_map = build_shot_id_sequence_map(shot_ids, sequences)

    aligned = align_sequences_to_shot_ids(np.array([30, 10]), sequence_map)

    assert np.array_equal(aligned[0], sequence_map[30])
    assert np.array_equal(aligned[1], sequence_map[10])


def test_verify_split_shot_sequence_coverage_requires_exact_match() -> None:
    sequences = np.zeros((2, SEQUENCE_LENGTH, len(EVENT_FEATURE_COLUMNS)), dtype=np.float32)
    sequence_map = build_shot_id_sequence_map(np.array([1, 2]), sequences)

    verify_split_shot_sequence_coverage(np.array([2, 1]), sequence_map, split_name="test")

    with pytest.raises(KeyError, match="missing sequences"):
        verify_split_shot_sequence_coverage(np.array([1, 3]), sequence_map, split_name="test")


def test_shot_make_sequence_dataset_returns_three_tensors() -> None:
    tabular = np.ones((2, 4), dtype=np.float32)
    sequences = np.zeros((2, SEQUENCE_LENGTH, 3), dtype=np.float32)
    targets = np.array([1.0, 0.0], dtype=np.float32)

    dataset = ShotMakeSequenceDataset(tabular, sequences, targets)
    sample_tabular, sample_sequence, sample_target = dataset[0]

    assert sample_tabular.shape == (4,)
    assert sample_sequence.shape == (SEQUENCE_LENGTH, 3)
    assert sample_target.shape == ()


def test_prepare_sequence_dataloaders_yields_tabular_sequence_target_batch() -> None:
    shot_ids = [101, 102, 103, 104]
    frame = _synthetic_tabular_frame(shot_ids)
    feature_columns = mlp_feature_columns(FEATURE_SET_SPATIAL)
    x = frame[feature_columns]
    y = frame["shot_made_flag"]
    sequences = np.arange(
        len(shot_ids) * SEQUENCE_LENGTH * len(EVENT_FEATURE_COLUMNS),
        dtype=np.float32,
    ).reshape(len(shot_ids), SEQUENCE_LENGTH, len(EVENT_FEATURE_COLUMNS))

    bundle = prepare_sequence_dataloaders(
        x.iloc[:2],
        y.iloc[:2],
        sequences[:2],
        feature_columns=feature_columns,
        x_val=x.iloc[2:3],
        y_val=y.iloc[2:3],
        val_sequences=sequences[2:3],
        batch_size=2,
    )

    tabular_batch, sequence_batch, target_batch = next(iter(bundle.train))
    assert tabular_batch.shape == (2, len(feature_columns))
    assert sequence_batch.shape == (2, SEQUENCE_LENGTH, len(EVENT_FEATURE_COLUMNS))
    assert target_batch.shape == (2,)


def test_shot_make_gru_forward_pass() -> None:
    tabular_dim = len(mlp_feature_columns(FEATURE_SET_SPATIAL))
    event_dim = len(EVENT_FEATURE_COLUMNS)
    model = ShotMakeGRU(tabular_dim, event_dim)
    tabular = torch.randn(4, tabular_dim)
    sequences = torch.randn(4, SEQUENCE_LENGTH, event_dim)
    logits = model(tabular, sequences)
    assert logits.shape == (4,)
