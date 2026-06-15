"""PyTorch datasets and DataLoaders for tabular + sequence GRU training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from courtvision.models.sequence_features import EVENT_FEATURE_COLUMNS, SEQUENCE_LENGTH
from courtvision.models.torch_data import FeaturePreprocessor

FEATURE_SET_SPATIAL_SEQUENCE = "spatial_sequence"


class SequencePreprocessor:
    """Standardize sequence event features fitted on inner-train sequences only."""

    def __init__(self, event_feature_count: int) -> None:
        self._event_feature_count = event_feature_count
        self._scaler = StandardScaler()
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, sequences: np.ndarray) -> SequencePreprocessor:
        if sequences.ndim != 3:
            raise ValueError(f"Expected sequences (n, seq, features), got shape {sequences.shape}")
        if sequences.shape[2] != self._event_feature_count:
            raise ValueError(
                f"Expected {self._event_feature_count} event features, got {sequences.shape[2]}"
            )
        flat = sequences.reshape(-1, self._event_feature_count)
        self._scaler.fit(flat)
        self._fitted = True
        return self

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SequencePreprocessor must be fit before transform.")
        if sequences.ndim != 3:
            raise ValueError(f"Expected sequences (n, seq, features), got shape {sequences.shape}")
        n_samples, seq_len, feature_count = sequences.shape
        flat = sequences.reshape(-1, feature_count)
        scaled = self._scaler.transform(flat).astype(np.float32)
        return scaled.reshape(n_samples, seq_len, feature_count)

    def fit_transform(self, sequences: np.ndarray) -> np.ndarray:
        return self.fit(sequences).transform(sequences)


class ShotMakeSequenceDataset(Dataset):
    """Tabular + prior-event sequence pairs for binary shot-make classification."""

    def __init__(
        self,
        tabular_features: np.ndarray,
        sequence_features: np.ndarray,
        targets: pd.Series | np.ndarray,
    ) -> None:
        if len(tabular_features) != len(sequence_features):
            raise ValueError("Tabular and sequence rows must align.")
        self._tabular = torch.as_tensor(tabular_features, dtype=torch.float32)
        self._sequences = torch.as_tensor(sequence_features, dtype=torch.float32)
        self._targets = torch.as_tensor(np.asarray(targets, dtype=np.float32))

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._tabular[index], self._sequences[index], self._targets[index]


@dataclass(frozen=True)
class SequenceDataLoaderBundle:
    """Train/validation/test loaders plus fitted tabular and sequence preprocessors."""

    train: DataLoader
    validation: DataLoader | None
    test: DataLoader | None
    tabular_preprocessor: FeaturePreprocessor
    sequence_preprocessor: SequencePreprocessor
    tabular_feature_count: int
    sequence_length: int
    event_feature_count: int


def build_shot_id_sequence_map(
    shot_ids: np.ndarray,
    sequences: np.ndarray,
) -> dict[int, np.ndarray]:
    """Map each shot_id to its (sequence_length, event_feature_count) tensor."""
    if len(shot_ids) != len(sequences):
        raise ValueError(
            f"shot_ids length ({len(shot_ids)}) must match sequences length ({len(sequences)})"
        )

    unique_ids, counts = np.unique(shot_ids, return_counts=True)
    duplicate_ids = unique_ids[counts != 1]
    if len(duplicate_ids) > 0:
        sample = ", ".join(str(int(value)) for value in duplicate_ids[:5])
        raise ValueError(f"Sequence builder produced duplicate shot_ids, e.g. {sample}")

    return {int(shot_id): sequences[index] for index, shot_id in enumerate(shot_ids)}


def align_sequences_to_shot_ids(
    shot_ids: pd.Series | np.ndarray,
    sequence_map: dict[int, np.ndarray],
) -> np.ndarray:
    """Return sequences ordered to match ``shot_ids`` without assuming parquet row order."""
    shot_ids_array = np.asarray(shot_ids, dtype=np.int64)
    missing = [int(shot_id) for shot_id in shot_ids_array if int(shot_id) not in sequence_map]
    if missing:
        sample = ", ".join(str(value) for value in missing[:5])
        raise KeyError(f"Missing sequences for {len(missing)} shot_ids, e.g. {sample}")

    return np.stack([sequence_map[int(shot_id)] for shot_id in shot_ids_array], axis=0)


def verify_split_shot_sequence_coverage(
    shot_ids: pd.Series | np.ndarray,
    sequence_map: dict[int, np.ndarray],
    *,
    split_name: str,
) -> None:
    """Ensure every shot in a split has exactly one sequence in the map."""
    shot_ids_array = np.asarray(shot_ids, dtype=np.int64)
    if len(np.unique(shot_ids_array)) != len(shot_ids_array):
        raise ValueError(f"{split_name} split contains duplicate shot_ids")

    missing = [int(shot_id) for shot_id in shot_ids_array if int(shot_id) not in sequence_map]
    if missing:
        sample = ", ".join(str(value) for value in missing[:5])
        raise KeyError(
            f"{split_name} split missing sequences for {len(missing)} shot_ids, e.g. {sample}"
        )


def build_shot_make_sequence_dataloader(
    tabular_features: np.ndarray,
    sequence_features: np.ndarray,
    targets: pd.Series | np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = ShotMakeSequenceDataset(tabular_features, sequence_features, targets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def prepare_sequence_dataloaders(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    train_sequences: np.ndarray,
    *,
    feature_columns: list[str],
    x_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    val_sequences: np.ndarray | None = None,
    x_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    test_sequences: np.ndarray | None = None,
    batch_size: int = 1024,
    num_workers: int = 0,
) -> SequenceDataLoaderBundle:
    """Fit preprocessors on inner train, transform splits, and build DataLoaders."""
    event_feature_count = len(EVENT_FEATURE_COLUMNS)
    tabular_preprocessor = FeaturePreprocessor(feature_columns).fit(x_train)
    sequence_preprocessor = SequencePreprocessor(event_feature_count).fit(train_sequences)

    train_loader = build_shot_make_sequence_dataloader(
        tabular_preprocessor.transform(x_train),
        sequence_preprocessor.transform(train_sequences),
        y_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_loader = None
    if x_val is not None and y_val is not None and val_sequences is not None:
        validation_loader = build_shot_make_sequence_dataloader(
            tabular_preprocessor.transform(x_val),
            sequence_preprocessor.transform(val_sequences),
            y_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    test_loader = None
    if x_test is not None and y_test is not None and test_sequences is not None:
        test_loader = build_shot_make_sequence_dataloader(
            tabular_preprocessor.transform(x_test),
            sequence_preprocessor.transform(test_sequences),
            y_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return SequenceDataLoaderBundle(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        tabular_preprocessor=tabular_preprocessor,
        sequence_preprocessor=sequence_preprocessor,
        tabular_feature_count=len(feature_columns),
        sequence_length=SEQUENCE_LENGTH,
        event_feature_count=event_feature_count,
    )
