"""PyTorch datasets, preprocessing, and DataLoaders for shot-make MLP training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

IMPUTER_STRATEGY = "median"


class FeaturePreprocessor:
    """Median imputation and standard scaling fitted on training features only."""

    def __init__(self, feature_columns: list[str]) -> None:
        self._feature_columns = feature_columns
        self._imputer = SimpleImputer(strategy=IMPUTER_STRATEGY)
        self._scaler = StandardScaler()
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, x_train: pd.DataFrame | np.ndarray) -> FeaturePreprocessor:
        values = self._to_numpy_features(x_train)
        imputed = self._imputer.fit_transform(values)
        self._scaler.fit(imputed)
        self._fitted = True
        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeaturePreprocessor must be fit before transform.")
        values = self._to_numpy_features(x)
        imputed = self._imputer.transform(values)
        return self._scaler.transform(imputed).astype(np.float32)

    def fit_transform(self, x_train: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.fit(x_train).transform(x_train)

    def _to_numpy_features(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            return x[self._feature_columns].to_numpy(dtype=np.float64)
        array = np.asarray(x, dtype=np.float64)
        if array.ndim == 1:
            if len(array) != len(self._feature_columns):
                raise ValueError(
                    f"Expected {len(self._feature_columns)} features, got {len(array)}."
                )
            return array.reshape(1, -1)
        if array.shape[1] != len(self._feature_columns):
            raise ValueError(
                f"Expected {len(self._feature_columns)} features, got {array.shape[1]}."
            )
        return array


class ShotMakeDataset(Dataset):
    """Feature/target pairs for binary shot-make classification."""

    def __init__(self, features: np.ndarray, targets: pd.Series | np.ndarray) -> None:
        self._features = torch.as_tensor(features, dtype=torch.float32)
        self._targets = torch.as_tensor(np.asarray(targets, dtype=np.float32))

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._features[index], self._targets[index]


@dataclass(frozen=True)
class DataLoaderBundle:
    """Train/validation/test loaders plus the fitted preprocessor."""

    train: DataLoader
    validation: DataLoader | None
    test: DataLoader | None
    preprocessor: FeaturePreprocessor
    feature_count: int


def build_shot_make_dataloader(
    features: np.ndarray,
    targets: pd.Series | np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = ShotMakeDataset(features, targets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def prepare_dataloaders(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    feature_columns: list[str],
    x_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    x_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    batch_size: int = 1024,
    num_workers: int = 0,
) -> DataLoaderBundle:
    """Fit preprocessor on train, transform splits, and build DataLoaders."""
    preprocessor = FeaturePreprocessor(feature_columns).fit(x_train)

    train_loader = build_shot_make_dataloader(
        preprocessor.transform(x_train),
        y_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = None
    if x_val is not None and y_val is not None:
        validation_loader = build_shot_make_dataloader(
            preprocessor.transform(x_val),
            y_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    test_loader = None
    if x_test is not None and y_test is not None:
        test_loader = build_shot_make_dataloader(
            preprocessor.transform(x_test),
            y_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return DataLoaderBundle(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        preprocessor=preprocessor,
        feature_count=len(feature_columns),
    )
