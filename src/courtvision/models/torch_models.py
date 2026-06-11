"""PyTorch model definitions for CourtVision shot-make prediction."""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_HIDDEN_DIMS: tuple[int, ...] = (128, 64, 32)
DEFAULT_DROPOUT = 0.2


class ShotMakeMLP(nn.Module):
    """Feed-forward MLP that outputs logits for binary shot-make classification."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self._network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._network(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def build_shot_make_mlp(
    input_dim: int,
    *,
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    dropout: float = DEFAULT_DROPOUT,
) -> ShotMakeMLP:
    """Construct an MLP with project defaults."""
    return ShotMakeMLP(
        input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )


DEFAULT_GRU_HIDDEN_SIZE = 64
DEFAULT_TABULAR_EMBED_DIM = 64
DEFAULT_GRU_HEAD_DIMS: tuple[int, ...] = (64, 32)


class ShotMakeGRU(nn.Module):
    """Tabular/spatial branch + GRU sequence branch for shot-make classification."""

    def __init__(
        self,
        tabular_dim: int,
        event_feature_dim: int,
        *,
        gru_hidden_size: int = DEFAULT_GRU_HIDDEN_SIZE,
        tabular_embed_dim: int = DEFAULT_TABULAR_EMBED_DIM,
        head_dims: tuple[int, ...] = DEFAULT_GRU_HEAD_DIMS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if tabular_dim < 1:
            raise ValueError(f"tabular_dim must be positive, got {tabular_dim}")
        if event_feature_dim < 1:
            raise ValueError(f"event_feature_dim must be positive, got {event_feature_dim}")
        if not head_dims:
            raise ValueError("head_dims must contain at least one layer size.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self._tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, tabular_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self._gru = nn.GRU(
            input_size=event_feature_dim,
            hidden_size=gru_hidden_size,
            batch_first=True,
        )

        combined_dim = tabular_embed_dim + gru_hidden_size
        head_layers: list[nn.Module] = []
        in_dim = combined_dim
        for hidden_dim in head_dims:
            head_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        head_layers.append(nn.Linear(in_dim, 1))
        self._head = nn.Sequential(*head_layers)

    def forward(self, tabular_features: torch.Tensor, sequence_features: torch.Tensor) -> torch.Tensor:
        tabular_embedding = self._tabular_branch(tabular_features)
        _sequence_output, hidden_state = self._gru(sequence_features)
        gru_embedding = hidden_state.squeeze(0)
        combined = torch.cat([tabular_embedding, gru_embedding], dim=1)
        return self._head(combined).squeeze(-1)

    @torch.no_grad()
    def predict_proba(
        self,
        tabular_features: torch.Tensor,
        sequence_features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(tabular_features, sequence_features))


def build_shot_make_gru(
    tabular_dim: int,
    event_feature_dim: int,
    *,
    gru_hidden_size: int = DEFAULT_GRU_HIDDEN_SIZE,
    tabular_embed_dim: int = DEFAULT_TABULAR_EMBED_DIM,
    head_dims: tuple[int, ...] = DEFAULT_GRU_HEAD_DIMS,
    dropout: float = DEFAULT_DROPOUT,
) -> ShotMakeGRU:
    """Construct a tabular + GRU model with project defaults."""
    return ShotMakeGRU(
        tabular_dim,
        event_feature_dim,
        gru_hidden_size=gru_hidden_size,
        tabular_embed_dim=tabular_embed_dim,
        head_dims=head_dims,
        dropout=dropout,
    )
