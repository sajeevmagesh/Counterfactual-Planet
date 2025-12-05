"""Multimodal fusion model for CO₂ forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class FusionConfig:
    numeric_dim: int
    policy_dim: int
    time_series_dim: int
    sequence_length: int
    hidden_dim: int = 256
    lstm_layers: int = 2
    dropout: float = 0.3
    modality_dropout: float = 0.2
    num_attention_heads: int = 4


class TemporalEncoder(nn.Module):
    """Encode time-series tensors via LSTM."""

    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class PolicyEncoder(nn.Module):
    """Project policy embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalFusionModel(nn.Module):
    """Fusion network with cross-modal attention and gating."""

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        self.numeric_net = nn.Sequential(
            nn.Linear(config.numeric_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
        )
        self.temporal_encoder = TemporalEncoder(
            input_dim=config.time_series_dim,
            hidden_dim=config.hidden_dim,
            layers=config.lstm_layers,
            dropout=config.dropout,
        )
        self.policy_encoder = PolicyEncoder(config.policy_dim, config.hidden_dim, config.dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )
        self.gate = nn.Linear(config.hidden_dim, config.hidden_dim)
        fusion_dim = config.hidden_dim + config.hidden_dim // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Linear(config.hidden_dim, 1)
        self.regressor = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        numeric_x: torch.Tensor,
        time_x: torch.Tensor,
        policy_x: torch.Tensor,
        modality_dropout: bool = True,
    ) -> dict:
        numeric_repr = self.numeric_net(numeric_x)
        temporal_repr = self.temporal_encoder(time_x)
        policy_repr = self.policy_encoder(policy_x)

        if self.training and modality_dropout and self.config.modality_dropout > 0:
            mask = torch.rand(policy_repr.size(0), 1, device=policy_repr.device)
            keep = (mask >= self.config.modality_dropout).float()
            policy_repr = policy_repr * keep

        query = temporal_repr.unsqueeze(1)
        key = policy_repr.unsqueeze(1)
        value = policy_repr.unsqueeze(1)
        attn_output, attn_weights = self.attention(query, key, value)
        attn_output = attn_output.squeeze(1)

        gate = torch.sigmoid(self.gate(policy_repr))
        gated_policy = gate * policy_repr
        fused = torch.cat([numeric_repr, attn_output + gated_policy], dim=1)
        fused = self.fusion(fused)
        logits = self.classifier(fused).squeeze(1)
        regression = self.regressor(fused).squeeze(1)
        probs = torch.sigmoid(logits)
        return {
            "logits": logits,
            "prob": probs,
            "regression": regression,
            "attention": attn_weights.mean(dim=1),
        }


def load_config_from_metadata(metadata: dict, policy_dim: Optional[int] = None) -> FusionConfig:
    """Construct config from metadata file."""

    numeric_dim = len(metadata["numeric_columns"])
    time_dim = len(metadata["time_series_columns"])
    policy_dim = policy_dim if policy_dim is not None else len(metadata["policy_columns"])
    sequence_length = metadata["sequence_length"]
    return FusionConfig(
        numeric_dim=numeric_dim,
        policy_dim=policy_dim,
        time_series_dim=time_dim,
        sequence_length=sequence_length,
    )
