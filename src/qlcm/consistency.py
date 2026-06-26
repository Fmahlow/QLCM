from __future__ import annotations

import math

import torch
from torch import nn

from src.models import QuantumScoreHead, SinusoidalTimeEmbedding


class ClassicalConsistencyModel(nn.Module):
    """Classical parameter-matched consistency student."""

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 256, time_dim: int = 64):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, self.time(t)], dim=1))


class QuantumLatentConsistencyModel(nn.Module):
    """Hybrid quantum-classical latent consistency student.

    placement:
      - head: quantum features are the main consistency bottleneck.
      - residual: classical prediction plus quantum residual correction.
      - time: quantum features condition the time embedding path.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        time_dim: int = 64,
        n_qubits: int = 4,
        n_layers: int = 2,
        placement: str = "residual",
    ):
        super().__init__()
        if placement not in {"head", "residual", "time"}:
            raise ValueError(f"Unknown QLCM placement: {placement}")
        self.latent_dim = latent_dim
        self.placement = placement
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.quantum_encoder = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * n_qubits),
            nn.Tanh(),
        )
        self.quantum = QuantumScoreHead(n_qubits=n_qubits, n_layers=n_layers)
        self.classical_backbone = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.quantum_head = nn.Sequential(
            nn.Linear(n_qubits + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.time_modulator = nn.Sequential(
            nn.Linear(n_qubits + time_dim, time_dim),
            nn.SiLU(),
        )

    def quantum_features(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time(t)
        q_inputs = math.pi * self.quantum_encoder(torch.cat([z_t, time_emb], dim=1))
        return self.quantum(q_inputs)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time(t)
        q_features = self.quantum_features(z_t, t)
        if self.placement == "head":
            return self.quantum_head(torch.cat([q_features, time_emb], dim=1))
        if self.placement == "residual":
            base = self.classical_backbone(torch.cat([z_t, time_emb], dim=1))
            residual = self.quantum_head(torch.cat([q_features, time_emb], dim=1))
            return base + residual
        modulated_time = self.time_modulator(torch.cat([q_features, time_emb], dim=1))
        return self.classical_backbone(torch.cat([z_t, modulated_time], dim=1))
