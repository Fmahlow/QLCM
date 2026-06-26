from __future__ import annotations

import torch
from torch import nn

from src.models import SinusoidalTimeEmbedding


class LatentTeacherScoreNet(nn.Module):
    """Classical latent diffusion teacher predicting Gaussian noise in latent space."""

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
        features = torch.cat([z_t, self.time(t)], dim=1)
        return self.net(features)


class LatentNoiseSchedule:
    def __init__(
        self,
        timesteps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str = "cpu",
    ):
        self.timesteps = timesteps
        self.device = torch.device(device)
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size: int, min_t: int = 1) -> torch.Tensor:
        return torch.randint(min_t, self.timesteps, (batch_size,), device=self.device)

    def sample_pairs(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.sample_timesteps(batch_size, min_t=1)
        s = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        s = torch.minimum(s, t - 1)
        return t, s

    def add_noise(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(z0)
        alpha_bar = self.alpha_bars[t].view(-1, 1)
        z_t = torch.sqrt(alpha_bar) * z0 + torch.sqrt(1.0 - alpha_bar) * noise
        return z_t, noise
