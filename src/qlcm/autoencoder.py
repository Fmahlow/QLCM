from __future__ import annotations

import torch
from torch import nn


class MlpAutoencoder(nn.Module):
    """Small autoencoder for 8x8 smoke datasets."""

    def __init__(self, image_size: int = 8, latent_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        pixels = image_size * image_size
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pixels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pixels),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(z.shape[0], 1, self.image_size, self.image_size)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z


class ConvAutoencoder(nn.Module):
    """Compact 28x28 autoencoder for MNIST and Fashion-MNIST."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.encoder_head = nn.Linear(32 * 7 * 7, latent_dim)
        self.decoder_head = nn.Linear(latent_dim, 32 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        return self.encoder_head(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_head(z).view(z.shape[0], 32, 7, 7)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z
