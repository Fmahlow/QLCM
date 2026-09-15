from __future__ import annotations

import torch
from torch.utils.data import Dataset


class BarsAndStripes(Dataset):
    """Small 8x8 synthetic image dataset for smoke tests."""

    def __init__(self, n_samples: int = 1024, size: int = 8, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        images = []
        labels = []
        for _ in range(n_samples):
            is_bar = bool(torch.randint(0, 2, (), generator=generator).item())
            pattern = torch.zeros(size, size)
            active = torch.rand(size, generator=generator) > 0.5
            if not active.any():
                active[torch.randint(0, size, (), generator=generator)] = True
            if is_bar:
                pattern[:, active] = 1.0
                labels.append(0)
            else:
                pattern[active, :] = 1.0
                labels.append(1)
            pattern = pattern * 2.0 - 1.0
            images.append(pattern.unsqueeze(0))
        self.images = torch.stack(images).float()
        self.labels = torch.tensor(labels).long()

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]
