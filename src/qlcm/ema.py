from __future__ import annotations

import copy

import torch


class EMATarget:
    """Exponential moving average target network for consistency
    distillation (Song et al. 2023, consistency models), used as the target
    side of the consistency loss instead of the online student under
    `no_grad`. v1 used the online student as its own target, which is a
    documented deviation from the original formulation
    (v1/paper/main.tex, Discussion, limitation 5); v2 tests the real EMA
    target as a deliberate methodology fix, validated on MNIST before any
    decision is made about whether it resolves Fashion-MNIST's mode
    collapse (docs/research_design.md).
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for param in self.shadow.parameters():
            param.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_param, param in zip(self.shadow.parameters(), model.parameters()):
            ema_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        for ema_buffer, buffer in zip(self.shadow.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)

    def to(self, device):
        self.shadow.to(device)
        return self
