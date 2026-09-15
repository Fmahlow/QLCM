from __future__ import annotations

import torch


class DiffusionSchedule:
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

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
        return xt, noise


@torch.no_grad()
def sample_ddpm(model, schedule: DiffusionSchedule, shape, device: str = "cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    for step in reversed(range(schedule.timesteps)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        predicted_noise = model(x, t)
        beta = schedule.betas[step]
        alpha = schedule.alphas[step]
        alpha_bar = schedule.alpha_bars[step]
        noise = torch.randn_like(x) if step > 0 else torch.zeros_like(x)
        x = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise) / torch.sqrt(alpha)
        x = x + torch.sqrt(beta) * noise
    return x.clamp(-1.0, 1.0)
