from __future__ import annotations

import torch


@torch.no_grad()
def sample_consistency(student, decoder, latent_dim: int, n_samples: int, steps: int, timesteps: int, device: str):
    student.eval()
    decoder.eval()
    z = torch.randn(n_samples, latent_dim, device=device)
    if steps <= 1:
        t = torch.full((n_samples,), timesteps - 1, device=device, dtype=torch.long)
        z0 = student(z, t)
        return decoder.decode(z0)

    schedule = torch.linspace(timesteps - 1, 0, steps, device=device).long()
    for t_value in schedule:
        t = torch.full((n_samples,), int(t_value.item()), device=device, dtype=torch.long)
        z = student(z, t)
        if t_value > 0:
            noise_scale = (t_value.float() / max(timesteps - 1, 1)).sqrt() * 0.05
            z = z + noise_scale * torch.randn_like(z)
    return decoder.decode(z)
