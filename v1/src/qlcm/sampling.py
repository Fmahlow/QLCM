from __future__ import annotations

import torch


@torch.no_grad()
def sample_consistency(student, decoder, latent_dim: int, n_samples: int, steps: int, schedule, device: str):
    """Multistep consistency sampling (Algorithm 1, Song et al. 2023).

    Each step re-noises the current clean-latent estimate back onto the
    teacher's forward trajectory at the next timestep via `schedule.add_noise`,
    the same process used during training, instead of an arbitrary small
    perturbation. This matters because the student is only ever trained on
    `schedule`-consistent noise levels; feeding it off-distribution inputs
    (as a small ad hoc perturbation would) degrades rather than refines the
    estimate.
    """
    student.eval()
    decoder.eval()
    timesteps = schedule.timesteps
    z = torch.randn(n_samples, latent_dim, device=device)
    t = torch.full((n_samples,), timesteps - 1, device=device, dtype=torch.long)
    z0_hat = student(z, t)
    if steps <= 1:
        return decoder.decode(z0_hat)

    step_schedule = torch.linspace(timesteps - 1, 0, steps, device=device).long()
    for t_value in step_schedule[1:]:
        t = torch.full((n_samples,), int(t_value.item()), device=device, dtype=torch.long)
        if int(t_value.item()) > 0:
            z_t, _ = schedule.add_noise(z0_hat, t)
        else:
            z_t = z0_hat
        z0_hat = student(z_t, t)
    return decoder.decode(z0_hat)
