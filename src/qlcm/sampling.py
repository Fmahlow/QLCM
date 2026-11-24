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


@torch.no_grad()
def sample_diffusion(score_net, decoder, latent_dim: int, n_samples: int, steps: int, schedule, device: str):
    """Standard iterative DDIM sampling from a noise-predicting score net.

    Unlike `sample_consistency`, the score net predicts noise (not a clean
    latent) at every step, and each step advances the trajectory by one
    subsampled increment via `schedule.ddim_step` rather than jumping
    straight to the clean endpoint. This is the naive multi-step diffusion
    baseline `steps` controls the number of NFEs directly, which is the
    quantity being compared against QLCM's 1/2/4-step consistency sampling.
    """
    score_net.eval()
    decoder.eval()
    timesteps = schedule.timesteps
    z = torch.randn(n_samples, latent_dim, device=device)
    step_schedule = torch.linspace(timesteps - 1, 0, max(steps, 1) + 1, device=device).long()
    for idx in range(len(step_schedule) - 1):
        t = torch.full((n_samples,), int(step_schedule[idx].item()), device=device, dtype=torch.long)
        s = torch.full((n_samples,), int(step_schedule[idx + 1].item()), device=device, dtype=torch.long)
        eps_hat = score_net(z, t)
        z, z0_hat = schedule.ddim_step(z, t, s, eps_hat)
    return decoder.decode(z0_hat)
