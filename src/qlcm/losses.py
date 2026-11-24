from __future__ import annotations

import torch
import torch.nn.functional as F


def consistency_loss(student, target_model, z_t: torch.Tensor, t: torch.Tensor, z_s: torch.Tensor, s: torch.Tensor):
    """`target_model` is an EMATarget (recommended, v2 default) or, for
    direct comparison against v1's approximation, the online `student`
    itself passed twice -- either way it is only ever called under
    `no_grad` here."""
    pred_t = student(z_t, t)
    with torch.no_grad():
        pred_s = target_model(z_s, s)
    return F.mse_loss(pred_t, pred_s)


def target_loss(student, z_t: torch.Tensor, t: torch.Tensor, z0: torch.Tensor):
    return F.mse_loss(student(z_t, t), z0)


def combined_qlcm_loss(
    student,
    target_model,
    teacher,
    z0: torch.Tensor,
    schedule,
    lambda_target: float = 0.5,
):
    """Consistency distillation loss (Song et al. 2023, Algorithm 2).

    z_s is not obtained by re-noising z0 independently at level s (which
    would make s and t two unrelated noisy views of the same clean sample,
    not two points on one trajectory); it is obtained by taking one DDIM
    step of the frozen `teacher` score model from z_t at level t down to
    level s, so student(z_t,t) and target_model(z_s,s) are trained to agree
    along the teacher's actual denoising trajectory. `teacher` must be in
    eval mode; gradients are not tracked through it. `target_model` is
    typically an `EMATarget` wrapping `student` (see src/qlcm/ema.py); using
    the online student as its own target instead is v1's approximation, kept
    available only for direct before/after comparison.
    """
    t, s = schedule.sample_pairs(z0.shape[0])
    z_t, _ = schedule.add_noise(z0, t)
    with torch.no_grad():
        eps_hat = teacher(z_t, t)
        z_s, _ = schedule.ddim_step(z_t, t, s, eps_hat)
    loss_cons = consistency_loss(student, target_model, z_t, t, z_s, s)
    loss_target = target_loss(student, z_t, t, z0)
    return loss_cons + lambda_target * loss_target, {
        "consistency_loss": float(loss_cons.detach().cpu()),
        "target_loss": float(loss_target.detach().cpu()),
    }
