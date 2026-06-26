from __future__ import annotations

import torch
import torch.nn.functional as F


def consistency_loss(student, z_t: torch.Tensor, t: torch.Tensor, z_s: torch.Tensor, s: torch.Tensor):
    pred_t = student(z_t, t)
    with torch.no_grad():
        pred_s = student(z_s, s)
    return F.mse_loss(pred_t, pred_s)


def target_loss(student, z_t: torch.Tensor, t: torch.Tensor, z0: torch.Tensor):
    return F.mse_loss(student(z_t, t), z0)


def combined_qlcm_loss(
    student,
    z0: torch.Tensor,
    schedule,
    lambda_target: float = 0.5,
):
    t, s = schedule.sample_pairs(z0.shape[0])
    noise = torch.randn_like(z0)
    z_t, _ = schedule.add_noise(z0, t, noise=noise)
    z_s, _ = schedule.add_noise(z0, s, noise=noise)
    loss_cons = consistency_loss(student, z_t, t, z_s, s)
    loss_target = target_loss(student, z_t, t, z0)
    return loss_cons + lambda_target * loss_target, {
        "consistency_loss": float(loss_cons.detach().cpu()),
        "target_loss": float(loss_target.detach().cpu()),
    }
