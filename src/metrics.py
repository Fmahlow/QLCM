from __future__ import annotations

import time

import torch


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def batch_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.mean((x - y) ** 2).detach().cpu())


def time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return value, elapsed
