"""Metrics for reconstruction, calibration, and Bayesian sweeps."""

from __future__ import annotations

import math
import torch


def mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    err = (pred - target).square()
    if mask is not None:
        err = err[mask]
    return err.mean()


def gaussian_nll_diag(mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    var = var.clamp_min(eps)
    return 0.5 * (math.log(2 * math.pi) + torch.log(var) + (target - mean).square() / var).mean()


def interval_coverage(mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor, z: float = 1.96) -> torch.Tensor:
    radius = z * torch.sqrt(var.clamp_min(0.0))
    return ((target >= mean - radius) & (target <= mean + radius)).float().mean()


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return 1.0 / weights.square().sum(dim=-1).clamp_min(1e-12)
