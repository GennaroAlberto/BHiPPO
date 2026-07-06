"""Bounds and diagnostics for Bayesian HiPPO predictors.

The functions here are deliberately small: they mirror the theorems in the
paper and make it easy to compute the terms that appear in PAC-Bayes and
readout-stability guarantees.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def gaussian_kl_diag(
    q_mean: torch.Tensor,
    q_var: torch.Tensor,
    p_mean: Optional[torch.Tensor] = None,
    p_var: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """KL(N(q_mean, diag(q_var)) || N(p_mean, diag(p_var))).

    Returns a tensor with the last dimension summed out, preserving leading
    batch dimensions.
    """
    if p_mean is None:
        p_mean = torch.zeros_like(q_mean)
    else:
        p_mean = p_mean.to(dtype=q_mean.dtype, device=q_mean.device)
    q_var = q_var.to(dtype=q_mean.dtype, device=q_mean.device).clamp_min(1e-30)
    p_var_t = torch.as_tensor(p_var, dtype=q_mean.dtype, device=q_mean.device).clamp_min(1e-30)
    term = (q_var + (q_mean - p_mean).square()) / p_var_t - 1.0 + torch.log(p_var_t / q_var)
    return 0.5 * term.sum(dim=-1)


def gaussian_kl_full(
    q_mean: torch.Tensor,
    q_cov: torch.Tensor,
    p_mean: Optional[torch.Tensor] = None,
    p_cov: Optional[torch.Tensor] = None,
    jitter: float = 1e-8,
) -> torch.Tensor:
    """KL(N(q_mean, q_cov) || N(p_mean, p_cov)) for full covariances.

    Shapes: q_mean (..., N), q_cov (..., N, N). p_mean/p_cov may be broadcastable.
    """
    dtype, device = q_mean.dtype, q_mean.device
    n = q_mean.shape[-1]
    if p_mean is None:
        p_mean = torch.zeros_like(q_mean)
    else:
        p_mean = p_mean.to(dtype=dtype, device=device).expand_as(q_mean)
    eye = torch.eye(n, dtype=dtype, device=device)
    if p_cov is None:
        p_cov = eye.expand(q_cov.shape)
    else:
        p_cov = p_cov.to(dtype=dtype, device=device)
        if p_cov.ndim == 2:
            p_cov = p_cov.expand(q_cov.shape)
    q_cov = q_cov + jitter * eye.expand(q_cov.shape)
    p_cov = p_cov + jitter * eye.expand(p_cov.shape)
    chol_p = torch.linalg.cholesky(p_cov)
    # tr(P^{-1}Q)
    p_inv_q = torch.cholesky_solve(q_cov, chol_p)
    trace_term = torch.diagonal(p_inv_q, dim1=-2, dim2=-1).sum(dim=-1)
    # quadratic term
    diff = (p_mean - q_mean).unsqueeze(-1)
    quad = torch.matmul(diff.transpose(-1, -2), torch.cholesky_solve(diff, chol_p)).squeeze(-1).squeeze(-1)
    logdet_p = 2.0 * torch.log(torch.diagonal(chol_p, dim1=-2, dim2=-1)).sum(dim=-1)
    chol_q = torch.linalg.cholesky(q_cov)
    logdet_q = 2.0 * torch.log(torch.diagonal(chol_q, dim1=-2, dim2=-1)).sum(dim=-1)
    return 0.5 * (trace_term + quad - n + logdet_p - logdet_q)


def pac_bayes_pinsker_bound(
    empirical_loss: float | torch.Tensor,
    kl: float | torch.Tensor,
    n: int,
    delta: float = 0.05,
) -> torch.Tensor:
    """A simple PAC-Bayes upper bound for losses in [0, 1].

    With probability at least 1-delta over n samples, for any posterior Q,

        L(Q) <= Lhat(Q) + sqrt((KL(Q||P)+log(2 sqrt(n)/delta))/(2n)).

    This is the standard Pinsker-relaxed McAllester-style form. It is not the
    tightest possible bound, but it is stable and easy to compute.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    emp = torch.as_tensor(empirical_loss, dtype=torch.float64)
    kl_t = torch.as_tensor(kl, dtype=torch.float64).clamp_min(0.0)
    complexity = kl_t + math.log(2.0 * math.sqrt(n) / delta)
    return (emp + torch.sqrt(complexity / (2.0 * n))).clamp(max=1.0)


def linear_readout_variance(weight: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """Exact output variance trace for a linear readout z = W c.

    Args:
        weight: W with shape (K, N) or (..., K, N).
        cov: coefficient covariance, diagonal (..., N) or full (..., N, N).

    Returns:
        tr(W P W^T), the expected squared logit perturbation around the mean.
    """
    if cov.ndim == weight.ndim - 1:  # diagonal covariance, e.g. cov (..., N)
        return (weight.square() * cov.unsqueeze(-2)).sum(dim=(-2, -1))
    if cov.ndim == weight.ndim:  # full covariance, e.g. (..., N, N)
        wp = torch.matmul(weight, cov)
        return (wp * weight).sum(dim=(-2, -1))
    raise ValueError("cov must be diagonal (..., N) or full (..., N, N) compatible with weight")


def lipschitz_readout_loss_gap_bound(
    cov: torch.Tensor,
    readout_lipschitz: float,
    loss_lipschitz: float = 1.0,
) -> torch.Tensor:
    """Bound |E loss(r(C)) - loss(r(E C))| for Lipschitz readouts/losses.

    If r is L_r-Lipschitz and the loss is L_ell-Lipschitz in readout space,
    then the gap is at most L_ell L_r sqrt(tr(P)).
    """
    if cov.ndim >= 2 and cov.shape[-1] == cov.shape[-2]:
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1)
    else:
        trace = cov.sum(dim=-1)
    return float(readout_lipschitz) * float(loss_lipschitz) * torch.sqrt(trace.clamp_min(0.0))
