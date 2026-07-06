"""Priors over translated-window lengths for Bayesian HiPPO.

The object of interest is the LegT window length theta.  A prior over theta
induces a mixture of sliding-window relevance measures

    mu_theta^{(t)}(x) = 1/theta 1{x in [t-theta, t]}.

This module provides finite quadrature grids for weakly-informative priors over
theta and utilities for posterior weighting.  The most useful prior in sweeps is
log-uniform on [theta_min, theta_max], which approximates the scale-invariant
improper prior dtheta/theta after finite-time truncation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch

WindowPriorKind = Literal["uniform", "log_uniform", "pareto"]
GridScale = Literal["linear", "log"]


@dataclass(frozen=True)
class WindowPriorGrid:
    """Finite quadrature approximation to a prior over window lengths.

    Attributes:
        theta: Positive window lengths, shape (M,).
        weights: Normalized prior masses, shape (M,).
        kind: Prior family.
        theta_min/theta_max: Truncation limits.
        scale: Grid spacing used for theta.
        pareto_alpha: Tail parameter for the Pareto/log-grid prior.
    """

    theta: torch.Tensor
    weights: torch.Tensor
    kind: str
    theta_min: float
    theta_max: float
    scale: str
    pareto_alpha: float = 1.0

    @property
    def logits(self) -> torch.Tensor:
        return torch.log(self.weights.clamp_min(1e-45))

    @property
    def num_windows(self) -> int:
        return int(self.theta.numel())


def make_window_prior_grid(
    theta_min: float,
    theta_max: float,
    num_windows: int,
    *,
    kind: WindowPriorKind = "log_uniform",
    scale: GridScale | None = None,
    pareto_alpha: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> WindowPriorGrid:
    """Construct a finite prior over LegT window lengths.

    ``kind='log_uniform'`` corresponds to the proper truncation of the
    scale-invariant prior p(theta) proportional to 1/theta.  On a logarithmic
    grid this gives approximately equal prior mass to each multiplicative
    interval, which is what we want for unknown sequence lengths.

    ``kind='uniform'`` is uniform in theta on [theta_min, theta_max].
    ``kind='pareto'`` uses density proportional to theta^{-1-alpha}; on a log
    grid the cell mass is proportional to theta^{-alpha}.
    """
    if theta_min <= 0:
        raise ValueError("theta_min must be positive")
    if theta_max <= theta_min:
        raise ValueError("theta_max must be larger than theta_min")
    if num_windows <= 0:
        raise ValueError("num_windows must be positive")
    if pareto_alpha <= 0:
        raise ValueError("pareto_alpha must be positive")
    if scale is None:
        scale = "log" if kind in {"log_uniform", "pareto"} else "linear"
    device = torch.device("cpu") if device is None else torch.device(device)
    if num_windows == 1:
        theta = torch.tensor([(theta_min * theta_max) ** 0.5], dtype=dtype, device=device)
    elif scale == "log":
        theta = torch.exp(torch.linspace(torch.log(torch.tensor(theta_min, dtype=dtype)),
                                         torch.log(torch.tensor(theta_max, dtype=dtype)),
                                         num_windows, dtype=dtype, device=device))
    elif scale == "linear":
        theta = torch.linspace(theta_min, theta_max, num_windows, dtype=dtype, device=device)
    else:
        raise ValueError("scale must be 'linear' or 'log'")

    if kind == "log_uniform":
        if scale == "log":
            weights = torch.ones_like(theta)
        else:
            weights = 1.0 / theta.clamp_min(1e-30)
    elif kind == "uniform":
        if scale == "log":
            # dtheta = theta d(log theta), so mass per log cell is proportional to theta.
            weights = theta.clone()
        else:
            weights = torch.ones_like(theta)
    elif kind == "pareto":
        if scale == "log":
            weights = theta.pow(-pareto_alpha)
        else:
            weights = theta.pow(-(1.0 + pareto_alpha))
    else:
        raise ValueError(f"unknown prior kind {kind}")
    weights = weights / weights.sum().clamp_min(1e-30)
    return WindowPriorGrid(
        theta=theta,
        weights=weights,
        kind=kind,
        theta_min=float(theta_min),
        theta_max=float(theta_max),
        scale=scale,
        pareto_alpha=float(pareto_alpha),
    )


def posterior_window_weights(
    losses: torch.Tensor,
    prior_weights: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Posterior weights over window lengths from losses.

    Implements w(theta) proportional to pi(theta) exp(-loss(theta)/temperature).
    The temperature plays the role of observation variance or inverse learning
    rate, depending on whether the losses are probabilistic or decision losses.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    losses = torch.as_tensor(losses)
    prior_weights = prior_weights.to(dtype=losses.dtype, device=losses.device)
    if losses.shape[-1] != prior_weights.numel():
        raise ValueError("last loss dimension must match number of prior weights")
    logits = torch.log(prior_weights.clamp_min(1e-45)) - losses / temperature
    return torch.softmax(logits, dim=-1)


def mixture_age_density(age: torch.Tensor, theta: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Marginal density of memory age under a mixture of uniform windows.

    Conditional on window length theta, the age A=t-S is Uniform(0, theta).
    Hence q(a) = sum_j weights_j * 1{0 <= a <= theta_j}/theta_j.
    """
    age = torch.as_tensor(age)
    theta = theta.to(dtype=age.dtype, device=age.device)
    weights = weights.to(dtype=age.dtype, device=age.device)
    valid = (age[..., None] >= 0.0) & (age[..., None] <= theta)
    return (valid.to(age.dtype) * weights / theta.clamp_min(1e-30)).sum(dim=-1)


def continuous_log_uniform_complexity(
    theta_min: float,
    theta_max: float,
    center: float,
    rel_radius: float,
) -> float:
    """KL complexity of localizing a log-uniform prior near a center.

    For prior pi(dtheta) proportional to dtheta/theta on [theta_min, theta_max]
    and a posterior rho that is log-uniform on
    [center/(1+r), center*(1+r)] intersected with the support, the KL is simply
    log(prior log-width / posterior log-width).  This is the scale-free term in
    the continuous-window oracle bound.
    """
    import math

    if theta_min <= 0 or theta_max <= theta_min:
        raise ValueError("invalid support")
    if center <= 0 or rel_radius <= 0:
        raise ValueError("center and rel_radius must be positive")
    lo = max(theta_min, center / (1.0 + rel_radius))
    hi = min(theta_max, center * (1.0 + rel_radius))
    if hi <= lo:
        return float("inf")
    return math.log(math.log(theta_max / theta_min) / math.log(hi / lo))
