"""Bayesian wrappers around HiPPO projection coefficients.

The core distinction in Bayesian HiPPO is between

    mu_t  : a normalized relevance distribution over past times;
    nu_t  : an unnormalized information measure, nu_t = alpha_t mu_t.

Deterministic HiPPO only sees ``mu_t`` because scaling a projection objective by
``alpha_t`` does not change its minimizer. Bayesian HiPPO sees ``alpha_t``
through posterior covariance: more evidence contracts uncertainty even if the
relevance distribution is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch

from .hippo import HiPPOMeasure, hippo_scan


@dataclass
class BayesianProjectionPosterior:
    """Gaussian posterior for projection coefficients.

    ``cov`` is either diagonal with shape ``(..., N)`` or full with shape
    ``(..., N, N)``. Most fast sweeps use diagonal covariance; empirical-Gram
    posteriors use full covariance.
    """

    mean: torch.Tensor
    cov: torch.Tensor

    @property
    def is_diag(self) -> bool:
        return self.cov.ndim == self.mean.ndim

    @staticmethod
    def from_projection(
        y: torch.Tensor,
        *,
        obs_var: float | torch.Tensor = 1.0,
        evidence_mass: float | torch.Tensor = 1.0,
        prior_mean: Optional[torch.Tensor] = None,
        prior_cov: float | torch.Tensor = 1e6,
    ) -> "BayesianProjectionPosterior":
        """Posterior from an orthonormal projection statistic.

        Model:
            y = c_star + eps, eps ~ N(0, obs_var / evidence_mass * I).

        Args:
            y: Projection statistic, shape ``(..., N)``.
            obs_var: Per-unit-evidence coefficient-statistic variance.
            evidence_mass: Unnormalized information mass alpha_t.
            prior_mean: Prior mean, broadcastable to y. Defaults to zero.
            prior_cov: Prior variance or diagonal covariance, broadcastable to y.

        Returns:
            Posterior with diagonal covariance represented as shape ``(..., N)``.
        """
        if prior_mean is None:
            prior_mean = torch.zeros_like(y)
        else:
            prior_mean = prior_mean.to(dtype=y.dtype, device=y.device)
        obs_var_t = torch.as_tensor(obs_var, dtype=y.dtype, device=y.device)
        mass_t = torch.as_tensor(evidence_mass, dtype=y.dtype, device=y.device).clamp_min(1e-30)
        prior_cov_t = torch.as_tensor(prior_cov, dtype=y.dtype, device=y.device)
        obs_prec = mass_t / obs_var_t
        prior_prec = 1.0 / prior_cov_t
        cov = 1.0 / (prior_prec + obs_prec)
        mean = cov * (prior_prec * prior_mean + obs_prec * y)
        return BayesianProjectionPosterior(mean=mean, cov=cov.expand_as(y))

    @staticmethod
    def from_weighted_samples(
        basis: torch.Tensor,
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        *,
        obs_var: float | torch.Tensor = 1.0,
        prior_mean: Optional[torch.Tensor] = None,
        prior_cov: float | torch.Tensor = 1e6,
        jitter: float = 1e-6,
    ) -> "BayesianProjectionPosterior":
        """Full-covariance posterior for weighted finite observations.

        Model:
            y_i = phi(s_i)^T c + eps_i, eps_i ~ N(0, obs_var / w_i).

        This is the empirical-Gram replacement for the ideal orthonormal
        continuum projection. It is the right object for irregular timestamps,
        missing blocks, quadrature mismatch, and finite/noisy samples.

        Args:
            basis: Tensor shape ``(T, N)``.
            values: Tensor shape ``(T,)`` or ``(T, D)``. Currently each output
                channel is handled independently by broadcasting.
            weights: Nonnegative weights shape ``(T,)``. Defaults to uniform 1.
            obs_var: Observation noise variance.
            prior_mean: Shape ``(N,)`` or ``(D, N)``. Defaults to zero.
            prior_cov: Scalar/diagonal prior variance or full ``(N, N)``.
            jitter: Added to precision before Cholesky/solve.
        """
        if basis.ndim != 2:
            raise ValueError("basis must have shape (T, N)")
        dtype, device = basis.dtype, basis.device
        T, N = basis.shape
        y = values.to(dtype=dtype, device=device)
        if y.shape[0] != T:
            raise ValueError("values first dimension must match basis")
        if y.ndim == 1:
            y = y[:, None]
            squeeze = True
        else:
            squeeze = False
        D = y.shape[1]
        if weights is None:
            w = torch.ones(T, dtype=dtype, device=device)
        else:
            w = weights.to(dtype=dtype, device=device).clamp_min(0.0)
        obs_prec = 1.0 / torch.as_tensor(obs_var, dtype=dtype, device=device)
        weighted_phi = basis * w[:, None]
        gram = obs_prec * (basis.T @ weighted_phi)
        rhs = obs_prec * (basis.T @ (w[:, None] * y)).T  # D, N

        if prior_mean is None:
            pm = torch.zeros(D, N, dtype=dtype, device=device)
        else:
            pm = prior_mean.to(dtype=dtype, device=device)
            if pm.ndim == 1:
                pm = pm.expand(D, N)

        pc = torch.as_tensor(prior_cov, dtype=dtype, device=device)
        eye = torch.eye(N, dtype=dtype, device=device)
        if pc.ndim == 0:
            prior_prec = eye / pc.clamp_min(1e-30)
        elif pc.ndim == 1:
            prior_prec = torch.diag(1.0 / pc.clamp_min(1e-30))
        elif pc.ndim == 2:
            prior_prec = torch.linalg.inv(pc + jitter * eye)
        else:
            raise ValueError("prior_cov must be scalar, diagonal, or full")

        precision = prior_prec + gram + jitter * eye
        chol = torch.linalg.cholesky(precision)
        nat = rhs + (prior_prec @ pm.T).T
        mean = torch.cholesky_solve(nat.T, chol).T
        cov = torch.cholesky_inverse(chol)
        if squeeze:
            mean = mean[0]
        return BayesianProjectionPosterior(mean=mean, cov=cov)

    def credible_radius(self, level: float = 1.96) -> torch.Tensor:
        """Elementwise Gaussian credible radius."""
        if self.is_diag:
            return level * torch.sqrt(self.cov.clamp_min(0.0))
        return level * torch.sqrt(torch.diagonal(self.cov, dim1=-2, dim2=-1).clamp_min(0.0))

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """Draw samples from the posterior."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.is_diag:
            eps = torch.randn(num_samples, *self.mean.shape, dtype=self.mean.dtype, device=self.mean.device)
            return self.mean.unsqueeze(0) + eps * torch.sqrt(self.cov.clamp_min(0.0)).unsqueeze(0)
        dist = torch.distributions.MultivariateNormal(self.mean, covariance_matrix=self.cov)
        return dist.rsample((num_samples,))

    def predictive_moments(self, basis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean and variance of reconstructed values phi(u)^T C.

        Args:
            basis: Tensor shape ``(T, N)``.
        """
        phi = basis.to(dtype=self.mean.dtype, device=self.mean.device)
        mean = torch.einsum("tn,...n->...t", phi, self.mean)
        if self.is_diag:
            var = torch.einsum("tn,...n->...t", phi.square(), self.cov)
        else:
            var = torch.einsum("tn,nm,tm->t", phi, self.cov, phi)
        return mean, var.clamp_min(0.0)


class BayesianHiPPOFilter:
    """Deterministic HiPPO state plus conjugate Gaussian coefficient posterior."""

    def __init__(
        self,
        N: int,
        measure: HiPPOMeasure | str = "legs",
        *,
        theta: float = 1.0,
        obs_var: float = 1.0,
        evidence_mass: float = 1.0,
        prior_var: float = 1e6,
    ):
        self.N = N
        self.measure = HiPPOMeasure(measure)
        self.theta = theta
        self.obs_var = obs_var
        self.evidence_mass = evidence_mass
        self.prior_var = prior_var

    def filter(self, x: torch.Tensor, return_sequence: bool = True) -> BayesianProjectionPosterior:
        y = hippo_scan(x, self.N, self.measure, theta=self.theta, return_sequence=return_sequence)
        return BayesianProjectionPosterior.from_projection(
            y,
            obs_var=self.obs_var,
            evidence_mass=self.evidence_mass,
            prior_cov=self.prior_var,
        )


class MeasureMixture:
    """Bayesian model averaging over a finite bank of HiPPO measures.

    Each component produces a reconstruction or prediction loss. Component
    weights are updated by a pseudo-likelihood proportional to
    ``exp(-loss / temperature)``. This is both a practical model-selection
    mechanism and the implementation counterpart of the finite-mixture regret
    theorem in the paper.
    """

    def __init__(
        self,
        components: Iterable[BayesianHiPPOFilter],
        *,
        temperature: float = 1.0,
        prior_logits: Optional[torch.Tensor] = None,
    ):
        self.components: List[BayesianHiPPOFilter] = list(components)
        if not self.components:
            raise ValueError("Need at least one component")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        if prior_logits is None:
            prior_logits = torch.zeros(len(self.components))
        self.prior_logits = prior_logits

    def posterior_weights(self, losses: torch.Tensor) -> torch.Tensor:
        logits = self.prior_logits.to(losses.device, losses.dtype) - losses / self.temperature
        return torch.softmax(logits, dim=-1)

    def combine(self, posteriors: List[BayesianProjectionPosterior], losses: torch.Tensor):
        """Moment-match a diagonal/full Gaussian mixture posterior."""
        if len(posteriors) != len(self.components):
            raise ValueError("posteriors must match components")
        weights = self.posterior_weights(losses)
        means = torch.stack([p.mean for p in posteriors], dim=-2)  # ..., M, N
        diag_covs = []
        for p in posteriors:
            if p.is_diag:
                diag_covs.append(p.cov)
            else:
                diag_covs.append(torch.diagonal(p.cov, dim1=-2, dim2=-1).expand_as(p.mean))
        covs = torch.stack(diag_covs, dim=-2)
        w = weights[..., None]
        mean = (w * means).sum(dim=-2)
        second = (w * (covs + means.square())).sum(dim=-2)
        cov = (second - mean.square()).clamp_min(0.0)
        return weights, BayesianProjectionPosterior(mean=mean, cov=cov)
