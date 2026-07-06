"""UnHiPPO baseline (Lienen, Saydemir & Guennemann, ICML 2025, arXiv:2506.05065).

Mirrors the paper exactly:

- Eq. (11): data-free coefficient dynamics dc/dt = (1/t)(A_H^T - I) c.
- Eqs. (17)-(18): regularized matrix
      A_R = pinv([I; B^T; Q^T]) @ [A_H^T - I; 2 Q^T; Q^T],
  with Q_i = sqrt(2i+1) * i(i+1)/2 (linear-extrapolation regularization).
- Eq. (21): exact propagator Abar_{R,k} = expm(log(t_k / t_{k-1}) A_R),
  with t_0 = t_1 so Abar_{R,1} = I.
- Eqs. (22)-(23): Kalman filter with prior m_0 = 0, P_0 = I, transition noise
  Sigma = I (their fixed choice), observation model y = B_H^T c + eps,
  eps ~ N(0, sigma^2) with sigma^2 as the smoothing hyperparameter
  (their sweeps use the range 1e4 .. 1e14).
- Eq. (24): the collapsed recurrence m_k = Abar_U m_{k-1} + Bbar_U y_k.

The filter runs in float64 and symmetrizes P after each update, as in the
paper's numerical-stability recipe. We additionally keep P so the (data-
independent) covariance can be used for NLL/coverage comparisons.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.linalg import expm

from .srif import legs_basis_matrix, legs_matrices


def unhippo_regularized_matrix(N: int) -> np.ndarray:
    """A_R from Eq. (18)."""
    A, B = legs_matrices(N)
    i = np.arange(N, dtype=np.float64)
    Q = np.sqrt(2 * i + 1) * i * (i + 1) / 2.0
    left = np.vstack([np.eye(N), B[None, :], Q[None, :]])
    right = np.vstack([A.T - np.eye(N), 2.0 * Q[None, :], Q[None, :]])
    return np.linalg.pinv(left) @ right


class UnHiPPOFilter:
    """Kalman filter over the regularized UnHiPPO dynamics (Eqs. 19-24)."""

    def __init__(
        self,
        N: int,
        *,
        sigma2: float = 1e8,
        transition_noise: float = 1.0,
        dtype=np.float64,
    ) -> None:
        self.N = N
        self.sigma2 = float(sigma2)
        self.Sigma = transition_noise * np.eye(N, dtype=dtype)
        self.A_R = unhippo_regularized_matrix(N).astype(dtype)
        _, B = legs_matrices(N)
        self.B = B.astype(dtype)
        self.dtype = dtype

        self.m = np.zeros(N, dtype=dtype)
        self.P = np.eye(N, dtype=dtype)
        self.t: Optional[float] = None
        self._expm_cache: dict[float, np.ndarray] = {}

    def _propagator(self, log_rho: float) -> np.ndarray:
        T = self._expm_cache.get(log_rho)
        if T is None:
            T = expm(log_rho * self.A_R.astype(np.float64)).astype(self.dtype)
            if len(self._expm_cache) < 4096:
                self._expm_cache[log_rho] = T
        return T

    def step(self, t_new: float, y: float) -> Tuple[float, float]:
        """One predict/update cycle; returns the pre-update predictive (mean, var)."""
        if t_new <= 0:
            raise ValueError("times must be positive")
        if self.t is None:
            self.t = float(t_new)  # t_0 = t_1  =>  Abar_{R,1} = I
            Abar = np.eye(self.N, dtype=self.dtype)
        else:
            if t_new < self.t:
                raise ValueError("times must be nondecreasing")
            Abar = self._propagator(float(np.log(t_new / self.t)))
            self.t = float(t_new)

        m_pred = Abar @ self.m
        P_pred = Abar @ self.P @ Abar.T + self.Sigma

        v = y - self.B @ m_pred
        s = float(self.B @ P_pred @ self.B + self.sigma2)
        K = (P_pred @ self.B) / s
        self.m = m_pred + K * v
        self.P = P_pred - s * np.outer(K, K)
        self.P = (self.P + self.P.T) / 2.0
        return float(self.B @ m_pred), s

    def run(self, times: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        preds = np.empty_like(ys)
        pvars = np.empty_like(ys)
        for i in range(len(ys)):
            preds[i], pvars[i] = self.step(times[i], ys[i])
        return preds, pvars

    def reconstruct(self, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean/variance of the represented function at times xs in [0, t]."""
        if self.t is None:
            raise RuntimeError("no observations yet")
        xs = np.asarray(xs, dtype=self.dtype)
        Phi = legs_basis_matrix(2.0 * xs / self.t - 1.0, self.N, dtype=self.dtype)
        mean = Phi @ self.m
        var = np.sum((Phi @ self.P) * Phi, axis=1)
        return mean, np.maximum(var, 0.0)
