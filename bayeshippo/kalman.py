"""Linear-Gaussian state-space updates for Kalman-HiPPO.

This module is generic: the state may be a HiPPO coefficient vector, and the
transition matrices can be supplied by a discretized HiPPO operator. It provides
exact Gaussian filtering recursions for adding a physical dynamics prior on top
of polynomial memory coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class KalmanState:
    """Gaussian filtering state."""

    mean: torch.Tensor  # (..., N)
    cov: torch.Tensor  # (..., N, N)


class LinearGaussianFilter:
    """Batched Kalman predict/update recursions."""

    def __init__(self, jitter: float = 1e-7):
        self.jitter = jitter

    def predict(
        self,
        state: KalmanState,
        transition: torch.Tensor,
        process_cov: torch.Tensor,
        control_matrix: Optional[torch.Tensor] = None,
        control: Optional[torch.Tensor] = None,
    ) -> KalmanState:
        """Prediction step: c_k = F c_{k-1} + B u_k + q_k."""
        F = transition.to(dtype=state.mean.dtype, device=state.mean.device)
        Q = process_cov.to(dtype=state.mean.dtype, device=state.mean.device)
        mean = torch.matmul(F, state.mean.unsqueeze(-1)).squeeze(-1)
        if control_matrix is not None and control is not None:
            B = control_matrix.to(dtype=state.mean.dtype, device=state.mean.device)
            u = control.to(dtype=state.mean.dtype, device=state.mean.device)
            mean = mean + torch.matmul(B, u.unsqueeze(-1)).squeeze(-1)
        cov = F @ state.cov @ F.transpose(-1, -2) + Q
        return KalmanState(mean=mean, cov=cov)

    def update(
        self,
        state: KalmanState,
        observation: torch.Tensor,
        observation_matrix: torch.Tensor,
        observation_cov: torch.Tensor,
    ) -> KalmanState:
        """Update step: y_k = H c_k + r_k."""
        dtype, device = state.mean.dtype, state.mean.device
        H = observation_matrix.to(dtype=dtype, device=device)
        R = observation_cov.to(dtype=dtype, device=device)
        y = observation.to(dtype=dtype, device=device)
        pred_y = torch.matmul(H, state.mean.unsqueeze(-1)).squeeze(-1)
        innovation = y - pred_y
        S = H @ state.cov @ H.transpose(-1, -2) + R
        m = S.shape[-1]
        eye_y = torch.eye(m, dtype=dtype, device=device).expand(S.shape)
        S = S + self.jitter * eye_y
        # K = P H^T S^{-1}. Use solve for numerical stability.
        PHt = state.cov @ H.transpose(-1, -2)
        K_t = torch.linalg.solve(S, PHt.transpose(-1, -2))
        K = K_t.transpose(-1, -2)
        mean = state.mean + torch.matmul(K, innovation.unsqueeze(-1)).squeeze(-1)
        n = state.mean.shape[-1]
        eye_x = torch.eye(n, dtype=dtype, device=device).expand(state.cov.shape)
        # Joseph form preserves PSD better: (I-KH)P(I-KH)^T + K R K^T.
        I_KH = eye_x - K @ H
        cov = I_KH @ state.cov @ I_KH.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
        return KalmanState(mean=mean, cov=cov)

    def step(
        self,
        state: KalmanState,
        transition: torch.Tensor,
        process_cov: torch.Tensor,
        observation: torch.Tensor,
        observation_matrix: torch.Tensor,
        observation_cov: torch.Tensor,
        control_matrix: Optional[torch.Tensor] = None,
        control: Optional[torch.Tensor] = None,
    ) -> KalmanState:
        """Combined predict/update step."""
        pred = self.predict(state, transition, process_cov, control_matrix, control)
        return self.update(pred, observation, observation_matrix, observation_cov)
