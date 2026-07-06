"""HiPPO matrix constructors and online recurrences.

The sign convention follows the HiPPO paper:

- LegT/LagT continuous dynamics: dc/dt = -A c + B f(t)
- LegS continuous dynamics:     dc/dt = -(1/t) A c + (1/t) B f(t)

For discrete LegS the default is the per-step bilinear (generalized bilinear
transform, alpha = 1/2) recurrence used in the HiPPO paper's experiments,
    c_k = (I + A/(2k))^{-1} [(I - A/(2k)) c_{k-1} + (B/k) f_k],
with k starting at 1. The forward-Euler variant
    c_k = (I - A/k) c_{k-1} + (B/k) f_k
is kept as ``discretization="euler"`` for reference only: it is violently
unstable for k <~ N because I - A/k has spectral radius >> 1 there.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from scipy.linalg import expm, solve


class HiPPOMeasure(str, Enum):
    LEGS = "legs"  # scaled Legendre: uniform on [0,t]
    LEGT = "legt"  # translated Legendre: uniform on [t-theta,t]
    LAGT = "lagt"  # translated Laguerre: exponential decay into the past


Discretization = Literal["euler", "backward_euler", "bilinear", "zoh"]


@dataclass(frozen=True)
class HiPPOMatrices:
    A: torch.Tensor
    B: torch.Tensor
    measure: HiPPOMeasure
    theta: float = 1.0


def _as_dtype_device(dtype: torch.dtype | None, device: torch.device | str | None):
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cpu")
    return dtype, torch.device(device)


def make_hippo_matrices(
    N: int,
    measure: HiPPOMeasure | str = HiPPOMeasure.LEGS,
    *,
    theta: float = 1.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> HiPPOMatrices:
    """Create continuous-time HiPPO matrices A, B.

    Args:
        N: Memory order.
        measure: ``legs``, ``legt``, or ``lagt``.
        theta: Sliding-window length for LegT. Ignored otherwise.
        dtype, device: Torch dtype/device.

    Returns:
        HiPPOMatrices with A in the paper's positive convention; dynamics use
        ``-A`` for LegT/LagT and ``-A/t`` for LegS.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    measure = HiPPOMeasure(measure)
    dtype, device = _as_dtype_device(dtype, device)

    n = torch.arange(N, dtype=dtype, device=device)
    if measure == HiPPOMeasure.LEGS:
        A = torch.zeros(N, N, dtype=dtype, device=device)
        for row in range(N):
            for col in range(N):
                if row > col:
                    A[row, col] = torch.sqrt(torch.tensor(2 * row + 1.0, dtype=dtype, device=device)) * torch.sqrt(
                        torch.tensor(2 * col + 1.0, dtype=dtype, device=device)
                    )
                elif row == col:
                    A[row, col] = row + 1.0
        B = torch.sqrt(2 * n + 1).unsqueeze(-1)
    elif measure == HiPPOMeasure.LEGT:
        A = torch.zeros(N, N, dtype=dtype, device=device)
        for row in range(N):
            for col in range(N):
                if row >= col:
                    A[row, col] = ((-1.0) ** (row - col)) * (2 * row + 1) / theta
                else:
                    A[row, col] = (2 * row + 1) / theta
        B = (((-1.0) ** n) * (2 * n + 1) / theta).unsqueeze(-1)
    elif measure == HiPPOMeasure.LAGT:
        A = torch.tril(torch.ones(N, N, dtype=dtype, device=device))
        B = torch.ones(N, 1, dtype=dtype, device=device)
    else:  # pragma: no cover
        raise ValueError(f"Unknown measure {measure}")

    return HiPPOMatrices(A=A, B=B, measure=measure, theta=theta)


def discretize_lti(
    A: torch.Tensor,
    B: torch.Tensor,
    dt: float,
    method: Discretization = "bilinear",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Discretize LTI dynamics dc/dt = A c + B u.

    This function expects the actual continuous matrix ``A``. For LegT/LagT,
    pass ``-A_paper``.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    N = A.shape[0]
    I = torch.eye(N, dtype=A.dtype, device=A.device)

    if method == "euler":
        return I + dt * A, dt * B
    if method == "backward_euler":
        M = I - dt * A
        return torch.linalg.solve(M, I), torch.linalg.solve(M, dt * B)
    if method == "bilinear":
        M = I - 0.5 * dt * A
        return torch.linalg.solve(M, I + 0.5 * dt * A), torch.linalg.solve(M, dt * B)
    if method == "zoh":
        # scipy for numerical stability and simplicity in the research scaffold
        A_np = A.detach().cpu().numpy()
        B_np = B.detach().cpu().numpy()
        Ad = expm(dt * A_np)
        try:
            Bd = solve(A_np, (Ad - np.eye(N)) @ B_np, assume_a="gen")
        except Exception:
            # block-matrix fallback also works when A is singular
            Z = np.zeros((N + B_np.shape[1], N + B_np.shape[1]), dtype=A_np.dtype)
            Z[:N, :N] = A_np
            Z[:N, N:] = B_np
            E = expm(dt * Z)
            Ad = E[:N, :N]
            Bd = E[:N, N:]
        return (
            torch.as_tensor(Ad, dtype=A.dtype, device=A.device),
            torch.as_tensor(Bd, dtype=B.dtype, device=B.device),
        )
    raise ValueError(f"Unknown discretization method {method}")


def hippo_scan(
    x: torch.Tensor,
    N: int,
    measure: HiPPOMeasure | str = HiPPOMeasure.LEGS,
    *,
    theta: float = 1.0,
    dt: float = 1.0,
    discretization: Discretization = "bilinear",
    return_sequence: bool = True,
    c0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a HiPPO recurrence over a scalar or multichannel sequence.

    Args:
        x: Tensor of shape ``(T,)``, ``(T, B)``, or ``(T, B, D)``.
           A separate N-dimensional memory is kept for each batch/channel.
        N: Memory order.
        measure: Which HiPPO measure to use.
        theta: LegT window length.
        dt: LTI discretization step size for LegT/LagT.
        discretization: LTI discretization method for LegT/LagT. For LegS,
            "euler" selects the unstable reference recurrence; anything else
            uses the per-step bilinear update.
        return_sequence: Return all states if true, else final state.
        c0: Optional initial state with shape ``(..., N)`` matching x batch dims.

    Returns:
        If return_sequence: shape ``(T, ..., N)``. Else ``(..., N)``.
    """
    if x.ndim == 1:
        x_in = x[:, None]  # T, B=1
        squeeze_batch = True
    else:
        x_in = x
        squeeze_batch = False
    T = x_in.shape[0]
    batch_shape = x_in.shape[1:]
    dtype, device = x_in.dtype, x_in.device
    mats = make_hippo_matrices(N, measure, theta=theta, dtype=dtype, device=device)
    A_p, B = mats.A, mats.B.squeeze(-1)

    if c0 is None:
        c = torch.zeros(*batch_shape, N, dtype=dtype, device=device)
    else:
        c = c0.to(dtype=dtype, device=device)
    states = []

    measure_enum = HiPPOMeasure(measure)
    if measure_enum == HiPPOMeasure.LEGS:
        I = torch.eye(N, dtype=dtype, device=device)
        if discretization == "euler":
            for k in range(1, T + 1):
                u = x_in[k - 1]
                Abar = I - A_p / float(k)
                bbar = B / float(k)
                c = torch.einsum("ij,...j->...i", Abar, c) + u[..., None] * bbar
                if return_sequence:
                    states.append(c)
        else:
            # Bilinear (GBT alpha=1/2) at each step; A_c = -A/t evaluated at t=k.
            for k in range(1, T + 1):
                u = x_in[k - 1]
                M = I + A_p / (2.0 * k)
                rhs = torch.einsum("ij,...j->...i", I - A_p / (2.0 * k), c) + u[..., None] * (B / float(k))
                c = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)
                if return_sequence:
                    states.append(c)
    else:
        Ad, Bd = discretize_lti(-A_p, B[:, None], dt=dt, method=discretization)
        Bd = Bd.squeeze(-1)
        for k in range(T):
            u = x_in[k]
            c = torch.einsum("ij,...j->...i", Ad, c) + u[..., None] * Bd
            if return_sequence:
                states.append(c)

    out = torch.stack(states, dim=0) if return_sequence else c
    if squeeze_batch:
        return out[:, 0, :] if return_sequence else out[0, :]
    return out


class HiPPOLayer(torch.nn.Module):
    """A tiny torch module wrapper around ``hippo_scan``."""

    def __init__(self, N: int, measure: str = "legs", theta: float = 1.0):
        super().__init__()
        self.N = N
        self.measure = measure
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hippo_scan(x, self.N, self.measure, theta=self.theta, return_sequence=True)
