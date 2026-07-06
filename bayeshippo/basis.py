"""Polynomial basis evaluation and reconstruction helpers.

The HiPPO-LegS coefficients implemented in :mod:`bayeshippo.hippo` use the
orthonormal shifted Legendre basis on the normalized interval [0, 1],

    phi_n(u) = sqrt(2n+1) P_n(2u-1),    u in [0, 1],

where P_n is the ordinary Legendre polynomial on [-1, 1]. These utilities make
it easy to turn final memory coefficients back into reconstructed functions for
experiments and diagnostics.
"""

from __future__ import annotations

import torch


def legendre_basis(u: torch.Tensor, N: int, *, clamp: bool = True) -> torch.Tensor:
    """Evaluate normalized shifted Legendre basis functions.

    Args:
        u: Tensor of normalized positions. The meaningful domain is [0, 1].
        N: Number of basis functions.
        clamp: Clamp u into [0, 1] before evaluation.

    Returns:
        Tensor with shape ``u.shape + (N,)``.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    u = torch.as_tensor(u)
    if clamp:
        u = u.clamp(0.0, 1.0)
    z = 2.0 * u - 1.0
    basis = []
    p0 = torch.ones_like(z)
    basis.append(p0)
    if N > 1:
        p1 = z
        basis.append(p1)
        pm2, pm1 = p0, p1
        for n in range(1, N - 1):
            # (n+1)P_{n+1}(z) = (2n+1)zP_n(z) - nP_{n-1}(z)
            p = ((2 * n + 1) * z * pm1 - n * pm2) / (n + 1)
            basis.append(p)
            pm2, pm1 = pm1, p
    out = torch.stack(basis, dim=-1)
    n = torch.arange(N, dtype=out.dtype, device=out.device)
    return out * torch.sqrt(2.0 * n + 1.0)


def reconstruct_from_coeffs(coeff: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Reconstruct values from shifted-Legendre coefficients.

    Args:
        coeff: Tensor ``(..., N)``.
        u: Tensor of normalized locations ``(T,)`` or any shape.

    Returns:
        Tensor with shape ``coeff.shape[:-1] + u.shape``.
    """
    coeff = torch.as_tensor(coeff)
    u = torch.as_tensor(u, dtype=coeff.dtype, device=coeff.device)
    phi = legendre_basis(u, coeff.shape[-1]).to(dtype=coeff.dtype, device=coeff.device)
    # phi shape U..., N; coeff shape C..., N -> C..., U...
    return torch.einsum("...n,un->...u", coeff, phi.reshape(-1, phi.shape[-1])).reshape(
        *coeff.shape[:-1], *u.shape
    )


def reconstruct_legs(coeff: torch.Tensor, times: torch.Tensor, *, final_time: float | None = None) -> torch.Tensor:
    """Reconstruct a LegS memory state on absolute times in [0, final_time]."""
    times = torch.as_tensor(times, dtype=coeff.dtype, device=coeff.device)
    if final_time is None:
        final_time = float(times.max().item()) if times.numel() else 1.0
    denom = max(float(final_time), 1e-12)
    u = times / denom
    return reconstruct_from_coeffs(coeff, u)


def reconstruct_legt(
    coeff: torch.Tensor,
    times: torch.Tensor,
    *,
    final_time: float,
    theta: float,
    outside_value: float = 0.0,
) -> torch.Tensor:
    """Reconstruct a LegT memory state on absolute times.

    The LegT/LMU recurrence in :mod:`bayeshippo.hippo` maintains coefficients
    with respect to the *unnormalized, time-reversed* shifted Legendre basis
    P_n(2 (t - x) / theta - 1) (Voelker et al.'s delay parameterization), i.e.
    phi_n(u) = (-1)^n P_n(2u - 1) with u = (x - (t - theta)) / theta -- NOT
    the orthonormal forward basis used by LegS.

    Values outside the final sliding window ``[final_time-theta, final_time]``
    are set to ``outside_value`` because LegT has intentionally forgotten them.
    """
    if theta <= 0:
        raise ValueError("theta must be positive")
    coeff = torch.as_tensor(coeff)
    times = torch.as_tensor(times, dtype=coeff.dtype, device=coeff.device)
    u = (times - (final_time - theta)) / theta
    n = torch.arange(coeff.shape[-1], dtype=coeff.dtype, device=coeff.device)
    # convert to the reversed unnormalized basis: divide the orthonormal
    # evaluation by sqrt(2n+1) and alternate signs
    coeff_reversed = coeff * ((-1.0) ** n) / torch.sqrt(2.0 * n + 1.0)
    rec = reconstruct_from_coeffs(coeff_reversed, u.clamp(0.0, 1.0))
    mask = ((u >= 0.0) & (u <= 1.0)).to(dtype=rec.dtype, device=rec.device)
    return rec * mask + float(outside_value) * (1.0 - mask)
