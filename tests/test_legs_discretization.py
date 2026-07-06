"""P0.1 acceptance: bilinear LegS is stable where forward Euler blew up."""

import torch

from bayeshippo.basis import reconstruct_legs, reconstruct_legt
from bayeshippo.hippo import hippo_scan


def _smooth_signal(T: int) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, T, dtype=torch.float64)
    return torch.sin(6.0 * torch.pi * t) + 0.5 * torch.cos(11.0 * torch.pi * t)


def test_bilinear_legs_beats_legt_full_window():
    T, N = 512, 32
    x = _smooth_signal(T)
    times = torch.arange(1, T + 1, dtype=torch.float64)

    c_legs = hippo_scan(x, N, "legs", return_sequence=False)
    rec_legs = reconstruct_legs(c_legs, times, final_time=float(T))
    mse_legs = torch.mean((rec_legs - x) ** 2).item()

    c_legt = hippo_scan(x, N, "legt", theta=float(T), dt=1.0, return_sequence=False)
    rec_legt = reconstruct_legt(c_legt, times, final_time=float(T), theta=float(T))
    mse_legt = torch.mean((rec_legt - x) ** 2).item()

    assert torch.isfinite(c_legs).all()
    assert mse_legs < float("inf")
    assert mse_legs < mse_legt, (mse_legs, mse_legt)
    # bilinear LegS should reconstruct a smooth signal well in absolute terms
    assert mse_legs < 1e-2, mse_legs


def test_euler_reference_is_unstable_and_bilinear_is_not():
    # Euler's instability is a transient: states blow up for k <~ N and are
    # slowly contracted again, so assert on the whole trajectory.
    T, N = 512, 32
    x = _smooth_signal(T)
    seq_euler = hippo_scan(x, N, "legs", discretization="euler")
    seq_bilinear = hippo_scan(x, N, "legs")
    assert seq_bilinear.abs().max() < 1e3
    assert seq_euler.abs().max() > 1e6
