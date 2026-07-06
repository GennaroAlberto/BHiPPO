"""Regression test for the LegT decode convention.

hippo.py's LegT recurrence follows the LMU convention: coefficients are with
respect to the unnormalized time-reversed shifted Legendre basis
P_n(2(t-x)/theta - 1). Decoding with the orthonormal forward basis (the old
reconstruct_legt) gave MSE ~ 8 on a unit-amplitude sine; the corrected decode
gives ~3e-3.
"""

import numpy as np
import torch

from bayeshippo.basis import reconstruct_legt
from bayeshippo.hippo import hippo_scan


def test_legt_window_reconstruction_smooth_signal():
    N, theta, T = 32, 64.0, 64
    ks = np.arange(1, T + 1, dtype=np.float64)
    y = np.sin(2 * np.pi * ks / 32)
    c = hippo_scan(torch.as_tensor(y), N, "legt", theta=theta, dt=1.0, return_sequence=False)
    rec = reconstruct_legt(c, torch.as_tensor(ks), final_time=float(T), theta=theta).numpy()
    mse = float(np.mean((rec - y) ** 2))
    assert mse < 0.05, mse


def test_legt_reconstruction_sliding_window_forgets():
    # values before the window must be masked to the outside value
    N, theta, T = 16, 32.0, 128
    ks = np.arange(1, T + 1, dtype=np.float64)
    y = np.cos(2 * np.pi * ks / 16)
    c = hippo_scan(torch.as_tensor(y), N, "legt", theta=theta, dt=1.0, return_sequence=False)
    rec = reconstruct_legt(c, torch.as_tensor(ks), final_time=float(T), theta=theta, outside_value=0.0).numpy()
    assert np.allclose(rec[: int(T - theta) - 1], 0.0)
    inside = slice(int(T - theta) + 2, T)
    assert float(np.mean((rec[inside] - y[inside]) ** 2)) < 0.05
