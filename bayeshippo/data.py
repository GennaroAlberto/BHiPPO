"""Synthetic time-series generators for BayesHiPPO experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

SignalKind = Literal["smooth", "rough", "piecewise", "local_burst"]


@dataclass(frozen=True)
class SyntheticSeries:
    times: torch.Tensor
    clean: torch.Tensor
    observed: torch.Tensor
    mask: torch.Tensor


def make_synthetic_series(
    T: int,
    *,
    kind: SignalKind = "smooth",
    noise_std: float = 0.0,
    missing_prob: float = 0.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> SyntheticSeries:
    """Generate a normalized synthetic signal on t in [0, 1].

    The generators are intentionally small and deterministic under a seed; they
    are meant for fast sweeps of memory behavior, not as final benchmarks.
    """
    if T <= 1:
        raise ValueError("T must be > 1")
    if not (0.0 <= missing_prob < 1.0):
        raise ValueError("missing_prob must be in [0, 1)")
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, T, dtype=np.float64)

    if kind == "smooth":
        phases = rng.uniform(0, 2 * np.pi, size=4)
        amps = np.array([1.0, 0.45, 0.25, 0.15])
        freqs = np.array([1, 2, 4, 7])
        clean = sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in zip(amps, freqs, phases))
    elif kind == "rough":
        phases = rng.uniform(0, 2 * np.pi, size=10)
        freqs = np.arange(1, 11)
        amps = 1.0 / np.sqrt(freqs)
        clean = sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in zip(amps, freqs, phases))
    elif kind == "piecewise":
        clean = np.sin(2 * np.pi * 2 * t)
        clean += (t > 0.35) * 0.7
        clean += (t > 0.65) * (-1.2)
        clean += 0.25 * np.sin(2 * np.pi * 13 * t) * (t > 0.5)
    elif kind == "local_burst":
        clean = 0.4 * np.sin(2 * np.pi * 2 * t)
        burst = np.exp(-0.5 * ((t - 0.82) / 0.045) ** 2)
        clean += 1.8 * burst * np.sin(2 * np.pi * 35 * t)
    else:
        raise ValueError(f"Unknown signal kind {kind}")

    clean = clean / (np.std(clean) + 1e-12)
    observed = clean + noise_std * rng.normal(size=T)
    mask = rng.uniform(size=T) >= missing_prob
    observed_missing = observed.copy()
    observed_missing[~mask] = 0.0

    return SyntheticSeries(
        times=torch.as_tensor(t, dtype=dtype),
        clean=torch.as_tensor(clean, dtype=dtype),
        observed=torch.as_tensor(observed_missing, dtype=dtype),
        mask=torch.as_tensor(mask, dtype=torch.bool),
    )


def sample_gp_rbf(
    times: np.ndarray,
    *,
    lengthscale: float,
    seed: int = 0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Sample a zero-mean RBF-kernel GP at the given (possibly irregular) times.

    ``lengthscale`` is in the same units as ``times``. The sample is normalized
    to unit standard deviation so noise levels are comparable across draws.
    """
    times = np.asarray(times, dtype=np.float64)
    rng = np.random.default_rng(seed)
    d = times[:, None] - times[None, :]
    K = amplitude**2 * np.exp(-0.5 * (d / lengthscale) ** 2)
    K[np.diag_indices_from(K)] += 1e-10 * amplitude**2
    L = np.linalg.cholesky(K)
    f = L @ rng.standard_normal(len(times))
    return f / (np.std(f) + 1e-12)
