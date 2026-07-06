"""Acceptance tests for the LegS transport SRIF (EXPERIMENTS_PLAN Phase 1)."""

import numpy as np
import pytest
import torch

from bayeshippo.hippo import hippo_scan
from bayeshippo.srif import LegSTransportFilter, cholupdate, legs_basis_matrix
from bayeshippo.unhippo import UnHiPPOFilter, unhippo_regularized_matrix


def test_cholupdate_matches_direct():
    rng = np.random.default_rng(0)
    L = np.zeros((6, 6))
    acc = np.zeros((6, 6))
    for _ in range(10):
        x = rng.standard_normal(6)
        cholupdate(L, x)
        acc += np.outer(x, x)
        assert np.abs(np.triu(L, 1)).max() == 0.0
        assert np.abs(L @ L.T - acc).max() < 1e-10


def test_matches_batch_posterior_irregular():
    """(i) filter == batch conjugate posterior on random irregular data."""
    rng = np.random.default_rng(1)
    N, M = 12, 60
    s = np.sort(rng.uniform(0.5, 25.0, M))
    y = np.sin(s) + 0.1 * rng.standard_normal(M)
    w = rng.uniform(0.5, 2.0, M)
    tau2, sig2 = 2.0, 0.05

    f = LegSTransportFilter(N, tau2=tau2, sigma2_init=sig2, learn_sigma2=False, prior="fixed")
    f.run(s, y, w)
    m_f, P_f = f.posterior()

    tf = s[-1]
    Phi = legs_basis_matrix(2.0 * s / tf - 1.0, N)
    G = (Phi * w[:, None]).T @ Phi
    b = Phi.T @ (w * y)
    lam = np.eye(N) / tau2 + G / sig2
    m_b = np.linalg.solve(lam, b / sig2)
    P_b = np.linalg.inv(lam)

    assert np.abs(m_f - m_b).max() < 1e-8
    assert np.abs(P_f - P_b).max() < 1e-8
    assert abs(f.sw - w.sum()) < 1e-12


def test_substepping_is_exact():
    """rho_max sub-stepping must not change results (transport is exact)."""
    rng = np.random.default_rng(2)
    N, M = 8, 25
    s = np.sort(rng.uniform(0.1, 300.0, M))  # ratios far beyond rho_max
    y = np.cos(0.1 * s)

    outs = []
    for rho_max in (1.3, 2.0, 1e9):
        f = LegSTransportFilter(N, sigma2_init=0.01, learn_sigma2=False, prior="fixed", rho_max=rho_max)
        f.run(s, y)
        outs.append(f.posterior()[0])
    assert np.abs(outs[0] - outs[1]).max() < 1e-9
    assert np.abs(outs[0] - outs[2]).max() < 1e-9


def test_dense_uniform_flat_prior_recovers_deterministic_legs():
    """(ii) dense uniform grid + flat prior -> deterministic LegS scan state."""
    T, N = 2000, 8
    tgrid = np.arange(1, T + 1, dtype=np.float64)
    x = np.sin(2 * np.pi * 3 * tgrid / T) + 0.5 * np.cos(2 * np.pi * 5 * tgrid / T)

    c_det = hippo_scan(torch.as_tensor(x), N, "legs", return_sequence=False).numpy()

    f = LegSTransportFilter(N, tau2=1e12, sigma2_init=1.0, learn_sigma2=False, prior="fixed")
    f.run(tgrid, x)  # w=1 = dt on the integer grid
    m, _ = f.posterior()

    rel = np.abs(m - c_det).max() / np.abs(c_det).max()
    assert rel < 5e-3, rel


def test_domain_extension_invariance():
    """(iii) transported prior: transport with no data changes nothing on [0, t]."""
    rng = np.random.default_rng(3)
    N, M = 10, 40
    s = np.sort(rng.uniform(1.0, 10.0, M))
    y = np.sin(s)

    f = LegSTransportFilter(N, sigma2_init=0.01, learn_sigma2=False, prior="transported")
    f.run(s, y)
    xs = np.linspace(0.1, f.t, 50)
    mean0, var0 = f.reconstruct(xs)

    f.transport(f.t * 2.3)
    mean1, var1 = f.reconstruct(xs)
    # Invariance is exact in exact arithmetic; the residual is the conditioning
    # of the extended frame (the data design crowds towards z = -1), not bias.
    assert np.abs(mean0 - mean1).max() < 1e-4
    # variance is quadratic in the ill-conditioned solve, so its residual is larger
    assert np.abs(var0 - var1).max() / var0.max() < 2e-3

    # fixed convention is *not* invariant (documents the difference)
    g = LegSTransportFilter(N, sigma2_init=0.01, learn_sigma2=False, prior="fixed")
    g.run(s, y)
    mean0f, _ = g.reconstruct(xs)
    g.transport(g.t * 2.3)
    mean1f, _ = g.reconstruct(xs)
    assert np.abs(mean0f - mean1f).max() > 1e-4


def test_factor_stays_lower_triangular():
    rng = np.random.default_rng(4)
    N = 16
    s = np.sort(rng.uniform(0.5, 200.0, 50))
    f = LegSTransportFilter(N, sigma2_init=0.01, learn_sigma2=False)
    f.run(s, np.sin(s))
    assert np.abs(np.triu(f.L, 1)).max() == 0.0


def test_empirical_bayes_sigma2_recovers_noise_scale():
    rng = np.random.default_rng(5)
    T, N = 1500, 16
    tgrid = np.arange(1, T + 1, dtype=np.float64)
    sigma_true = 0.3
    clean = np.sin(2 * np.pi * 2 * tgrid / T)
    y = clean + sigma_true * rng.standard_normal(T)

    f = LegSTransportFilter(N, sigma2_init=1.0, learn_sigma2=True, burn_in=20)
    f.run(tgrid, y)
    assert 0.5 * sigma_true**2 < f.sigma2 < 2.0 * sigma_true**2, f.sigma2


def test_unhippo_regularized_matrix_shape_and_filter_smooths():
    N = 24
    A_R = unhippo_regularized_matrix(N)
    assert A_R.shape == (N, N)
    assert np.isfinite(A_R).all()

    # Fig. 3 regime: memory order large relative to signal complexity, so
    # deterministic HiPPO tracks the noise while UnHiPPO filters it out.
    rng = np.random.default_rng(6)
    T, N_big = 600, 64
    tgrid = np.arange(1, T + 1, dtype=np.float64)
    clean = np.sin(2 * np.pi * 2 * tgrid / T) + 0.4 * np.cos(2 * np.pi * 5 * tgrid / T)
    y = clean + 0.4 * rng.standard_normal(T)

    # deterministic HiPPO on the noisy data
    from bayeshippo.basis import reconstruct_legs

    c = hippo_scan(torch.as_tensor(y), N_big, "legs", return_sequence=False)
    rec_h = reconstruct_legs(c, torch.as_tensor(tgrid), final_time=float(T)).numpy()
    mse_h = np.mean((rec_h - clean) ** 2)

    best_mse_u = np.inf
    for sigma2 in (1e5, 1e6, 1e7, 1e8, 1e9):
        uf = UnHiPPOFilter(N_big, sigma2=sigma2)
        uf.run(tgrid, y)
        rec_u, _ = uf.reconstruct(tgrid)
        best_mse_u = min(best_mse_u, np.mean((rec_u - clean) ** 2))
    # Fig. 3 qualitative behavior: UnHiPPO visibly denoises vs HiPPO
    assert best_mse_u < mse_h, (best_mse_u, mse_h)
