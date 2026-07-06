"""P0.2: transport identities from scripts/verify_transport.py as pytest checks.

Numpy-only. These are the ground-truth identities the SRIF filter relies on;
they must stay green forever.
"""

import numpy as np
from numpy.polynomial.legendre import legval
from scipy.linalg import expm

TOL = 1e-8
N = 8


def _legs_A(N: int) -> np.ndarray:
    A = np.zeros((N, N))
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = np.sqrt((2 * n + 1) * (2 * k + 1))
            elif n == k:
                A[n, k] = n + 1
    return A


A = _legs_A(N)
B = np.sqrt(2 * np.arange(N) + 1)
At = A - np.eye(N)


def phi(t: float, x: float, N: int) -> np.ndarray:
    """LegS basis phi_n(t, x) = sqrt(2n+1) P_n(2x/t - 1)."""
    z = 2 * x / t - 1
    out = np.zeros(N)
    for n in range(N):
        c = np.zeros(n + 1)
        c[n] = 1
        out[n] = np.sqrt(2 * n + 1) * legval(z, c)
    return out


def bvec(rho: float, N: int) -> np.ndarray:
    """Closed form rho^{-At} B = sqrt(2n+1) P_n(2/rho - 1), O(N)."""
    z = 2.0 / rho - 1.0
    P = np.zeros(N)
    P[0] = 1.0
    if N > 1:
        P[1] = z
    for n in range(1, N - 1):
        P[n + 1] = ((2 * n + 1) * z * P[n] - n * P[n - 1]) / (n + 1)
    return np.sqrt(2 * np.arange(N) + 1) * P


def test_consistency_identity():
    assert np.abs(B[:, None] * B[None, :] - At - At.T - np.eye(N)).max() < TOL


def test_basis_time_derivative():
    t, x, h = 3.7, 1.234, 1e-6
    fd = (phi(t + h, x, N) - phi(t - h, x, N)) / (2 * h)
    an = -(1 / t) * At @ phi(t, x, N)
    assert np.abs(fd - an).max() < 1e-4  # finite-difference limited


def test_finite_transport_and_lower_triangular():
    t, t2, x = 3.7, 6.1, 1.234
    T = expm(-np.log(t2 / t) * At)
    assert np.abs(T @ phi(t, x, N) - phi(t2, x, N)).max() < TOL
    assert np.abs(np.triu(T, 1)).max() < TOL
    # diagonal is (t/t')^n
    assert np.abs(np.diag(T) - (t / t2) ** np.arange(N)).max() < TOL


def test_right_endpoint_is_B():
    for t in (0.3, 1.0, 3.7, 42.0):
        assert np.abs(phi(t, t, N) - B).max() < TOL


def test_recurrence_matches_batch_statistics_and_posterior():
    rng = np.random.default_rng(0)
    s = np.sort(rng.uniform(0.5, 10.0, 40))
    y = np.sin(s) + 0.1 * rng.standard_normal(40)
    w = rng.uniform(0.5, 2.0, 40)
    tf = s[-1]
    Phi = np.stack([phi(tf, si, N) for si in s])
    G_batch = (Phi * w[:, None]).T @ Phi
    b_batch = Phi.T @ (w * y)

    G = w[0] * np.outer(B, B)
    b = w[0] * B * y[0]
    for k in range(1, len(s)):
        Tk = expm(-np.log(s[k] / s[k - 1]) * At)
        G = Tk @ G @ Tk.T + w[k] * np.outer(B, B)
        b = Tk @ b + w[k] * B * y[k]

    assert np.abs(G - G_batch).max() / np.abs(G_batch).max() < TOL
    assert np.abs(b - b_batch).max() / np.abs(b_batch).max() < TOL

    tau2, sig2 = 1.0, 0.01
    m_rec = np.linalg.solve(np.eye(N) / tau2 + G / sig2, b / sig2)
    m_bat = np.linalg.solve(np.eye(N) / tau2 + G_batch / sig2, b_batch / sig2)
    assert np.abs(m_rec - m_bat).max() < TOL


def test_closed_form_injection():
    for rho in (1.3, 2.0, 5.0, 20.0):
        ref = expm(-np.log(rho) * At) @ B
        err = np.abs(bvec(rho, N) - ref).max() / np.abs(ref).max()
        assert err < TOL, (rho, err)


def test_frame_conditioning():
    # forward (current-frame) design vectors are bounded by sqrt(2N-1)
    for rho in (1.5, 3.0, 10.0):
        assert np.abs(bvec(rho, N)).max() <= np.sqrt(2 * N - 1) + TOL
    # backward transport explodes -- documents why it is forbidden
    assert np.abs(bvec(1 / 1.5, N)).max() > 1e3
