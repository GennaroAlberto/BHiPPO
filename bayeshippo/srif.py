"""Exact online Bayesian HiPPO-LegS via sufficient-statistic transport.

This implements the square-root information filter (SRIF) from the paper's
transport theorem. The carried state is the *data* sufficient statistic

    Ghat = sum_i w_i phi_t(s_i) phi_t(s_i)^T   (as a lower-triangular factor L)
    bhat = sum_i w_i phi_t(s_i) y_i
    sw   = sum_i w_i                            (evidence mass)

expressed in the current-time frame. Between observations the statistics are
transported exactly with T = expm(-log(t'/t) (A - I)), which is
lower-triangular and contracting (diag (t/t')^n). At an observation the
right-endpoint identity phi_t(t) = B makes the data injection a rank-one
Cholesky update with the constant vector B.

Priors and the observation noise sigma^2 are folded in only at query time, in
one of two conventions:

- "transported" (a, default): the Gaussian prior N(0, tau^2 I) is placed in
  the frame of the first observation and transported with the data. The
  implied posterior over the *function* is then invariant under domain
  extension (transport with no data changes nothing on [0, t]).
- "fixed" (b): the prior N(0, tau^2 I) applies to the coefficients in the
  current frame at query time.

Frame rule: everything is evaluated in the current-time frame. Design vectors
phi_t(x) for x <= t are bounded by sqrt(2n+1); backward transport explodes
like (2 t/x)^n and is never performed.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, expm, solve_triangular

PriorConvention = Literal["transported", "fixed"]


def legs_matrices(N: int, dtype=np.float64) -> Tuple[np.ndarray, np.ndarray]:
    """HiPPO-LegS (A, B) in the positive paper convention."""
    A = np.zeros((N, N), dtype=dtype)
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = np.sqrt((2 * n + 1) * (2 * k + 1))
            elif n == k:
                A[n, k] = n + 1
    B = np.sqrt(2 * np.arange(N, dtype=dtype) + 1)
    return A, B


def legs_basis_matrix(z: np.ndarray, N: int, dtype=np.float64) -> np.ndarray:
    """Rows phi_n = sqrt(2n+1) P_n(z) for z = 2x/t - 1, vectorized over z."""
    z = np.asarray(z, dtype=dtype)
    out = np.empty(z.shape + (N,), dtype=dtype)
    out[..., 0] = 1.0
    if N > 1:
        out[..., 1] = z
    for n in range(1, N - 1):
        out[..., n + 1] = ((2 * n + 1) * z * out[..., n] - n * out[..., n - 1]) / (n + 1)
    return out * np.sqrt(2 * np.arange(N, dtype=dtype) + 1)


def bvec(rho: float, N: int, dtype=np.float64) -> np.ndarray:
    """Closed-form injection rho^{-(A-I)} B = sqrt(2n+1) P_n(2/rho - 1), O(N)."""
    return legs_basis_matrix(np.asarray(2.0 / rho - 1.0, dtype=dtype), N, dtype=dtype)


def cholupdate(L: np.ndarray, x: np.ndarray) -> None:
    """In-place rank-one update L <- chol(L L^T + x x^T), Givens form.

    The Givens formulation has no division by the diagonal, so it handles
    rank-deficient factors (e.g. L = 0 at initialization) exactly.
    """
    x = x.copy()
    n = L.shape[0]
    for k in range(n):
        g = np.hypot(L[k, k], x[k])
        if g == 0.0:
            continue
        c = L[k, k] / g
        s = x[k] / g
        L[k, k] = g
        if k + 1 < n:
            col = L[k + 1 :, k].copy()
            tail = x[k + 1 :]
            L[k + 1 :, k] = c * col + s * tail
            x[k + 1 :] = c * tail - s * col


class LegSTransportFilter:
    """Exact online Bayesian HiPPO-LegS square-root information filter.

    Args:
        N: memory order.
        tau2: prior coefficient variance tau^2.
        sigma2_init: initial observation-noise variance guess.
        learn_sigma2: empirical-Bayes update of sigma^2. The default "rice"
            method is the model-free robust first-difference (Rice/MAD)
            estimator sigma^2 = median(|y_k - y_{k-1}|)^2 / (2 * 0.6745^2);
            "covmatch" is innovation covariance matching
            e = (y - yhat)^2 - B^T P^- B (model-based, biased when the
            polynomial model over-disperses at the endpoint).
        burn_in: observations before the estimator replaces sigma2_init.
        freeze_after: if set, stop updating sigma^2 after this many observations.
        prior: prior convention, "transported" (a) or "fixed" (b).
        prior_decay: exponent p of the diagonal smoothness prior
            Lambda_0 = diag((2n+1)^p) / tau^2. p = 0 is the flat N(0, tau^2 I)
            prior; p = 1 shrinks high-order coefficients like the projection
            coefficients of a bounded function and tames polynomial
            oscillation inside long observation gaps.
        rho_max: maximum transport ratio per sub-step (numerical guard).
        prior_floor: identity floor added to the prior information matrix in
            the transported convention (guards underflow of the transported
            prior over extreme time ratios; 0 disables).
        dtype: np.float64 (default) or np.float32 to probe the stability envelope.
    """

    def __init__(
        self,
        N: int,
        *,
        tau2: float = 1.0,
        sigma2_init: float = 1e-2,
        learn_sigma2: bool = True,
        sigma2_method: Literal["rice", "covmatch"] = "rice",
        burn_in: int = 20,
        freeze_after: Optional[int] = None,
        prior: PriorConvention = "transported",
        prior_decay: float = 0.0,
        rho_max: float = 2.0,
        prior_floor: float = 0.0,
        dtype=np.float64,
    ) -> None:
        if N <= 0:
            raise ValueError("N must be positive")
        if rho_max <= 1.0:
            raise ValueError("rho_max must exceed 1")
        self.N = N
        self.tau2 = float(tau2)
        self.prior = prior
        self.prior_decay = float(prior_decay)
        self.rho_max = float(rho_max)
        self.prior_floor = float(prior_floor)
        self.dtype = dtype

        A, B = legs_matrices(N, dtype=np.float64)
        self.Atilde = (A - np.eye(N)).astype(dtype)
        self.B = B.astype(dtype)

        self.L = np.zeros((N, N), dtype=dtype)
        self.b = np.zeros(N, dtype=dtype)
        self.t: Optional[float] = None
        self.t0: Optional[float] = None
        self.sw = 0.0
        self.nobs = 0

        self.learn_sigma2 = learn_sigma2
        self.sigma2_method = sigma2_method
        self.burn_in = int(burn_in)
        self.freeze_after = freeze_after
        self.sigma2 = float(sigma2_init)
        self.sigma2_clamp = (1e-6, 1e6)
        self._burn_ys: list[float] = []
        self._innov_sum = 0.0
        self._innov_count = 0
        self._last_y: Optional[float] = None
        self._diff_buffer: list[float] = []
        self.sigma2_trajectory: list[float] = []

        self._expm_cache: dict[float, np.ndarray] = {}

    # ---------------- transport ----------------

    def _propagator(self, log_rho: float) -> np.ndarray:
        T = self._expm_cache.get(log_rho)
        if T is None:
            T = expm(-log_rho * self.Atilde.astype(np.float64)).astype(self.dtype)
            T = np.tril(T)  # exactly lower-triangular by theory
            if len(self._expm_cache) < 4096:
                self._expm_cache[log_rho] = T
        return T

    def transport(self, t_new: float) -> None:
        """Advance the frame to t_new with no data (exact, sub-stepped)."""
        if self.t is None:
            raise RuntimeError("filter has no reference time yet")
        if t_new < self.t:
            raise ValueError("never transport backwards (frame rule)")
        if t_new == self.t:
            return
        log_rho = np.log(t_new / self.t)
        n_sub = max(1, int(np.ceil(log_rho / np.log(self.rho_max))))
        T = self._propagator(log_rho / n_sub)
        for _ in range(n_sub):
            self.L = np.tril(T @ self.L)
            self.b = T @ self.b
        self.t = float(t_new)

    # ---------------- posterior algebra ----------------

    def _information_cholesky(self) -> np.ndarray:
        """Cholesky factor of Lambda = prior_info + Ghat / sigma^2."""
        G = self.L @ self.L.T
        n = np.arange(self.N, dtype=self.dtype)
        lam0 = np.diag((2.0 * n + 1.0) ** self.prior_decay) / self.tau2
        if self.prior == "fixed" or self.t is None or self.t0 is None:
            prior_info = lam0
        else:
            # precision transports by congruence: Lambda' = T Lambda_0 T^T
            T0 = self._propagator(float(np.log(self.t / self.t0)))
            prior_info = T0 @ lam0 @ T0.T
            if self.prior_floor > 0.0:
                prior_info = prior_info + self.prior_floor * np.eye(self.N, dtype=self.dtype)
        lam = prior_info + G / self.sigma2
        # The transported prior decays like rho^{-2n}, so along directions with
        # no data yet Lambda can dip below roundoff of its largest eigenvalues
        # and lose numerical positive-definiteness. Jitter only on failure so
        # the exact (jitter-free) posterior is returned whenever it exists.
        try:
            return np.linalg.cholesky(lam)
        except np.linalg.LinAlgError:
            eye = np.eye(self.N, dtype=self.dtype)
            jitter = 1e-15 * np.trace(lam) / self.N
            for _ in range(12):
                try:
                    return np.linalg.cholesky(lam + jitter * eye)
                except np.linalg.LinAlgError:
                    jitter *= 10.0
            raise

    def posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean and covariance of the coefficients (current frame)."""
        Lc = self._information_cholesky()
        m = cho_solve((Lc, True), self.b / self.sigma2)
        P = cho_solve((Lc, True), np.eye(self.N, dtype=self.dtype))
        return m, P

    def _predict_at_design(self, Phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predictive mean and *function* variance at design rows Phi."""
        Lc = self._information_cholesky()
        m = cho_solve((Lc, True), self.b / self.sigma2)
        Z = solve_triangular(Lc, Phi.T, lower=True)
        return Phi @ m, np.sum(Z * Z, axis=0)

    def predict_observation(self, t_new: Optional[float] = None) -> Tuple[float, float]:
        """Predictive mean and variance of a noisy observation at time t_new.

        Uses the right-endpoint identity phi_t(t) = B, so this costs two
        triangular solves. Does not mutate state unless transport is needed.
        """
        if t_new is not None and self.t is not None and t_new > self.t:
            self.transport(t_new)
        mean, fvar = self._predict_at_design(self.B[None, :])
        return float(mean[0]), float(fvar[0] + self.sigma2)

    def reconstruct(
        self, xs: np.ndarray, *, include_obs_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predictive mean/variance of the represented function at times xs.

        xs must lie in [0, t] (current frame); phi_t(x) is evaluated by the
        Legendre recurrence at z = 2 x / t - 1, entries bounded by sqrt(2n+1).
        """
        if self.t is None:
            raise RuntimeError("no observations yet")
        xs = np.asarray(xs, dtype=self.dtype)
        if np.any(xs > self.t * (1 + 1e-12)) or np.any(xs < 0):
            raise ValueError("reconstruction points must lie in [0, t]")
        Phi = legs_basis_matrix(2.0 * xs / self.t - 1.0, self.N, dtype=self.dtype)
        mean, var = self._predict_at_design(Phi)
        if include_obs_noise:
            var = var + self.sigma2
        return mean, var

    # ---------------- data path ----------------

    def _update_sigma2(self, y: float, yhat: float, s: float) -> None:
        if not self.learn_sigma2:
            self.sigma2_trajectory.append(self.sigma2)
            return
        if self.freeze_after is not None and self.nobs > self.freeze_after:
            self.sigma2_trajectory.append(self.sigma2)
            return
        lo, hi = self.sigma2_clamp

        if self.sigma2_method == "rice":
            # model-free robust first-difference estimator: for consecutive
            # observations y_k - y_{k-1} ~ N(signal diff, 2 sigma^2); the
            # median of |diffs| resists the few long-gap pairs where the
            # signal term dominates. |N(0, 2s2)| has median 0.6745*sqrt(2s2).
            if self._last_y is not None:
                self._diff_buffer.append(abs(y - self._last_y))
                if len(self._diff_buffer) > 4096:
                    self._diff_buffer = self._diff_buffer[-2048:]
                if len(self._diff_buffer) >= max(self.burn_in, 8):
                    med = float(np.median(self._diff_buffer))
                    self.sigma2 = float(np.clip(med**2 / (2 * 0.6745**2), lo, hi))
            self._last_y = float(y)
        else:  # covmatch
            if self.nobs <= self.burn_in:
                self._burn_ys.append(y)
                if self.nobs == self.burn_in and len(self._burn_ys) >= 3:
                    d = np.diff(np.asarray(self._burn_ys))
                    self.sigma2 = float(np.clip(0.5 * np.mean(d * d), lo, hi))
            else:
                # covariance matching: E[(y - yhat)^2] = B^T P^- B + sigma^2,
                # so e = (y - yhat)^2 - B^T P^- B is unbiased for sigma^2 when
                # P is well-specified. Only accumulate in the data-dominated
                # regime; biased low when the polynomial model over-disperses
                # at the right endpoint (see E1 notes) -- kept as an ablation.
                s_f = s - self.sigma2
                if s_f <= self.sigma2:
                    self._innov_sum += (y - yhat) ** 2 - s_f
                    self._innov_count += 1
                    self.sigma2 = float(np.clip(self._innov_sum / max(self._innov_count, 1), lo, hi))
        self.sigma2_trajectory.append(self.sigma2)

    def step(self, t_new: float, y: float, w: float = 1.0) -> Tuple[float, float]:
        """Assimilate one weighted observation at absolute time t_new > 0.

        Returns the one-step predictive (mean, variance) for y computed
        *before* the update (for scoring and empirical Bayes).
        """
        if t_new <= 0:
            raise ValueError("LegS times must be positive")
        if w < 0:
            raise ValueError("weights must be nonnegative")
        if self.t is None:
            self.t = float(t_new)
            self.t0 = float(t_new)
        else:
            self.transport(float(t_new))

        yhat, s = self.predict_observation()

        cholupdate(self.L, np.sqrt(np.asarray(w, dtype=self.dtype)) * self.B)
        self.b = self.b + w * self.B * y
        self.sw += float(w)
        self.nobs += 1
        self._update_sigma2(float(y), yhat, s)
        return yhat, s

    def run(
        self, times: np.ndarray, ys: np.ndarray, ws: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter a whole sequence; returns one-step predictive means/vars."""
        times = np.asarray(times, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        if ws is None:
            ws = np.ones_like(ys)
        preds = np.empty_like(ys)
        pvars = np.empty_like(ys)
        for i in range(len(ys)):
            preds[i], pvars[i] = self.step(times[i], ys[i], ws[i])
        return preds, pvars
