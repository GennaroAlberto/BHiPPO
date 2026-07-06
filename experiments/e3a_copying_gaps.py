"""E3a - Copying with random gaps: adaptive-measure mixture vs fixed baselines.

Redesigned after two structural findings (see RESULTS.md):

1. Capacity is a decode problem, not a state problem: the noiseless discrete
   state of ANY covering component retains the prefix exactly (pinv of the
   known LTI map recovers it at 100%). The honest per-component readout is
   therefore the closed-form *conditional decode* D_i(T) = ridge-pinv of
   [Ad_i^{T-k} Bd_i]_k, which uses only the component's own dynamics and the
   known elapsed time T (no training). Under observation noise this decode
   exposes the real capacity limit: conditioning. Matched theta ~ G + K is
   noise-robust; theta >> G collapses to chance (measured: 0.90 vs 0.14 at
   G=16, N=64, sigma=0.05).

2. Identification is impossible on pure copying streams: tokens are iid and
   the gap is zeros, so the stream contains no statistical dependence on the
   past -- eta=1 log-loss BMA on next-value predictive densities provably
   favors the most sluggish component (best predictor of the marginal mean),
   ESS -> 1 on theta_max regardless of the gap. We therefore run two modes:

   - mode "pure": the negative result, reported as such.
   - mode "feedback": the first RECALL_LEN prefix tokens reappear (observed)
     after the gap; the stream now carries evidence about which memory covers
     the gap, and the BMA mass can shift onto covering components. Readout is
     scored only on the held-back second half of the prefix.

Bank: LegT theta in a near-geometric grid [16, 256] + LegS, N = 64,
gap ~ log-uniform [8, 96] with out-of-range probes {128, 192};
observation noise sigma = 0.05; scoring sigma matched.

Methods (per mode): mixture_direct (BMA-weighted conditional decodes),
map_direct (argmax-w), legs_direct, fixed_direct_t<theta> (-> oracle/worst),
mixture_mlp / s4_bank_mlp / legs_mlp (trained heads, reviewer track).

    PYTHONPATH=. python experiments/e3a_copying_gaps.py --quick --out runs/e3a_quick
    PYTHONPATH=. python experiments/e3a_copying_gaps.py --out runs/e3a_full
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bayeshippo.sweep import append_jsonl

VOCAB = 6
PREFIX_LEN = 8
RECALL_LEN = 4  # tokens replayed in feedback mode; readout scored on the rest
GAP_BINS = [(8, 16, "g8_16"), (16, 32, "g16_32"), (32, 64, "g32_64"), (64, 97, "g64_96"), (97, 512, "g_ood")]


# ---------------- bank construction ----------------


def legt_matrices(N: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    n = np.arange(N, dtype=np.float64)
    A = np.zeros((N, N))
    for r in range(N):
        for c in range(N):
            A[r, c] = ((-1.0) ** (r - c) if r >= c else 1.0) * (2 * r + 1) / theta
    B = ((-1.0) ** n) * (2 * n + 1) / theta
    return A, B


def bilinear(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    N = A.shape[0]
    M0 = np.eye(N) - 0.5 * dt * A
    return np.linalg.solve(M0, np.eye(N) + 0.5 * dt * A), np.linalg.solve(M0, dt * B)


def legs_step_matrices(N: int, T_max: int) -> tuple[np.ndarray, np.ndarray]:
    A = np.zeros((N, N))
    for r in range(N):
        for c in range(N):
            if r > c:
                A[r, c] = np.sqrt((2 * r + 1) * (2 * c + 1))
            elif r == c:
                A[r, c] = r + 1
    Bv = np.sqrt(2 * np.arange(N) + 1.0)
    I = np.eye(N)
    Abars = np.empty((T_max + 1, N, N))
    bbars = np.empty((T_max + 1, N))
    for k in range(1, T_max + 1):
        M0 = I + A / (2.0 * k)
        Abars[k] = np.linalg.solve(M0, I - A / (2.0 * k))
        bbars[k] = np.linalg.solve(M0, Bv / k)
    return Abars, bbars


class CopyBank:
    """LegT bank + LegS: online eta=1 BMA + closed-form conditional decodes."""

    def __init__(self, N: int, thetas: list[float], T_max: int, sigma_score: float = 0.05):
        self.N = N
        self.thetas = list(thetas)
        self.M = len(thetas) + 1
        self.sigma2 = sigma_score**2
        Ads, Bds = [], []
        for th in thetas:
            A, B = legt_matrices(N, th)
            Ad, Bd = bilinear(-A, B, dt=1.0)
            Ads.append(Ad)
            Bds.append(Bd)
        self.Ad = np.stack(Ads)
        self.Bd = np.stack(Bds)
        self.legs_Abar, self.legs_bbar = legs_step_matrices(N, T_max)
        self.eval_legt = (-1.0) ** np.arange(N)
        self.eval_legs = np.sqrt(2 * np.arange(N) + 1.0)
        self._legt_pow: list[dict[int, np.ndarray]] = [dict() for _ in thetas]
        self._decode_cache: dict[tuple[int, int], np.ndarray] = {}

    # -- filtering --

    def run(self, y: np.ndarray):
        Tn = len(y)
        states_t = np.zeros((self.M - 1, self.N))
        state_s = np.zeros(self.N)
        cumloss = np.zeros(self.M)
        mix_loss = 0.0
        logw = np.zeros(self.M)
        c0 = 0.5 * np.log(2 * np.pi * self.sigma2)
        for k in range(1, Tn + 1):
            yk = y[k - 1]
            preds = np.empty(self.M)
            preds[:-1] = states_t @ self.eval_legt
            preds[-1] = state_s @ self.eval_legs
            ll = -c0 - 0.5 * (yk - preds) ** 2 / self.sigma2
            wnorm = logw - logw.max()
            mix_loss -= float(np.logaddexp.reduce(wnorm + ll) - np.logaddexp.reduce(wnorm))
            cumloss -= ll
            logw += ll
            states_t = np.einsum("mij,mj->mi", self.Ad, states_t) + yk * self.Bd
            state_s = self.legs_Abar[k] @ state_s + yk * self.legs_bbar[k]
        states = np.vstack([states_t, state_s[None, :]])
        return states, logw, mix_loss - float(cumloss.min())

    # -- closed-form conditional decodes --

    def _apow(self, i: int, p: int) -> np.ndarray:
        cache = self._legt_pow[i]
        if p == 0:
            return np.eye(self.N)
        if p in cache:
            return cache[p]
        half = self._apow(i, p // 2)
        r = half @ half
        if p % 2:
            r = r @ self.Ad[i]
        cache[p] = r
        return r

    def _ridge_decode(self, Wmap: np.ndarray) -> np.ndarray:
        lam = self.N * self.sigma2
        return Wmap.T @ np.linalg.inv(Wmap @ Wmap.T + lam * np.eye(self.N))

    def legt_decoder(self, i: int, T: int) -> np.ndarray:
        """(PREFIX_LEN, N) ridge decode of tokens at steps 1..PREFIX_LEN from c_T."""
        key = (i, T)
        D = self._decode_cache.get(key)
        if D is None:
            Wmap = np.stack([self._apow(i, T - k) @ self.Bd[i] for k in range(1, PREFIX_LEN + 1)], axis=1)
            D = self._ridge_decode(Wmap)
            if len(self._decode_cache) < 100_000:
                self._decode_cache[key] = D
        return D

    def legs_decoder(self, T: int) -> np.ndarray:
        key = (-1, T)
        D = self._decode_cache.get(key)
        if D is None:
            P = np.eye(self.N)
            prods = {}
            for j in range(T, 0, -1):
                prods[j] = P.copy()  # prod Abar_{j+1..T}
                P = P @ self.legs_Abar[j]
            Wmap = np.stack([prods[k] @ self.legs_bbar[k] for k in range(1, PREFIX_LEN + 1)], axis=1)
            D = self._ridge_decode(Wmap)
            if len(self._decode_cache) < 100_000:
                self._decode_cache[key] = D
        return D

    def direct_readout(self, states: np.ndarray, T: int) -> np.ndarray:
        out = np.empty((self.M, PREFIX_LEN))
        for i in range(self.M - 1):
            out[i] = self.legt_decoder(i, T) @ states[i]
        out[-1] = self.legs_decoder(T) @ states[-1]
        return out


# ---------------- data ----------------


def make_sequence(gap: int, rng: np.random.Generator, mode: str, sigma_obs: float):
    tokens = rng.integers(0, VOCAB, PREFIX_LEN)
    values = (tokens - (VOCAB - 1) / 2.0) / ((VOCAB - 1) / 2.0)
    parts = [values, np.zeros(gap)]
    if mode == "feedback":
        parts.append(values[:RECALL_LEN])
    y = np.concatenate(parts)
    y = y + sigma_obs * rng.standard_normal(len(y))
    return y, values, tokens


def sample_gaps(n: int, rng: np.random.Generator, lo=8, hi=96) -> np.ndarray:
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n)).astype(int)


def token_accuracy(values_hat: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    raw = values_hat * ((VOCAB - 1) / 2.0) + (VOCAB - 1) / 2.0
    pred = np.clip(np.round(raw), 0, VOCAB - 1).astype(int)
    return (pred == tokens).mean(axis=1)


def mlp_fit_predict(Xtr, Ytr, Xte, seed=0, hidden=128, epochs=300, lr=1e-3):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_t = torch.as_tensor((Xtr - mu) / sd, dtype=torch.float32)
    Xte_t = torch.as_tensor((Xte - mu) / sd, dtype=torch.float32)
    Ytr_t = torch.as_tensor(Ytr, dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, Ytr.shape[1]),
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.mse_loss(net(Xtr_t), Ytr_t).backward()
        opt.step()
    with torch.no_grad():
        return net(Xte_t).numpy()


# ---------------- experiment ----------------


def run_seed(seed: int, mode: str, sigma_obs: float, n_train: int, n_test: int, N: int,
             thetas: list[float], bank: CopyBank, out_rows: list, result_path: Path):
    rng = np.random.default_rng(seed)
    M = bank.M
    theta_arr = np.array(thetas)
    eval_slice = slice(RECALL_LEN, None) if mode == "feedback" else slice(None)

    def build(n, gaps, rng_local):
        S = np.empty((n, M * N))
        W = np.empty((n, M))
        direct = np.empty((n, M, PREFIX_LEN))
        Y = np.empty((n, PREFIX_LEN))
        toks = np.empty((n, PREFIX_LEN), int)
        diags = []
        for i in range(n):
            y, values, tokens = make_sequence(int(gaps[i]), rng_local, mode, sigma_obs)
            states, logw, regret = bank.run(y)
            w = np.exp(logw - np.logaddexp.reduce(logw))
            S[i] = states.ravel()
            W[i] = w
            direct[i] = bank.direct_readout(states, len(y))
            Y[i] = values
            toks[i] = tokens
            covering = np.append(theta_arr >= gaps[i] + PREFIX_LEN + (RECALL_LEN if mode == "feedback" else 0), True)
            geo_theta = float(np.exp(np.sum(w[:-1] * np.log(theta_arr)) / max(w[:-1].sum(), 1e-12)))
            diags.append({"gap": int(gaps[i]), "mass_covering": float(w[covering].sum()),
                          "geo_theta": geo_theta, "w_legs": float(w[-1]),
                          "ess": float(1.0 / np.sum(w**2)), "regret": regret})
        return S, W, direct, Y, toks, pd.DataFrame(diags)

    gaps_tr = sample_gaps(n_train, rng)
    n_ood = max(n_test // 5, 8)
    gaps_te = np.concatenate([sample_gaps(n_test - n_ood, rng), np.repeat([128, 192], n_ood // 2 + 1)[:n_ood]])
    t0 = time.perf_counter()
    Str, Wtr, _, Ytr, _, _ = build(n_train, gaps_tr, rng)
    Ste, Wte, Dte, Yte, tok_te, diag_te = build(n_test, gaps_te, np.random.default_rng(seed + 10_000))
    bank_secs = time.perf_counter() - t0
    gaps_test = diag_te["gap"].to_numpy()
    logT_tr = np.log(gaps_tr + PREFIX_LEN)[:, None]
    logT_te = np.log(gaps_te + PREFIX_LEN)[:, None]

    preds: dict[str, np.ndarray] = {}
    preds["mixture_direct"] = np.einsum("nm,nmk->nk", Wte, Dte)
    preds["map_direct"] = Dte[np.arange(len(Dte)), Wte.argmax(axis=1)]
    preds["legs_direct"] = Dte[:, -1]
    for j, th in enumerate(thetas):
        preds[f"fixed_direct_t{int(th)}"] = Dte[:, j]
    preds["mixture_mlp"] = mlp_fit_predict(np.hstack([Str, Wtr, logT_tr]), Ytr, np.hstack([Ste, Wte, logT_te]), seed)
    preds["s4_bank_mlp"] = mlp_fit_predict(np.hstack([Str, logT_tr]), Ytr, np.hstack([Ste, logT_te]), seed)
    preds["legs_mlp"] = mlp_fit_predict(np.hstack([Str[:, -N:], logT_tr]), Ytr, np.hstack([Ste[:, -N:], logT_te]), seed)

    for name, pred in preds.items():
        acc = token_accuracy(pred[:, eval_slice], tok_te[:, eval_slice])
        se = (pred[:, eval_slice] - Yte[:, eval_slice]) ** 2
        for lo, hi, bname in GAP_BINS:
            sel = (gaps_test >= lo) & (gaps_test < hi)
            if sel.sum() == 0:
                continue
            row = {"seed": seed, "mode": mode, "method": name, "bin": bname, "n_seq": int(sel.sum()),
                   "acc": float(acc[sel].mean()), "mse": float(se[sel].mean())}
            out_rows.append(row)
            append_jsonl(result_path, row)
    for lo, hi, bname in GAP_BINS:
        sel = (gaps_test >= lo) & (gaps_test < hi)
        if sel.sum() == 0:
            continue
        d = diag_te[sel]
        row = {"seed": seed, "mode": mode, "method": "_diagnostics", "bin": bname, "n_seq": int(sel.sum()),
               "mass_covering": float(d.mass_covering.mean()), "geo_theta": float(d.geo_theta.mean()),
               "w_legs": float(d.w_legs.mean()), "ess": float(d.ess.mean()),
               "regret_max": float(d.regret.max()), "log_M_bound": float(np.log(M)), "bank_secs": bank_secs}
        out_rows.append(row)
        append_jsonl(result_path, row)
    diag_te.assign(mode=mode).to_csv(result_path.parent / f"diag_{mode}_seed{seed}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="runs/e3a")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.jsonl"
    if result_path.exists():
        result_path.unlink()

    thetas = [16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0]
    N = 64
    sigma_obs = 0.05
    if args.quick:
        seeds, n_train, n_test = [0, 1], 300, 150
    else:
        seeds, n_train, n_test = [0, 1, 2, 3, 4], 800, 300

    rows: list[dict] = []
    bank = CopyBank(N, thetas, PREFIX_LEN + RECALL_LEN + 512, sigma_score=sigma_obs)
    for mode in ("pure", "feedback"):
        for seed in seeds:
            run_seed(seed, mode, sigma_obs, n_train, n_test, N, thetas, bank, rows, result_path)
            print(f"mode {mode} seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    summary = df[df.method != "_diagnostics"].groupby(["mode", "method", "bin"], as_index=False)[["acc", "mse"]].mean()
    diag = df[df.method == "_diagnostics"].groupby(["mode", "bin"], as_index=False)[
        ["mass_covering", "geo_theta", "w_legs", "ess", "regret_max", "log_M_bound"]
    ].mean()
    summary.to_csv(out / "summary.csv", index=False)
    diag.to_csv(out / "diagnostics.csv", index=False)
    (out / "config.json").write_text(json.dumps(
        {"quick": args.quick, "thetas": thetas, "N": N, "sigma_obs": sigma_obs, "n_train": n_train,
         "n_test": n_test, "seeds": seeds, "vocab": VOCAB, "prefix_len": PREFIX_LEN,
         "recall_len": RECALL_LEN}, indent=2))

    cols = [b for *_, b in GAP_BINS]
    for mode in ("pure", "feedback"):
        acc_piv = summary[summary["mode"] == mode].pivot(index="method", columns="bin", values="acc")
        fixed = acc_piv[acc_piv.index.str.startswith("fixed_direct_")]
        acc_piv.loc["oracle_fixed_direct"] = fixed.max()
        acc_piv.loc["worst_fixed_direct"] = fixed.min()
        order = (["mixture_direct", "map_direct", "legs_direct", "oracle_fixed_direct", "worst_fixed_direct"]
                 + ["mixture_mlp", "s4_bank_mlp", "legs_mlp"])
        print(f"\n=== mode {mode}: token accuracy by gap bin (chance = {1/VOCAB:.3f}) ===")
        print(acc_piv.loc[[m for m in order if m in acc_piv.index], [c for c in cols if c in acc_piv.columns]].round(3).to_string())
        print(f"\n=== mode {mode}: diagnostics ===")
        print(diag[diag["mode"] == mode].round(3).to_string(index=False))
    print(f"\nWrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
