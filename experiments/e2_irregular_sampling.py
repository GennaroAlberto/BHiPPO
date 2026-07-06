"""E2 - Irregular sampling & missing blocks.

GP signals observed on-grid but irregularly: (a) Poisson/Bernoulli thinning,
(b) contiguous missing blocks, (c) bursty clusters with gaps. Baselines run on
an imputed uniform grid (zero-order hold and linear interpolation); the SRIF
consumes the irregular times natively. Metrics are stratified by distance to
the nearest observation -- the interesting regime is inside the gaps, where a
calibrated method must inflate its variance.

    PYTHONPATH=. python experiments/e2_irregular_sampling.py --quick --out runs/e2_quick
    PYTHONPATH=. python experiments/e2_irregular_sampling.py --out runs/e2_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bayeshippo.basis import reconstruct_legs
from bayeshippo.data import sample_gp_rbf
from bayeshippo.hippo import hippo_scan
from bayeshippo.srif import LegSTransportFilter
from bayeshippo.sweep import append_jsonl, grid_product
from bayeshippo.unhippo import UnHiPPOFilter

DIST_BINS = [(0.0, 0.5, "observed"), (0.5, 5.5, "d1_5"), (5.5, 20.5, "d6_20"), (20.5, 50.5, "d21_50"), (50.5, np.inf, "d50p")]


def make_mask(pattern: str, T: int, missing_frac: float, rng: np.random.Generator) -> np.ndarray:
    if pattern == "poisson":
        mask = rng.uniform(size=T) >= missing_frac
    elif pattern == "blocks":
        mask = np.ones(T, bool)
        n_blocks = 4
        total_missing = int(missing_frac * T)
        lengths = rng.multinomial(total_missing, np.ones(n_blocks) / n_blocks)
        for ln in lengths:
            if ln == 0:
                continue
            start = rng.integers(T // 20, T - ln)  # keep a little data at the start
            mask[start : start + ln] = False
    elif pattern == "bursty":
        mask = np.zeros(T, bool)
        n_clusters = 8
        keep = int((1 - missing_frac) * T)
        width = max(keep // n_clusters, 2)
        centers = np.sort(rng.choice(np.arange(width, T - width), size=n_clusters, replace=False))
        for c in centers:
            mask[c - width // 2 : c + width // 2] = True
    else:
        raise ValueError(pattern)
    mask[0] = True  # anchor: at least one early observation
    if mask.sum() < 8:
        mask[rng.choice(T, 8, replace=False)] = True
    return mask


def impute(tobs: np.ndarray, yobs: np.ndarray, tgrid: np.ndarray, how: str) -> np.ndarray:
    if how == "linear":
        return np.interp(tgrid, tobs, yobs)
    if how == "zoh":
        idx = np.clip(np.searchsorted(tobs, tgrid, side="right") - 1, 0, len(tobs) - 1)
        return yobs[idx]
    raise ValueError(how)


def stratified(mean, var, clean, dist) -> list[dict]:
    out = []
    for lo, hi, name in [(-1.0, np.inf, "all")] + DIST_BINS:
        sel = (dist > lo) & (dist <= hi) if np.isfinite(hi) else (dist > lo)
        if name != "all":
            sel = (dist >= lo) & (dist < hi)
        if sel.sum() == 0:
            continue
        row = {"stratum": name, "n_points": int(sel.sum()), "mse": float(np.mean((mean[sel] - clean[sel]) ** 2))}
        if var is not None:
            v = np.maximum(var[sel], 1e-12)
            e = clean[sel] - mean[sel]
            row["nll"] = float(np.mean(0.5 * (np.log(2 * np.pi * v) + e**2 / v)))
            row["cov90"] = float(np.mean(np.abs(e) <= 1.645 * np.sqrt(v)))
            row["cov95"] = float(np.mean(np.abs(e) <= 1.96 * np.sqrt(v)))
            row["mean_std"] = float(np.mean(np.sqrt(v)))
        out.append(row)
    return out


def run_one(cfg: dict, unhippo_sigma2s: list[float]) -> list[dict]:
    T, N = int(cfg["T"]), int(cfg["N"])
    noise_std = float(cfg["noise_std"])
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    tgrid = np.arange(1, T + 1, dtype=np.float64)
    clean = sample_gp_rbf(tgrid / T, lengthscale=float(cfg["lengthscale"]), seed=seed)
    noisy = clean + noise_std * rng.standard_normal(T)

    mask = make_mask(cfg["pattern"], T, float(cfg["missing_frac"]), rng)
    tobs, yobs = tgrid[mask], noisy[mask]
    dist = np.abs(tgrid[:, None] - tobs[None, :]).min(axis=1)

    rows = []

    def add(method, mean, var, extra=None):
        for srow in stratified(mean, var, clean, dist):
            rows.append({**cfg, "method": method, "n_obs": int(mask.sum()), **srow, **(extra or {})})

    # deterministic LegS on imputed grids
    for how in ("zoh", "linear"):
        yimp = impute(tobs, yobs, tgrid, how)
        c = hippo_scan(torch.as_tensor(yimp), N, "legs", return_sequence=False)
        rec = reconstruct_legs(c, torch.as_tensor(tgrid), final_time=float(T)).numpy()
        add(f"legs_{how}", rec, None)

    # UnHiPPO on imputed grids
    for how in ("zoh", "linear"):
        yimp = impute(tobs, yobs, tgrid, how)
        for s2 in unhippo_sigma2s:
            uf = UnHiPPOFilter(N, sigma2=s2)
            uf.run(tgrid, yimp)
            mean, var = uf.reconstruct(tgrid)
            add(f"unhippo_{how}", mean, var, {"sigma2_obs": s2})

    # UnHiPPO on the raw irregular times (no imputation; exact propagator
    # handles uneven gaps) -- the strongest fair variant of the baseline.
    for s2 in unhippo_sigma2s:
        uf = UnHiPPOFilter(N, sigma2=s2)
        uf.run(tobs, yobs)
        mean, var = uf.reconstruct(tgrid)
        add("unhippo_native", mean, var, {"sigma2_obs": s2})

    # SRIF natively on irregular times: headline (fixed smoothness prior p=1)
    # plus prior-convention ablations.
    srif_variants = {
        "srif": {"prior": "fixed", "prior_decay": 1.0},
        "srif_flat": {"prior": "fixed", "prior_decay": 0.0},
        "srif_transported": {"prior": "transported", "prior_decay": 0.0},
    }
    for name, kw in srif_variants.items():
        f = LegSTransportFilter(N, tau2=1.0, sigma2_init=1.0, learn_sigma2=True, burn_in=20, **kw)
        f.run(tobs, yobs)
        if f.t < float(T):  # data-free domain extension to the evaluation horizon
            f.transport(float(T))
        mean, var = f.reconstruct(tgrid)
        add(name, mean, var, {"sigma2_hat": f.sigma2})

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="runs/e2")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.jsonl"
    if result_path.exists():
        result_path.unlink()

    if args.quick:
        grid = {
            "seed": [0, 1, 2],
            "T": [1000],
            "N": [32],
            "lengthscale": [0.1],
            "noise_std": [0.25],
            "pattern": ["poisson", "blocks", "bursty"],
            "missing_frac": [0.5],
        }
        unhippo_sigma2s = [10.0**k for k in range(4, 13, 2)]
    else:
        grid = {
            "seed": list(range(5)),
            "T": [1000],
            "N": [32],
            "lengthscale": [0.1, 0.2],
            "noise_std": [0.1, 0.25],
            "pattern": ["poisson", "blocks", "bursty"],
            "missing_frac": [0.3, 0.5, 0.7],
        }
        unhippo_sigma2s = [10.0**k for k in range(4, 14)]

    rows = []
    for cfg in grid_product(grid):
        for row in run_one(cfg, unhippo_sigma2s):
            rows.append(row)
            append_jsonl(result_path, row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    # summary on the gap-interior stratum and overall; UnHiPPO at its best
    # sigma^2 per (pattern, missing_frac, stratum) cell, chosen by gap MSE.
    keys = ["pattern", "missing_frac", "noise_std", "stratum"]
    is_un = df.method.str.startswith("unhippo")
    un = df[is_un].groupby(keys + ["method", "sigma2_obs"], as_index=False).mean(numeric_only=True)
    un_best = un.loc[un.groupby(keys + ["method"])["mse"].idxmin()]
    un_best = un_best.assign(method=un_best.method + "_best")
    rest = df[~is_un].groupby(keys + ["method"], as_index=False).mean(numeric_only=True)
    summary = pd.concat([rest, un_best], ignore_index=True).sort_values(keys + ["method"])
    cols = keys + ["method", "n_points", "mse", "nll", "cov90", "cov95", "mean_std", "sigma2_obs", "sigma2_hat"]
    summary = summary[[c for c in cols if c in summary.columns]]
    summary.to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps({"quick": args.quick, "grid": grid, "unhippo_sigma2s": unhippo_sigma2s}, indent=2))

    show = summary[summary.stratum.isin(["all", "d21_50", "d50p"])]
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(show.to_string(index=False))
    print(f"\nWrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
