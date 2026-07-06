"""E1 - Noise robustness vs UnHiPPO (their home turf).

GP samples (RBF, a few lengthscales) + iid Gaussian noise on a uniform grid.
Compared methods:

- ``legs``: deterministic HiPPO-LegS (bilinear scan) -- point estimate, MSE only.
- ``unhippo_<s2>``: UnHiPPO Kalman filter at each sigma^2 in their sweep range;
  its (data-independent) covariance gives NLL/coverage.
- ``srif``: our transport SRIF with empirical-Bayes sigma^2 -- no tuning allowed.
- ``srif_oracle``: SRIF with sigma^2 fixed to the true noise variance (upper bound).

Metrics: final-state reconstruction MSE against the clean signal, Gaussian NLL
and 90/95% empirical coverage of the clean signal under the function-value
variance, wall-clock per step.

    PYTHONPATH=. python experiments/e1_noise_robustness.py --quick --out runs/e1_quick
    PYTHONPATH=. python experiments/e1_noise_robustness.py --out runs/e1_full
"""

from __future__ import annotations

import argparse
import json
import time
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


def gaussian_nll(mean, var, target, eps=1e-12):
    var = np.maximum(var, eps)
    return float(np.mean(0.5 * (np.log(2 * np.pi * var) + (target - mean) ** 2 / var)))


def coverage(mean, var, target, z):
    r = z * np.sqrt(np.maximum(var, 0.0))
    return float(np.mean((target >= mean - r) & (target <= mean + r)))


def eval_method(mean, var, clean):
    out = {"mse": float(np.mean((mean - clean) ** 2))}
    if var is not None:
        out["nll"] = gaussian_nll(mean, var, clean)
        out["cov90"] = coverage(mean, var, clean, 1.645)
        out["cov95"] = coverage(mean, var, clean, 1.96)
    return out


def run_one(cfg: dict, unhippo_sigma2s: list[float]) -> list[dict]:
    T, N = int(cfg["T"]), int(cfg["N"])
    noise_std = float(cfg["noise_std"])
    rng = np.random.default_rng(int(cfg["seed"]))
    tgrid = np.arange(1, T + 1, dtype=np.float64)
    clean = sample_gp_rbf(tgrid / T, lengthscale=float(cfg["lengthscale"]), seed=int(cfg["seed"]))
    y = clean + noise_std * rng.standard_normal(T)

    rows = []

    def add(method, metrics, wall_ms_step, extra=None):
        rows.append({**cfg, "method": method, **metrics, "wall_ms_per_step": wall_ms_step, **(extra or {})})

    # deterministic LegS
    t0 = time.perf_counter()
    c = hippo_scan(torch.as_tensor(y), N, "legs", return_sequence=False)
    wall = (time.perf_counter() - t0) / T * 1e3
    rec = reconstruct_legs(c, torch.as_tensor(tgrid), final_time=float(T)).numpy()
    add("legs", eval_method(rec, None, clean), wall)

    # UnHiPPO sweep
    for s2 in unhippo_sigma2s:
        t0 = time.perf_counter()
        uf = UnHiPPOFilter(N, sigma2=s2)
        uf.run(tgrid, y)
        wall = (time.perf_counter() - t0) / T * 1e3
        mean, var = uf.reconstruct(tgrid)
        add(f"unhippo", eval_method(mean, var, clean), wall, {"sigma2_obs": s2})

    # SRIF empirical Bayes (headline: no tuning; fixed smoothness prior p=1)
    t0 = time.perf_counter()
    f = LegSTransportFilter(
        N, tau2=1.0, prior="fixed", prior_decay=1.0, sigma2_init=1.0, learn_sigma2=True, burn_in=20
    )
    f.run(tgrid, y)
    wall = (time.perf_counter() - t0) / T * 1e3
    mean, var = f.reconstruct(tgrid)
    add("srif", eval_method(mean, var, clean), wall, {"sigma2_hat": f.sigma2})

    # SRIF with oracle noise variance
    f = LegSTransportFilter(
        N, tau2=1.0, prior="fixed", prior_decay=1.0, sigma2_init=max(noise_std**2, 1e-8), learn_sigma2=False
    )
    f.run(tgrid, y)
    mean, var = f.reconstruct(tgrid)
    add("srif_oracle", eval_method(mean, var, clean), None, {"sigma2_hat": f.sigma2})

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="runs/e1")
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
            "noise_std": [0.1, 0.5],
        }
        unhippo_sigma2s = [10.0**k for k in range(4, 13, 2)]
    else:
        grid = {
            "seed": list(range(5)),
            "T": [1000],
            "N": [32, 64],
            "lengthscale": [0.05, 0.15],
            "noise_std": [0.05, 0.1, 0.25, 0.5],
        }
        unhippo_sigma2s = [10.0**k for k in range(4, 15)]

    rows = []
    for cfg in grid_product(grid):
        for row in run_one(cfg, unhippo_sigma2s):
            rows.append(row)
            append_jsonl(result_path, row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    # summary: per (N, lengthscale, noise_std) average over seeds; UnHiPPO at its
    # best sigma^2 per cell (oracle tuning for the baseline, none for SRIF).
    keys = ["N", "lengthscale", "noise_std"]
    un = df[df.method == "unhippo"].groupby(keys + ["sigma2_obs"], as_index=False).mean(numeric_only=True)
    un_best = un.loc[un.groupby(keys)["mse"].idxmin()].assign(method="unhippo_best")
    rest = df[df.method != "unhippo"].groupby(keys + ["method"], as_index=False).mean(numeric_only=True)
    summary = pd.concat([rest, un_best], ignore_index=True).sort_values(keys + ["method"])
    cols = keys + ["method", "mse", "nll", "cov90", "cov95", "sigma2_obs", "sigma2_hat", "wall_ms_per_step"]
    summary = summary[[c for c in cols if c in summary.columns]]
    summary.to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps({"quick": args.quick, "grid": grid, "unhippo_sigma2s": unhippo_sigma2s}, indent=2))

    with pd.option_context("display.width", 200, "display.max_rows", 300):
        print(summary.to_string(index=False))
    print(f"\nWrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
