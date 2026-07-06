"""E5 - Ablations & engineering measurements.

Three compact studies, each written to its own CSV under --out:

1. evidence_mass.csv: sampling density x {1,4,16,64} on the same signal.
   Deterministic LegS coefficients are invariant (normalized measure);
   SRIF posterior trace(P) contracts like 1/alpha (log-log slope -1).
   This is the paper's Proposition-pair figure.
2. wallclock.csv: per-step wall-clock vs N in {16,32,64,128} for the
   deterministic scan, UnHiPPO Kalman step, and SRIF step.
3. float32.csv: float32 vs float64 reconstruction divergence across
   (N, gap-ratio) -- the stability envelope for single-precision transport.

    PYTHONPATH=. python experiments/e5_ablations.py --out runs/e5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bayeshippo.data import sample_gp_rbf
from bayeshippo.hippo import hippo_scan
from bayeshippo.srif import LegSTransportFilter
from bayeshippo.unhippo import UnHiPPOFilter


def evidence_mass_study(out: Path, seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    rows = []
    base_T = 250
    alphas = [1, 4, 16, 64]
    N = 16
    sigma = 0.1
    for seed in seeds:
        rng = np.random.default_rng(seed)
        # one underlying function per seed, sampled at the finest grid and
        # subsampled for lower densities so all alphas see the same signal
        T_max = base_T * max(alphas)
        t_max_grid = np.arange(1, T_max + 1, dtype=np.float64)
        clean_max = sample_gp_rbf(t_max_grid / T_max, lengthscale=0.2, seed=seed)
        c_ref = None
        for alpha in alphas:
            stride = max(alphas) // alpha
            tgrid = t_max_grid[stride - 1 :: stride] / stride  # rescale to 1..T
            clean = clean_max[stride - 1 :: stride]
            T = len(tgrid)
            y = clean + sigma * rng.standard_normal(T)

            # deterministic: normalized projection, invariant to density
            c_det = hippo_scan(torch.as_tensor(clean), N, "legs", return_sequence=False).numpy()
            if alpha == alphas[0]:
                c_ref = c_det
            det_drift = float(np.abs(c_det - c_ref).max())

            f = LegSTransportFilter(N, tau2=1.0, prior="fixed", sigma2_init=sigma**2, learn_sigma2=False)
            f.run(tgrid, y)
            _, P = f.posterior()
            rows.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "n_obs": T,
                    "det_coeff_drift_vs_alpha1": det_drift,
                    "srif_trace_P": float(np.trace(P)),
                    "srif_evidence_mass": f.sw,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "evidence_mass.csv", index=False)
    # log-log slope of trace(P) vs alpha
    g = df.groupby("alpha")["srif_trace_P"].mean()
    slope = np.polyfit(np.log(g.index.values.astype(float)), np.log(g.values), 1)[0]
    print(f"[evidence mass] mean det drift: {df.det_coeff_drift_vs_alpha1.max():.2e}; "
          f"trace(P) log-log slope vs alpha: {slope:.3f} (theory: -1)")
    return df


def wallclock_study(out: Path, T=2000) -> pd.DataFrame:
    rows = []
    for N in (16, 32, 64, 128):
        tgrid = np.arange(1, T + 1, dtype=np.float64)
        y = np.sin(2 * np.pi * 3 * tgrid / T)

        t0 = time.perf_counter()
        hippo_scan(torch.as_tensor(y), N, "legs", return_sequence=False)
        det_ms = (time.perf_counter() - t0) / T * 1e3

        uf = UnHiPPOFilter(N, sigma2=1e8)
        t0 = time.perf_counter()
        uf.run(tgrid, y)
        un_ms = (time.perf_counter() - t0) / T * 1e3

        f = LegSTransportFilter(N, prior="fixed", prior_decay=1.0, sigma2_init=1.0)
        t0 = time.perf_counter()
        f.run(tgrid, y)
        srif_ms = (time.perf_counter() - t0) / T * 1e3

        rows.append({"N": N, "det_ms_per_step": det_ms, "unhippo_ms_per_step": un_ms, "srif_ms_per_step": srif_ms})
        print(f"[wallclock] N={N}: det {det_ms:.3f} ms, unhippo {un_ms:.3f} ms, srif {srif_ms:.3f} ms")
    df = pd.DataFrame(rows)
    df.to_csv(out / "wallclock.csv", index=False)
    return df


def float32_study(out: Path, seeds=(0, 1, 2)) -> pd.DataFrame:
    """Single mid-sequence gap of ratio rho; float32 vs float64 reconstruction."""
    rows = []
    for N in (16, 32, 64):
        for rho in (2.0, 8.0, 32.0, 128.0):
            for seed in seeds:
                rng = np.random.default_rng(seed)
                # 200 obs on [1, 100], gap, 200 obs on [100*rho, 101*rho]
                t1 = np.linspace(1.0, 100.0, 200)
                t2 = np.linspace(100.0 * rho, 101.0 * rho, 200)
                ts = np.concatenate([t1, t2])
                ys = np.sin(0.05 * ts) + 0.1 * rng.standard_normal(len(ts))

                recs = {}
                ok = {}
                for dtype in (np.float64, np.float32):
                    f = LegSTransportFilter(
                        N, prior="fixed", prior_decay=1.0, sigma2_init=0.01, learn_sigma2=False, dtype=dtype
                    )
                    try:
                        f.run(ts, ys)
                        xs = np.linspace(1.0, f.t, 500)
                        m, _ = f.reconstruct(xs)
                        recs[dtype] = m.astype(np.float64)
                        ok[dtype] = bool(np.isfinite(m).all())
                    except Exception:
                        recs[dtype] = None
                        ok[dtype] = False
                if recs[np.float64] is not None and recs[np.float32] is not None:
                    gap = float(np.abs(recs[np.float64] - recs[np.float32]).max())
                else:
                    gap = np.inf
                rows.append({"seed": seed, "N": N, "gap_ratio": rho, "f32_ok": ok[np.float32], "f64_ok": ok[np.float64], "max_recon_diff_f32_f64": gap})
    df = pd.DataFrame(rows)
    df.to_csv(out / "float32.csv", index=False)
    piv = df.groupby(["N", "gap_ratio"])["max_recon_diff_f32_f64"].median().unstack()
    print("[float32] median |recon_f32 - recon_f64| by (N, gap ratio):")
    print(piv.to_string(float_format=lambda v: f"{v:.2e}"))
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="runs/e5")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    evidence_mass_study(out)
    wallclock_study(out)
    float32_study(out)
    (out / "config.json").write_text(json.dumps({"studies": ["evidence_mass", "wallclock", "float32"]}, indent=2))
    print(f"Wrote E5 CSVs to {out}")


if __name__ == "__main__":
    main()
