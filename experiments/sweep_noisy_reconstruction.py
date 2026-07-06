"""Sweep noisy reconstruction for deterministic vs Bayesian HiPPO.

This is the first experiment for the reframed paper: deterministic HiPPO fixes a
normalized relevance measure and returns a point estimate. Bayesian HiPPO sees
both relevance and evidence mass, allowing shrinkage and calibrated covariance.

Example:
    python experiments/sweep_noisy_reconstruction.py --quick --out runs/noisy
    python experiments/sweep_noisy_reconstruction.py --out runs/noisy_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from bayeshippo.basis import legendre_basis, reconstruct_legs
from bayeshippo.bayesian import BayesianProjectionPosterior
from bayeshippo.data import make_synthetic_series
from bayeshippo.hippo import hippo_scan
from bayeshippo.metrics import gaussian_nll_diag, interval_coverage, mse
from bayeshippo.sweep import append_jsonl, grid_product, set_seed


def run_one(cfg: dict) -> dict:
    set_seed(int(cfg["seed"]))
    T = int(cfg["T"])
    N = int(cfg["N"])
    noise_std = float(cfg["noise_std"])
    prior_var = float(cfg["prior_var"])
    evidence_mass = float(cfg["evidence_mass"])

    series = make_synthetic_series(
        T,
        kind=cfg["kind"],
        noise_std=noise_std,
        missing_prob=float(cfg.get("missing_prob", 0.0)),
        seed=int(cfg["seed"]),
    )
    coeff_noisy = hippo_scan(series.observed, N, "legs", return_sequence=False)
    coeff_clean = hippo_scan(series.clean, N, "legs", return_sequence=False)
    post = BayesianProjectionPosterior.from_projection(
        coeff_noisy,
        obs_var=max(noise_std**2, 1e-8),
        evidence_mass=evidence_mass,
        prior_cov=prior_var,
    )

    rec_det = reconstruct_legs(coeff_noisy, series.times, final_time=1.0)
    rec_bayes = reconstruct_legs(post.mean, series.times, final_time=1.0)
    rec_oracle = reconstruct_legs(coeff_clean, series.times, final_time=1.0)

    phi = legendre_basis(series.times, N)
    pred_mean, pred_var_coeff = post.predictive_moments(phi)
    # Add observation noise for predictive NLL/coverage on noisy observations.
    pred_var_obs = pred_var_coeff + noise_std**2

    return {
        **cfg,
        "mse_det_clean": mse(rec_det, series.clean),
        "mse_bayes_clean": mse(rec_bayes, series.clean),
        "mse_oracle_clean": mse(rec_oracle, series.clean),
        "mse_det_noisy": mse(rec_det, series.observed),
        "mse_bayes_noisy": mse(rec_bayes, series.observed),
        "nll_bayes_noisy": gaussian_nll_diag(pred_mean, pred_var_obs, series.observed),
        "coverage95_bayes_noisy": interval_coverage(pred_mean, pred_var_obs, series.observed),
        "posterior_trace": post.cov.sum(),
        "shrinkage_norm_ratio": post.mean.norm() / coeff_noisy.norm().clamp_min(1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="runs/noisy_reconstruction")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.jsonl"
    if result_path.exists():
        result_path.unlink()

    if args.quick:
        grid = {
            "seed": [0, 1],
            "T": [512],
            "N": [16, 32],
            "noise_std": [0.05, 0.25],
            "prior_var": [0.1, 1.0, 1e6],
            "evidence_mass": [1.0, 64.0],
            "kind": ["smooth", "rough"],
            "missing_prob": [0.0],
        }
    else:
        grid = {
            "seed": list(range(10)),
            "T": [512, 2048, 8192],
            "N": [8, 16, 32, 64, 128],
            "noise_std": [0.0, 0.05, 0.10, 0.25, 0.50],
            "prior_var": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 1e6],
            "evidence_mass": [1.0, 8.0, 64.0, 512.0],
            "kind": ["smooth", "rough", "piecewise"],
            "missing_prob": [0.0],
        }

    rows = []
    for cfg in grid_product(grid):
        row = run_one(cfg)
        rows.append({k: (float(v) if torch.is_tensor(v) and v.numel() == 1 else v) for k, v in row.items()})
        append_jsonl(result_path, row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    summary = (
        df.groupby(["kind", "noise_std", "N", "evidence_mass"], as_index=False)
        [["mse_det_clean", "mse_bayes_clean", "mse_oracle_clean", "nll_bayes_noisy", "coverage95_bayes_noisy"]]
        .mean()
    )
    summary.to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps({"quick": args.quick, "grid": grid}, indent=2))
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
