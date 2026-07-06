#!/usr/bin/env python
"""Noisy function reconstruction experiment.

Goal: test the first Bayesian claim in a controlled setting. The true signal is
smooth; observations are noisy. Deterministic HiPPO projects the noisy signal.
Bayesian HiPPO treats the projection coefficients as noisy sufficient statistics
and shrinks them according to a Gaussian coefficient prior.

This experiment does not claim to reproduce Table 3 of the HiPPO paper; it is a
new diagnostic designed for the Bayesian view.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bayeshippo.basis import discrete_legendre_projection, reconstruct_legendre, smooth_random_function
from bayeshippo.bayesian import BayesianProjectionPosterior
from bayeshippo.hippo import hippo_scan
from bayeshippo.metrics import mse


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in range(args.seeds):
        true_np = smooth_random_function(args.T, bandwidth=args.bandwidth, seed=seed)
        rng = np.random.default_rng(seed + 17)
        noisy_np = true_np + args.noise * rng.normal(size=args.T).astype(np.float32)
        true = torch.tensor(true_np)
        noisy = torch.tensor(noisy_np)

        # Online HiPPO statistic from noisy signal.
        c_noisy = hippo_scan(noisy, args.N, measure="legs", return_sequence=False)

        # Offline target projection of the clean signal, used only for evaluation.
        c_true = discrete_legendre_projection(true, args.N).squeeze(0)

        # Bayesian shrinkage posterior. Under orthonormal basis with projected
        # noise variance approximately noise^2 / T_eff; we expose a multiplier
        # because discrete sampling and Euler HiPPO are imperfect.
        obs_var = args.obs_var if args.obs_var is not None else (args.noise**2) * args.obs_var_scale
        post = BayesianProjectionPosterior.from_projection(
            c_noisy, obs_var=obs_var, prior_cov=args.prior_var
        )

        grid, rec_true = reconstruct_legendre(c_true, M=args.T)
        _, rec_det = reconstruct_legendre(c_noisy, grid=grid)
        _, rec_bayes = reconstruct_legendre(post.mean, grid=grid)

        row = {
            "seed": seed,
            "coef_mse_det": float(mse(c_noisy, c_true)),
            "coef_mse_bayes": float(mse(post.mean, c_true)),
            "signal_mse_det": float(mse(rec_det, true)),
            "signal_mse_bayes": float(mse(rec_bayes, true)),
            "posterior_avg_sd": float(torch.sqrt(post.cov).mean()),
        }
        rows.append(row)

        if seed == 0:
            plt.figure(figsize=(10, 4))
            plt.plot(true.numpy(), label="true", linewidth=2)
            plt.plot(noisy.numpy(), label="noisy", alpha=0.35)
            plt.plot(rec_det.detach().numpy(), label="deterministic HiPPO")
            plt.plot(rec_bayes.detach().numpy(), label="Bayesian HiPPO")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / "reconstruction_seed0.png", dpi=160)
            plt.close()

    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "seed"}
    summary.update({f"{k}_std": float(np.std([r[k] for r in rows])) for k in rows[0] if k != "seed"})
    with open(out / "metrics.json", "w") as f:
        json.dump({"args": vars(args), "rows": rows, "summary": summary}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--bandwidth", type=int, default=16)
    p.add_argument("--noise", type=float, default=0.2)
    p.add_argument("--prior-var", type=float, default=1.0)
    p.add_argument("--obs-var", type=float, default=None)
    p.add_argument("--obs-var-scale", type=float, default=1.0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", type=str, default="runs/synthetic")
    run(p.parse_args())
