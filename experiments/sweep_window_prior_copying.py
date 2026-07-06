"""Sweep weakly-informative priors over LegT window lengths on a copying-style memory task.

This is a memory-layer analogue of the Copying task: a random prefix is shown,
then a long zero gap appears, and at the final time the memory is queried to
reconstruct the prefix.  The experiment asks whether a prior over window lengths
can avoid specifying the correct LegT theta.

Example:
    PYTHONPATH=. python experiments/sweep_window_prior_copying.py --quick --out runs/window_copy_quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from bayeshippo.basis import reconstruct_legs, reconstruct_legt
from bayeshippo.hippo import hippo_scan
from bayeshippo.metrics import effective_sample_size, mse
from bayeshippo.sweep import append_jsonl, grid_product, set_seed
from bayeshippo.window_priors import make_window_prior_grid, posterior_window_weights


def make_copy_memory_sequence(prefix_len: int, gap_len: int, cue_len: int, *, vocab: int, seed: int):
    """Return a continuous-valued copying sequence and its prefix target."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, vocab + 1, (prefix_len,), generator=g)
    # Center/scale tokens for polynomial reconstruction.
    values = (tokens.to(torch.float32) - (vocab + 1) / 2.0) / (vocab / 2.0)
    T = prefix_len + gap_len + cue_len
    seq = torch.zeros(T, dtype=torch.float32)
    seq[:prefix_len] = values
    # A cue marker is included so the stream has the same qualitative shape as Copying.
    # The reconstruction query is evaluated only on the prefix values.
    if cue_len > 0:
        seq[-cue_len:] = 0.0
    return seq, values, tokens


def nearest_token_accuracy(values_hat: torch.Tensor, tokens: torch.Tensor, *, vocab: int) -> torch.Tensor:
    raw = values_hat * (vocab / 2.0) + (vocab + 1) / 2.0
    pred = raw.round().clamp(1, vocab).to(torch.long)
    return (pred == tokens).to(torch.float32).mean()


def legt_prefix_reconstruction(seq: torch.Tensor, prefix_times: torch.Tensor, N: int, theta: float) -> torch.Tensor:
    coeff = hippo_scan(seq, N, "legt", theta=float(theta), dt=1.0, return_sequence=False)
    return reconstruct_legt(coeff, prefix_times, final_time=float(seq.numel() - 1), theta=float(theta))


def legs_prefix_reconstruction(seq: torch.Tensor, prefix_times: torch.Tensor, N: int) -> torch.Tensor:
    coeff = hippo_scan(seq, N, "legs", return_sequence=False)
    return reconstruct_legs(coeff, prefix_times, final_time=float(seq.numel() - 1))


def run_one(cfg: dict) -> dict:
    set_seed(int(cfg["seed"]))
    prefix_len = int(cfg["prefix_len"])
    gap_len = int(cfg["gap_len"])
    cue_len = int(cfg["cue_len"])
    vocab = int(cfg["vocab"])
    N = int(cfg["N"])
    seq, target, tokens = make_copy_memory_sequence(prefix_len, gap_len, cue_len, vocab=vocab, seed=int(cfg["seed"]))
    T = seq.numel()
    prefix_times = torch.arange(prefix_len, dtype=torch.float32)

    theta_min = float(cfg["theta_min"])
    theta_max = float(cfg["theta_max_factor"]) * float(T)
    prior = make_window_prior_grid(
        theta_min,
        theta_max,
        int(cfg["num_windows"]),
        kind=cfg["prior"],
        scale="log" if cfg["grid_scale"] == "log" else "linear",
    )

    recs = []
    losses = []
    accs = []
    for theta in prior.theta:
        rec = legt_prefix_reconstruction(seq, prefix_times, N, float(theta.item()))
        recs.append(rec)
        losses.append(mse(rec, target))
        accs.append(nearest_token_accuracy(rec, tokens, vocab=vocab))
    losses_t = torch.stack(losses)
    accs_t = torch.stack(accs)
    weights = posterior_window_weights(losses_t, prior.weights, temperature=float(cfg["temperature"]))
    rec_stack = torch.stack(recs)
    rec_mix = (weights[:, None] * rec_stack).sum(dim=0)

    rec_legs = legs_prefix_reconstruction(seq, prefix_times, N)
    legs_loss = mse(rec_legs, target)
    legs_acc = nearest_token_accuracy(rec_legs, tokens, vocab=vocab)

    best_idx = int(torch.argmin(losses_t).item())
    post_mean_theta = (weights * prior.theta).sum()
    post_log_mean_theta = torch.exp((weights * torch.log(prior.theta)).sum())
    required_theta = float(T - 1)
    posterior_mass_covering_required = weights[prior.theta >= required_theta].sum()

    return {
        **cfg,
        "T": T,
        "required_theta": required_theta,
        "theta_grid": prior.theta,
        "prior_weights": prior.weights,
        "posterior_weights": weights,
        "posterior_mean_theta": post_mean_theta,
        "posterior_log_mean_theta": post_log_mean_theta,
        "posterior_mass_covering_required": posterior_mass_covering_required,
        "ess": effective_sample_size(weights),
        "best_theta": prior.theta[best_idx],
        "best_loss": losses_t[best_idx],
        "best_acc": accs_t[best_idx],
        "mix_loss": mse(rec_mix, target),
        "mix_acc": nearest_token_accuracy(rec_mix, tokens, vocab=vocab),
        "legs_loss": legs_loss,
        "legs_acc": legs_acc,
        "short_theta_loss": losses_t[0],
        "short_theta_acc": accs_t[0],
        "long_theta_loss": losses_t[-1],
        "long_theta_acc": accs_t[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="runs/window_copying")
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
            "prefix_len": [10],
            "gap_len": [64, 200],
            "cue_len": [10],
            "vocab": [8],
            "N": [32],
            "theta_min": [1.0],
            "theta_max_factor": [2.0],
            "num_windows": [12],
            "prior": ["log_uniform", "uniform"],
            "grid_scale": ["log"],
            "temperature": [0.05],
        }
    else:
        grid = {
            "seed": list(range(20)),
            "prefix_len": [10],
            "gap_len": [64, 128, 200, 512],
            "cue_len": [10],
            "vocab": [8],
            "N": [8, 16, 32, 64],
            "theta_min": [1.0],
            "theta_max_factor": [1.0, 1.25, 2.0, 4.0],
            "num_windows": [16, 32, 64],
            "prior": ["log_uniform", "uniform", "pareto"],
            "grid_scale": ["log"],
            "temperature": [0.005, 0.02, 0.10],
        }

    rows = []
    for cfg in grid_product(grid):
        row = run_one(cfg)
        rows.append({k: (float(v) if torch.is_tensor(v) and v.numel() == 1 else v) for k, v in row.items()})
        append_jsonl(result_path, row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    summary = (
        df.groupby(["gap_len", "N", "prior", "theta_max_factor", "num_windows"], as_index=False)
        [["best_loss", "mix_loss", "legs_loss", "short_theta_loss", "long_theta_loss", "mix_acc", "best_acc", "ess", "posterior_mass_covering_required"]]
        .mean()
    )
    summary.to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps({"quick": args.quick, "grid": grid}, indent=2))
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
