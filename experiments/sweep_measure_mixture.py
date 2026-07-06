"""Sweep finite mixtures over HiPPO time-relevance measures.

This experiment targets the paper's main new claim: HiPPO fixes a relevance
prior, while Bayesian HiPPO can infer or average over relevance priors.

We build a bank containing LegS and LegT memories at several window lengths,
score each component under a query distribution, and report posterior weights
and mixture reconstruction error.

Example:
    python experiments/sweep_measure_mixture.py --quick --out runs/mixture
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from bayeshippo.basis import reconstruct_legs, reconstruct_legt
from bayeshippo.data import make_synthetic_series
from bayeshippo.hippo import hippo_scan
from bayeshippo.metrics import effective_sample_size, mse
from bayeshippo.sweep import append_jsonl, grid_product, set_seed


def query_mask(times: torch.Tensor, mode: str, theta: float = 0.25) -> torch.Tensor:
    if mode == "global":
        return torch.ones_like(times, dtype=torch.bool)
    if mode == "recent":
        return times >= (1.0 - theta)
    if mode == "middle":
        return (times >= 0.35) & (times <= 0.65)
    raise ValueError(f"Unknown query mode {mode}")


def component_reconstruction(observed: torch.Tensor, times: torch.Tensor, N: int, component: str) -> torch.Tensor:
    if component == "legs":
        coeff = hippo_scan(observed, N, "legs", return_sequence=False)
        return reconstruct_legs(coeff, times, final_time=1.0)
    if component.startswith("legt:"):
        theta = float(component.split(":", 1)[1])
        coeff = hippo_scan(observed, N, "legt", theta=theta, return_sequence=False)
        return reconstruct_legt(coeff, times, final_time=1.0, theta=theta)
    raise ValueError(component)


def run_one(cfg: dict) -> dict:
    set_seed(int(cfg["seed"]))
    T = int(cfg["T"])
    N = int(cfg["N"])
    series = make_synthetic_series(
        T,
        kind=cfg["kind"],
        noise_std=float(cfg["noise_std"]),
        seed=int(cfg["seed"]),
    )
    components = ["legs", "legt:0.05", "legt:0.10", "legt:0.25", "legt:0.50", "legt:1.00"]
    mask = query_mask(series.times, cfg["query"], theta=float(cfg.get("query_theta", 0.25)))
    recs = []
    losses = []
    for comp in components:
        rec = component_reconstruction(series.observed, series.times, N, comp)
        recs.append(rec)
        losses.append(mse(rec, series.clean, mask=mask))
    losses_t = torch.stack(losses)
    temperature = float(cfg["temperature"])
    weights = torch.softmax(-losses_t / temperature, dim=0)
    rec_stack = torch.stack(recs, dim=0)
    rec_mix = (weights[:, None] * rec_stack).sum(dim=0)
    best_idx = int(torch.argmin(losses_t).item())
    return {
        **cfg,
        "components": components,
        "component_losses": losses_t,
        "weights": weights,
        "best_component": components[best_idx],
        "best_loss": losses_t[best_idx],
        "mix_loss": mse(rec_mix, series.clean, mask=mask),
        "legs_loss": losses_t[0],
        "ess": effective_sample_size(weights),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="runs/measure_mixture")
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
            "T": [512],
            "N": [16, 32],
            "noise_std": [0.05, 0.25],
            "kind": ["smooth", "local_burst"],
            "query": ["global", "recent"],
            "temperature": [0.02, 0.10],
        }
    else:
        grid = {
            "seed": list(range(20)),
            "T": [512, 2048],
            "N": [8, 16, 32, 64],
            "noise_std": [0.0, 0.05, 0.10, 0.25],
            "kind": ["smooth", "rough", "piecewise", "local_burst"],
            "query": ["global", "recent", "middle"],
            "temperature": [0.005, 0.02, 0.10, 0.50],
        }

    rows = []
    for cfg in grid_product(grid):
        row = run_one(cfg)
        rows.append({k: (float(v) if torch.is_tensor(v) and v.numel() == 1 else v) for k, v in row.items()})
        append_jsonl(result_path, row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    summary = (
        df.groupby(["kind", "query", "noise_std", "N"], as_index=False)
        [["best_loss", "mix_loss", "legs_loss", "ess"]]
        .mean()
    )
    summary.to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps({"quick": args.quick, "grid": grid}, indent=2))
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
