"""Make quick diagnostic plots from sweep CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    csv = Path(args.csv)
    out = Path(args.out) if args.out else csv.parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)

    if {"noise_std", "mse_det_clean", "mse_bayes_clean"}.issubset(df.columns):
        agg = df.groupby("noise_std")[["mse_det_clean", "mse_bayes_clean", "mse_oracle_clean"]].mean()
        ax = agg.plot(marker="o")
        ax.set_xlabel("Noise std")
        ax.set_ylabel("Clean reconstruction MSE")
        ax.set_title("Noisy reconstruction sweep")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(out / "noisy_reconstruction_mse.png", dpi=180)
        plt.close(fig)

    if {"query", "best_loss", "mix_loss", "legs_loss"}.issubset(df.columns):
        agg = df.groupby("query")[["best_loss", "mix_loss", "legs_loss"]].mean()
        ax = agg.plot(kind="bar")
        ax.set_ylabel("Query reconstruction MSE")
        ax.set_title("Measure-mixture sweep")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(out / "measure_mixture_loss.png", dpi=180)
        plt.close(fig)

    print(f"Wrote plots to {out}")


if __name__ == "__main__":
    main()
