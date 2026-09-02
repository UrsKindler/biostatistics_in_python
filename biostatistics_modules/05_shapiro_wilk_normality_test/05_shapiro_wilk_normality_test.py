from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def test_normality_feature(data: np.ndarray) -> dict[str, tuple[float, float]]:
    clean = data[~np.isnan(data)]
    if len(clean) < 3:
        return {}
    sw_stat, sw_p = stats.shapiro(clean)
    dp_stat, dp_p = stats.normaltest(clean) if len(clean) >= 8 else (np.nan, np.nan)
    return {"Shapiro-Wilk": (sw_stat, sw_p), "D'Agostino-Pearson": (dp_stat, dp_p)}


def plot_normality_diagnostics(norm_data: np.ndarray, non_norm_data: np.ndarray, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # A: Normal Histogram
    axes[0, 0].hist(norm_data, bins=25, density=True, color="#2C7FB8", alpha=0.7, edgecolor="black")
    xmin, xmax = axes[0, 0].get_xlim()
    x_axis = np.linspace(xmin, xmax, 100)
    axes[0, 0].plot(x_axis, stats.norm.pdf(x_axis, np.mean(norm_data), np.std(norm_data)), "r-", lw=2, label="Gaussian Fit")
    axes[0, 0].set_title("A: Normal Feature Histogram (Gaussian)", fontweight="bold", fontsize=11)
    axes[0, 0].legend()

    # B: Normal Q-Q Plot
    stats.probplot(norm_data, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("B: Normal Q-Q Plot", fontweight="bold", fontsize=11)
    axes[0, 1].grid(alpha=0.3)

    # C: Non-Normal Histogram
    axes[1, 0].hist(non_norm_data, bins=25, density=True, color="#E6550D", alpha=0.7, edgecolor="black")
    axes[1, 0].set_title("C: Skewed Feature Histogram (Non-Gaussian)", fontweight="bold", fontsize=11)

    # D: Non-Normal Q-Q Plot
    stats.probplot(non_norm_data, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("D: Skewed Q-Q Plot (Heavy Tails)", fontweight="bold", fontsize=11)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    norm_vals = np.random.normal(15, 2, 80)
    non_norm_vals = np.random.exponential(3, 80)

    out_dir = Path(__file__).parent
    outpath = out_dir / "05_shapiro_wilk_normality.png"
    plot_normality_diagnostics(norm_vals, non_norm_vals, outpath)
    print(f"Normality diagnostics saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
