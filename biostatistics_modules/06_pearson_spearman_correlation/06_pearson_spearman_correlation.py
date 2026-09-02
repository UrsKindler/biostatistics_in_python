from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_correlation_heatmaps(df: pd.DataFrame, outpath: Path) -> None:
    corr_pearson = df.corr(method="pearson")
    corr_spearman = df.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=axes[0], cbar_kws={"label": "Pearson r"})
    axes[0].set_title("A: Pearson Correlation (Linear Association)", fontweight="bold", fontsize=11)

    sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=axes[1], cbar_kws={"label": "Spearman ρ"})
    axes[1].set_title("B: Spearman Rank Correlation (Monotonic Association)", fontweight="bold", fontsize=11)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    n = 60
    x1 = np.random.normal(10, 2, n)
    x2 = x1 * 1.5 + np.random.normal(0, 1, n)
    x3 = np.exp(x1 / 4) + np.random.normal(0, 0.5, n)
    x4 = np.random.normal(5, 2, n)

    df = pd.DataFrame({"Biomarker_A": x1, "Biomarker_B": x2, "Biomarker_C": x3, "Biomarker_D": x4})
    out_dir = Path(__file__).parent
    outpath = out_dir / "06_correlation_analysis.png"
    plot_correlation_heatmaps(df, outpath)
    print(f"Correlation plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
