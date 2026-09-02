from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def normalize_median(df: pd.DataFrame) -> pd.DataFrame:
    sample_medians = df.median(axis=0)
    target_median = sample_medians.mean()
    return df.sub(sample_medians, axis=1).add(target_median)


def normalize_quantile(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = pd.DataFrame(np.sort(df.values, axis=0), index=df.index, columns=df.columns)
    df_mean = df_sorted.mean(axis=1)
    df_ranked = df.rank(method="min").astype(int) - 1
    return pd.DataFrame(df_mean.values[df_ranked.values], index=df.index, columns=df.columns)


def scale_zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1).replace(0, 1.0)
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


def plot_normalization_comparison(df_raw: pd.DataFrame, df_norm: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(data=df_raw, ax=axes[0], palette="Blues_d", fliersize=2)
    axes[0].set_title("A: Pre-Normalization (Raw Abundance)", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Intensity")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    sns.boxplot(data=df_norm, ax=axes[1], palette="Greens_d", fliersize=2)
    axes[1].set_title("B: Post-Median Normalization", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Normalized Intensity")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    cols = [f"Sample_{i+1:02d}" for i in range(10)]
    shifts = np.random.uniform(0.7, 1.4, size=10)
    raw_data = np.random.normal(20, 2, size=(500, 10)) * shifts
    df_raw = pd.DataFrame(raw_data, columns=cols)

    df_norm = normalize_median(df_raw)
    out_dir = Path(__file__).parent
    outpath = out_dir / "04_normalization_and_scaling.png"
    plot_normalization_comparison(df_raw, df_norm, outpath)
    print(f"Normalization plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
