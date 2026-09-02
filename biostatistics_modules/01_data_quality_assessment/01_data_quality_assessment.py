from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns


def generate_synthetic_biodata(n_samples: int = 60, n_proteins: int = 15, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    base_data = np.random.normal(loc=22.0, scale=2.5, size=(n_samples, n_proteins))
    base_data[5, 2] = 34.0
    base_data[12, 4] = 8.5
    mask = (base_data < 18.5) & (np.random.rand(n_samples, n_proteins) < 0.7)
    base_data[mask] = np.nan
    columns = [f"Protein_{i+1:02d}" for i in range(n_proteins)]
    index = [f"Sample_{i+1:02d}" for i in range(n_samples)]
    return pd.DataFrame(base_data, columns=columns, index=index)


def detect_iqr_outliers(df: pd.DataFrame) -> pd.Series:
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return ((df < lower_bound) | (df > upper_bound)).sum()


def plot_data_quality(df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    msno.matrix(df, ax=axes[0], sparkline=False, fontsize=9, color=(0.2, 0.4, 0.6))
    axes[0].set_title("A: Missingness Pattern (MNAR / Limit of Detection)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Biomarkers / Proteins", fontsize=11)
    axes[0].set_ylabel("Samples", fontsize=11)

    sample_cols = df.columns[:8]
    sns.boxplot(data=df[sample_cols], ax=axes[1], palette="Set2", fliersize=5)
    axes[1].set_title("B: Distribution & Outlier Inspection (Selected Features)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel("Log2 Abundance", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = generate_synthetic_biodata()
    outliers = detect_iqr_outliers(df)
    out_dir = Path(__file__).parent
    outpath = out_dir / "01_data_quality_assessment.png"
    plot_data_quality(df, outpath)
    print(f"Data quality plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
