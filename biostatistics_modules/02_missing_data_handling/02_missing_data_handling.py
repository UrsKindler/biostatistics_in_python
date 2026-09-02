from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer


def generate_synthetic_proteomics_missing(n_samples: int = 50, n_proteins: int = 20, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    latent = np.random.normal(loc=0, scale=1, size=(n_samples, 3))
    weights = np.random.normal(loc=0, scale=1, size=(3, n_proteins))
    base = 20.0 + np.dot(latent, weights) + np.random.normal(loc=0, scale=0.5, size=(n_samples, n_proteins))

    # MCAR
    mask_mcar = np.random.rand(*base.shape) < 0.05
    base[mask_mcar] = np.nan

    # MNAR: low abundance -> missing
    mask_mnar = (base < 18.5) & (np.random.rand(*base.shape) < 0.6)
    base[mask_mnar] = np.nan

    columns = [f"Protein_{i+1:02d}" for i in range(n_proteins)]
    index = [f"Sample_{i+1:02d}" for i in range(n_samples)]
    return pd.DataFrame(base, columns=columns, index=index)


def impute_knn(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_arr = imputer.fit_transform(df)
    return pd.DataFrame(imputed_arr, columns=df.columns, index=df.index)


def plot_imputation_diagnostics(df_raw: pd.DataFrame, df_knn: pd.DataFrame, df_median: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.heatmap(df_raw.isna(), cmap="YlOrRd", cbar=False, yticklabels=False, ax=axes[0, 0])
    axes[0, 0].set_title("A: Raw Data Missingness (Missing = Red)", fontweight="bold", fontsize=11)
    axes[0, 0].set_xlabel("Proteins")
    axes[0, 0].set_ylabel("Samples")

    sns.heatmap(df_knn.isna(), cmap="Blues", cbar=False, yticklabels=False, ax=axes[0, 1])
    axes[0, 1].set_title("B: Post KNN Imputation (Complete)", fontweight="bold", fontsize=11)
    axes[0, 1].set_xlabel("Proteins")

    prot = df_raw.columns[0]
    sns.kdeplot(df_raw[prot].dropna(), label="Original Observed", color="black", lw=2, ax=axes[1, 0])
    sns.kdeplot(df_knn[prot], label="KNN Imputed", color="#2C7FB8", linestyle="--", lw=2, ax=axes[1, 0])
    sns.kdeplot(df_median[prot], label="Median Imputed", color="#E6550D", linestyle=":", lw=2, ax=axes[1, 0])
    axes[1, 0].set_title(f"C: Density Distribution ({prot})", fontweight="bold", fontsize=11)
    axes[1, 0].set_xlabel("Abundance")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    raw_corr = df_raw.corr().values[np.triu_indices(df_raw.shape[1], k=1)]
    knn_corr = df_knn.corr().values[np.triu_indices(df_knn.shape[1], k=1)]
    valid_mask = ~np.isnan(raw_corr)
    axes[1, 1].scatter(raw_corr[valid_mask], knn_corr[valid_mask], color="#2C7FB8", alpha=0.7, edgecolors="k")
    axes[1, 1].plot([-1, 1], [-1, 1], "r--", label="Identity (Ideal)")
    axes[1, 1].set_title("D: Correlation Preservation (Pairwise r)", fontweight="bold", fontsize=11)
    axes[1, 1].set_xlabel("Correlation in Original Data")
    axes[1, 1].set_ylabel("Correlation after KNN Imputation")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df_raw = generate_synthetic_proteomics_missing()
    df_knn = impute_knn(df_raw, n_neighbors=5)
    df_median = df_raw.fillna(df_raw.median())
    out_dir = Path(__file__).parent
    outpath = out_dir / "02_missing_data_handling.png"
    plot_imputation_diagnostics(df_raw, df_knn, df_median, outpath)
    print(f"Missing data handling plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
