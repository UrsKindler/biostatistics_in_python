from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def nipals_algorithm(
    X_matrix: np.ndarray,
    n_comps: int = 2,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = X_matrix.copy()
    n_samples, n_features = X.shape
    scores = np.zeros((n_samples, n_comps))
    loadings = np.zeros((n_features, n_comps))
    eigenvalues = np.zeros(n_comps)

    for k in range(n_comps):
        var_per_col = np.nanvar(X, axis=0)
        best_col = int(np.nanargmax(var_per_col))
        t = np.nan_to_num(X[:, best_col], nan=0.0)
        if np.all(t == 0):
            t = np.random.randn(n_samples)

        for _ in range(max_iter):
            t_old = t.copy()

            numerator = np.nansum(X * t[:, None], axis=0)
            obs_mask = ~np.isnan(X)
            denominator = np.sum(obs_mask * (t[:, None] ** 2), axis=0)
            denominator = np.where(denominator == 0, 1e-10, denominator)
            p = numerator / denominator
            p_norm = np.linalg.norm(p)
            if p_norm > 0:
                p = p / p_norm

            numerator_t = np.nansum(X * p[None, :], axis=1)
            denominator_t = np.sum(obs_mask * (p[None, :] ** 2), axis=1)
            denominator_t = np.where(denominator_t == 0, 1e-10, denominator_t)
            t = numerator_t / denominator_t

            if np.linalg.norm(t - t_old) < tol:
                break

        scores[:, k] = t
        loadings[:, k] = p
        eigenvalues[k] = np.nanvar(t) * np.sum(p ** 2)

        X -= np.outer(t, p)

    return scores, loadings, eigenvalues


def perform_nipals_pca(
    data: pd.DataFrame,
    max_components: int = 5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    mean = data.mean(axis=0)
    std = data.std(axis=0).replace(0, 1.0)
    scaled_df = (data - mean) / std

    n_components = min(max_components, data.shape[0], data.shape[1])
    scores, loadings, eigenvalues = nipals_algorithm(scaled_df.values, n_comps=n_components)

    total_variance = data.shape[1]
    explained_variance_ratio = (eigenvalues / total_variance) * 100.0

    score_df = pd.DataFrame(
        scores[:, :n_components],
        index=data.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    return scores, explained_variance_ratio, score_df


def plot_nipals_pca_comparison(
    score_df: pd.DataFrame,
    explained_var: np.ndarray,
    sample_groups: pd.Series,
    color_map: dict[str, str],
    outpath: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for group in sample_groups.unique():
        mask = sample_groups == group
        clr = color_map.get(group, "#333333")
        ax1.scatter(
            score_df.loc[mask, "PC1"],
            score_df.loc[mask, "PC2"],
            label=group,
            color=clr,
            s=100,
            edgecolor="black",
            alpha=0.85,
        )
        for idx in score_df[mask].index:
            ax1.annotate(
                str(idx),
                (score_df.loc[idx, "PC1"], score_df.loc[idx, "PC2"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel(f"PC1 ({explained_var[0]:.1f}% Variance)")
    ax1.set_ylabel(f"PC2 ({explained_var[1]:.1f}% Variance)")
    ax1.set_title("NIPALS PCA (Native Missing-Value Handling)", fontweight="bold", fontsize=12)
    ax1.legend(loc="best", frameon=True)
    ax1.grid(alpha=0.3)

    pcs = np.arange(1, len(explained_var) + 1)
    ax2.bar(pcs, explained_var, color="#2C7FB8", alpha=0.8, label="Individual %")
    ax2.plot(pcs, np.cumsum(explained_var), marker="o", color="#D9534F", lw=2, label="Cumulative %")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance (%)")
    ax2.set_title("Scree Plot (Variance Explained)", fontweight="bold", fontsize=12)
    ax2.set_xticks(pcs)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    out_dir = Path(__file__).parent
    np.random.seed(42)
    n_samples, n_features = 24, 50
    X = np.random.normal(loc=10, scale=2, size=(n_samples, n_features))
    mask = np.random.rand(*X.shape) < 0.15
    X[mask] = np.nan

    groups = ["Control"] * 8 + ["Treatment_A"] * 8 + ["Treatment_B"] * 8
    df = pd.DataFrame(X, index=[f"Sample_{i+1:02d}" for i in range(n_samples)], columns=[f"Protein_{j+1:02d}" for j in range(n_features)])
    sample_groups = pd.Series(groups, index=df.index)

    color_map = {"Control": "#2C7FB8", "Treatment_A": "#D9534F", "Treatment_B": "#5CB85C"}

    scores, exp_var, score_df = perform_nipals_pca(df, max_components=5)
    out_path = out_dir / "05b_nipals_no_imputation_pca.png"
    plot_nipals_pca_comparison(score_df, exp_var, sample_groups, color_map, out_path)
    print(f"NIPALS PCA plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
