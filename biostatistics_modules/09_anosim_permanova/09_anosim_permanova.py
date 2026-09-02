from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.manifold import MDS


def calculate_permanova_pseudo_f(dist_matrix: np.ndarray, groupings: np.ndarray) -> float:
    n = len(groupings)
    unique_groups = np.unique(groupings)
    a = len(unique_groups)

    ss_total = np.sum(dist_matrix ** 2) / (2 * n)
    ss_within = 0.0

    for g in unique_groups:
        idx = np.where(groupings == g)[0]
        n_g = len(idx)
        if n_g > 1:
            sub_dist = dist_matrix[np.ix_(idx, idx)]
            ss_within += np.sum(sub_dist ** 2) / (2 * n_g)

    ss_between = ss_total - ss_within
    df_between = a - 1
    df_within = n - a

    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 1e-9
    return float(ms_between / ms_within)


def run_permanova(dist_matrix: np.ndarray, groupings: np.ndarray, n_permutations: int = 999) -> tuple[float, float]:
    f_obs = calculate_permanova_pseudo_f(dist_matrix, groupings)
    count_ge = 0
    rng = np.random.default_rng(42)

    for _ in range(n_permutations):
        perm_groups = rng.permutation(groupings)
        f_perm = calculate_permanova_pseudo_f(dist_matrix, perm_groups)
        if f_perm >= f_obs:
            count_ge += 1

    p_val = (count_ge + 1) / (n_permutations + 1)
    return f_obs, p_val


def plot_permanova_ordination(dist_matrix: np.ndarray, groupings: np.ndarray, f_stat: float, p_val: float, outpath: Path) -> None:
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(dist_matrix)

    fig, ax = plt.subplots(figsize=(9, 7))
    unique_groups = np.unique(groupings)
    colors = ["#2C7FB8", "#D9534F", "#5CB85C"]

    for idx, g in enumerate(unique_groups):
        mask = groupings == g
        ax.scatter(coords[mask, 0], coords[mask, 1], label=g, color=colors[idx % len(colors)], s=100, edgecolor="black", alpha=0.85)

    ax.set_title(f"PERMANOVA Distance Ordination (PCoA/MDS)\nPseudo-F = {f_stat:.3f}, p = {p_val:.4f}", fontweight="bold", fontsize=12)
    ax.set_xlabel("MDS Dimension 1")
    ax.set_ylabel("MDS Dimension 2")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    g1 = np.random.normal(10, 1.5, size=(12, 10))
    g2 = np.random.normal(12, 1.5, size=(12, 10))
    g3 = np.random.normal(14, 1.5, size=(12, 10))
    X = np.vstack([g1, g2, g3])
    groupings = np.array(["Control"] * 12 + ["Treatment_A"] * 12 + ["Treatment_B"] * 12)

    dist_mat = squareform(pdist(X, metric="euclidean"))
    f_stat, p_val = run_permanova(dist_mat, groupings, n_permutations=999)
    print(f"PERMANOVA: Pseudo-F = {f_stat:.4f}, p-value = {p_val:.4e}")

    out_dir = Path(__file__).parent
    outpath = out_dir / "09_multivariate_anosim_permanova.png"
    plot_permanova_ordination(dist_mat, groupings, f_stat, p_val, outpath)
    print(f"Plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
