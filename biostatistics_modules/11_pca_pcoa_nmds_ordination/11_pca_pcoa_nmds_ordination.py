from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler


def run_multivariate_ordination(X: np.ndarray, labels: np.ndarray, outpath: Path) -> None:
    X_scaled = StandardScaler().fit_transform(X)

    # 1. PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    exp_var = pca.explained_variance_ratio_ * 100

    # 2. PCoA / MDS (Euclidean)
    dist_mat = squareform(pdist(X_scaled, "euclidean"))
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    pcoa_coords = mds.fit_transform(dist_mat)

    # 3. NMDS
    nmds = MDS(n_components=2, dissimilarity="precomputed", metric=False, random_state=42)
    nmds_coords = nmds.fit_transform(dist_mat)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    unique_labels = np.unique(labels)
    colors = ["#2C7FB8", "#D9534F", "#5CB85C"]

    for idx, lab in enumerate(unique_labels):
        mask = labels == lab
        clr = colors[idx % len(colors)]
        axes[0].scatter(pcs[mask, 0], pcs[mask, 1], label=lab, color=clr, s=80, edgecolor="black")
        axes[1].scatter(pcoa_coords[mask, 0], pcoa_coords[mask, 1], label=lab, color=clr, s=80, edgecolor="black")
        axes[2].scatter(nmds_coords[mask, 0], nmds_coords[mask, 1], label=lab, color=clr, s=80, edgecolor="black")

    axes[0].set_title(f"A: PCA (PC1: {exp_var[0]:.1f}%, PC2: {exp_var[1]:.1f}%)", fontweight="bold")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("B: PCoA (Classical Metric MDS)", fontweight="bold")
    axes[1].set_xlabel("PCoA Axis 1")
    axes[1].set_ylabel("PCoA Axis 2")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].set_title("C: NMDS (Non-Metric Multidimensional Scaling)", fontweight="bold")
    axes[2].set_xlabel("NMDS 1")
    axes[2].set_ylabel("NMDS 2")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    g1 = np.random.normal(10, 1.2, (15, 20))
    g2 = np.random.normal(12, 1.5, (15, 20))
    g3 = np.random.normal(14, 1.8, (15, 20))
    X = np.vstack([g1, g2, g3])
    labels = np.array(["Control"] * 15 + ["Treatment_A"] * 15 + ["Treatment_B"] * 15)

    out_dir = Path(__file__).parent
    outpath = out_dir / "11_multivariate_ordination.png"
    run_multivariate_ordination(X, labels, outpath)
    print(f"Multivariate ordination plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
