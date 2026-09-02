from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


def plot_clustering_comparison(X: np.ndarray, labels_true: np.ndarray, outpath: Path) -> None:
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(X)

    agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels_agg = agg.fit_predict(X)

    dbscan = DBSCAN(eps=1.2, min_samples=4)
    labels_db = dbscan.fit_predict(X)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # A: K-Means
    axes[0, 0].scatter(X[:, 0], X[:, 1], c=labels_km, cmap="Set2", s=80, edgecolor="black")
    axes[0, 0].set_title(f"A: K-Means Clustering (Silhouette = {silhouette_score(X, labels_km):.3f})", fontweight="bold")

    # B: Hierarchical Agglomerative
    axes[0, 1].scatter(X[:, 0], X[:, 1], c=labels_agg, cmap="Set2", s=80, edgecolor="black")
    axes[0, 1].set_title(f"B: Hierarchical Ward (Silhouette = {silhouette_score(X, labels_agg):.3f})", fontweight="bold")

    # C: DBSCAN
    axes[1, 0].scatter(X[:, 0], X[:, 1], c=labels_db, cmap="Set2", s=80, edgecolor="black")
    axes[1, 0].set_title("C: DBSCAN Density-Based (Noise = -1)", fontweight="bold")

    # D: Hierarchical Dendrogram
    Z = linkage(X, method="ward")
    dendrogram(Z, ax=axes[1, 1], truncate_mode="lastp", p=12)
    axes[1, 1].set_title("D: Dendrogram Truncated Tree", fontweight="bold")

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    c1 = np.random.normal([2, 2], 0.8, size=(25, 2))
    c2 = np.random.normal([7, 2], 0.8, size=(25, 2))
    c3 = np.random.normal([4.5, 7], 0.8, size=(25, 2))
    X = np.vstack([c1, c2, c3])
    labels_true = np.array([0]*25 + [1]*25 + [2]*25)

    out_dir = Path(__file__).parent
    outpath = out_dir / "10_unsupervised_clustering.png"
    plot_clustering_comparison(X, labels_true, outpath)
    print(f"Clustering plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
