### Overview
- **Purpose**: Group unlabeled biological samples or molecular features into distinct clusters based on multivariate feature similarities, uncovering hidden disease subtypes, co-expressed gene modules, or patient stratifications without prior class labels.
- **Use Case**: Unsupervised patient cohort stratification, tumor subtyping from transcriptomic profiles, single-cell cluster discovery, or cell morphology classification.
- **Prerequisites**: 
	* Python 3.9+
	* Scaled/standardized continuous numerical matrix ($N \text{ samples} \times P \text{ features}$)
- **Data Types**: 
	* Continuous numerical features (standardized via `StandardScaler`)
- **Output**:
	* Discrete cluster membership labels for each sample
	* Silhouette validation scores and optimal $k$ elbow diagnostic curves
	* 3-Panel cluster comparison figure (`10_unsupervised_clustering.png`)

#### When to Use

| Algorithm | Cluster Geometry | Scalability | Requires $k$? | Recommended Python Implementation | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **K-Means** | Spherical, convex clusters | Very High ($O(N \cdot k \cdot D)$) | Yes | `sklearn.cluster.KMeans(n_clusters=k)` | Minimizes within-cluster sum of squares; sensitive to outliers |
| **Hierarchical (Agglomerative)** | Arbitrary nested clusters | Moderate ($O(N^2 \log N)$) | Optional | `sklearn.cluster.AgglomerativeClustering(linkage='ward')` | Produces intuitive tree dendrograms; deterministic |
| **DBSCAN** | Arbitrary non-convex shapes | High ($O(N \log N)$) | No | `sklearn.cluster.DBSCAN(eps=..., min_samples=...)` | Density-based; labels noise and outliers as $-1$ automatically |

#### Decision Criteria
- **Use K-Means when**: Dataset is large, clusters are expected to be roughly isotropic/spherical, and optimal $k$ can be determined via Silhouette Score or Elbow curve.
- **Use Hierarchical (Ward) when**: Wanting to inspect hierarchical relationships, taxonomies, or co-expression heatmaps with dendrograms.
- **Use DBSCAN when**: Dealing with spatial or density-varying biological data where noise and technical outliers must be filtered out without distorting cluster centroids.

### Python Libraries & Methods

| Aspect | `KMeans` | `AgglomerativeClustering` | `DBSCAN` |
| :--- | :--- | :--- | :--- |
| **Module** | `sklearn.cluster` | `sklearn.cluster` | `sklearn.cluster` |
| **Distance Metric** | Euclidean | Euclidean, Manhattan, Cosine | Any metric (`eps` distance) |
| **Outlier Handling** | Pulls centroids towards outliers | Outliers form isolated branches | Outliers marked as `-1` (Noise) |
| **Cluster Evaluation** | `silhouette_score(X, labels)` | `silhouette_score(X, labels)` | `silhouette_score(X[labels != -1], ...)` |

### Quick Start Code

```bash
python -m pip install scikit-learn numpy pandas matplotlib seaborn scipy
```

```python
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def generate_cohort_data(n_samples: int = 120, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic patient biomarker profiles with 3 underlying disease subtypes."""
    X, _ = make_blobs(n_samples=n_samples, n_features=6, centers=3, cluster_std=1.2, random_state=seed)
    # Add noise / outliers
    noise = np.random.uniform(low=-10, high=10, size=(10, 6))
    X_all = np.vstack([X, noise])
    X_scaled = StandardScaler().fit_transform(X_all)
    return X_scaled, X_all


def main() -> None:
    X_scaled, _ = generate_cohort_data()
    
    # 1. K-Means mit optimaler k-Suche
    silhouette_scores = []
    k_range = range(2, 7)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, km.labels_))
        
    optimal_k = list(k_range)[np.argmax(silhouette_scores)]
    best_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto").fit(X_scaled)
    
    # 2. Hierarchisches Agglomeratives Clustering (Ward Linkage)
    hier = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward").fit(X_scaled)
    
    # 3. DBSCAN
    dbscan = DBSCAN(eps=1.2, min_samples=5).fit(X_scaled)
    n_db_clusters = len(set(dbscan.labels_) - {-1})
    n_db_noise = np.sum(dbscan.labels_ == -1)
    
    print("=== UNSUPERVISED CLUSTERING EVALUATION ===")
    print(f"Optimales k für K-Means:    {optimal_k} (Silhouette Score: {max(silhouette_scores):.3f})")
    print(f"Hierarchical Silhouette:    {silhouette_score(X_scaled, hier.labels_):.3f}")
    print(f"DBSCAN Cluster gefunden:    {n_db_clusters} (Ausreißer/Noise-Punkte: {n_db_noise})")
    
    # 4. Visualisierung der Cluster im 2D-Projektionsraum (Erste 2 Hauptachsen)
    from sklearn.decomposition import PCA
    coords_2d = PCA(n_components=2).fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel A: K-Means
    axes[0].scatter(coords_2d[:, 0], coords_2d[:, 1], c=best_kmeans.labels_, cmap="Set1", s=50, edgecolors="k", alpha=0.85)
    axes[0].set_title(f"A: K-Means Clustering (k={optimal_k})\nSilhouette = {max(silhouette_scores):.3f}", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    axes[0].grid(alpha=0.3)
    
    # Panel B: Hierarchical Ward
    axes[1].scatter(coords_2d[:, 0], coords_2d[:, 1], c=hier.labels_, cmap="Set2", s=50, edgecolors="k", alpha=0.85)
    axes[1].set_title(f"B: Hierarchisches Ward-Clustering (k={optimal_k})\nSilhouette = {silhouette_score(X_scaled, hier.labels_):.3f}", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("PCA 1")
    axes[1].grid(alpha=0.3)
    
    # Panel C: DBSCAN
    axes[2].scatter(coords_2d[:, 0], coords_2d[:, 1], c=dbscan.labels_, cmap="Paired", s=50, edgecolors="k", alpha=0.85)
    axes[2].set_title(f"C: DBSCAN (eps=1.2)\n{n_db_clusters} Cluster, {n_db_noise} Noise-Punkte (-1)", fontweight="bold", fontsize=11)
    axes[2].set_xlabel("PCA 1")
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("10_unsupervised_clustering.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[10_unsupervised_clustering.png]]
