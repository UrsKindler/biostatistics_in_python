### Overview
- **Purpose**: Reduce high-dimensional biological data ($P$ features) down to 2 or 3 interpretable latent coordinates (ordination), maximizing explained variance or preserving ecological/multivariate pairwise dissimilarity relationships.
- **Use Case**: Global sample overview, batch effect visualization, quality control in transcriptomics/proteomics, and microbiome community structure comparison.
- **Prerequisites**: 
	* Python 3.9+
	* Scaled continuous data (for PCA) or Distance/Dissimilarity matrix (for PCoA and NMDS)
- **Data Types**: 
	* Standardized continuous measurements (PCA)
	* Count/abundance matrices with custom distance metrics (PCoA, NMDS)
- **Output**:
	* 2D/3D ordination coordinates, explained variance ratios (PCA) or Stress value (NMDS)
	* 3-Panel ordination comparison figure (`11_multivariate_ordination.png`)

#### When to Use

| Ordination Method | Mathematical Basis | Metric / Distance Used | Preserves | Recommended Python Function |
| :--- | :--- | :--- | :--- | :--- |
| **PCA** (Principal Component Analysis) | Eigenvalue decomposition of covariance/correlation matrix | Euclidean (implicit) | Maximum linear variance across all features | `sklearn.decomposition.PCA()` |
| **PCoA** (Principal Coordinate Analysis / Metric MDS) | Classical multidimensional scaling on distance matrix | Any distance (Bray-Curtis, Jaccard, UniFrac) | Exact pairwise distance values | `sklearn.manifold.MDS(metric=True)` |
| **NMDS** (Non-Metric Multidimensional Scaling) | Rank-order monotonic regression | Any distance | Rank order of pairwise dissimilarities (Stress minimization) | `sklearn.manifold.MDS(metric=False)` |

#### Decision Criteria
- **Use PCA when**: Variables are continuous, standardized, linear combinations are meaningful, and you want to know which specific features contribute most to PC1/PC2 (via Loadings).
- **Use PCoA when**: Dealing with sparse ecological or microbiome count matrices where Bray-Curtis or Jaccard distances are mandatory.
- **Use NMDS when**: Data is highly non-linear, zero-inflated, and preserving rank orders of similarity is more important than absolute distances (ensure Stress $< 0.20$).

### Python Libraries & Methods

| Aspect | `sklearn.decomposition.PCA` | `sklearn.manifold.MDS` (PCoA) | `sklearn.manifold.MDS` (NMDS) |
| :--- | :--- | :--- | :--- |
| **Linearity** | Linear | Linear on distances | Non-linear rank-based |
| **Variance Metric** | `pca.explained_variance_ratio_` | Eigenvalues | `mds.stress_` |
| **Feature Loadings** | `pca.components_` | Not directly available | Not directly available |
| **Determinism** | Exact / Deterministic | Iterative / Random initialization | Iterative / Gradient descent |

### Quick Start Code

```bash
python -m pip install scikit-learn scipy numpy pandas matplotlib seaborn
```

```python
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler


def generate_multivariate_cohorts(n_samples: int = 60, n_features: int = 80, seed: int = 42) -> tuple[pd.DataFrame, list[str]]:
    """Generates synthetic high-dimensional multi-omics data across 3 biological conditions."""
    np.random.seed(seed)
    
    n_per = n_samples // 3
    labels = ["Healthy Control"] * n_per + ["Mild Phenotype"] * n_per + ["Severe Phenotype"] * n_per
    
    # Baseline
    data = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Condition-specific signatures
    data[:n_per, :20] += 1.8         # Healthy biomarker signature
    data[n_per:2*n_per, 20:40] += 2.2 # Mild signature
    data[2*n_per:, 40:60] += 3.5     # Severe signature
    
    df = pd.DataFrame(data, columns=[f"Biomarker_{i+1:02d}" for i in range(n_features)])
    return df, labels


def main() -> None:
    df, labels = generate_multivariate_cohorts()
    palette = {"Healthy Control": "#2C7FB8", "Mild Phenotype": "#41B6C4", "Severe Phenotype": "#E6550D"}
    
    # 1. PCA (Standardisiert)
    X_scaled = StandardScaler().fit_transform(df)
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    
    # 2. PCoA (Metrisches MDS auf Euklidischer Distanz)
    dist_mat = squareform(pdist(X_scaled, metric="euclidean"))
    pcoa = MDS(n_components=2, metric=True, dissimilarity="precomputed", random_state=42)
    pcoa_coords = pcoa.fit_transform(dist_mat)
    
    # 3. NMDS (Nicht-metrisches MDS)
    nmds = MDS(n_components=2, metric=False, dissimilarity="precomputed", random_state=42, max_iter=500)
    nmds_coords = nmds.fit_transform(dist_mat)
    
    print("=== MULTIVARIATE ORDINATION SUMMARY ===")
    print(f"PCA Erklärte Varianz: PC1 = {var_exp[0]:.1%}, PC2 = {var_exp[1]:.1%} (Gesamt: {sum(var_exp):.1%})")
    print(f"NMDS Stress-Wert:     {nmds.stress_:.3f} ({'Gut (< 0.2)' if nmds.stress_ < 0.2 else 'Überprüfen'})")
    
    # 4. Visualisierung der 3 Ordinationsmethoden
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel A: PCA
    sns.scatterplot(x=pca_coords[:, 0], y=pca_coords[:, 1], hue=labels, palette=palette,
                    s=80, edgecolor="k", alpha=0.85, ax=axes[0])
    axes[0].set_title(f"A: PCA (Linear)\nPC1 ({var_exp[0]:.1%}) & PC2 ({var_exp[1]:.1%})", fontweight="bold", fontsize=11)
    axes[0].set_xlabel(f"PC1 ({var_exp[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({var_exp[1]:.1%})")
    axes[0].grid(alpha=0.3)
    
    # Panel B: PCoA
    sns.scatterplot(x=pcoa_coords[:, 0], y=pcoa_coords[:, 1], hue=labels, palette=palette,
                    s=80, edgecolor="k", alpha=0.85, ax=axes[1])
    axes[1].set_title("B: PCoA (Metrische Distanz-Erhaltung)", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("PCoA 1")
    axes[1].set_ylabel("PCoA 2")
    axes[1].grid(alpha=0.3)
    
    # Panel C: NMDS
    sns.scatterplot(x=nmds_coords[:, 0], y=nmds_coords[:, 1], hue=labels, palette=palette,
                    s=80, edgecolor="k", alpha=0.85, ax=axes[2])
    axes[2].set_title(f"C: NMDS (Rangordnung)\nStress = {nmds.stress_:.3f}", fontweight="bold", fontsize=11)
    axes[2].set_xlabel("NMDS 1")
    axes[2].set_ylabel("NMDS 2")
    axes[2].grid(alpha=0.3)
    
    for ax in axes:
        ax.legend(title="Kohorte", frameon=True)
        
    plt.tight_layout()
    
    output_file = Path("11_multivariate_ordination.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[11_multivariate_ordination.png]]
