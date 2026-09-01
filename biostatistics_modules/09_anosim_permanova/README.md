### Overview
- **Purpose**: Test for statistically significant differences in multivariate ecological community structures, microbial compositions, or high-dimensional biomarker profiles between experimental categories using permutation-based distance tests.
- **Use Case**: 16S / Metagenomic microbiome beta-diversity comparisons across phenotypes, untargeted metabolomic profile shifts, or environmental gradient analyses.
- **Prerequisites**: 
	* Python 3.9+
	* Sample-by-feature count/abundance matrix ($N \times P$)
	* Categorical sample grouping metadata
	* Distance/dissimilarity metric choice (e.g., Bray-Curtis, Jaccard, Euclidean)
- **Data Types**: 
	* Non-negative abundance/count matrices
	* Distance matrices (`skbio.DistanceMatrix`)
- **Output**:
	* PERMANOVA pseudo-$F$ statistic, ANOSIM $R$ statistic, and permutation $p$-values
	* PCoA Ordination plot color-coded by experimental group (`09_multivariate_anosim_permanova.png`)

#### When to Use

| Test Method | Primary Focus | Metric Property | Recommended Python Function | Note |
| :--- | :--- | :--- | :--- | :--- |
| **PERMANOVA (`adonis`)** | Multivariate Centroid Separation (Location) | Distance-based multivariate ANOVA | `skbio.stats.distance.permanova(dm, grouping)` | Partitions sums of squares; robust, highly flexible; sensitive to dispersion differences |
| **ANOSIM** | Rank-Order Distance Differences (Within vs. Between) | Rank-based dissimilarity | `skbio.stats.distance.anosim(dm, grouping)` | $R \in [-1, 1]$: $R > 0.75$ strong separation, $R > 0.5$ moderate, $R < 0.25$ weak |
| **Distance Selection** | Bray-Curtis / Jaccard | Abundance / Binary presence-absence | `scipy.spatial.distance.pdist(df, 'braycurtis')` | Standard for sparse microbiome / ecological count data |

#### Decision Criteria
- **Use PERMANOVA when**: Testing whether multi-omics or microbiome profiles are significantly altered by disease status, treatment, or diet.
- **Use ANOSIM when**: Wanting an intuitive, rank-based metric ($R$-statistic) of how distinct two or more multivariate clusters are.
- **Important Check**: Always check for multivariate homogeneity of group dispersions (`betadisper` equivalent) to avoid confounding location shifts with dispersion differences.

### Python Libraries & Methods

| Aspect | `skbio.stats.distance.permanova` | `skbio.stats.distance.anosim` |
| :--- | :--- | :--- |
| **Library** | `scikit-bio` | `scikit-bio` |
| **Test Statistic** | Pseudo-$F = \frac{SS_A / (a-1)}{SS_W / (N-a)}$ | $R = \frac{\bar{r}_B - \bar{r}_W}{N(N-1)/4}$ |
| **Significance** | Permutation testing (e.g. 999 permutations) | Permutation testing (e.g. 999 permutations) |
| **Input Format** | `skbio.DistanceMatrix` + 1D categorical array | `skbio.DistanceMatrix` + 1D categorical array |

### Quick Start Code

```bash
python -m pip install scikit-bio scipy numpy pandas matplotlib seaborn scikit-learn
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
from skbio import DistanceMatrix
from skbio.stats.distance import anosim, permanova
from sklearn.manifold import MDS


def generate_microbiome_data(n_samples: int = 40, n_taxa: int = 60, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Generates synthetic microbiome count matrix with group-specific community shifts."""
    np.random.seed(seed)
    
    half = n_samples // 2
    groups = ["Control"] * half + ["Treated"] * half
    
    # Baseline microbial community
    counts = np.random.negative_binomial(n=2, p=0.1, size=(n_samples, n_taxa))
    
    # Specific taxa altered in treated condition
    counts[half:, :15] += np.random.poisson(lam=45, size=(half, 15))   # Enriched in treated
    counts[:half, 15:30] += np.random.poisson(lam=35, size=(half, 15)) # Depleted in treated
    
    df_taxa = pd.DataFrame(
        counts,
        columns=[f"ASV_{i+1:03d}" for i in range(n_taxa)],
        index=[f"Sample_{i+1:02d}" for i in range(n_samples)]
    )
    return df_taxa, pd.Series(groups, index=df_taxa.index, name="Cohort")


def main() -> None:
    df_counts, metadata = generate_microbiome_data()
    
    # 1. Bray-Curtis Distanzmatrix berechnen
    bray_dists = pdist(df_counts.values, metric="braycurtis")
    dm = DistanceMatrix(squareform(bray_dists), ids=df_counts.index)
    
    # 2. PERMANOVA Test
    perm_res = permanova(dm, grouping=metadata.values, permutations=999)
    f_stat = perm_res["test statistic"]
    p_perm = perm_res["p-value"]
    
    # 3. ANOSIM Test
    anosim_res = anosim(dm, grouping=metadata.values, permutations=999)
    r_stat = anosim_res["test statistic"]
    p_anosim = anosim_res["p-value"]
    
    print("=== MULTIVARIATE HYPOTHESIS TESTING RESULTS ===")
    print(f"Distanz-Metrik:       Bray-Curtis")
    print(f"PERMANOVA pseudo-F:   {f_stat:.3f}, p-Wert = {p_perm:.4f} (999 Permutationen)")
    print(f"ANOSIM R-Statistik:   {r_stat:.3f}, p-Wert = {p_anosim:.4f}")
    
    if r_stat > 0.75:
        print("-> Interpretation: Sehr starke, eindeutige Cluster-Trennung.")
    elif r_stat > 0.50:
        print("-> Interpretation: Moderate, aber signifikante Gruppen-Trennung.")
    else:
        print("-> Interpretation: Schwache oder überlappende Gruppenunterschiede.")
        
    # 4. PCoA / MDS Ordination zur Visualisierung
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    pcoa_coords = mds.fit_transform(squareform(bray_dists))
    
    plot_df = pd.DataFrame({
        "Axis_1": pcoa_coords[:, 0],
        "Axis_2": pcoa_coords[:, 1],
        "Cohort": metadata.values
    })
    
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = {"Control": "#2C7FB8", "Treated": "#E6550D"}
    
    sns.scatterplot(data=plot_df, x="Axis_1", y="Axis_2", hue="Cohort", style="Cohort",
                    palette=palette, s=120, edgecolor="black", alpha=0.85, ax=ax)
    
    # Konfidenz-Ellipsen / Zentroiden
    for g, col in palette.items():
        sub = plot_df[plot_df["Cohort"] == g]
        ax.plot(sub["Axis_1"].mean(), sub["Axis_2"].mean(), marker="X", markersize=14,
                color=col, markeredgecolor="black", label=f"{g} Zentroid")
        
    ax.set_title(f"PCoA Ordination (Bray-Curtis Distanz)\nPERMANOVA F={f_stat:.2f} (p={p_perm:.4f}) | ANOSIM R={r_stat:.2f}",
                 fontweight="bold", fontsize=12, pad=12)
    ax.set_xlabel("PCoA Dimension 1")
    ax.set_ylabel("PCoA Dimension 2")
    ax.legend(title="Gruppe", frameon=True)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("09_multivariate_anosim_permanova.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[09_multivariate_anosim_permanova.png]]
