### Overview
- **Purpose**: Filter out uninformative, low-abundance, low-prevalence, or near-constant features (genes, peptides, OTUs/taxa) in high-throughput datasets to minimize measurement noise and drastically decrease multiple testing burden.
- **Use Case**: RNA-Seq count matrices, 16S microbiome ASV/OTU tables, shotgun metagenomics, LC-MS/MS bottom-up proteomics.
- **Prerequisites**: 
	* Python 3.9+
	* Count matrix, relative abundance matrix, or normalized intensity matrix ($N \text{ samples} \times P \text{ features}$)
- **Data Types**: 
	* Non-negative counts ($\ge 0$)
	* Relative abundance proportions ($[0, 1]$)
	* Continuous signal intensities
- **Output**:
	* Filtered `pd.DataFrame` containing only informative, statistically robust features
	* Diagnostic filtering curve and summary figure (`03_abundance_filtering.png`)

#### When to Use

| Situation | Recommended Filter Strategy | Note |
| :--- | :--- | :--- |
| **Microbiome / 16S Sequencing** | Prevalence Threshold ($\ge 10-20\%$ of samples) + Relative Abundance ($\ge 0.1\%$) | Removes sporadic spurious taxa and PCR sequencing artifacts |
| **Transcriptomics (RNA-Seq)** | CPM (Counts Per Million) / Read Depth Filtering | Ensures sufficient statistical power for negative binomial GLMs (DESeq2/edgeR equivalents) |
| **Proteomics / Mass Spectrometry** | Valid Value Filter ($\ge 70\%$ valid values in at least one experimental group) | Preserves condition-specific biomarkers while filtering global dropouts |
| **Machine Learning Pipelines** | `VarianceThreshold(threshold=...)` | Eliminates uninformative invariant predictors before model training |

#### Decision Criteria
- **Use always**: In omics pipelines where number of features $P \gg$ number of samples $N$.
- **Essential when**: Applying False Discovery Rate (FDR) corrections (fewer tests $\implies$ lower p-value penalty).
- **Critical for**: Preventing zero-inflated distortion in distance-based multivariate analyses (PERMANOVA, PCoA).
- **Don't skip when**: Dealing with sparse count matrices where $>60\%$ of entries are zeroes.

### Python Libraries & Methods

| Aspect | `pandas` Vectorized | `sklearn.feature_selection.VarianceThreshold` |
| :--- | :--- | :--- |
| **Filter Logic** | Prevalence ($>0$), Relative sum ($> \text{threshold}$) | Empirical feature variance ($s^2 > \text{threshold}$) |
| **Input Type** | `pd.DataFrame` (Counts or Ints) | 2D Array / DataFrame |
| **Flexibility** | Custom condition logic across metadata groups | Simple threshold cut-off |
| **Return Value** | Filtered DataFrame via boolean indexing | Transformed 2D Array + `.get_support()` mask |

### Quick Start Code

```bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn
```

```python
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold


def generate_synthetic_omics(n_samples: int = 40, n_features: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generates a sparse high-dimensional count matrix with technical noise."""
    np.random.seed(seed)
    
    # 50 biologically relevant features with high signal
    signal = np.random.negative_binomial(n=10, p=0.3, size=(n_samples, 50)) * 50
    
    # 450 sparse/noise features with low prevalence
    noise = np.random.negative_binomial(n=1, p=0.9, size=(n_samples, 450))
    noise[np.random.rand(*noise.shape) < 0.75] = 0
    
    data = np.hstack([signal, noise])
    columns = [f"Feature_{i+1:03d}" for i in range(n_features)]
    index = [f"Sample_{i+1:02d}" for i in range(n_samples)]
    return pd.DataFrame(data, columns=columns, index=index)


def main() -> None:
    df_raw = generate_synthetic_omics()
    n_initial = df_raw.shape[1]
    
    # 1. Prävalenz-Filter (Feature in mind. 20% aller Proben vorhanden)
    prevalence = (df_raw > 0).sum(axis=0) / len(df_raw)
    min_prev = 0.20
    prev_mask = prevalence >= min_prev
    df_prev = df_raw.loc[:, prev_mask]
    
    # 2. Relative Abundanz (Feature macht mind. 0.05% der Proben-Gesamtsumme in mind. 5 Proben aus)
    rel_abundance = df_prev.div(df_prev.sum(axis=1), axis=0)
    min_rel = 0.0005
    abund_mask = (rel_abundance >= min_rel).sum(axis=0) >= 5
    df_abund = df_prev.loc[:, abund_mask]
    
    # 3. Varianzfilter
    selector = VarianceThreshold(threshold=10.0)
    selector.fit(df_abund)
    df_final = df_abund.loc[:, selector.get_support()]
    
    print("=== ABUNDANCE & PREVALENCE FILTERING SUMMARY ===")
    print(f"Features initial:               {n_initial}")
    print(f"Nach Prävalenzfilter (>= {min_prev:.0%}):    {df_prev.shape[1]} (entfernt: {n_initial - df_prev.shape[1]})")
    print(f"Nach Abundanzfilter (>= {min_rel:.2%}):  {df_abund.shape[1]} (entfernt: {df_prev.shape[1] - df_abund.shape[1]})")
    print(f"Nach Varianzfilter:             {df_final.shape[1]} (entfernt: {df_abund.shape[1] - df_final.shape[1]})")
    print(f"Gesamte Rauschreduktion:        {(1 - df_final.shape[1]/n_initial):.1%}")
    
    # 4. Visualisierung der Filterkaskade
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Panel A: Filterkaskade (Balkendiagramm)
    stages = ["Raw Features", "Prevalence >= 20%", "Relative Abund.", "Final Retained"]
    counts = [n_initial, df_prev.shape[1], df_abund.shape[1], df_final.shape[1]]
    colors = ["#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]
    
    bars = axes[0].bar(stages, counts, color=colors, edgecolor="black", width=0.55)
    axes[0].bar_label(bars, fmt="%d", padding=3, fontweight="bold")
    axes[0].set_title("A: Feature-Reduktion durch Filterkaskade", fontweight="bold", fontsize=12)
    axes[0].set_ylabel("Anzahl aktiver Features")
    axes[0].grid(axis="y", alpha=0.3)
    
    # Panel B: Prävalenz vs. mittlere relative Abundanz (Scatter Plot)
    raw_rel_abund = df_raw.div(df_raw.sum(axis=1), axis=0)
    mean_abund = raw_rel_abund.mean(axis=0)
    retained_mask = df_raw.columns.isin(df_final.columns)
    axes[1].scatter(prevalence, np.log10(mean_abund + 1e-6), c=retained_mask,
                   cmap="coolwarm", alpha=0.6, edgecolors="k", s=35)
    axes[1].axvline(min_prev, color="red", linestyle="--", label=f"Min. Prävalenz ({min_prev:.0%})")
    axes[1].axhline(np.log10(min_rel), color="orange", linestyle="--", label=f"Min. Abundanz ({min_rel:.2%})")
    axes[1].set_title("B: Prävalenz vs. Mittlere relative Abundanz", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Prävalenz (Anteil Proben > 0)")
    axes[1].set_ylabel("Mittlere Abundanz (Log10)")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("03_abundance_filtering.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[03_abundance_filtering.png]]
