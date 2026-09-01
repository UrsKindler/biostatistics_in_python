### Overview
- **Purpose**: Systematically handle missing values in biological and clinical datasets through complete case filtering, univariate imputation, or multivariate iterative/KNN imputation while preserving correlation structures.
- **Use Case**: Incomplete mass spectrometry proteomics runs, clinical survey datasets with sporadic missingness, or longitudinal sample profiles.
- **Prerequisites**: 
	* Python 3.9+
	* Understanding of missingness mechanism:
		- **MCAR** (Missing Completely at Random): Dropouts unrelated to any observed or unobserved variable.
		- **MAR** (Missing at Random): Missingness can be explained by other measured variables.
		- **MNAR** (Missing Not at Random): Missingness depends on the unobserved value itself (e.g., values below detection limit).
- **Data Types**: 
	* Continuous numerical (protein abundances, gene expression levels, clinical labs)
	* Discrete counts / ordinal scores
- **Output**:
	* Imputed `pd.DataFrame` without missing values
	* Comparative diagnostic plot (`02_missing_data_handling.png`) showing data matrices before/after imputation, feature distribution preservation, and correlation preservation

#### When to Use

| Situation | Recommended Approach | Note |
| :--- | :--- | :--- |
| **MCAR & $<5\%$ Missing** | Complete Case Analysis (`df.dropna()`) | Simple and unbiased when missingness is completely random and low |
| **MAR & Moderate Missingness ($<10\%$)** | Simple Imputation (`SimpleImputer(strategy='median')`) | Fast baseline, but reduces feature variance and ignores correlations |
| **MAR & High Missingness ($>10\%$)** | KNN Imputation (`KNNImputer(n_neighbors=5)`) | Exploits co-expression / correlation between similar biological samples |
| **Complex Inter-Feature Dependencies** | Iterative MICE Imputation (`IterativeImputer()`) | Models each feature with missing values as a function of other features |
| **MNAR (Below Detection Limit)** | Left-Censored / Half-Min Imputation | Imputes values near the bottom quantile or estimated limit of detection (LOD) |

#### Decision Criteria
- **Use always**: When downstream multivariate methods (PCA, t-SNE, Clustering, Random Forest) require complete matrices without `NaN`.
- **Essential when**: Working with small sample cohorts where dropping rows would destroy statistical power.
- **Critical for**: Verifying that imputation does not distort empirical distributions or create artificial correlation artifacts.
- **Don't skip when**: Comparing multiple experimental batches with uneven coverage.

### Python Libraries & Methods

| Aspect | `sklearn.impute.SimpleImputer` | `sklearn.impute.KNNImputer` | `sklearn.impute.IterativeImputer` |
| :--- | :--- | :--- | :--- |
| **Algorithm** | Mean, Median, Mode substitution | k-Nearest Neighbors distance-weighted average | Multivariate Imputation by Chained Equations (MICE) |
| **Preserves Correlation** | No (attenuates correlations) | Yes (preserves local neighborhoods) | Yes (preserves covariance structure) |
| **Computational Speed** | Instantaneous ($O(N)$) | Moderate ($O(N^2 \cdot D)$) | Iterative / Slower ($O(M \cdot N \cdot D)$) |
| **Parameter Tuning** | `strategy='median'` | `n_neighbors=5`, `weights='distance'` | `max_iter=10`, `estimator=BayesianRidge()` |
| **Installation** | `pip install scikit-learn` | `pip install scikit-learn` | `pip install scikit-learn` (enable experimental) |

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
from sklearn.impute import KNNImputer, SimpleImputer


def generate_missing_biodata(n_samples: int = 50, n_features: int = 12, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic correlated biological data with MAR/MNAR missingness."""
    np.random.seed(seed)
    
    # Latent biological signal
    latent = np.random.normal(0, 1, size=(n_samples, 1))
    weights = np.random.uniform(0.5, 2.0, size=(1, n_features))
    noise = np.random.normal(0, 0.5, size=(n_samples, n_features))
    
    data = (latent @ weights + noise) * 5 + 50
    
    # Introduce ~15% missingness
    missing_mask = np.random.rand(n_samples, n_features) < 0.15
    data[missing_mask] = np.nan
    
    columns = [f"Protein_{i+1:02d}" for i in range(n_features)]
    index = [f"Sample_{i+1:02d}" for i in range(n_samples)]
    return pd.DataFrame(data, columns=columns, index=index)


def main() -> None:
    df_raw = generate_missing_biodata()
    
    print("=== MISSING DATA IMPUTATION WORKFLOW ===")
    print(f"Fehlende Werte vor Imputation: {df_raw.isna().sum().sum()} / {df_raw.size} ({df_raw.isna().sum().sum()/df_raw.size:.1%})")
    
    # 1. KNN-Imputation
    knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
    df_knn = pd.DataFrame(
        knn_imputer.fit_transform(df_raw),
        columns=df_raw.columns,
        index=df_raw.index
    )
    
    # 2. Simple Median Imputation zum Vergleich
    med_imputer = SimpleImputer(strategy="median")
    df_median = pd.DataFrame(
        med_imputer.fit_transform(df_raw),
        columns=df_raw.columns,
        index=df_raw.index
    )
    
    print(f"Fehlende Werte nach KNN-Imputation: {df_knn.isna().sum().sum()}")
    
    # 3. 4-Panel Diagnose-Visualisierung
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # A: Matrix vor Imputation
    sns.heatmap(df_raw.isna(), cmap="YlOrRd", cbar=False, yticklabels=False, ax=axes[0, 0])
    axes[0, 0].set_title("A: Rohdaten (Fehlwerte = Rot)", fontweight="bold", fontsize=12)
    axes[0, 0].set_xlabel("Proteine")
    axes[0, 0].set_ylabel("Proben")
    
    # B: Matrix nach Imputation
    sns.heatmap(df_knn.isna(), cmap="Blues", cbar=False, yticklabels=False, ax=axes[0, 1])
    axes[0, 1].set_title("B: Nach KNN-Imputation (Vollständig)", fontweight="bold", fontsize=12)
    axes[0, 1].set_xlabel("Proteine")
    
    # C: Verteilungsvergleich (Protein_01)
    prot = "Protein_01"
    sns.kdeplot(df_raw[prot].dropna(), label="Original (vorhanden)", color="black", lw=2, ax=axes[1, 0])
    sns.kdeplot(df_knn[prot], label="KNN Imputation", color="#2C7FB8", linestyle="--", lw=2, ax=axes[1, 0])
    sns.kdeplot(df_median[prot], label="Median Imputation", color="#E6550D", linestyle=":", lw=2, ax=axes[1, 0])
    axes[1, 0].set_title(f"C: Dichteverteilung ({prot})", fontweight="bold", fontsize=12)
    axes[1, 0].set_xlabel("Abundanz")
    axes[1, 0].set_ylabel("Dichte")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # D: Korrelationserhaltung
    raw_corr = df_raw.corr().values[np.triu_indices(df_raw.shape[1], k=1)]
    knn_corr = df_knn.corr().values[np.triu_indices(df_knn.shape[1], k=1)]
    
    valid_mask = ~np.isnan(raw_corr)
    axes[1, 1].scatter(raw_corr[valid_mask], knn_corr[valid_mask], color="#2C7FB8", alpha=0.7, edgecolors="k")
    axes[1, 1].plot([-1, 1], [-1, 1], "r--", label="Identitätslinie (Ideal)")
    axes[1, 1].set_title("D: Korrelationserhaltung (Pairwise r)", fontweight="bold", fontsize=12)
    axes[1, 1].set_xlabel("Korrelation Originaldaten")
    axes[1, 1].set_ylabel("Korrelation nach KNN-Imputation")
    axes[1, 1].set_xlim(-0.2, 1.0)
    axes[1, 1].set_ylim(-0.2, 1.0)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("02_missing_data_handling.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[02_missing_data_handling.png]]
