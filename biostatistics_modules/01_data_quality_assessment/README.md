### Overview
- **Purpose**: Evaluate dataset completeness, detect missingness mechanisms (MCAR, MAR, MNAR), identify univariate/multivariate outliers, and verify data integrity prior to statistical modeling.
- **Use Case**: Initial exploratory data analysis (EDA) for proteomics, transcriptomics, metabolomics, or clinical cohort studies.
- **Prerequisites**: 
	* Python 3.9+
	* Tabular dataset (samples as rows, features/biomarkers as columns)
- **Data Types**: 
	* Continuous numerical (intensities, abundances, concentrations)
	* Discrete numerical (counts, read depths)
	* Categorical / metadata factors (cohorts, batches, phenotypes)
- **Output**:
	* Summary statistics and missingness metrics in terminal
	* Dual-panel quality plot (`01_data_quality_assessment.png`) featuring missingness matrix and outlier boxplots

#### When to Use

| Situation | Recommended Approach | Note |
| :--- | :--- | :--- |
| **New / Unprocessed Dataset** | Complete Data Quality Pipeline (`df.info()`, `df.describe()`, `missingno`) | Always the mandatory first step before any downstream transformation |
| **Suspected Systematic Loss** | Missingness Matrix & Heatmap (`missingno.matrix()`, `missingno.heatmap()`) | Identifies MNAR patterns (e.g., low-abundance proteins missing below LOD) |
| **Heavy-Tailed / Skewed Data** | IQR-based Outlier Detection ($Q_1 - 1.5 \cdot IQR$, $Q_3 + 1.5 \cdot IQR$) | Robust against non-normal distributions unlike standard Z-score |
| **High-Dimensional Omics** | Zero & Sparsity Inspection | Identifies dropouts vs. biological true zeros |

#### Decision Criteria
- **Use always**: As the foundation of every biostatistical analysis workflow.
- **Essential when**: Integrating multi-center cohorts, different extraction batches, or longitudinal timepoints.
- **Critical for**: Preventing biased effect sizes caused by technical dropouts or undetected measurement artifacts.
- **Don't skip when**: Working with real-world biological and clinical data.

### Python Libraries & Methods

| Aspect | `missingno` | `pandas` | `seaborn` / `matplotlib` |
| :--- | :--- | :--- | :--- |
| **Main Function** | Missing data visualization | Quality metrics & filtering | Distribution & outlier plots |
| **Key Methods** | `msno.matrix()`, `msno.heatmap()` | `df.isna().sum()`, `df.quantile()` | `sns.boxplot()`, `sns.histplot()` |
| **Installation** | `pip install missingno` | `pip install pandas` | `pip install seaborn matplotlib` |
| **Input Format** | `pd.DataFrame` | `pd.DataFrame` | Array / DataFrame columns |
| **Best For** | Sparsity & pattern inspection | IQR outlier calculation & stats | High-resolution publication plots |
| **Output** | Matplotlib Axes | Numeric summaries / Boolean masks | Rendered PNG Figures |

### Quick Start Code

```bash
python -m pip install pandas numpy missingno matplotlib seaborn
```

```python
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns


def generate_synthetic_biodata(n_samples: int = 60, n_proteins: int = 15, seed: int = 42) -> pd.DataFrame:
    """Generates realistic biological data with MNAR missingness and outliers."""
    np.random.seed(seed)
    
    # Baseline protein intensities (log2 scale)
    base_data = np.random.normal(loc=22.0, scale=2.5, size=(n_samples, n_proteins))
    
    # Introduce outliers in selected proteins
    base_data[5, 2] = 34.0  # High outlier
    base_data[12, 4] = 8.5   # Low outlier
    
    # Introduce Limit of Detection (LOD) / MNAR missingness (low intensity -> missing)
    mask = (base_data < 18.5) & (np.random.rand(n_samples, n_proteins) < 0.7)
    base_data[mask] = np.nan
    
    columns = [f"Protein_{i+1:02d}" for i in range(n_proteins)]
    index = [f"Sample_{i+1:02d}" for i in range(n_samples)]
    return pd.DataFrame(base_data, columns=columns, index=index)


def detect_iqr_outliers(df: pd.DataFrame) -> pd.Series:
    """Calculates number of outliers per column using the 1.5 * IQR rule."""
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return ((df < lower_bound) | (df > upper_bound)).sum()


def main() -> None:
    # 1. Daten generieren
    df = generate_synthetic_biodata()
    
    print("=== DATA QUALITY ASSESSMENT SUMMARY ===")
    print(f"Dimensionen: {df.shape[0]} Proben x {df.shape[1]} Proteine")
    print(f"Gesamtanzahl Werte: {df.size}")
    print(f"Fehlende Werte gesamt: {df.isna().sum().sum()} ({df.isna().sum().sum()/df.size:.1%})")
    
    # 2. Ausreißer analysieren
    outliers = detect_iqr_outliers(df)
    print("\nAusreißer pro Protein (Top 5):")
    print(outliers.sort_values(ascending=False).head(5))
    
    # 3. Visualisierung erstellen
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel A: Missing Data Matrix
    msno.matrix(df, ax=axes[0], sparkline=False, fontsize=9, color=(0.2, 0.4, 0.6))
    axes[0].set_title("A: Fehlwertmuster (MNAR / Limit of Detection)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Biomarker / Proteine", fontsize=11)
    axes[0].set_ylabel("Proben", fontsize=11)
    
    # Panel B: Outlier Detection Boxplot
    sample_cols = df.columns[:8]
    sns.boxplot(data=df[sample_cols], ax=axes[1], palette="Set2", fliersize=5)
    axes[1].set_title("B: Verteilung & Ausreißer-Inspektion (Auswahl)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel("Log2-Abundanz", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    
    output_file = Path("01_data_quality_assessment.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[01_data_quality_assessment.png]]
