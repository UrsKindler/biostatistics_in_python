### Overview
- **Purpose**: Transform and scale feature distributions onto comparable dynamic ranges, eliminating technical scale differences, heteroscedasticity, and dynamic range dominance in multivariate algorithms.
- **Use Case**: Pre-scaling before Principal Component Analysis (PCA), K-Means / Hierarchical Clustering, Regularized Ridge/Lasso Regression, and Deep Neural Networks.
- **Prerequisites**: 
	* Python 3.9+
	* Numeric feature matrix without missing values (or after imputation)
- **Data Types**: 
	* Continuous numerical (concentrations, intensities, physical measurements)
	* Discrete non-negative counts
- **Output**:
	* Normalized `pd.DataFrame`
	* Comparative 4-panel scaling diagnosis figure (`04_normalization_and_scaling.png`)

#### When to Use

| Scaling Method | Mathematical Formula | Recommended Use Case | Note |
| :--- | :--- | :--- | :--- |
| **Z-Score (`StandardScaler`)** | $z = \frac{x - \mu}{\sigma}$ | Standard for PCA, Clustering, Linear Models | Centers to mean $\mu=0$ and standard deviation $\sigma=1$; sensitive to severe outliers |
| **Min-Max (`MinMaxScaler`)** | $x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | Neural Networks, Image processing, Bounded range $[0, 1]$ | Compresses outliers into extreme edges; destroys original variance scale |
| **Robust Scaler (`RobustScaler`)** | $x_{\text{scaled}} = \frac{x - \text{median}}{IQR}$ | Datasets with unavoidable biological/technical outliers | Centers by Median and scales by Interquartile Range ($Q_3 - Q_1$) |
| **Log-Transformation** | $y = \log_2(x + 1)$ oder $\log_{10}(x + 1)$ | Skewed count/intensity data (RNA-Seq, Proteomics, qPCR) | Compresses right tail, linearizes multiplicative effects into additive fold-changes |

#### Decision Criteria
- **Use always**: Before running distance-based algorithms (Euclidean, Manhattan) or dimensionality reduction.
- **Essential when**: Features have vastly different units (e.g., Blood pressure [mmHg] vs. CRP [mg/L] vs. White blood cells [cells/$\mu$L]).
- **Critical for**: Preventing high-variance features from completely dominating PCA Component 1.
- **Don't skip when**: Training gradient-based neural networks or distance-sensitive SVMs.

### Python Libraries & Methods

| Aspect | `StandardScaler` | `MinMaxScaler` | `RobustScaler` | `np.log1p` |
| :--- | :--- | :--- | :--- | :--- |
| **Module** | `sklearn.preprocessing` | `sklearn.preprocessing` | `sklearn.preprocessing` | `numpy` |
| **Target Distribution** | Mean 0, Std 1 | Range $[0, 1]$ | Median 0, IQR 1 | Normalizes right-skewed counts |
| **Outlier Robustness** | Low | Low | High | High |
| **Inverse Transform** | `scaler.inverse_transform()` | `scaler.inverse_transform()` | `scaler.inverse_transform()` | `np.expm1()` |

### Quick Start Code

```bash
python -m pip install scikit-learn numpy pandas matplotlib seaborn
```

```python
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def generate_heterogeneous_biodata(n_samples: int = 80, seed: int = 42) -> pd.DataFrame:
    """Generates multi-feature biological dataset with wildly different scales and outliers."""
    np.random.seed(seed)
    
    # Feature 1: Highly skewed exponential data (e.g., cytokine pg/mL)
    feat_skewed = np.random.exponential(scale=50.0, size=n_samples) + 2.0
    
    # Feature 2: High magnitude gaussian (e.g., blood cell counts ~ 5000)
    feat_high_mag = np.random.normal(loc=5000, scale=400, size=n_samples)
    
    # Feature 3: Small scale ratio (e.g., biomarker index ~ 0.5) with extreme outliers
    feat_outlier = np.random.normal(loc=0.5, scale=0.1, size=n_samples)
    feat_outlier[3] = 4.5
    feat_outlier[27] = -2.2
    
    return pd.DataFrame({
        "Cytokine_pg_mL": feat_skewed,
        "Cell_Count": feat_high_mag,
        "Biomarker_Index": feat_outlier
    })


def main() -> None:
    df_raw = generate_heterogeneous_biodata()
    
    # 1. Scaler anwenden
    # A: Z-Score
    df_zscore = pd.DataFrame(StandardScaler().fit_transform(df_raw), columns=df_raw.columns)
    
    # B: Min-Max
    df_minmax = pd.DataFrame(MinMaxScaler().fit_transform(df_raw), columns=df_raw.columns)
    
    # C: Robust Scaler
    df_robust = pd.DataFrame(RobustScaler().fit_transform(df_raw), columns=df_raw.columns)
    
    # D: Log1p Transformation
    df_log = np.log1p(df_raw)
    
    print("=== SCALING METHOD COMPARISON (MEANS & STDS) ===")
    print("\nOriginal Means:\n", df_raw.mean())
    print("\nZ-Score Means (≈0):\n", df_zscore.mean().round(4))
    print("\nMin-Max Bounds:\nMin:", df_minmax.min().round(2), "\nMax:", df_minmax.max().round(2))
    
    # 2. Diagnose-Plots erstellen
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Panel A: Rohdaten (stark verzerrte Skalen)
    sns.boxplot(data=df_raw, ax=axes[0, 0], palette="Pastel1")
    axes[0, 0].set_title("A: Originaldaten (Skalen dominieren einander)", fontweight="bold", fontsize=12)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Wertebereich (Log-Skala)")
    axes[0, 0].grid(axis="y", alpha=0.3)
    
    # Panel B: Z-Score Standardisierung
    sns.boxplot(data=df_zscore, ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_title("B: Z-Score Standardisierung (Mean=0, Std=1)", fontweight="bold", fontsize=12)
    axes[0, 1].axhline(0, color="red", linestyle="--", alpha=0.7)
    axes[0, 1].set_ylabel("Standardabweichungen (Z)")
    axes[0, 1].grid(axis="y", alpha=0.3)
    
    # Panel C: Min-Max Skalierung
    sns.boxplot(data=df_minmax, ax=axes[1, 0], palette="Accent")
    axes[1, 0].set_title("C: Min-Max Skalierung (Bereich [0, 1])", fontweight="bold", fontsize=12)
    axes[1, 0].set_ylabel("Skalierter Wert")
    axes[1, 0].grid(axis="y", alpha=0.3)
    
    # Panel D: Robust Scaler (Median=0, IQR=1)
    sns.boxplot(data=df_robust, ax=axes[1, 1], palette="Pastel2")
    axes[1, 1].set_title("D: Robust Scaler (Resistent gegen Ausreißer)", fontweight="bold", fontsize=12)
    axes[1, 1].axhline(0, color="red", linestyle="--", alpha=0.7)
    axes[1, 1].set_ylabel("IQR-Einheiten")
    axes[1, 1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("04_normalization_and_scaling.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[04_normalization_and_scaling.png]]
