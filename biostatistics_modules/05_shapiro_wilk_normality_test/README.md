### Overview
- **Purpose**: Test the null hypothesis ($H_0$) that a sample came from a normally distributed population, directing the choice between parametric (T-Test, ANOVA, Pearson) and non-parametric (Mann-Whitney, Kruskal-Wallis, Spearman) statistical methods.
- **Use Case**: Testing continuous clinical laboratory values, gene/protein intensities, physiological metrics, or regression model residuals.
- **Prerequisites**: 
	* Python 3.9+
	* Univariate continuous numerical vector ($N \ge 3$)
- **Data Types**: 
	* Continuous numerical floats / integers
- **Output**:
	* Test statistic $W$ and $p$-value
	* Clear decision verdict (Parametric vs. Non-Parametric)
	* Multi-panel normality diagnostic plot (`05_shapiro_wilk_normality.png`) showing empirical histograms, fitted Gaussian curves, and Q-Q probability plots

#### When to Use

| Sample Size ($N$) | Recommended Normality Test | Python Function | Note |
| :--- | :--- | :--- | :--- |
| **$3 \le N \le 50$** | Shapiro-Wilk Test | `scipy.stats.shapiro(x)` | Highest statistical power for small to medium biological sample sizes |
| **$50 < N \le 200$** | Anderson-Darling Test | `scipy.stats.anderson(x, dist='norm')` | Gives more weight to distribution tails |
| **$N > 200$** | D’Agostino & Pearson Test | `scipy.stats.normaltest(x)` | Combines skewness and kurtosis; Shapiro-Wilk becomes overly sensitive at large $N$ |
| **All Sample Sizes** | Quantile-Quantile (Q-Q) Plot | `scipy.stats.probplot(x, plot=plt)` | Visual inspection is mandatory (detects multimodality, skew, heavy tails) |

#### Decision Criteria
- **Use always**: Prior to conducting parametric hypothesis tests or computing Pearson correlations.
- **Essential when**: Small clinical cohort studies ($N < 30$) where Central Limit Theorem (CLT) cannot yet be assumed.
- **Critical rule**: 
  - $p > 0.05 \implies H_0$ not rejected $\implies$ **Parametric tests allowed**.
  - $p \le 0.05 \implies H_0$ rejected (non-Gaussian) $\implies$ **Use Non-Parametric tests or Log-Transform**.
- **Don't skip when**: Reporting biomarker significance in peer-reviewed biological journals.

### Python Libraries & Methods

| Aspect | `scipy.stats.shapiro` | `scipy.stats.normaltest` | `scipy.stats.probplot` |
| :--- | :--- | :--- | :--- |
| **Mathematical Basis** | Correlation between ordered sample and normal order statistics | Omnibus test based on Skewness & Kurtosis | Graphical correlation against theoretical Gaussian quantiles |
| **Sample Size Limits** | $3 \le N \le 5000$ | $N \ge 20$ (recommended) | Any $N \ge 5$ |
| **Sensitivity to Outliers** | High | High | High (visible on plot edges) |
| **Output** | `ShapiroResult(statistic, pvalue)` | `NormaltestResult(statistic, pvalue)` | `(osm, osr), (slope, intercept, r)` |

### Quick Start Code

```bash
python -m pip install scipy numpy pandas matplotlib seaborn
```

```python
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def generate_normality_datasets(n_samples: int = 45, seed: int = 42) -> dict[str, np.ndarray]:
    """Generates a Gaussian and a Log-Normal biological distribution."""
    np.random.seed(seed)
    
    # Gaussian Biomarker (e.g., Blood pH / Albumin)
    normal_data = np.random.normal(loc=4.2, scale=0.4, size=n_samples)
    
    # Skewed Biomarker (e.g., Serum IL-6 pg/mL)
    skewed_data = np.random.lognormal(mean=2.0, sigma=0.8, size=n_samples)
    
    return {
        "Gaussian Biomarker": normal_data,
        "Skewed Biomarker (IL-6)": skewed_data
    }


def main() -> None:
    datasets = generate_normality_datasets()
    
    print("=== SHAPIRO-WILK NORMALITY TEST SUMMARY ===")
    
    results = {}
    for name, data in datasets.items():
        stat, p_val = stats.shapiro(data)
        is_normal = p_val > 0.05
        results[name] = {"W": stat, "p": p_val, "Normal": is_normal}
        
        print(f"\n{name}:")
        print(f"  Shapiro-Wilk W: {stat:.4f}, p-value: {p_val:.4e}")
        if is_normal:
            print("  ✓ Normalverteilung bestätigt (p > 0.05) -> Parametrische Tests (T-Test, ANOVA) empfohlen.")
        else:
            print("  ✗ Abweichung von Normalverteilung (p <= 0.05) -> Nicht-parametrische Tests (Mann-Whitney, Kruskal) empfohlen.")
            
    # Visualisierung: Histogramm + Q-Q Plot für beide Variablen
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for row_idx, (name, data) in enumerate(datasets.items()):
        res = results[name]
        
        # 1. Histogramm + Fitted Normal
        sns.histplot(data, kde=True, stat="density", ax=axes[row_idx, 0], color="#2C7FB8", edgecolor="k")
        x_grid = np.linspace(data.min(), data.max(), 100)
        axes[row_idx, 0].plot(x_grid, stats.norm.pdf(x_grid, np.mean(data), np.std(data)),
                              "r--", lw=2, label="Theoretische Normalverteilung")
        axes[row_idx, 0].set_title(f"{name}: Dichte vs. Normal (p={res['p']:.3e})", fontweight="bold", fontsize=11)
        axes[row_idx, 0].set_ylabel("Dichte")
        axes[row_idx, 0].legend()
        axes[row_idx, 0].grid(alpha=0.3)
        
        # 2. Q-Q Plot
        stats.probplot(data, dist="norm", plot=axes[row_idx, 1])
        axes[row_idx, 1].get_lines()[0].set_markerfacecolor("#2C7FB8")
        axes[row_idx, 1].get_lines()[0].set_markeredgecolor("k")
        axes[row_idx, 1].get_lines()[1].set_color("red")
        axes[row_idx, 1].get_lines()[1].set_linewidth(2)
        axes[row_idx, 1].set_title(f"{name}: Q-Q Plot", fontweight="bold", fontsize=11)
        axes[row_idx, 1].grid(alpha=0.3)
        
    plt.tight_layout()
    
    output_file = Path("05_shapiro_wilk_normality.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[05_shapiro_wilk_normality.png]]
