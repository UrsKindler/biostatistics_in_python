### Overview
- **Purpose**: Measure the strength and direction of association between two continuous or ranked variables, evaluating either linear relationships (Pearson) or monotonic relationships (Spearman / Kendall).
- **Use Case**: Co-expression network construction, multi-omics biomarker association studies, dose-response relationships, or clinical feature selection.
- **Prerequisites**: 
	* Python 3.9+
	* Paired numerical observations $(x_i, y_i)$
- **Data Types**: 
	* Continuous numerical (Pearson, Spearman)
	* Ordinal ranked variables (Spearman, Kendall)
- **Output**:
	* Correlation coefficient ($r, \rho, \tau \in [-1, 1]$) and two-sided $p$-value
	* Automated smart-selection correlation function
	* Dual-panel correlation scatter plot with regression curves (`06_correlation_analysis.png`)

#### When to Use

| Correlation Method | Relationship Assumed | Distribution Requirement | Recommended Python Function | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Pearson ($r$)** | Linear | Bivariate Normal; no heavy outliers | `scipy.stats.pearsonr(x, y)` | Measures how close points lie to a straight line; sensitive to outliers |
| **Spearman ($\rho$)** | Monotonic | Non-parametric (any continuous/ordinal) | `scipy.stats.spearmanr(x, y)` | Pearson correlation on ranked values; robust to non-linear monotonic trends |
| **Kendall ($\tau$)** | Monotonic | Non-parametric (small sample size) | `scipy.stats.kendalltau(x, y)` | Concordant vs. discordant pairs; best for small $N$ with tied ranks |
| **Matrix Correlation** | Linear / Monotonic | Batch processing | `df.corr(method='pearson'/'spearman')` | Efficient pairwise matrix computation across hundreds of features |

#### Decision Criteria
- **Use Pearson when**: Both variables are continuous, normally distributed (Shapiro-Wilk $p > 0.05$), and linear relationship is expected.
- **Use Spearman when**: Data exhibits non-Gaussian skew, contains ordinal ranks, or the relationship is curved/monotonic.
- **Critical rule**: Correlation $\neq$ Causation. Always visually inspect scatter plots to avoid Anscombe's Quartet pitfalls.
- **Don't skip when**: Performing gene-gene or metabolite-protein association scans.

### Python Libraries & Methods

| Aspect | `scipy.stats.pearsonr` | `scipy.stats.spearmanr` | `pandas.DataFrame.corr` |
| :--- | :--- | :--- | :--- |
| **Input** | 1D Arrays $(x, y)$ | 1D Arrays $(x, y)$ | 2D DataFrame |
| **Statistical Test** | Exact $t$-distribution test for $r=0$ | Permutation / $t$-approximation for $\rho=0$ | Matrix of pairwise coefficients |
| **Returns $p$-value** | Yes | Yes | No (requires custom wrapper / scipy) |
| **Missing Value Handling** | Requires clean arrays | Requires clean arrays | `dropna` / pairwise complete |

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


def generate_correlation_data(n_samples: int = 60, seed: int = 42) -> pd.DataFrame:
    """Simulates linear and non-linear biological associations."""
    np.random.seed(seed)
    
    # 1. Linear Biomarker Pair (e.g., Protein A vs. Target Pathway B)
    x_linear = np.random.normal(loc=50, scale=10, size=n_samples)
    y_linear = 0.75 * x_linear + np.random.normal(0, 4, size=n_samples)
    
    # 2. Non-Linear Monotonic Exponential Pair (e.g., Drug Dosage vs. Cell Inhibition)
    x_nonlin = np.random.uniform(1, 10, size=n_samples)
    y_nonlin = 100 / (1 + np.exp(-1.2 * (x_nonlin - 5))) + np.random.normal(0, 3, size=n_samples)
    
    return pd.DataFrame({
        "Protein_A": x_linear,
        "Protein_B": y_linear,
        "Drug_Concentration": x_nonlin,
        "Inhibition_Rate": y_nonlin
    })


def smart_correlation(x: pd.Series, y: pd.Series) -> tuple[float, float, str]:
    """Automatically selects Pearson or Spearman based on Shapiro-Wilk normality tests."""
    clean_df = pd.DataFrame({"x": x, "y": y}).dropna()
    x_c, y_c = clean_df["x"], clean_df["y"]
    
    _, p_norm_x = stats.shapiro(x_c)
    _, p_norm_y = stats.shapiro(y_c)
    
    if p_norm_x > 0.05 and p_norm_y > 0.05:
        res = stats.pearsonr(x_c, y_c)
        return res.statistic, res.pvalue, "Pearson (Normal)"
    else:
        res = stats.spearmanr(x_c, y_c)
        return res.statistic, res.pvalue, "Spearman (Non-Normal)"


def main() -> None:
    df = generate_correlation_data()
    
    # 1. Berechnungen
    r_lin, p_lin, method_lin = smart_correlation(df["Protein_A"], df["Protein_B"])
    r_nonlin, p_nonlin, method_nonlin = smart_correlation(df["Drug_Concentration"], df["Inhibition_Rate"])
    
    print("=== KORRELATIONSANALYSE ERGEBNISSE ===")
    print(f"\n1. Protein_A vs. Protein_B:")
    print(f"   Methode: {method_lin}")
    print(f"   Koeffizient: {r_lin:.3f}, p-Wert: {p_lin:.4e}")
    
    print(f"\n2. Drug_Concentration vs. Inhibition_Rate:")
    print(f"   Methode: {method_nonlin}")
    print(f"   Koeffizient: {r_nonlin:.3f}, p-Wert: {p_nonlin:.4e}")
    
    # 2. Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel A: Linearer Zusammenhang
    sns.regplot(data=df, x="Protein_A", y="Protein_B", ax=axes[0], color="#2C7FB8",
                scatter_kws={"alpha": 0.7, "edgecolor": "k", "s": 50})
    axes[0].set_title(f"A: Lineare Assoziation ({method_lin})\nr = {r_lin:.3f}, p = {p_lin:.2e}",
                      fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Protein A Abundanz")
    axes[0].set_ylabel("Protein B Abundanz")
    axes[0].grid(alpha=0.3)
    
    # Panel B: Monoton-nichtlinearer Zusammenhang
    sns.scatterplot(data=df, x="Drug_Concentration", y="Inhibition_Rate", ax=axes[1],
                    color="#E6550D", alpha=0.8, edgecolor="k", s=55)
    sns.lineplot(data=df.sort_values("Drug_Concentration"), x="Drug_Concentration", y="Inhibition_Rate",
                 ax=axes[1], color="red", lw=2, linestyle="--", label="Monotone Kurve")
    axes[1].set_title(f"B: Sigmoidaler / Monotoner Trend ({method_nonlin})\nrho = {r_nonlin:.3f}, p = {p_nonlin:.2e}",
                      fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Drug Konzentration (uM)")
    axes[1].set_ylabel("Inhibitionsrate (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("06_correlation_analysis.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[06_correlation_analysis.png]]
