### Overview
- **Purpose**: Statistically test whether the variance of a continuous dependent variable is equal across two or more categorical groups (homoscedasticity assumption).
- **Use Case**: Prerequisite testing before applying standard Student's T-Test, Pooled Two-Sample Tests, or Classic One-Way ANOVA.
- **Prerequisites**: 
	* Python 3.9+
	* Continuous numerical measurement grouped by a discrete factor ($\ge 2$ levels)
- **Data Types**: 
	* Continuous numeric response variable
	* Categorical / discrete group labels
- **Output**:
	* Test statistics and p-values (Levene, Bartlett, Fligner-Killeen)
	* Decision recommendation (Standard ANOVA vs. Welch's ANOVA)
	* Multi-panel diagnostic plot (`02_variance_homogeneity_tests.png`)

#### When to Use

| Situation | Recommended Test | Python Implementation | Note |
| :--- | :--- | :--- | :--- |
| **Normal Distribution Confirmed** | Bartlett's Test | `scipy.stats.bartlett(*groups)` | Highest statistical power for strictly Gaussian data; sensitive to non-normality |
| **Moderate Skewness / General Case** | Levene's Test (Center = Median) | `scipy.stats.levene(*groups, center='median')` | Brown-Forsythe variant; robust against outliers and moderate distribution departures |
| **Heavy-Tailed / Non-Normal Data** | Fligner-Killeen Test | `scipy.stats.fligner(*groups)` | Fully non-parametric rank-based test for equal scale parameters |
| **Heteroscedasticity Detected ($p < 0.05$)** | Welch's Correction | `scipy.stats.ttest_ind(equal_var=False)` | Use Welch's T-Test or Welch's ANOVA; do not force standard pooled models |

#### Decision Criteria
- **Use always**: Before running pooled T-tests or standard One-Way ANOVA with unequal group sizes ($N_1 \neq N_2$).
- **Essential when**: Sample sizes vary significantly across experimental arms (heteroscedasticity severely inflates Type-I error rates).
- **Critical for**: Selecting between standard Tukey HSD and Games-Howell post-hoc pairwise comparisons.
- **Don't skip when**: Analyzing primary biological endpoints across distinct disease severities.

### Python Libraries & Methods

| Aspect | `scipy.stats.levene` | `scipy.stats.bartlett` | `scipy.stats.fligner` |
| :--- | :--- | :--- | :--- |
| **Underlying Principle** | ANOVA on absolute residuals from median | Likelihood-ratio test of pooled variances | Chi-square test on normal-score transformed ranks |
| **Normality Assumption** | No (robust) | Yes (strict) | No (distribution-free) |
| **Robustness to Outliers** | High | Low | High |
| **Key Arguments** | `center='median'`, `center='mean'` | Positional group arrays `*groups` | Positional group arrays `*groups` |
| **Output** | `(statistic, pvalue)` | `(statistic, pvalue)` | `(statistic, pvalue)` |

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


def generate_heteroscedastic_cohorts(seed: int = 42) -> pd.DataFrame:
    """Simulates 3 clinical groups with distinct variance structures."""
    np.random.seed(seed)
    
    n_ctrl, n_mild, n_severe = 35, 30, 25
    
    # Control: low variance, Mild: moderate variance, Severe: high variance
    ctrl = np.random.normal(loc=100.0, scale=8.0, size=n_ctrl)
    mild = np.random.normal(loc=108.0, scale=16.0, size=n_mild)
    severe = np.random.normal(loc=115.0, scale=28.0, size=n_severe)
    
    data = np.concatenate([ctrl, mild, severe])
    labels = ["Control"] * n_ctrl + ["Mild Disease"] * n_mild + ["Severe Disease"] * n_severe
    
    return pd.DataFrame({"Group": labels, "Biomarker_Level": data})


def main() -> None:
    df = generate_heteroscedastic_cohorts()
    
    # Gruppen trennen
    group_names = ["Control", "Mild Disease", "Severe Disease"]
    groups = [df[df["Group"] == g]["Biomarker_Level"].values for g in group_names]
    
    # 1. Statistische Tests durchführen
    stat_levene, p_levene = stats.levene(*groups, center="median")
    stat_bartlett, p_bartlett = stats.bartlett(*groups)
    stat_fligner, p_fligner = stats.fligner(*groups)
    
    print("=== VARIANZHOMOGENITÄTS-PRÜFUNG ===")
    print(f"Levene-Test (Brown-Forsythe): W = {stat_levene:.3f}, p = {p_levene:.4e}")
    print(f"Bartlett-Test:                T = {stat_bartlett:.3f}, p = {p_bartlett:.4e}")
    print(f"Fligner-Killeen-Test:         X² = {stat_fligner:.3f}, p = {p_fligner:.4e}")
    
    if p_levene > 0.05:
        print("\n✓ Entscheidung: Varianzen sind homogen (p > 0.05). Standard-ANOVA zulässig.")
    else:
        print("\n✗ Entscheidung: Heteroskedastizität nachgewiesen (p <= 0.05).")
        print("  -> Empfehlung: Welch's ANOVA bzw. Welch's T-Test und Games-Howell Post-Hoc verwenden!")
    
    # 2. Diagnose-Plots erstellen
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: Boxplot mit Einzelwerten
    sns.boxplot(data=df, x="Group", y="Biomarker_Level", ax=axes[0], palette="Blues", width=0.5)
    sns.stripplot(data=df, x="Group", y="Biomarker_Level", ax=axes[0], color="black", alpha=0.5, jitter=0.2)
    axes[0].set_title("A: Gruppen-Streuung (Boxplot)", fontweight="bold", fontsize=12)
    axes[0].set_ylabel("Biomarker Level (pg/mL)")
    axes[0].grid(axis="y", alpha=0.3)
    
    # Panel B: Dichteverteilungen
    for g in group_names:
        sns.kdeplot(df[df["Group"] == g]["Biomarker_Level"], ax=axes[1], label=g, lw=2)
    axes[1].set_title("B: Verteilungsbreite (KDE)", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Biomarker Level")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Panel C: Gruppen-Varianzen (Balkendiagramm)
    variances = [np.var(g, ddof=1) for g in groups]
    bars = axes[2].bar(group_names, variances, color=["#6BAED6", "#3182BD", "#08519C"], edgecolor="k")
    axes[2].set_title(f"C: Varianzen (Levene p={p_levene:.3e})", fontweight="bold", fontsize=12)
    axes[2].set_ylabel("Empirische Varianz (s²)")
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].grid(axis="y", alpha=0.3)
    
    # Text-Overlay
    verdict = "Heteroskedastisch (p < 0.05)" if p_levene < 0.05 else "Homogen (p >= 0.05)"
    axes[2].text(0.5, 0.90, verdict, transform=axes[2].transAxes, ha="center",
                 fontweight="bold", color="red" if p_levene < 0.05 else "green",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    output_file = Path("02_variance_homogeneity_tests.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[02_variance_homogeneity_tests.png]]
