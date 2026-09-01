### Overview
- **Purpose**: Test for global differences in central tendencies (means or medians) across three or more independent biological groups simultaneously, eliminating Family-Wise Error Rate (FWER) inflation.
- **Use Case**: Dose-response studies (Vehicle, Low, Mid, High dose), multi-timepoint longitudinal studies, or multi-cohort disease stage comparisons.
- **Prerequisites**: 
	* Python 3.9+
	* One continuous numeric response variable and one categorical grouping factor ($\ge 3$ levels)
- **Data Types**: 
	* Continuous numerical floats (ANOVA, Kruskal-Wallis)
	* Ordinal scores (Kruskal-Wallis)
- **Output**:
	* Omnibus $F$-statistic / $H$-statistic with associated $p$-values
	* Effect size estimates ($\eta^2$, $\epsilon^2$)
	* Comparative 3-group distribution and post-hoc evaluation figure (`08_anova_kruskal_wallis.png`)

#### When to Use

| Test Method | Distribution Requirement | Variance Assumption | Recommended Python Function | Note |
| :--- | :--- | :--- | :--- | :--- |
| **One-Way ANOVA** | Gaussian within each group | Homoscedastic (Levene $p > 0.05$) | `scipy.stats.f_oneway(*groups)` | Tests $H_0: \mu_1 = \mu_2 = \dots = \mu_k$ against at least one difference |
| **Welch’s ANOVA** | Gaussian | Heteroscedastic | `pingouin.welch_anova()` / Custom | Robust against unequal variances and unbalanced group sizes |
| **Kruskal-Wallis H-Test** | Non-Gaussian / Skewed | Distribution-free | `scipy.stats.kruskal(*groups)` | Non-parametric extension of Mann-Whitney for $\ge 3$ groups |
| **Post-Hoc Followup** | Significant Omnibus ($p < 0.05$) | Dictated by variance equality | Tukey HSD (ANOVA) / Dunn’s Test (Kruskal) | Mandatory follow-up step to pinpoint specific pair differences |

#### Decision Criteria
- **Use One-Way ANOVA when**: Normality is confirmed across all groups (Shapiro-Wilk $p > 0.05$) and variances are equal (Levene $p > 0.05$).
- **Use Kruskal-Wallis when**: Small sample size ($N < 10$ per group), ordinal endpoint, or non-normal distribution.
- **Crucial Rule**: Never perform post-hoc tests if the omnibus ANOVA / Kruskal-Wallis test is not statistically significant ($p \ge 0.05$).

### Python Libraries & Methods

| Aspect | `scipy.stats.f_oneway` | `scipy.stats.kruskal` |
| :--- | :--- | :--- |
| **Test Statistic** | $F = \frac{MS_{\text{between}}}{MS_{\text{within}}}$ | $H = \frac{12}{N(N+1)} \sum \frac{R_i^2}{n_i} - 3(N+1)$ |
| **Degrees of Freedom** | $(k - 1, N - k)$ | $k - 1$ ($\chi^2$ distribution) |
| **Input Format** | Unpacked 1D arrays `*groups` | Unpacked 1D arrays `*groups` |
| **Sensitivity to Outliers** | Moderate to High | Low (rank-based) |

### Quick Start Code

```bash
python -m pip install scipy numpy pandas statsmodels scikit-posthocs matplotlib seaborn
```

```python
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def generate_multigroup_data(seed: int = 42) -> pd.DataFrame:
    """Generates 3 dosage cohorts with dose-dependent biomarker response."""
    np.random.seed(seed)
    
    n_per_group = 24
    ctrl = np.random.normal(loc=100.0, scale=12.0, size=n_per_group)
    low_dose = np.random.normal(loc=112.0, scale=11.5, size=n_per_group)
    high_dose = np.random.normal(loc=135.0, scale=14.0, size=n_per_group)
    
    values = np.concatenate([ctrl, low_dose, high_dose])
    groups = ["Control"] * n_per_group + ["Low Dose (10mg)"] * n_per_group + ["High Dose (50mg)"] * n_per_group
    return pd.DataFrame({"Treatment": groups, "Enzyme_Activity": values})


def main() -> None:
    df = generate_multigroup_data()
    
    group_names = ["Control", "Low Dose (10mg)", "High Dose (50mg)"]
    groups = [df[df["Treatment"] == g]["Enzyme_Activity"].values for g in group_names]
    
    # 1. Assumption Checks
    p_norms = [stats.shapiro(g).pvalue for g in groups]
    stat_lev, p_lev = stats.levene(*groups, center="median")
    
    print("=== ASSUMPTION TESTING ===")
    for g, p in zip(group_names, p_norms):
        print(f"Shapiro-Wilk {g}: p = {p:.4f}")
    print(f"Levene Homoscedasticity: p = {p_lev:.4f}")
    
    # 2. Omnibus Tests
    f_stat, p_anova = stats.f_oneway(*groups)
    h_stat, p_kruskal = stats.kruskal(*groups)
    
    # Eta-squared calculation for ANOVA
    all_data = np.concatenate(groups)
    ss_total = np.sum((all_data - all_data.mean())**2)
    ss_between = sum(len(g) * (g.mean() - all_data.mean())**2 for g in groups)
    eta_sq = ss_between / ss_total
    
    print("\n=== OMNIBUS TEST RESULTS ===")
    print(f"One-Way ANOVA:       F = {f_stat:.3f}, p = {p_anova:.4e} (Eta² = {eta_sq:.3f})")
    print(f"Kruskal-Wallis:      H = {h_stat:.3f}, p = {p_kruskal:.4e}")
    
    # 3. Post-Hoc Pairwise Tests (Tukey HSD)
    tukey = pairwise_tukeyhsd(endog=df["Enzyme_Activity"], groups=df["Treatment"], alpha=0.05)
    print("\n=== TUKEY HSD POST-HOC COMPARISONS ===")
    print(tukey)
    
    # 4. Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Panel A: Boxplot mit Einzelwerten
    sns.boxplot(data=df, x="Treatment", y="Enzyme_Activity", ax=axes[0], palette="Blues", width=0.5)
    sns.stripplot(data=df, x="Treatment", y="Enzyme_Activity", ax=axes[0], color="black", alpha=0.5, jitter=0.2)
    axes[0].set_title(f"A: Multi-Gruppen Vergleich\nOne-Way ANOVA F={f_stat:.2f} (p={p_anova:.2e})", fontweight="bold", fontsize=12)
    axes[0].set_ylabel("Enzymaktivität (U/L)")
    axes[0].grid(axis="y", alpha=0.3)
    
    # Panel B: Post-Hoc Konfidenzintervalle (Tukey Differences)
    res_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    comps = [f"{row['group1']} vs.\n{row['group2']}" for _, row in res_df.iterrows()]
    diffs = res_df["meandiff"]
    errs = [res_df["meandiff"] - res_df["lower"], res_df["upper"] - res_df["meandiff"]]
    
    axes[1].errorbar(diffs, range(len(comps)), xerr=errs, fmt="o", color="#08519C", ecolor="#3182BD",
                     elinewidth=2, capsize=5, markersize=8)
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.7)
    axes[1].set_yticks(range(len(comps)))
    axes[1].set_yticklabels(comps)
    axes[1].set_title("B: Tukey HSD 95% Konfidenzintervalle der Differenzen", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Mittelwert-Differenz (95% CI)")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("08_anova_kruskal_wallis.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[08_anova_kruskal_wallis.png]]
