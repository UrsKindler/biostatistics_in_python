### Overview
- **Purpose**: Compare location parameters (means or median ranks) between two independent or paired experimental groups, identifying statistically significant phenotypic or biomarker differences.
- **Use Case**: Wildtype vs. Knockout comparison, Treatment vs. Vehicle control, Pre- vs. Post-intervention clinical trials.
- **Prerequisites**: 
	* Python 3.9+
	* Two experimental cohorts (independent or matched pairs)
- **Data Types**: 
	* Continuous numerical (T-Test, Mann-Whitney)
	* Ordinal scores / ranks (Mann-Whitney)
- **Output**:
	* Test statistics ($t, U$), $p$-values, effect sizes
	* Automated assumption-checking workflow
	* Dual-panel comparative distribution figure (`07_two_group_comparisons.png`)

#### When to Use

| Experimental Design | Distribution Assumption | Recommended Test | Python Implementation | Note |
| :--- | :--- | :--- | :--- | :--- |
| **2 Independent Groups** | Gaussian & Equal Variances | Student’s Two-Sample T-Test | `scipy.stats.ttest_ind(g1, g2, equal_var=True)` | Classic pooled variance $t$-test |
| **2 Independent Groups** | Gaussian & Unequal Variances | Welch’s T-Test | `scipy.stats.ttest_ind(g1, g2, equal_var=False)` | **Default recommendation** for parametric 2-group comparisons |
| **2 Independent Groups** | Non-Gaussian / Skewed / Ordinal | Mann-Whitney U-Test | `scipy.stats.mannwhitneyu(g1, g2, alternative='two-sided')` | Rank-sum test; distribution-free |
| **Paired / Matched Groups** | Gaussian Differences | Paired T-Test | `scipy.stats.ttest_rel(pre, post)` | Evaluates $d_i = \text{post}_i - \text{pre}_i$ against zero |
| **Paired / Matched Groups** | Non-Gaussian Differences | Wilcoxon Signed-Rank Test | `scipy.stats.wilcoxon(pre, post)` | Non-parametric paired test on ranked signed differences |

#### Decision Criteria
- **Use Student / Welch T-Test when**: Normality is satisfied (Shapiro-Wilk $p > 0.05$). Prefer Welch's T-Test whenever group sample sizes differ ($N_1 \neq N_2$).
- **Use Mann-Whitney U-Test when**: Samples fail normality, sample sizes are very small ($N < 10$), or data is ordinal.
- **Don't use when**: Comparing $\ge 3$ groups (use ANOVA or Kruskal-Wallis instead to prevent Type-I error inflation).

### Python Libraries & Methods

| Aspect | `scipy.stats.ttest_ind` | `scipy.stats.mannwhitneyu` | `scipy.stats.ttest_rel` |
| :--- | :--- | :--- | :--- |
| **Design** | Independent 2 groups | Independent 2 groups | Paired (same subject across 2 conditions) |
| **Parameter Tested** | Difference between Means ($\mu_1 - \mu_2$) | Stochastic superiority / Median Rank | Mean of pairwise differences ($\bar{d}$) |
| **Key Arguments** | `equal_var=False`, `alternative='two-sided'` | `alternative='two-sided'`, `use_continuity=True` | `alternative='two-sided'` |
| **Returns** | `TtestResult(statistic, pvalue)` | `MannwhitneyuResult(statistic, pvalue)` | `TtestResult(statistic, pvalue)` |

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


def generate_two_group_data(seed: int = 42) -> pd.DataFrame:
    """Simulates independent control and treated biological samples."""
    np.random.seed(seed)
    
    n_ctrl, n_treat = 28, 25
    
    # Control group (Baseline)
    ctrl = np.random.normal(loc=12.4, scale=1.8, size=n_ctrl)
    
    # Treated group (Upregulated biomarker with increased variance)
    treat = np.random.normal(loc=15.1, scale=2.9, size=n_treat)
    
    data = np.concatenate([ctrl, treat])
    labels = ["Control"] * n_ctrl + ["Treated"] * n_treat
    return pd.DataFrame({"Condition": labels, "Log2_Intensity": data})


def main() -> None:
    df = generate_two_group_data()
    
    ctrl_vals = df[df["Condition"] == "Control"]["Log2_Intensity"].values
    treat_vals = df[df["Condition"] == "Treated"]["Log2_Intensity"].values
    
    # 1. Voraussetzungsprüfungen
    _, p_norm_ctrl = stats.shapiro(ctrl_vals)
    _, p_norm_treat = stats.shapiro(treat_vals)
    _, p_levene = stats.levene(ctrl_vals, treat_vals)
    
    print("=== ASSUMPTION CHECKS ===")
    print(f"Shapiro-Wilk Control: p = {p_norm_ctrl:.4f}")
    print(f"Shapiro-Wilk Treated: p = {p_norm_treat:.4f}")
    print(f"Levene Test (Varianz): p = {p_levene:.4f}")
    
    # 2. Statistische Tests durchführen
    # Welch's T-Test (Parametrisch)
    t_stat, p_welch = stats.ttest_ind(ctrl_vals, treat_vals, equal_var=False)
    
    # Mann-Whitney U-Test (Nicht-parametrisch)
    u_stat, p_mwu = stats.mannwhitneyu(ctrl_vals, treat_vals, alternative="two-sided")
    
    # Cohen's d Effektstärke
    n1, n2 = len(ctrl_vals), len(treat_vals)
    s_pool = np.sqrt(((n1-1)*ctrl_vals.var(ddof=1) + (n2-1)*treat_vals.var(ddof=1)) / (n1+n2-2))
    cohen_d = (treat_vals.mean() - ctrl_vals.mean()) / s_pool
    
    print("\n=== HYPOTHESIS TESTING RESULTS ===")
    print(f"Welch's T-Test:       t = {t_stat:.3f}, p = {p_welch:.4e}")
    print(f"Mann-Whitney U-Test:  U = {u_stat:.1f}, p = {p_mwu:.4e}")
    print(f"Cohen's d (Effekt):   d = {cohen_d:.3f} (Großer biologischer Effekt)")
    
    # 3. Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Boxplot + Stripplot
    sns.boxplot(data=df, x="Condition", y="Log2_Intensity", ax=axes[0], palette="Set2", width=0.45)
    sns.stripplot(data=df, x="Condition", y="Log2_Intensity", ax=axes[0], color="black", alpha=0.6, jitter=0.2, s=6)
    
    # Signifikanzbalken einzeichnen
    y_max = df["Log2_Intensity"].max() + 0.8
    axes[0].plot([0, 0, 1, 1], [y_max, y_max+0.3, y_max+0.3, y_max], color="black", lw=1.5)
    sig_label = f"Welch t-test: p = {p_welch:.2e} (d = {cohen_d:.2f})"
    axes[0].text(0.5, y_max+0.5, sig_label, ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    axes[0].set_title("A: Gruppenvergleich (Log2 Intensität)", fontweight="bold", fontsize=12)
    axes[0].set_ylabel("Log2 Intensität")
    axes[0].set_ylim(df["Log2_Intensity"].min() - 1, y_max + 2.0)
    axes[0].grid(axis="y", alpha=0.3)
    
    # Panel B: Kernel Density Estimation (KDE)
    sns.kdeplot(ctrl_vals, ax=axes[1], label=f"Control (Mean={ctrl_vals.mean():.2f})", fill=True, color="#66C2A5", alpha=0.4)
    sns.kdeplot(treat_vals, ax=axes[1], label=f"Treated (Mean={treat_vals.mean():.2f})", fill=True, color="#FC8D62", alpha=0.4)
    axes[1].set_title("B: Verteilungsverschiebung (Dichte)", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Log2 Intensität")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("07_two_group_comparisons.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[07_two_group_comparisons.png]]
