### Overview
- **Purpose**: Conduct pairwise multiple comparisons to determine exactly which specific pairs of experimental groups differ significantly after a statistically significant omnibus test (ANOVA or Kruskal-Wallis).
- **Use Case**: Multi-arm preclinical trials, multi-stage disease models, or gene expression across $>2$ cell types.
- **Prerequisites**: 
	* Python 3.9+
	* Statistically significant omnibus test ($p < 0.05$)
	* Categorical factor with $\ge 3$ levels and continuous/ordinal outcome
- **Data Types**: 
	* Continuous numerical (Tukey, Games-Howell, Bonferroni)
	* Ordinal ranks (Dunn's test)
- **Output**:
	* Pairwise difference estimates, adjusted $p$-values, and confidence intervals
	* 2-Panel pairwise significance matrix and difference plot (`09_1_posthoc_tests.png`)

#### When to Use

| Omnibus Test Used | Variance Assumption | Recommended Post-Hoc Test | Python Library & Function | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Standard ANOVA** | Homoscedastic (Equal Variances) | **Tukey's HSD** (Honestly Significant Difference) | `statsmodels.stats.multicomp.pairwise_tukeyhsd` | Optimal power; calculates studentized range distribution $q$; controls FWER |
| **Welch's ANOVA** | Heteroscedastic (Unequal Variances) | **Games-Howell Test** | `scikit_posthocs.posthoc_gameshowell` | Welch-like post-hoc test for unequal variances and unbalanced sample sizes |
| **Kruskal-Wallis** | Non-Parametric / Skewed | **Dunn's Test** with FDR/Holm adjustment | `scikit_posthocs.posthoc_dunn` | Non-parametric pairwise rank comparisons |
| **General Multiple $t$-Tests** | Any Distribution | Pairwise $t$-test + **Holm/Bonferroni** | `statsmodels.stats.multitest.multipletests` | Step-down Holm is strictly more powerful than standard Bonferroni |

#### Decision Criteria
- **Use always**: After an omnibus ANOVA ($F$-test) or Kruskal-Wallis test reaches $p < 0.05$.
- **Never use**: If the omnibus test was non-significant ($p \ge 0.05$), to prevent unadjusted Type-I error inflation.
- **Critical choice**: Always use Games-Howell instead of Tukey HSD if Levene's test rejected variance equality.

### Python Libraries & Methods

| Aspect | `pairwise_tukeyhsd` | `posthoc_dunn` | `posthoc_gameshowell` |
| :--- | :--- | :--- | :--- |
| **Package** | `statsmodels` | `scikit-posthocs` | `scikit-posthocs` |
| **Preceding Test** | Standard ANOVA | Kruskal-Wallis | Welch's ANOVA |
| **P-Value Adjustment** | Studentized Range ($q$) | FDR (BH), Bonferroni, Holm | Welch-Satterthwaite $t$ |
| **Returns** | `TukeyHSDResults` object with summary table | Pairwise $P \times P$ symmetric DataFrame | Pairwise $P \times P$ symmetric DataFrame |

### Quick Start Code

```bash
python -m pip install statsmodels scikit-posthocs pandas numpy scipy matplotlib seaborn
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


def generate_posthoc_cohorts(seed: int = 42) -> pd.DataFrame:
    """Simulates 4 treatment cohorts with distinct efficacy."""
    np.random.seed(seed)
    
    n = 20
    ctrl = np.random.normal(loc=50.0, scale=6.0, size=n)
    drug_a = np.random.normal(loc=53.0, scale=6.5, size=n)    # Mild effect (not sig vs ctrl)
    drug_b = np.random.normal(loc=68.0, scale=7.0, size=n)    # Strong effect
    drug_comb = np.random.normal(loc=76.0, scale=6.2, size=n) # Synergistic combination
    
    values = np.concatenate([ctrl, drug_a, drug_b, drug_comb])
    labels = ["Control"] * n + ["Drug A"] * n + ["Drug B"] * n + ["Combination"] * n
    return pd.DataFrame({"Treatment": labels, "Biomarker": values})


def main() -> None:
    df = generate_posthoc_cohorts()
    
    # 1. Omnibus ANOVA
    groups = [df[df["Treatment"] == g]["Biomarker"].values for g in df["Treatment"].unique()]
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"Omnibus ANOVA: F = {f_stat:.3f}, p = {p_anova:.4e}")
    
    # 2. Tukey HSD Post-Hoc
    tukey = pairwise_tukeyhsd(endog=df["Biomarker"], groups=df["Treatment"], alpha=0.05)
    print("\n=== TUKEY HSD SUMMARY ===")
    print(tukey)
    
    # 3. Dunn's Non-Parametric Post-Hoc Test
    dunn_matrix = sp.posthoc_dunn(df, val_col="Biomarker", group_col="Treatment", p_adjust="fdr_bh")
    print("\n=== DUNN'S TEST (ADJUSTED P-VALUE MATRIX) ===")
    print(dunn_matrix.round(4))
    
    # 4. Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel A: Tukey Konfidenzintervalle
    res_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    y_pos = range(len(res_df))
    comparisons = [f"{r['group1']} vs. {r['group2']}" for _, r in res_df.iterrows()]
    diffs = res_df["meandiff"]
    err_low = res_df["meandiff"] - res_df["lower"]
    err_high = res_df["upper"] - res_df["meandiff"]
    
    for i, (d, el, eh, sig) in enumerate(zip(diffs, err_low, err_high, res_df["reject"])):
        col = "#2CA02C" if sig else "#7F7F7F"
        axes[0].errorbar(d, i, xerr=[[el], [eh]], fmt="o", color=col, ecolor=col,
                         elinewidth=2.5, capsize=5, markersize=8)
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.7)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(comparisons)
    axes[0].set_title("A: Tukey HSD 95% Differenz-Konfidenzintervalle\n(Grün = Signifikant verwerfend)",
                      fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Mittelwertsdifferenz")
    axes[0].grid(axis="x", alpha=0.3)
    
    # Panel B: Dunn's p-Wert Signifikanz Heatmap
    sns.heatmap(dunn_matrix, annot=True, fmt=".3e", cmap="YlGnBu_r", vmin=0, vmax=0.05,
                cbar_kws={"label": "FDR-adjustierter p-Wert"}, ax=axes[1])
    axes[1].set_title("B: Dunn Post-Hoc Pairwise p-Werte (FDR-adjustiert)", fontweight="bold", fontsize=11)
    
    plt.tight_layout()
    
    output_file = Path("09_1_posthoc_tests.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[09_1_posthoc_tests.png]]
