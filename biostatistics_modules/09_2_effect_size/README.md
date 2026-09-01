### Overview
- **Purpose**: Quantify the absolute or standardized magnitude of an observed biological effect, experimental treatment difference, or association, completely independent of sample size $N$.
- **Use Case**: Distinguishing true biological relevance from mere statistical significance ($p < 0.05$ driven by large $N$), meta-analyses, and sample size / power calculations.
- **Prerequisites**: 
	* Python 3.9+
	* Summary statistics (means, standard deviations, sample counts) or raw observation arrays
- **Data Types**: 
	* Continuous numerical (Cohen's $d$, Hedges' $g$, Eta-squared $\eta^2$)
	* Categorical contingency tables (Cramér's $V$, Odds Ratio)
- **Output**:
	* Standardized effect size point estimates and rule-of-thumb qualitative classifications
	* Comparative effect size benchmark chart (`09_2_effect_size_analysis.png`)

#### When to Use

| Statistical Setting | Effect Size Metric | Mathematical Definition | Benchmark Thresholds | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Two Independent Groups ($t$-Test)** | **Cohen’s $d$** | $d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$ | Small: $0.2$, Medium: $0.5$, Large: $0.8$ | Standardized mean difference |
| **Two Small Groups ($N < 20$)** | **Hedges’ $g$** | $g \approx d \cdot \left(1 - \frac{3}{4(n_1+n_2) - 9}\right)$ | Small: $0.2$, Medium: $0.5$, Large: $0.8$ | Corrects for small sample overestimation bias in Cohen's $d$ |
| **Multi-Group ANOVA** | **Eta-Squared ($\eta^2$)** | $\eta^2 = \frac{SS_{\text{between}}}{SS_{\text{total}}}$ | Small: $0.01$, Medium: $0.06$, Large: $0.14$ | Proportion of total variance explained |
| **Multi-Group ANOVA (Unbiased)** | **Omega-Squared ($\omega^2$)** | $\omega^2 = \frac{SS_{\text{between}} - (k-1)MS_{\text{within}}}{SS_{\text{total}} + MS_{\text{within}}}$ | Small: $0.01$, Medium: $0.06$, Large: $0.14$ | Unbiased population variance proportion estimate |
| **Contingency Tables / $\chi^2$** | **Cramér’s $V$** | $V = \sqrt{\frac{\chi^2}{N \cdot \min(r-1, c-1)}}$ | Small: $0.1$, Medium: $0.3$, Large: $0.5$ | Association between categorical nominal variables |

#### Decision Criteria
- **Use always**: Alongside every reported $p$-value in scientific publications and clinical reports.
- **Essential when**: Very large sample sizes ($N > 1000$) where trivial, biologically meaningless differences yield $p < 0.001$.
- **Critical for**: Power analysis and calculating required sample size for future grant proposals.

### Python Libraries & Methods

| Aspect | Custom NumPy Vectorized | `scipy.stats.contingency.association` |
| :--- | :--- | :--- |
| **Scope** | Cohen’s $d$, Hedges’ $g$, $\eta^2$, $\omega^2$ | Cramér’s $V$, Pearson’s contingency coefficient |
| **Speed** | Highly vectorized ($<1$ ms) | Fast ($<5$ ms) |
| **Dependencies** | `numpy`, `scipy` | `scipy >= 1.7` |

### Quick Start Code

```bash
python -m pip install numpy scipy pandas matplotlib seaborn
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


def calculate_cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    """Calculates Cohen's d for two independent groups."""
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return float((np.mean(g1) - np.mean(g2)) / s_pooled)


def calculate_hedges_g(g1: np.ndarray, g2: np.ndarray) -> float:
    """Calculates Hedges' g (bias-corrected Cohen's d for small samples)."""
    d = calculate_cohens_d(g1, g2)
    n = len(g1) + len(g2)
    correction = 1.0 - (3.0 / (4.0 * n - 9.0))
    return d * correction


def calculate_eta_squared(groups: list[np.ndarray]) -> float:
    """Calculates Eta-squared (SS_between / SS_total) for ANOVA."""
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    return float(ss_between / ss_total)


def interpret_cohen_d(d_val: float) -> str:
    val = abs(d_val)
    if val < 0.2:
        return "Vernachlässigbar (< 0.2)"
    elif val < 0.5:
        return "Klein (0.2 - 0.5)"
    elif val < 0.8:
        return "Mittel (0.5 - 0.8)"
    else:
        return "Groß (>= 0.8)"


def main() -> None:
    np.random.seed(42)
    
    # 2 Gruppen Simulation (z.B. Placebo vs. Drug)
    placebo = np.random.normal(loc=100.0, scale=15.0, size=25)
    drug = np.random.normal(loc=114.5, scale=14.0, size=25)
    
    d_val = calculate_cohens_d(drug, placebo)
    g_val = calculate_hedges_g(drug, placebo)
    
    # 3 Gruppen ANOVA Simulation
    ctrl = np.random.normal(100, 10, 20)
    dose_low = np.random.normal(105, 10, 20)
    dose_high = np.random.normal(120, 10, 20)
    eta_sq = calculate_eta_squared([ctrl, dose_low, dose_high])
    
    print("=== EFFEKTSTÄRKEN-BERECHNUNG ===")
    print(f"Cohen's d (Drug vs. Placebo):  {d_val:.3f} -> {interpret_cohen_d(d_val)}")
    print(f"Hedges' g (bias-corrected):    {g_val:.3f}")
    print(f"Eta-Squared (ANOVA 3-Gruppen): {eta_sq:.3f} ({(eta_sq*100):.1f}% Varianzaufklärung)")
    
    # Visualisierung der Effektstärken-Benchmarks
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ["Cohen's d\n(2-Gruppen)", "Hedges' g\n(Small Sample)", "Eta-Squared (x10)\n(Multi-Gruppen)"]
    values = [d_val, g_val, eta_sq * 10]
    colors = ["#2C7FB8", "#41B6C4", "#7FCDBB"]
    
    bars = ax.bar(metrics, values, color=colors, edgecolor="black", width=0.45)
    ax.bar_label(bars, fmt="%.3f", padding=5, fontweight="bold", fontsize=11)
    
    # Benchmark-Schwellenwerte als Referenzlinien
    ax.axhline(0.2, color="gray", linestyle=":", alpha=0.8, label="Schwacher Effekt (d=0.2)")
    ax.axhline(0.5, color="orange", linestyle="--", alpha=0.8, label="Mittlerer Effekt (d=0.5)")
    ax.axhline(0.8, color="red", linestyle="-.", alpha=0.8, label="Starker Effekt (d=0.8)")
    
    ax.set_title("Effektstärken-Quantifizierung & Benchmarks", fontweight="bold", fontsize=13, pad=12)
    ax.set_ylabel("Standardisierte Effektstärke")
    ax.set_ylim(0, max(values) + 0.3)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("09_2_effect_size_analysis.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[09_2_effect_size_analysis.png]]
