### Overview
- **Purpose**: Adjust $p$-values across thousands of simultaneous hypothesis tests to rigorously control either the False Discovery Rate (FDR: expected proportion of false positives among all discoveries) or Family-Wise Error Rate (FWER: probability of making at least one false positive).
- **Use Case**: Differential gene expression (RNA-Seq, microarray), proteomics biomarker discovery, metabolomics profiling, genome-wide association studies (GWAS).
- **Prerequisites**: 
	* Python 3.9+
	* Array of raw unadjusted $p$-values from univariate statistical tests
- **Data Types**: 
	* 1D array of float probabilities $p \in [0, 1]$
- **Output**:
	* Adjusted $p$-values ($q$-values) and binary hypothesis rejection masks
	* Multi-panel comparison figure (`09_3_false_discovery_rate.png`) comparing raw vs. Benjamini-Hochberg vs. Bonferroni vs. Holm corrections

#### When to Use

| Correction Method | Target Metric Controlled | Mathematical Formula | Statistical Power | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Benjamini-Hochberg (BH / FDR)** | False Discovery Rate ($E[\frac{V}{R}]$) | $p_{(i)} \le \frac{i}{m} \cdot Q$ | **High** (Optimal for Omics) | **Standard for high-throughput biology**; balances discoveries and false alarms |
| **Benjamini-Yekutieli (BY)** | FDR under arbitrary dependence | $p_{(i)} \le \frac{i}{m \sum_{j=1}^m 1/j} \cdot Q$ | Moderate | Robust against complex negative correlation structures |
| **Bonferroni** | Family-Wise Error Rate (FWER) | $p_{\text{adj}} = \min(1, p \cdot m)$ | Very Low (Ultra-conservative) | High risk of false negatives (Type-II error); only for $<10$ tests |
| **Holm-Bonferroni** | Family-Wise Error Rate (FWER) | Step-down: $p_{(i)} \cdot (m - i + 1)$ | Moderate | Strictly dominates standard Bonferroni without additional assumptions |

#### Decision Criteria
- **Use Benjamini-Hochberg (FDR) always**: When screening $>50$ features (genes, proteins, OTUs) for candidate biomarkers.
- **Use Bonferroni / Holm**: When false positives have fatal clinical/legal consequences (e.g. diagnostic confirmatory tests).
- **Critical rule**: Never report raw uncorrected $p$-values in high-dimensional omics screens (at $\alpha=0.05$, testing 10,000 genes produces 500 false discoveries by pure chance!).

### Python Libraries & Methods

| Aspect | `statsmodels.stats.multitest.multipletests` |
| :--- | :--- |
| **Library** | `statsmodels` |
| **Key Methods Supported** | `'fdr_bh'` (Benjamini-Hochberg), `'fdr_by'` (Benjamini-Yekutieli), `'bonferroni'`, `'holm'`, `'fdr_tsbh'` |
| **Input Arguments** | `pvals`, `alpha=0.05`, `method='fdr_bh'` |
| **Output Tuple** | `(reject, pvals_corrected, alphacSidak, alphacBonf)` |

### Quick Start Code

```bash
python -m pip install statsmodels numpy pandas matplotlib seaborn
```

```python
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests


def generate_omics_pvalues(n_tests: int = 1000, n_true_signals: int = 80, seed: int = 42) -> np.ndarray:
    """Generates a realistic mixture distribution of true nulls (uniform) and true signals (beta)."""
    np.random.seed(seed)
    
    # 920 True Nulls (Uniformly distributed p-values on [0, 1])
    null_pvals = np.random.uniform(0.0, 1.0, size=n_tests - n_true_signals)
    
    # 80 True Biological Signals (Skewed towards 0)
    signal_pvals = np.random.beta(a=0.3, b=20.0, size=n_true_signals)
    
    pvals = np.concatenate([signal_pvals, null_pvals])
    np.random.shuffle(pvals)
    return pvals


def main() -> None:
    pvals = generate_omics_pvalues()
    alpha = 0.05
    
    # 1. Multiple Testing Corrections anwenden
    # A: Benjamini-Hochberg FDR
    reject_bh, pvals_bh, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    
    # B: Bonferroni FWER
    reject_bonf, pvals_bonf, _, _ = multipletests(pvals, alpha=alpha, method="bonferroni")
    
    # C: Holm-Bonferroni
    reject_holm, pvals_holm, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    
    raw_sig = np.sum(pvals < alpha)
    
    print("=== MULTIPLE TESTING CORRECTION SUMMARY (m = 1000 Tests) ===")
    print(f"Signifikant ohne Korrektur (Raw p < 0.05): {raw_sig} (enthält ~46 False Positives)")
    print(f"Signifikant mit FDR (Benjamini-Hochberg):   {np.sum(reject_bh)} (kontrolliert FDR auf 5%)")
    print(f"Signifikant mit Holm-Bonferroni:           {np.sum(reject_holm)}")
    print(f"Signifikant mit Bonferroni (Konservativ):  {np.sum(reject_bonf)}")
    
    # 2. Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel A: Sortierte p-Werte vs. Korrekturschwellen
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    sorted_bh = pvals_bh[sorted_idx]
    ranks = np.arange(1, len(pvals) + 1)
    
    axes[0].plot(ranks[:200], sorted_pvals[:200], label="Raw p-Values", color="black", lw=1.5)
    axes[0].plot(ranks[:200], sorted_bh[:200], label="Benjamini-Hochberg FDR", color="#2C7FB8", lw=2)
    axes[0].axhline(alpha, color="red", linestyle="--", label=f"Signifikanzgrenze (alpha={alpha})")
    axes[0].set_title("A: Geordnete p-Werte & FDR-Schwellen (Top 200)", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Rang (Rank i)")
    axes[0].set_ylabel("p-Wert / q-Wert")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Panel B: Vergleich der Entdeckungsraten
    methods = ["Unkorrigiert\n(Raw p)", "FDR (BH)\n[Empfohlen]", "Holm\n(Step-down)", "Bonferroni\n(Ultra-Konservativ)"]
    counts = [raw_sig, np.sum(reject_bh), np.sum(reject_holm), np.sum(reject_bonf)]
    colors = ["#E6550D", "#2C7FB8", "#41B6C4", "#7FCDBB"]
    
    bars = axes[1].bar(methods, counts, color=colors, edgecolor="black", width=0.5)
    axes[1].bar_label(bars, fmt="%d", padding=3, fontweight="bold", fontsize=11)
    axes[1].set_title(f"B: Anzahl entdeckter Gene / Proteine (alpha={alpha})", fontweight="bold", fontsize=12)
    axes[1].set_ylabel("Anzahl signifikante Befunde")
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("09_3_false_discovery_rate.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[09_3_false_discovery_rate.png]]
