from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def run_tukey_hsd(df_long: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    res = pairwise_tukeyhsd(df_long[value_col], df_long[group_col], alpha=0.05)
    return pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])


def run_dunn_test(df_long: pd.DataFrame, value_col: str, group_col: str, p_adjust: str = "fdr_bh") -> pd.DataFrame:
    return sp.posthoc_dunn(df_long, val_col=value_col, group_col=group_col, p_adjust=p_adjust)


def plot_posthoc_summary(tukey_df: pd.DataFrame, dunn_matrix: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Tukey Confidence Intervals
    for idx, row in tukey_df.iterrows():
        pair = f"{row['group1']} vs {row['group2']}"
        meandiff = row["meandiff"]
        lower = row["lower"]
        upper = row["upper"]
        color = "red" if row["reject"] else "blue"
        axes[0].plot([lower, upper], [idx, idx], color=color, lw=2)
        axes[0].plot(meandiff, idx, marker="o", color=color, markersize=8)

    axes[0].axvline(0, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_yticks(range(len(tukey_df)))
    axes[0].set_yticklabels([f"{r['group1']} vs {r['group2']}" for _, r in tukey_df.iterrows()])
    axes[0].set_xlabel("Mean Difference (95% CI)")
    axes[0].set_title("A: Tukey HSD Pairwise Differences", fontweight="bold", fontsize=11)
    axes[0].grid(axis="x", alpha=0.3)

    # Panel B: Dunn Adjusted P-values Heatmap
    sns.heatmap(dunn_matrix, annot=True, fmt=".3e", cmap="YlGnBu_r", cbar_kws={"label": "Adjusted p-value"}, ax=axes[1])
    axes[1].set_title("B: Dunn Non-Parametric Post-Hoc Matrix (FDR-BH)", fontweight="bold", fontsize=11)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    g1 = np.random.normal(10, 1.2, 20)
    g2 = np.random.normal(12, 1.2, 20)
    g3 = np.random.normal(16, 1.5, 20)

    df_long = pd.DataFrame({
        "Condition": ["Control"] * 20 + ["Treatment_A"] * 20 + ["Treatment_B"] * 20,
        "Abundance": np.concatenate([g1, g2, g3]),
    })

    tukey_res = run_tukey_hsd(df_long, "Abundance", "Condition")
    dunn_res = run_dunn_test(df_long, "Abundance", "Condition")

    out_dir = Path(__file__).parent
    outpath = out_dir / "09_1_posthoc_tests.png"
    plot_posthoc_summary(tukey_res, dunn_res, outpath)
    print(f"Post-hoc summary plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
