from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def run_two_group_tests(g1: np.ndarray, g2: np.ndarray) -> dict[str, tuple[float, float]]:
    g1_c = g1[~np.isnan(g1)]
    g2_c = g2[~np.isnan(g2)]

    t_stud, p_stud = stats.ttest_ind(g1_c, g2_c, equal_var=True)
    t_welch, p_welch = stats.ttest_ind(g1_c, g2_c, equal_var=False)
    u_stat, p_mwu = stats.mannwhitneyu(g1_c, g2_c, alternative="two-sided")

    return {
        "Student t-test": (t_stud, p_stud),
        "Welch t-test": (t_welch, p_welch),
        "Mann-Whitney U": (u_stat, p_mwu),
    }


def plot_two_group_comparison(g1: np.ndarray, g2: np.ndarray, label1: str, label2: str, outpath: Path) -> None:
    df_plot = pd.DataFrame({
        "Condition": [label1] * len(g1) + [label2] * len(g2),
        "Abundance": np.concatenate([g1, g2]),
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    sns.boxplot(data=df_plot, x="Condition", y="Abundance", ax=axes[0], palette=["#2C7FB8", "#D9534F"])
    sns.stripplot(data=df_plot, x="Condition", y="Abundance", ax=axes[0], color="black", alpha=0.6, jitter=0.2)
    axes[0].set_title("A: Group Distribution & Data Points", fontweight="bold", fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)

    sns.kdeplot(g1, label=label1, color="#2C7FB8", lw=2, ax=axes[1])
    sns.kdeplot(g2, label=label2, color="#D9534F", lw=2, ax=axes[1])
    axes[1].set_title("B: Kernel Density Overlay", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Abundance")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    g1 = np.random.normal(12, 1.5, 30)
    g2 = np.random.normal(14, 2.0, 30)

    res = run_two_group_tests(g1, g2)
    for test, (s, p) in res.items():
        print(f"  {test:<20}: Stat = {s:.4f}, p-value = {p:.4e}")

    out_dir = Path(__file__).parent
    outpath = out_dir / "07_two_group_comparisons.png"
    plot_two_group_comparison(g1, g2, "Control", "Treatment", outpath)
    print(f"Two-group comparison plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
