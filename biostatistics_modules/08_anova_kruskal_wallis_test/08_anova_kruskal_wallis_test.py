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


def run_multigroup_tests(groups: list[np.ndarray]) -> dict[str, tuple[float, float]]:
    clean_groups = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) >= 2]
    f_stat, p_anova = stats.f_oneway(*clean_groups)
    h_stat, p_kw = stats.kruskal(*clean_groups)
    return {"One-Way ANOVA": (f_stat, p_anova), "Kruskal-Wallis": (h_stat, p_kw)}


def plot_multigroup_comparison(df_long: pd.DataFrame, x_col: str, y_col: str, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.boxplot(data=df_long, x=x_col, y=y_col, ax=axes[0], palette="Set2")
    sns.stripplot(data=df_long, x=x_col, y=y_col, ax=axes[0], color="black", alpha=0.5, jitter=0.2)
    axes[0].set_title("A: Multi-Group Abundance Distribution", fontweight="bold", fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)

    sns.violinplot(data=df_long, x=x_col, y=y_col, ax=axes[1], palette="Set2", inner="quartile")
    axes[1].set_title("B: Violin Density Profiles", fontweight="bold", fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    g1 = np.random.normal(10, 1.5, 25)
    g2 = np.random.normal(12, 1.5, 25)
    g3 = np.random.normal(15, 1.8, 25)

    res = run_multigroup_tests([g1, g2, g3])
    for test, (s, p) in res.items():
        print(f"  {test:<20}: Stat = {s:.4f}, p = {p:.4e}")

    df_long = pd.DataFrame({
        "Condition": ["Control"] * 25 + ["Treatment_A"] * 25 + ["Treatment_B"] * 25,
        "Abundance": np.concatenate([g1, g2, g3]),
    })
    out_dir = Path(__file__).parent
    outpath = out_dir / "08_anova_kruskal_wallis.png"
    plot_multigroup_comparison(df_long, "Condition", "Abundance", outpath)
    print(f"ANOVA / Kruskal-Wallis plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
