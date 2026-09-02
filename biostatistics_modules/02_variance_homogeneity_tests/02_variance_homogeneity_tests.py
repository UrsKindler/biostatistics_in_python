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


def run_homogeneity_tests(groups_data: list[np.ndarray]) -> dict[str, tuple[float, float]]:
    clean_groups = [g[~np.isnan(g)] for g in groups_data if len(g[~np.isnan(g)]) >= 2]
    if len(clean_groups) < 2:
        return {}

    levene_stat, levene_p = stats.levene(*clean_groups, center="median")
    bartlett_stat, bartlett_p = stats.bartlett(*clean_groups)
    fligner_stat, fligner_p = stats.fligner(*clean_groups)

    return {
        "Levene (Brown-Forsythe)": (levene_stat, levene_p),
        "Bartlett": (bartlett_stat, bartlett_p),
        "Fligner-Killeen": (fligner_stat, fligner_p),
    }


def plot_variance_homogeneity(df_long: pd.DataFrame, group_col: str, value_col: str, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(data=df_long, x=group_col, y=value_col, ax=axes[0], palette="Set2")
    sns.stripplot(data=df_long, x=group_col, y=value_col, ax=axes[0], color="black", alpha=0.5, jitter=0.2)
    axes[0].set_title("A: Variance Across Groups (Boxplot & Data Points)", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Experimental Condition")
    axes[0].set_ylabel("Abundance Value")
    axes[0].grid(axis="y", alpha=0.3)

    # Residuals from group medians
    medians = df_long.groupby(group_col)[value_col].transform("median")
    df_long["abs_residual"] = (df_long[value_col] - medians).abs()

    sns.barplot(data=df_long, x=group_col, y="abs_residual", ax=axes[1], palette="Set2", ci=68, capsize=0.1)
    axes[1].set_title("B: Absolute Median Deviations (Homoscedasticity Metric)", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Experimental Condition")
    axes[1].set_ylabel("Mean |y - median|")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    g1 = np.random.normal(10, 1.2, 20)
    g2 = np.random.normal(10, 1.3, 20)
    g3 = np.random.normal(10, 3.5, 20)  # Heteroscedastic group

    res = run_homogeneity_tests([g1, g2, g3])
    print("Variance Homogeneity Test Results:")
    for test_name, (stat, p) in res.items():
        print(f"  {test_name:<25}: Stat = {stat:.4f}, p-value = {p:.4e}")

    df_long = pd.DataFrame({
        "group": ["Control"] * 20 + ["Treatment_A"] * 20 + ["Treatment_B"] * 20,
        "value": np.concatenate([g1, g2, g3]),
    })
    out_dir = Path(__file__).parent
    outpath = out_dir / "02_variance_homogeneity_tests.png"
    plot_variance_homogeneity(df_long, "group", "value", outpath)
    print(f"Plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
