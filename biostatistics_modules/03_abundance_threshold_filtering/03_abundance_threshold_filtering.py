from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def filter_abundance_and_missingness(
    df: pd.DataFrame,
    group_dict: dict[str, list[str]],
    min_valid_prop_per_group: float = 0.70,
    min_mean_intensity: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters features based on minimum valid observations in at least one group
    and overall mean abundance.
    """
    valid_mask_any_group = pd.Series(False, index=df.index)

    for group, cols in group_dict.items():
        present_cols = [c for c in cols if c in df.columns]
        if not present_cols:
            continue
        group_valid_rate = df[present_cols].notna().mean(axis=1)
        valid_mask_any_group |= (group_valid_rate >= min_valid_prop_per_group)

    all_cols = [c for cols in group_dict.values() for c in cols if c in df.columns]
    mean_intensity = df[all_cols].mean(axis=1)
    intensity_mask = mean_intensity >= min_mean_intensity

    keep_mask = valid_mask_any_group & intensity_mask
    return df[keep_mask].copy(), df[~keep_mask].copy()


def plot_filtering_summary(df_raw: pd.DataFrame, df_kept: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Feature count before and after
    counts = {"Pre-Filtering": len(df_raw), "Retained": len(df_kept), "Filtered Out": len(df_raw) - len(df_kept)}
    bars = axes[0].bar(counts.keys(), counts.values(), color=["#4C72B0", "#5CB85C", "#D9534F"], alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{int(yval)}", ha="center", va="bottom", fontweight="bold")
    axes[0].set_ylabel("Number of Features / Proteins")
    axes[0].set_title("A: Feature Retention Summary", fontweight="bold", fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel B: Missingness per feature histogram
    raw_missing = df_raw.isna().mean(axis=1) * 100
    kept_missing = df_kept.isna().mean(axis=1) * 100
    axes[1].hist(raw_missing, bins=25, alpha=0.5, label="Raw Features", color="#D9534F", edgecolor="black")
    axes[1].hist(kept_missing, bins=25, alpha=0.7, label="Filtered Features", color="#5CB85C", edgecolor="black")
    axes[1].set_xlabel("Missing Values per Feature (%)")
    axes[1].set_ylabel("Feature Count")
    axes[1].set_title("B: Missingness Distribution Shift", fontweight="bold", fontsize=11)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    n_features, n_samples = 500, 12
    X = np.random.normal(15, 3, size=(n_features, n_samples))
    mask = (X < 12) & (np.random.rand(*X.shape) < 0.75)
    X[mask] = np.nan
    cols = [f"Sample_{i+1:02d}" for i in range(n_samples)]
    df = pd.DataFrame(X, columns=cols, index=[f"Protein_{j+1:03d}" for j in range(n_features)])
    groups = {"Group1": cols[:6], "Group2": cols[6:]}

    df_kept, df_discarded = filter_abundance_and_missingness(df, groups, min_valid_prop_per_group=0.66)
    print(f"Features kept: {len(df_kept)}/{len(df)} ({len(df_kept)/len(df):.1%})")

    out_dir = Path(__file__).parent
    outpath = out_dir / "03_abundance_filtering.png"
    plot_filtering_summary(df, df_kept, outpath)
    print(f"Filtering plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
