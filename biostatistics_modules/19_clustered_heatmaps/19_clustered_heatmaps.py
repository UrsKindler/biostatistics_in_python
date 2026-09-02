from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_clustered_heatmap(
    expression_matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    group_col: str,
    color_map: dict[str, str],
    title: str,
    outpath: Path,
    z_score_rows: bool = True,
    top_n_features: int = 40,
) -> Path:
    data = expression_matrix.copy()

    if len(data) > top_n_features:
        var_order = data.var(axis=1).sort_values(ascending=False)
        data = data.loc[var_order.head(top_n_features).index]

    if z_score_rows:
        row_mean = data.mean(axis=1)
        row_std = data.std(axis=1).replace(0, np.nan)
        data = data.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)

    sample_groups = sample_metadata.loc[data.columns, group_col]
    col_colors = sample_groups.map(lambda g: color_map.get(g, "#999999"))

    sns.set_theme(context="notebook", style="white")
    g = sns.clustermap(
        data,
        cmap="vlag",
        center=0,
        col_colors=col_colors,
        yticklabels=True,
        xticklabels=True,
        figsize=(12, 10),
        cbar_kws={"label": "Row Z-score" if z_score_rows else "Expression"},
        dendrogram_ratio=(0.12, 0.12),
        cbar_pos=(0.91, 0.78, 0.03, 0.16),
    )

    g.ax_heatmap.tick_params(axis="y", labelsize=8)
    g.ax_heatmap.tick_params(axis="x", labelsize=9)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")

    g.fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")
    g.fig.subplots_adjust(left=0.2, right=0.88, top=0.92, bottom=0.08)

    outpath.parent.mkdir(exist_ok=True, parents=True)
    g.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    return outpath


def main() -> None:
    out_dir = Path(__file__).parent
    np.random.seed(42)
    n_features = 35
    n_samples = 18

    samples = [f"Sample_{i+1:02d}" for i in range(n_samples)]
    features = [f"Protein_{j+1:02d}" for j in range(n_features)]
    groups = ["Control"] * 6 + ["Treatment_A"] * 6 + ["Treatment_B"] * 6

    mat = np.random.normal(15, 2, size=(n_features, n_samples))
    mat[:12, 6:12] += 2.5
    mat[12:24, 12:] -= 2.0

    expr_df = pd.DataFrame(mat, index=features, columns=samples)
    meta_df = pd.DataFrame({"group": groups}, index=samples)
    color_map = {"Control": "#2C7FB8", "Treatment_A": "#D9534F", "Treatment_B": "#5CB85C"}

    out_path = out_dir / "19_clustered_heatmaps.png"
    plot_clustered_heatmap(
        expr_df,
        meta_df,
        group_col="group",
        color_map=color_map,
        title="Hierarchical Clustered Heatmap (Top Biomarkers)",
        outpath=out_path,
    )
    print(f"Clustered Heatmap saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
