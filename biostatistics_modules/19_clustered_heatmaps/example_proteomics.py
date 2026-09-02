from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODULE_DIR = Path(__file__).parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from Random_Proteomics_Dataset_Generator import (
    COLOR_MAP,
    GROUP_TO_RUNS,
    create_random_proteomics_table,
    load_template,
    save_template,
)

mod = importlib.import_module("19_clustered_heatmaps")
plot_clustered_heatmap = mod.plot_clustered_heatmap


def run_proteomics_heatmap() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    qty_cols = [f"{run}.raw.PG.Quantity" for runs in GROUP_TO_RUNS.values() for run in runs]
    valid_cols = [c for c in qty_cols if c in df_raw.columns]

    expr_mat = df_raw[valid_cols].copy()
    expr_mat.index = df_raw["PG.Genes"]
    expr_mat = np.log2(expr_mat.replace(0, np.nan))
    expr_mat = expr_mat.apply(lambda row: row.fillna(row.median()), axis=1).fillna(0)

    sample_to_group = {}
    for group, runs in GROUP_TO_RUNS.items():
        for run in runs:
            sample_to_group[f"{run}.raw.PG.Quantity"] = group

    meta_df = pd.DataFrame({"group": [sample_to_group.get(c, "Unknown") for c in valid_cols]}, index=valid_cols)

    out_dir = Path(__file__).parent
    out_path = out_dir / "19_clustered_heatmaps.png"
    plot_clustered_heatmap(
        expression_matrix=expr_mat,
        sample_metadata=meta_df,
        group_col="group",
        color_map=COLOR_MAP,
        title="Proteomics Quantitative Heatmap (Top 40 Variable Proteins)",
        outpath=out_path,
        top_n_features=40,
    )
    print(f"Proteomics Heatmap successfully saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_heatmap()
