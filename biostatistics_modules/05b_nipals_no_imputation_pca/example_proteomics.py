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

mod = importlib.import_module("05b_nipals_no_imputation_pca")
perform_nipals_pca = mod.perform_nipals_pca
plot_nipals_pca_comparison = mod.plot_nipals_pca_comparison


def run_proteomics_nipals() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    qty_cols = [f"{run}.raw.PG.Quantity" for runs in GROUP_TO_RUNS.values() for run in runs]
    valid_qty_cols = [c for c in qty_cols if c in df_raw.columns]

    intensity_matrix = df_raw[valid_qty_cols].T
    intensity_matrix = np.log2(intensity_matrix.replace(0, np.nan))

    sample_to_group = {}
    for group, runs in GROUP_TO_RUNS.items():
        for run in runs:
            sample_to_group[f"{run}.raw.PG.Quantity"] = group
    sample_groups = pd.Series([sample_to_group.get(s, "Unknown") for s in intensity_matrix.index], index=intensity_matrix.index)

    print("=== NIPALS PCA ON PROTEOMICS DATA ===")
    print(f"Samples: {intensity_matrix.shape[0]}, Proteins: {intensity_matrix.shape[1]}")
    print(f"Missing Values: {intensity_matrix.isna().sum().sum()} ({intensity_matrix.isna().sum().sum()/intensity_matrix.size:.1%})")

    scores, exp_var, score_df = perform_nipals_pca(intensity_matrix, max_components=5)

    out_dir = Path(__file__).parent
    out_path = out_dir / "05b_nipals_no_imputation_pca.png"
    plot_nipals_pca_comparison(score_df, exp_var, sample_groups, COLOR_MAP, out_path)
    print(f"Proteomics NIPALS PCA figure saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_nipals()
