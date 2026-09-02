from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Random_Proteomics_Dataset_Generator import (
    GROUP_TO_RUNS,
    create_random_proteomics_table,
    load_template,
    save_template,
)
MODULE_DIR = Path(__file__).parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
import importlib
mod = importlib.import_module("02_missing_data_handling")
plot_imputation_diagnostics = mod.plot_imputation_diagnostics


def run_proteomics_imputation() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    qty_cols = [f"{run}.raw.PG.Quantity" for runs in GROUP_TO_RUNS.values() for run in runs]
    valid_cols = [c for c in qty_cols if c in df_raw.columns]

    # Samples x Proteins matrix
    mat_raw = df_raw[valid_cols].T
    mat_log = np.log2(mat_raw.replace(0, np.nan))

    # Take subset of 25 proteins for clear 4-panel diagnostic
    subset_cols = mat_log.columns[:25]
    sub_raw = mat_log[subset_cols]

    imputer = KNNImputer(n_neighbors=4)
    sub_knn = pd.DataFrame(imputer.fit_transform(sub_raw), index=sub_raw.index, columns=sub_raw.columns)
    sub_median = sub_raw.fillna(sub_raw.median())

    print("=== MISSING DATA IMPUTATION ON PROTEOMICS DATA ===")
    print(f"Evaluated Submatrix: {sub_raw.shape[0]} runs x {sub_raw.shape[1]} proteins")
    print(f"Missing Values before: {sub_raw.isna().sum().sum()}, after KNN: {sub_knn.isna().sum().sum()}")

    out_path = MODULE_DIR / "02_missing_data_handling.png"
    plot_imputation_diagnostics(sub_raw, sub_knn, sub_median, out_path)
    print(f"Proteomics Imputation Diagnostic saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_imputation()
