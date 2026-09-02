from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
mod = importlib.import_module("05_shapiro_wilk_normality_test")
plot_normality_diagnostics = mod.plot_normality_diagnostics
test_normality_feature = mod.test_normality_feature


def run_proteomics_normality() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    qty_cols = [f"{run}.raw.PG.Quantity" for runs in GROUP_TO_RUNS.values() for run in runs]
    valid_cols = [c for c in qty_cols if c in df_raw.columns]

    df_log = np.log2(df_raw[valid_cols].replace(0, np.nan))

    # Test normality across all proteins
    p_values = [test_normality_feature(row.dropna().values).get("Shapiro-Wilk", (0, 0))[1] for _, row in df_log.iterrows()]
    normal_pct = np.mean(np.array(p_values) >= 0.05) * 100

    print("=== NORMALITY ASSESSMENT ON PROTEOMICS DATA ===")
    print(f"Proteins Conforming to Log-Normal Distribution: {normal_pct:.1f}%")

    # Pick representative normal vs skewed protein
    idx_norm = int(np.argmax(p_values))
    idx_skew = int(np.argmin(p_values))

    norm_vals = df_log.iloc[idx_norm].dropna().values
    skew_vals = df_log.iloc[idx_skew].dropna().values

    out_path = MODULE_DIR / "05_shapiro_wilk_normality.png"
    plot_normality_diagnostics(norm_vals, skew_vals, out_path)
    print(f"Proteomics normality figure saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_normality()
