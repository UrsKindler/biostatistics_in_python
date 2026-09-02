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
mod = importlib.import_module("02_variance_homogeneity_tests")
run_homogeneity_tests = mod.run_homogeneity_tests
plot_variance_homogeneity = mod.plot_variance_homogeneity


def run_proteomics_variance_tests() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    ctrl_cols = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Control"]]
    ta_cols = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Treatment_A"]]
    tb_cols = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Treatment_B"]]

    g_ctrl = np.log2(df_raw[ctrl_cols].replace(0, np.nan)).values.flatten()
    g_ta = np.log2(df_raw[ta_cols].replace(0, np.nan)).values.flatten()
    g_tb = np.log2(df_raw[tb_cols].replace(0, np.nan)).values.flatten()

    res = run_homogeneity_tests([g_ctrl, g_ta, g_tb])
    print("=== HOMOSCEDASTICITY ASSESSMENT ACROSS PROTEOMICS GROUPS ===")
    for k, (s, p) in res.items():
        print(f"  {k:<25}: Stat = {s:.4f}, p = {p:.4e}")

    df_long = pd.DataFrame({
        "Condition": ["Control"] * len(g_ctrl) + ["Treatment_A"] * len(g_ta) + ["Treatment_B"] * len(g_tb),
        "Log2_Intensity": np.concatenate([g_ctrl, g_ta, g_tb]),
    }).dropna()

    out_path = MODULE_DIR / "02_variance_homogeneity_tests.png"
    plot_variance_homogeneity(df_long, "Condition", "Log2_Intensity", out_path)
    print(f"Proteomics variance plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_variance_tests()
