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
mod = importlib.import_module("07_ttest_mann_whitney_utest")
plot_two_group_comparison = mod.plot_two_group_comparison
run_two_group_tests = mod.run_two_group_tests


def run_proteomics_two_group() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    ctrl_cols = [f"{run}.raw.PG.Quantity" for run in GROUP_TO_RUNS["Control"]]
    ta_cols = [f"{run}.raw.PG.Quantity" for run in GROUP_TO_RUNS["Treatment_A"]]

    log_ctrl = np.log2(df_raw[ctrl_cols].replace(0, np.nan))
    log_ta = np.log2(df_raw[ta_cols].replace(0, np.nan))

    # Test top differential candidate (first row)
    g1 = log_ctrl.iloc[0].dropna().values
    g2 = log_ta.iloc[0].dropna().values

    res = run_two_group_tests(g1, g2)
    print("=== TWO-GROUP HYPOTHESIS TESTING ON PROTEOMICS DATA ===")
    for k, (s, p) in res.items():
        print(f"  {k:<20}: Stat = {s:.4f}, p = {p:.4e}")

    out_path = MODULE_DIR / "07_two_group_comparisons.png"
    plot_two_group_comparison(g1, g2, "Control", "Treatment_A", out_path)
    print(f"Proteomics two-group comparison figure saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_two_group()
