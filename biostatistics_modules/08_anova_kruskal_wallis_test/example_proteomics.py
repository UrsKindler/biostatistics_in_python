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
mod = importlib.import_module("08_anova_kruskal_wallis_test")
plot_multigroup_comparison = mod.plot_multigroup_comparison
run_multigroup_tests = mod.run_multigroup_tests


def run_proteomics_anova() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    c_runs = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Control"]]
    ta_runs = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Treatment_A"]]
    tb_runs = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Treatment_B"]]

    g_ctrl = np.log2(df_raw[c_runs].replace(0, np.nan)).iloc[0].dropna().values
    g_ta = np.log2(df_raw[ta_runs].replace(0, np.nan)).iloc[0].dropna().values
    g_tb = np.log2(df_raw[tb_runs].replace(0, np.nan)).iloc[0].dropna().values

    res = run_multigroup_tests([g_ctrl, g_ta, g_tb])
    print("=== ANOVA / KRUSKAL-WALLIS ON PROTEOMICS DATA ===")
    for k, (s, p) in res.items():
        print(f"  {k:<20}: Stat = {s:.4f}, p = {p:.4e}")

    df_long = pd.DataFrame({
        "Group": ["Control"] * len(g_ctrl) + ["Treatment_A"] * len(g_ta) + ["Treatment_B"] * len(g_tb),
        "Intensity": np.concatenate([g_ctrl, g_ta, g_tb]),
    })

    out_path = MODULE_DIR / "08_anova_kruskal_wallis.png"
    plot_multigroup_comparison(df_long, "Group", "Intensity", out_path)
    print(f"Proteomics ANOVA plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_anova()
