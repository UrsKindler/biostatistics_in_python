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
mod = importlib.import_module("09_1_posthoc_tests")
plot_posthoc_summary = mod.plot_posthoc_summary
run_dunn_test = mod.run_dunn_test
run_tukey_hsd = mod.run_tukey_hsd


def run_proteomics_posthoc() -> None:
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

    df_long = pd.DataFrame({
        "Condition": ["Control"] * len(g_ctrl) + ["Treatment_A"] * len(g_ta) + ["Treatment_B"] * len(g_tb),
        "Abundance": np.concatenate([g_ctrl, g_ta, g_tb]),
    })

    tukey_res = run_tukey_hsd(df_long, "Abundance", "Condition")
    dunn_res = run_dunn_test(df_long, "Abundance", "Condition")

    print("=== POST-HOC PAIRWISE TESTING ON PROTEOMICS DATA ===")
    print(tukey_res)

    out_path = MODULE_DIR / "09_1_posthoc_tests.png"
    plot_posthoc_summary(tukey_res, dunn_res, out_path)
    print(f"Proteomics post-hoc plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_posthoc()
