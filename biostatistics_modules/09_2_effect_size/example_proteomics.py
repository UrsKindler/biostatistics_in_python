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
mod = importlib.import_module("09_2_effect_size")
cohens_d = mod.cohens_d
hedges_g = mod.hedges_g
plot_effect_sizes = mod.plot_effect_sizes


def run_proteomics_effect_size() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    c_runs = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Control"]]
    ta_runs = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Treatment_A"]]
    tb_runs = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Treatment_B"]]

    log_c = np.log2(df_raw[c_runs].replace(0, np.nan))
    log_ta = np.log2(df_raw[ta_runs].replace(0, np.nan))
    log_tb = np.log2(df_raw[tb_runs].replace(0, np.nan))

    # Top candidate
    p1_c = log_c.iloc[0].dropna().values
    p1_ta = log_ta.iloc[0].dropna().values
    p1_tb = log_tb.iloc[0].dropna().values

    effects = {
        "P00001 (Treatment_A vs Ctrl, d)": cohens_d(p1_ta, p1_c),
        "P00001 (Treatment_A vs Ctrl, g)": hedges_g(p1_ta, p1_c),
        "P00001 (Treatment_B vs Ctrl, d)": cohens_d(p1_tb, p1_c),
        "P00001 (Treatment_B vs Ctrl, g)": hedges_g(p1_tb, p1_c),
    }

    print("=== EFFECT SIZE ANALYSIS ON PROTEOMICS DATA ===")
    for k, v in effects.items():
        print(f"  {k:<35}: {v:.4f}")

    out_path = MODULE_DIR / "09_2_effect_size_analysis.png"
    plot_effect_sizes(effects, out_path)
    print(f"Proteomics effect size figure saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_effect_size()
