from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODULE_DIR = Path(__file__).parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from Random_Proteomics_Dataset_Generator import (
    GROUP_TO_RUNS,
    create_random_proteomics_table,
    load_template,
    save_template,
)

mod = importlib.import_module("18_ma_plots")
plot_ma = mod.plot_ma


def run_proteomics_ma() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    ctrl_cols = [f"{run}.raw.PG.Quantity" for run in GROUP_TO_RUNS["Control"]]
    treat_cols = [f"{run}.raw.PG.Quantity" for run in GROUP_TO_RUNS["Treatment_A"]]

    raw_ctrl = df_raw[ctrl_cols]
    raw_treat = df_raw[treat_cols]

    base_mean = pd.concat([raw_ctrl, raw_treat], axis=1).mean(axis=1)

    log_ctrl = np.log2(raw_ctrl.replace(0, np.nan))
    log_treat = np.log2(raw_treat.replace(0, np.nan))
    log2fc = log_treat.mean(axis=1) - log_ctrl.mean(axis=1)

    p_values = []
    for idx in range(len(df_raw)):
        c = log_ctrl.iloc[idx].dropna()
        t = log_treat.iloc[idx].dropna()
        if len(c) >= 2 and len(t) >= 2:
            _, p = stats.ttest_ind(t, c, equal_var=False)
            p_values.append(p)
        else:
            p_values.append(np.nan)

    p_arr = np.array(p_values)
    valid_mask = ~np.isnan(p_arr)
    padj = np.full_like(p_arr, np.nan)
    padj[valid_mask] = multipletests(p_arr[valid_mask], method="fdr_bh")[1]

    df_ma = pd.DataFrame({
        "baseMean": base_mean,
        "log2FoldChange": log2fc,
        "padj": padj,
    })

    out_dir = Path(__file__).parent
    out_path = plot_ma(df_ma, "TreatmentA_vs_Control", out_dir)
    print(f"Proteomics MA plot generated at: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_ma()
