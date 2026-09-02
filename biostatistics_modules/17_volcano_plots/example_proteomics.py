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

mod = importlib.import_module("17_volcano_plots")
plot_volcano = mod.plot_volcano


def run_proteomics_volcano() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    ctrl_cols = [f"{run}.raw.PG.Quantity" for run in GROUP_TO_RUNS["Control"]]
    treat_cols = [f"{run}.raw.PG.Quantity" for run in GROUP_TO_RUNS["Treatment_A"]]

    ctrl_mat = np.log2(df_raw[ctrl_cols].replace(0, np.nan))
    treat_mat = np.log2(df_raw[treat_cols].replace(0, np.nan))

    mean_ctrl = ctrl_mat.mean(axis=1)
    mean_treat = treat_mat.mean(axis=1)
    log2fc = mean_treat - mean_ctrl

    p_values = []
    for idx in range(len(df_raw)):
        c_vals = ctrl_mat.iloc[idx].dropna()
        t_vals = treat_mat.iloc[idx].dropna()
        if len(c_vals) >= 2 and len(t_vals) >= 2:
            _, p = stats.ttest_ind(t_vals, c_vals, equal_var=False)
            p_values.append(p)
        else:
            p_values.append(np.nan)

    p_arr = np.array(p_values)
    valid_mask = ~np.isnan(p_arr)
    padj = np.full_like(p_arr, np.nan)
    padj[valid_mask] = multipletests(p_arr[valid_mask], method="fdr_bh")[1]

    df_deg = pd.DataFrame({
        "gene_name": df_raw["PG.Genes"],
        "log2FoldChange": log2fc,
        "padj": padj,
    })

    out_dir = Path(__file__).parent
    out_path = plot_volcano(df_deg, "TreatmentA_vs_Control", out_dir, padj_thr=0.05, log2fc_thr=1.0)
    print(f"Proteomics Volcano plot generated at: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_volcano()
