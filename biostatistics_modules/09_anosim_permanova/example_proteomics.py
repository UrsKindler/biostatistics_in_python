from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

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
mod = importlib.import_module("09_anosim_permanova")
plot_permanova_ordination = mod.plot_permanova_ordination
run_permanova = mod.run_permanova


def run_proteomics_permanova() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    qty_cols = [f"{r}.raw.PG.Quantity" for runs in GROUP_TO_RUNS.values() for r in runs]
    valid_cols = [c for c in qty_cols if c in df_raw.columns]

    # Samples x Proteins matrix
    mat = df_raw[valid_cols].T
    mat_log = np.log2(mat.replace(0, np.nan)).fillna(0)

    # Group labels
    sample_to_group = {}
    for group, runs in GROUP_TO_RUNS.items():
        for run in runs:
            sample_to_group[f"{run}.raw.PG.Quantity"] = group
    groupings = np.array([sample_to_group.get(s, "Unknown") for s in mat.index])

    dist_mat = squareform(pdist(mat_log.values, metric="euclidean"))
    f_stat, p_val = run_permanova(dist_mat, groupings, n_permutations=999)

    print("=== PERMANOVA ON PROTEOMICS GLOBAL PROFILES ===")
    print(f"Pseudo-F Statistic: {f_stat:.4f}")
    print(f"Permutation p-value: {p_val:.4e}")

    out_path = MODULE_DIR / "09_multivariate_anosim_permanova.png"
    plot_permanova_ordination(dist_mat, groupings, f_stat, p_val, out_path)
    print(f"Proteomics PERMANOVA plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_permanova()
