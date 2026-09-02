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
mod = importlib.import_module("04_normalization_zscore_scaling")
normalize_median = mod.normalize_median
plot_normalization_comparison = mod.plot_normalization_comparison


def run_proteomics_norm() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    qty_cols = [f"{run}.raw.PG.Quantity" for runs in GROUP_TO_RUNS.values() for run in runs]
    valid_cols = [c for c in qty_cols if c in df_raw.columns]

    df_log = np.log2(df_raw[valid_cols].replace(0, np.nan))
    df_norm = normalize_median(df_log)

    print("=== NORMALIZATION ON PROTEOMICS DATA ===")
    print(f"Runs Processed: {df_log.shape[1]}, Proteins: {df_log.shape[0]}")

    out_path = MODULE_DIR / "04_normalization_and_scaling.png"
    plot_normalization_comparison(df_log, df_norm, out_path)
    print(f"Proteomics normalization plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_norm()
