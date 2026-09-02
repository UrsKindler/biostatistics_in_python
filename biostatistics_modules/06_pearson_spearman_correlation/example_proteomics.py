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
mod = importlib.import_module("06_pearson_spearman_correlation")
plot_correlation_heatmaps = mod.plot_correlation_heatmaps


def run_proteomics_correlation() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    qty_cols = [f"{run}.raw.PG.Quantity" for runs in GROUP_TO_RUNS.values() for run in runs]
    valid_cols = [c for c in qty_cols if c in df_raw.columns]

    # Sample-to-sample correlation
    df_log = np.log2(df_raw[valid_cols].replace(0, np.nan))
    sample_df = df_log.dropna()

    out_path = MODULE_DIR / "06_correlation_analysis.png"
    plot_correlation_heatmaps(sample_df, out_path)
    print(f"Proteomics sample correlation plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_correlation()
