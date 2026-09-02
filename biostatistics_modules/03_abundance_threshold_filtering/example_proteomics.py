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
mod = importlib.import_module("03_abundance_threshold_filtering")
filter_abundance_and_missingness = mod.filter_abundance_and_missingness
plot_filtering_summary = mod.plot_filtering_summary


def run_proteomics_filtering() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    group_qty_cols = {g: [f"{r}.raw.PG.Quantity" for r in runs] for g, runs in GROUP_TO_RUNS.items()}
    all_qty = [c for cols in group_qty_cols.values() for c in cols if c in df_raw.columns]

    df_qty = df_raw[all_qty].copy()
    df_log = np.log2(df_qty.replace(0, np.nan))

    df_kept, df_discarded = filter_abundance_and_missingness(df_log, group_qty_cols, min_valid_prop_per_group=0.66, min_mean_intensity=5.0)

    print("=== ABUNDANCE & MISSINGNESS FILTERING ON PROTEOMICS DATA ===")
    print(f"Total Initial Proteins: {len(df_raw)}")
    print(f"Proteins Passing Filter: {len(df_kept)} ({len(df_kept)/len(df_raw):.1%})")
    print(f"Proteins Removed:        {len(df_discarded)}")

    out_path = MODULE_DIR / "03_abundance_filtering.png"
    plot_filtering_summary(df_log, df_kept, out_path)
    print(f"Proteomics filtering plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_filtering()
