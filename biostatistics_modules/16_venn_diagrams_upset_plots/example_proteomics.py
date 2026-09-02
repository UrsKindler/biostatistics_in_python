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
mod = importlib.import_module("16_venn_diagrams_upset_plots")
plot_venn_and_upset = mod.plot_venn_and_upset
summarize_venn_regions = mod.summarize_venn_regions


def run_proteomics_venn() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    sets_dict = {}
    for group, runs in GROUP_TO_RUNS.items():
        qty_cols = [f"{r}.raw.PG.Quantity" for r in runs if f"{r}.raw.PG.Quantity" in df_raw.columns]
        # Protein is identified if observed in >= 2 replicates
        mask = df_raw[qty_cols].notna().sum(axis=1) >= 2
        sets_dict[group] = set(df_raw.loc[mask, "PG.Genes"])

    summary_df = summarize_venn_regions(sets_dict, set_type="Identified Proteins")
    print("=== PROTEOMICS IDENTIFICATION OVERLAPS (VENN) ===")
    print(summary_df)

    out_path = MODULE_DIR / "16_venn_upset_plots.png"
    plot_venn_and_upset(sets_dict, out_path)
    print(f"Proteomics Venn diagram saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_venn()
