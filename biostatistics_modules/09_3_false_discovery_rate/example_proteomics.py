from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

mod = importlib.import_module("09_3_false_discovery_rate")
plot_multiple_correction_benchmark = mod.plot_multiple_correction_benchmark
run_multiple_testing_correction = mod.run_multiple_testing_correction


def run_proteomics_fdr() -> None:
    try:
        df = load_template()
    except Exception:
        df = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df)

    p_values = df["PG.Pvalue"].to_numpy()
    print("=== MULTIPLE TESTING CORRECTION ON PROTEOMICS DATA ===")
    print(f"Total Proteins Tested: {len(p_values)}")

    df_corrected = run_multiple_testing_correction(p_values, alpha=0.05)

    out_dir = Path(__file__).parent
    out_path = out_dir / "09_3_false_discovery_rate.png"
    plot_multiple_correction_benchmark(df_corrected, alpha=0.05, outpath=out_path)
    print(f"Figure saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_fdr()
