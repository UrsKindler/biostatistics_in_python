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
mod = importlib.import_module("13_cross_validation")
plot_cv_schemes = mod.plot_cv_schemes


def run_proteomics_cv() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    runs = [r for runs_list in GROUP_TO_RUNS.values() for r in runs_list]
    y = np.array([0]*3 + [1]*3 + [2]*3)

    out_path = MODULE_DIR / "13_cross_validation.png"
    plot_cv_schemes(len(runs), y, out_path)
    print(f"Proteomics cross-validation scheme saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_cv()
