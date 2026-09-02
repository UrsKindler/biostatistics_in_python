from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
mod = importlib.import_module("15_random_forests")
plot_rf_feature_importance = mod.plot_rf_feature_importance


def run_proteomics_rf() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    c_runs = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Control"]]
    ta_runs = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Treatment_A"]]

    X_mat = df_raw[c_runs + ta_runs].T.fillna(0)
    y = np.array([0]*len(c_runs) + [1]*len(ta_runs))

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_mat.values, y)

    gene_names = df_raw["PG.Genes"].tolist()
    out_path = MODULE_DIR / "15_random_forests.png"
    plot_rf_feature_importance(rf, X_mat.values, y, gene_names, out_path)
    print(f"Proteomics Random Forest plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_rf()
