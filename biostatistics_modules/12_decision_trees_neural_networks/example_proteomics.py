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
mod = importlib.import_module("12_decision_trees_neural_networks")
plot_tree_and_nn_decision = mod.plot_tree_and_nn_decision


def run_proteomics_trees_nn() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    ctrl_runs = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Control"]]
    ta_runs = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Treatment_A"]]

    c_mat = df_raw[ctrl_runs].T
    ta_mat = df_raw[ta_runs].T

    X = pd.concat([c_mat, ta_mat], axis=0).fillna(0).values[:, :10]
    y = np.array([0] * len(ctrl_runs) + [1] * len(ta_runs))

    out_path = MODULE_DIR / "12_decision_trees_and_neural_nets.png"
    plot_tree_and_nn_decision(X, y, out_path)
    print(f"Proteomics Decision Tree / NN plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_trees_nn()
