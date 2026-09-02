from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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
mod = importlib.import_module("14_model_evaluation_metrics")
plot_model_metrics = mod.plot_model_metrics


def run_proteomics_eval() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    c_runs = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Control"]]
    ta_runs = [f"{r}.raw.PG.Quantity" for r in GROUP_TO_RUNS["Treatment_A"]]

    X = df_raw[c_runs + ta_runs].T.fillna(0).values[:, :5]
    y = np.array([0]*len(c_runs) + [1]*len(ta_runs))

    clf = LogisticRegression()
    clf.fit(X, y)
    y_score = clf.predict_proba(X)[:, 1]

    out_path = MODULE_DIR / "14_model_evaluation_metrics.png"
    plot_model_metrics(y, y_score, out_path)
    print(f"Proteomics model evaluation plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_eval()
