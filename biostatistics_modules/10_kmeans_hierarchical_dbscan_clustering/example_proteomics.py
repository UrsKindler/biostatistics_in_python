from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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
mod = importlib.import_module("10_kmeans_hierarchical_dbscan_clustering")
plot_clustering_comparison = mod.plot_clustering_comparison


def run_proteomics_clustering() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    qty_cols = [f"{r}.raw.PG.Quantity" for runs in GROUP_TO_RUNS.values() for r in runs]
    valid_cols = [c for c in qty_cols if c in df_raw.columns]

    mat = df_raw[valid_cols].T
    mat_log = np.log2(mat.replace(0, np.nan)).fillna(0)

    # Dimensionality reduction to 2D for clustering visual
    pca = PCA(n_components=2)
    coords = pca.fit_transform(mat_log.values)

    sample_to_group = {}
    for group, runs in GROUP_TO_RUNS.items():
        for run in runs:
            sample_to_group[f"{run}.raw.PG.Quantity"] = group
    groupings = np.array([sample_to_group.get(s, "Unknown") for s in mat.index])

    print("=== UNSUPERVISED CLUSTERING ON PROTEOMICS SAMPLES ===")
    out_path = MODULE_DIR / "10_unsupervised_clustering.png"
    plot_clustering_comparison(coords, groupings, out_path)
    print(f"Proteomics clustering figure saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_clustering()
