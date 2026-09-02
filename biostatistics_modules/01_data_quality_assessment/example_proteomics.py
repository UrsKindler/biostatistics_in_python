from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Random_Proteomics_Dataset_Generator import (
    GROUP_TO_RUNS,
    create_random_proteomics_table,
    load_template,
    save_template,
)


def run_proteomics_qc() -> None:
    try:
        df_raw = load_template()
    except Exception:
        df_raw = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df_raw)

    qty_cols = [f"{run}.raw.PG.Quantity" for runs in GROUP_TO_RUNS.values() for run in runs]
    valid_cols = [c for c in qty_cols if c in df_raw.columns]

    df_qty = df_raw[valid_cols].copy()
    df_log = np.log2(df_qty.replace(0, np.nan))

    print("=== DATA QUALITY ASSESSMENT ON PROTEOMICS DATA ===")
    print(f"Proteins: {df_qty.shape[0]}, Samples: {df_qty.shape[1]}")
    print(f"Total Missing Values: {df_qty.isna().sum().sum()} ({df_qty.isna().sum().sum()/df_qty.size:.1%})")

    out_dir = Path(__file__).parent
    outpath = out_dir / "01_data_quality_assessment.png"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    msno.matrix(df_log, ax=axes[0], sparkline=False, fontsize=8, color=(0.17, 0.49, 0.72))
    axes[0].set_title("A: Missingness Matrix (Proteins x Samples)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mass Spec Runs / Samples", fontsize=10)
    axes[0].set_ylabel("Protein Groups", fontsize=10)

    sample_subset = valid_cols[:8]
    sns.boxplot(data=df_log[sample_subset], ax=axes[1], palette="Blues_r", fliersize=3)
    axes[1].set_title("B: Log2 Intensity Distribution & Outliers (Selected Runs)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Log2(Quantity)", fontsize=10)
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Proteomics QC plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    run_proteomics_qc()
