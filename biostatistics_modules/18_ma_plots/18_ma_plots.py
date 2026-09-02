from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ma(
    df: pd.DataFrame,
    contrast_name: str,
    outdir: Path,
    log2fc_thr: float = 1.0,
    padj_thr: float = 0.05,
    base_mean_col: str = "baseMean",
    log2fc_col: str = "log2FoldChange",
    padj_col: str = "padj",
) -> Path:
    d = df.copy()

    conditions = [
        (d[padj_col] <= padj_thr) & (d[log2fc_col] >= log2fc_thr),
        (d[padj_col] <= padj_thr) & (d[log2fc_col] <= -log2fc_thr),
    ]
    choices = ["Up", "Down"]
    d["regulation"] = np.select(conditions, choices, default="Not significant")

    plt.figure(figsize=(9.5, 6.5))
    categories = [("Up", "#D9534F"), ("Down", "#2C7FB8"), ("Not significant", "#B0B0B0")]

    for label, color in categories:
        mask = d["regulation"] == label
        plt.scatter(
            d.loc[mask, base_mean_col].clip(lower=1e-3),
            d.loc[mask, log2fc_col],
            c=color,
            s=30 if label != "Not significant" else 14,
            alpha=0.8 if label != "Not significant" else 0.35,
            label=f"{label} ({mask.sum()})",
            edgecolors="black" if label != "Not significant" else "none",
            linewidth=0.4 if label != "Not significant" else 0,
        )

    plt.xscale("log")
    plt.axhline(0, color="black", linewidth=1.0)
    plt.axhline(log2fc_thr, color="black", linestyle="--", linewidth=0.8)
    plt.axhline(-log2fc_thr, color="black", linestyle="--", linewidth=0.8)

    plt.xlabel("Mean Intensity / Abundance (baseMean, log scale)", fontsize=11, fontweight="bold")
    plt.ylabel("log₂ Fold Change (M)", fontsize=11, fontweight="bold")
    plt.title(f"MA Plot (Abundance vs. Ratio): {contrast_name}", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True, fontsize=9.5)
    plt.grid(alpha=0.25)

    outdir.mkdir(exist_ok=True, parents=True)
    outpath = outdir / "18_ma_plots.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    return outpath


def main() -> None:
    out_dir = Path(__file__).parent
    np.random.seed(42)
    n = 1000
    base_mean = 10 ** np.random.uniform(2.0, 6.5, n)
    log2fc = np.random.normal(0, 0.45, n)
    padj = np.random.uniform(0.08, 1.0, n)

    # 40 UP and 40 DOWN
    up_idx = np.random.choice(n, size=40, replace=False)
    log2fc[up_idx] = np.random.uniform(1.5, 3.8, 40)
    padj[up_idx] = 10 ** np.random.uniform(-8, -2.5, 40)

    remaining = [i for i in range(n) if i not in up_idx]
    down_idx = np.random.choice(remaining, size=40, replace=False)
    log2fc[down_idx] = np.random.uniform(-3.8, -1.5, 40)
    padj[down_idx] = 10 ** np.random.uniform(-8, -2.5, 40)

    df = pd.DataFrame({
        "baseMean": base_mean,
        "log2FoldChange": log2fc,
        "padj": padj,
    })
    outpath = plot_ma(df, "Treatment_vs_Control", out_dir)
    print(f"MA plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
