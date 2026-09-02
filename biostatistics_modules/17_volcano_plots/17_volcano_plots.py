from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_volcano(
    df: pd.DataFrame,
    contrast_name: str,
    outdir: Path,
    padj_thr: float = 0.05,
    log2fc_thr: float = 1.0,
    top_n: int = 10,
    gene_col: str = "gene_name",
    log2fc_col: str = "log2FoldChange",
    padj_col: str = "padj",
) -> Path:
    d = df.copy()
    d["neg_log10_padj"] = -np.log10(d[padj_col].clip(lower=1e-300))

    conditions = [
        (d[padj_col] <= padj_thr) & (d[log2fc_col] >= log2fc_thr),
        (d[padj_col] <= padj_thr) & (d[log2fc_col] <= -log2fc_thr),
    ]
    choices = ["Up", "Down"]
    d["regulation"] = np.select(conditions, choices, default="Not significant")

    plt.figure(figsize=(9.5, 7.5))

    categories = [("Up", "#D9534F"), ("Down", "#2C7FB8"), ("Not significant", "#B0B0B0")]
    for label, color in categories:
        mask = d["regulation"] == label
        plt.scatter(
            d.loc[mask, log2fc_col],
            d.loc[mask, "neg_log10_padj"],
            c=color,
            s=35 if label != "Not significant" else 15,
            alpha=0.75 if label != "Not significant" else 0.4,
            label=f"{label} ({mask.sum()})",
            edgecolors="black" if label != "Not significant" else "none",
            linewidth=0.5 if label != "Not significant" else 0,
        )

    top_up = d[d["regulation"] == "Up"].sort_values([padj_col, log2fc_col], ascending=[True, False]).head(top_n)
    top_down = d[d["regulation"] == "Down"].sort_values([padj_col, log2fc_col], ascending=[True, True]).head(top_n)
    top_labels = pd.concat([top_up, top_down])

    for _, row in top_labels.iterrows():
        g_name = str(row[gene_col]) if gene_col in row else str(row.name)
        plt.annotate(
            g_name,
            xy=(row[log2fc_col], row["neg_log10_padj"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            alpha=0.9,
        )

    plt.axvline(log2fc_thr, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(-log2fc_thr, color="black", linestyle="--", linewidth=0.8)
    plt.axhline(-np.log10(padj_thr), color="black", linestyle="--", linewidth=0.8)

    n_up = int((d["regulation"] == "Up").sum())
    n_down = int((d["regulation"] == "Down").sum())

    plt.text(
        0.03,
        0.95,
        f"Significant Up: {n_up}\nSignificant Down: {n_down}\nCut-offs: |log2FC| $\\geq$ {log2fc_thr}, padj $\\leq$ {padj_thr}",
        transform=plt.gca().transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#CCCCCC"),
    )

    plt.xlabel("log₂ Fold Change", fontsize=11, fontweight="bold")
    plt.ylabel("-log₁₀(Adjusted P-value)", fontsize=11, fontweight="bold")
    plt.title(f"Volcano Plot: {contrast_name}", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, fontsize=9.5, loc="upper right")
    plt.grid(alpha=0.25)

    outdir.mkdir(exist_ok=True, parents=True)
    outpath = outdir / "17_volcano_plots.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    return outpath


def main() -> None:
    out_dir = Path(__file__).parent
    np.random.seed(42)
    n = 1000
    
    # Generate background non-significant proteins
    log2fc = np.random.normal(0, 0.6, n)
    pvals = np.random.uniform(0.06, 1.0, n)
    
    # Instate 45 clear UP and 45 clear DOWN significant proteins
    up_idx = np.random.choice(n, size=45, replace=False)
    log2fc[up_idx] = np.random.uniform(1.5, 4.0, 45)
    pvals[up_idx] = 10 ** np.random.uniform(-10, -2.5, 45)

    remaining = [i for i in range(n) if i not in up_idx]
    down_idx = np.random.choice(remaining, size=45, replace=False)
    log2fc[down_idx] = np.random.uniform(-4.0, -1.5, 45)
    pvals[down_idx] = 10 ** np.random.uniform(-10, -2.5, 45)

    df = pd.DataFrame({
        "gene_name": [f"GENE_{i+1:04d}" for i in range(n)],
        "log2FoldChange": log2fc,
        "padj": pvals,
    })
    outpath = plot_volcano(df, "Treatment_vs_Control", out_dir)
    print(f"Volcano plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
