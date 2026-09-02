from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    d = cohens_d(x, y)
    df = len(x) + len(y) - 2
    j = 1 - (3 / (4 * df - 1))
    return float(d * j)


def plot_effect_sizes(effect_dict: dict[str, float], outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(effect_dict.keys())
    values = list(effect_dict.values())
    colors = ["#2C7FB8" if v >= 0 else "#D9534F" for v in values]

    bars = ax.barh(names, values, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linestyle="-", lw=1)
    ax.axvline(0.2, color="gray", linestyle=":", label="Small (0.2)")
    ax.axvline(0.5, color="orange", linestyle="--", label="Medium (0.5)")
    ax.axvline(0.8, color="red", linestyle="-.", label="Large (0.8)")

    for bar in bars:
        w = bar.get_width()
        xpos = w + (0.05 if w >= 0 else -0.15)
        ax.text(xpos, bar.get_y() + bar.get_height()/2.0, f"{w:.2f}", va="center", fontweight="bold", fontsize=9)

    ax.set_xlabel("Effect Size (Cohen's d / Hedges' g)", fontweight="bold")
    ax.set_title("Biomarker Effect Size Magnitudes", fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    g_ctrl = np.random.normal(10, 1.2, 30)
    g_ta = np.random.normal(11.5, 1.3, 30)
    g_tb = np.random.normal(9.0, 1.1, 30)

    effects = {
        "Treatment_A vs Control (Cohen d)": cohens_d(g_ta, g_ctrl),
        "Treatment_A vs Control (Hedges g)": hedges_g(g_ta, g_ctrl),
        "Treatment_B vs Control (Cohen d)": cohens_d(g_tb, g_ctrl),
        "Treatment_B vs Control (Hedges g)": hedges_g(g_tb, g_ctrl),
    }
    for k, v in effects.items():
        print(f"  {k:<35}: {v:.4f}")

    out_dir = Path(__file__).parent
    outpath = out_dir / "09_2_effect_size_analysis.png"
    plot_effect_sizes(effects, outpath)
    print(f"Effect size plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
